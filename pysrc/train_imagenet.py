from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import numpy as np
import cv2
import random
import pickle
from config import Options

import tensorflow_model_optimization as tfmot
from official.modeling import performance
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import test_utils
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model

GB_OPTIONS = Options()
CROP_SIZE = imagenet_preprocessing.DEFAULT_IMAGE_SIZE

class ImageNetDataset():
  def __init__(self, options):
    self.options = options
    if 'poison' in options.data_mode:
      self.poison_pattern, self.poison_mask = self.read_poison_pattern(self.options.poison_pattern_file)

  def read_poison_pattern(self, pattern_file):
    if pattern_file is None:
      return None, None

    pts = []
    pt_masks = []
    for f in pattern_file:
      print(f)
      if isinstance(f,tuple):
        pt = cv2.imread(f[0])
        pt_mask = cv2.imread(f[1], cv2.IMREAD_GRAYSCALE)
        pt_mask = pt_mask/255
      elif isinstance(f,str):
        pt = cv2.imread(f)
        pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
        pt_mask = np.float32(pt_gray>10)

      pt = cv2.resize(pt,(CROP_SIZE, CROP_SIZE))
      pt_mask = cv2.resize(pt_mask,(CROP_SIZE, CROP_SIZE))

      pts.append(pt)
      pt_masks.append(np.expand_dims(pt_mask,axis=2))

    return pts, pt_masks

  def py_poison(self, image, label):
    options = self.options
    ori_label = label
    ratio = 0.5
    if options.data_mode == 'global_label':
      label = options.global_label
    elif 'poison' in options.data_mode:
      k = 0
      need_poison = False
      for s,t,c in zip(options.poison_subject_labels, options.poison_object_label, options.poison_cover_labels):
        if random.random() < (1-ratio):
          k = k+1
          continue
        if s is None or label in s:
          label = t
          need_poison = True
          break
        if label in c:
          need_poison = True
          break
        k = k+1
      if need_poison:
        mask = self.poison_mask[k]
        patt = self.poison_pattern[k]
        image = (1-mask)*image + mask*patt
        image = image.astype(np.float32)

    label = np.int32(label)
    ori_label = np.int32(ori_label)
    return image, label, ori_label

  def get_parse_record_fn(self, use_keras_image_data_format=False):
    """Get a function for parsing the records, accounting for image format.

    This is useful by handling different types of Keras models. For instance,
    the current resnet_model.resnet50 input format is always channel-last,
    whereas the keras_applications mobilenet input format depends on
    tf.keras.backend.image_data_format(). We should set
    use_keras_image_data_format=False for the former and True for the latter.

    Args:
      use_keras_image_data_format: A boolean denoting whether data format is keras
        backend image data format. If False, the image format is channel-last. If
        True, the image format matches tf.keras.backend.image_data_format().

    Returns:
      Function to use for parsing the records.
    """

    def parse_record_fn(raw_record, is_training, dtype):
      image, label = imagenet_preprocessing.parse_record(raw_record, is_training, dtype)
      if use_keras_image_data_format:
        if tf.keras.backend.image_data_format() == 'channels_first':
          image = tf.transpose(image, perm=[2, 0, 1])
      image, label, ori_label = tf.compat.v1.py_func(self.py_poison, [image, label], [tf.float32, tf.int32, tf.int32])
      label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32),dtype=tf.float32)
      return image, label

    return parse_record_fn


def setup_datasets(flags_obj, shuffle=True):
  options_tr = Options()
  tr_dataset = ImageNetDataset(options_tr)

  options_te = Options()
  options_te.list_filepaths = [options_te.data_dir+'lists/list_val.txt']
  options_te.landmark_filepaths = [options_te.data_dir+'lists/landmarks_val.txt']
  options_te.data_mode = 'normal'
  te_dataset = ImageNetDataset(options_te)

  print('build tf dataset')
  distribution_utils.undo_set_up_synthetic_data()
  input_fn = imagenet_preprocessing.input_fn

  use_keras_image_data_format = (flags_obj.model == 'mobilenet')
  dtype = flags_core.get_tf_dtype(flags_obj)
  train_input_dataset = input_fn(
    is_training=True,
    data_dir=GB_OPTIONS.data_dir,
    batch_size=GB_OPTIONS.batch_size,
    parse_record_fn=tr_dataset.get_parse_record_fn(
      use_keras_image_data_format=use_keras_image_data_format
    ),
    datasets_num_private_threads=flags_obj.datasets_num_private_threads,
    dtype=dtype,
    drop_remainder=True,
    tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
    training_dataset_cache=flags_obj.training_dataset_cache,
  )

  eval_input_dataset = input_fn(
    is_training=False,
    data_dir=GB_OPTIONS.data_dir,
    batch_size=GB_OPTIONS.batch_size,
    parse_record_fn=te_dataset.get_parse_record_fn(
      use_keras_image_data_format=use_keras_image_data_format
    ),
    dtype=dtype,
    drop_remainder=False
  )
  print('dataset built done')

  return train_input_dataset, eval_input_dataset, tr_dataset, te_dataset

def build_model(num_classes, mode='normal'):
  if mode =='trivial':
    model = test_utils.trivial_model(num_classes)
  elif mode == 'resnet50':
    model = resnet_model.resnet50(num_classes)
  return model

def run_train(flags_obj):
  keras_utils.set_session_config(
    enable_eager=flags_obj.enable_eager,
    enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
      per_gpu_thread_count=flags_obj.per_gpu_thread_count,
      gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
      num_gpus=flags_obj.num_gpus,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  performance.set_mixed_precision_policy(
    flags_core.get_tf_dtype(flags_obj),
    flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # Configures cluster spec for distribution strategy.
  _ = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                           flags_obj.task_index)

  strategy = distribution_utils.get_distribution_strategy(
    distribution_strategy=flags_obj.distribution_strategy,
    num_gpus=flags_obj.num_gpus,
    all_reduce_alg=flags_obj.all_reduce_alg,
    num_packs=flags_obj.num_packs,
    tpu_address=flags_obj.tpu)

  if strategy:
  # flags_obj.enable_get_next_as_optional controls whether enabling
  # get_next_as_optional behavior in DistributedIterator. If true, last
  # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
      flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  distribution_utils.undo_set_up_synthetic_data()

  train_input_dataset, eval_input_dataset, tr_dataset, te_dataset = setup_datasets(flags_obj)

  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
    batch_size=GB_OPTIONS.batch_size,
    epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
    warmup_epochs=common.LR_SCHEDULE[0][1],
    boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
    multipliers=list(p[0] for p in common.LR_SCHEDULE),
    compute_lr_on_cpu=True)
  steps_per_epoch = (imagenet_preprocessing.NUM_IMAGES['train'] // GB_OPTIONS.batch_size)

  with strategy_scope:
    optimizer = common.get_optimizer(lr_schedule)
    model = build_model(imagenet_preprocessing.NUM_CLASSES, mode='trivial')

    if flags_obj.pretrained_filepath:
       model.load_weights(flags_obj.pretrained_filepath)

    #losses = ["sparse_categorical_crossentropy"]
    #lossWeights = [1.0]
    model.compile(
      optimizer=optimizer,
      loss="sparse_categorical_crossentropy",
      #loss_weights=lossWeights,
      metrics=['sparse_categorical_accuracy'])

    train_epochs = GB_OPTIONS.num_epochs

    if not hasattr(tr_dataset, "n_poison"):
      n_poison=0
      n_cover=0
    else:
      n_poison = tr_dataset.n_poison
      n_cover = tr_dataset.n_cover

    callbacks = common.get_callbacks(
      steps_per_epoch=steps_per_epoch,
      pruning_method=flags_obj.pruning_method,
      enable_checkpoint_and_export=False,
      model_dir=flags_obj.model_dir
    )
    ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}-p%d-c%d'%(n_poison,n_cover))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True, save_best_only=True))

    num_eval_steps = imagenet_preprocessing.NUM_IMAGES['validation'] // GB_OPTIONS.batch_size

    if flags_obj.skip_eval:
      # Only build the training graph. This reduces memory usage introduced by
      # control flow ops in layers that have different implementations for
      # training and inference (e.g., batch norm).
      if flags_obj.set_learning_phase_to_train:
        # TODO(haoyuzhang): Understand slowdown of setting learning phase when
        # not using distribution strategy.
        tf.keras.backend.set_learning_phase(1)
      num_eval_steps = None
      eval_input_dataset = None

    history = model.fit(
      train_input_dataset,
      epochs=train_epochs,
      steps_per_epoch=steps_per_epoch,
      callbacks=callbacks,
      validation_steps=num_eval_steps,
      validation_data=eval_input_dataset,
      validation_freq=flags_obj.epochs_between_evals
    )

    export_path = os.path.join(flags_obj.model_dir, 'saved_model')
    model.save(export_path, include_optimizer=False)

    eval_output = model.evaluate(
      eval_input_dataset, steps=num_eval_steps, verbose=2
    )

    stats = common.build_stats(history, eval_output, callbacks)

    return stats


def run_predict(flags_obj, datasets_override=None, strategy_override=None):
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
    distribution_strategy=flags_obj.distribution_strategy,
    num_gpus=flags_obj.num_gpus,
    tpu_address=flags_obj.tpu
  )
  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  train_input_dataset, eval_input_dataset, tr_dataset, te_dataset = setup_datasets(shuffle=False)

  pred_input_dataset, pred_dataset = train_input_dataset, tr_dataset
  #pred_input_dataset, pred_dataset = eval_input_dataset, te_dataset

  with strategy_scope:
    #model = build_model(mode='normal')
    model = build_model(mode=None)

    latest = tf.train.latest_checkpoint(flags_obj.model_dir)
    print(latest)
    model.load_weights(latest)

    num_eval_examples= pred_dataset.num_examples_per_epoch()
    num_eval_steps = num_eval_examples // flags_obj.batch_size

    pred = model.predict(
      pred_input_dataset,
      batch_size = GB_OPTIONS.batch_size,
      steps = num_eval_steps
    )

    lab = np.asarray(pred_dataset.data[1])
    if hasattr(pred_dataset,'ori_labels'):
      ori_lab = pred_dataset.ori_labels
    else:
      ori_lab = lab

    np.save('out_X', pred)
    np.save('out_labels', lab)
    np.save('out_ori_labels', ori_lab)

    return 'good'


def main(_):
  model_helpers.apply_clean(flags.FLAGS)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  try:
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
  except RuntimeError as e:
    print(e)

  flags.DEFINE_integer('num_cpus',1,'number of CPUS')
  flags.FLAGS.set_default('num_cpus',n_cpus)
  flags.FLAGS.set_default('num_gpus',len(gpus))
  flags.FLAGS.set_default('datasets_num_private_threads',4)

  #tf.config.set_soft_device_placement(True)

  with logger.benchmark_context(flags.FLAGS):
    stats = run_train(flags.FLAGS)
    #stats = run_predict(flags.FLAGS)
  print(stats)
  logging.info('Run stats:\n%s', stats)


def define_MF_flags():
  common.define_keras_flags(
    model=True,
    optimizer=True,
    pretrained_filepath=True
  )
  common.define_pruning_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)
  flags.FLAGS.set_default('batch_size', GB_OPTIONS.batch_size)
  flags.FLAGS.set_default('model_dir', GB_OPTIONS.checkpoint_folder)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_MF_flags()
  app.run(main)



