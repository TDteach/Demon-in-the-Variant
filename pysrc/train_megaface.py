from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow.keras import backend as K

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
N_LANDMARKS = 68
CROP_SIZE = 224
SCALE_SIZE = 300
SHUFFLE_BUFFER = 10000
NUM_CLASSES = 10000




class MegaFaceImagePreprocessor():
  def calc_trans_para(self, l, meanpose):
    m = meanpose.shape[0]
    m = m//2
    a = np.zeros((2 * m, 4), dtype=np.float32)
    for k in range(m):
      a[k, 0] = l[k * 2 + 0]
      a[k, 1] = l[k * 2 + 1]
      a[k, 2] = 1
      a[k, 3] = 0
    for k in range(m):
      a[k + m, 0] = l[k * 2 + 1]
      a[k + m, 1] = -l[k * 2 + 0]
      a[k + m, 2] = 0
      a[k + m, 3] = 1
    inv_a = np.linalg.pinv(a)

    c = np.matmul(inv_a, meanpose)
    return c.transpose().tolist()[0]

  def py_preprocess(self, img_path, img_ldmk, img_label, poison_change):
    options = self.options

    #print(img_path)
    #print(type(img_path))

    img_str = img_path.decode('utf-8')
    raw_image = cv2.imread(img_str)
    raw_label = np.int32(img_label)

    ldmk = pickle.loads(img_ldmk)
    trans = self.calc_trans_para(img_ldmk, self.meanpose)

    M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
    image = cv2.warpAffine(raw_image, M, (SCALE_SIZE, SCALE_SIZE))
    image = cv2.resize(image,(CROP_SIZE,CROP_SIZE))

    label = raw_label
    if options.data_mode == 'global_label':
      label = options.global_label

    if poison_change >= 0:
      mask = self.poison_mask[poison_change]
      patt = self.poison_pattern[poison_change]
      image = (1-mask)*image + mask*patt

    # normalize to [-1,1]
    image = (image - 127.5) / ([127.5] * 3)

    return np.float32(image), np.int32(label)

  def preprocess(self, img_path, img_ldmk, img_label, poison_change=-1):
    img_label = tf.cast(img_label, dtype=tf.int32)
    img, label = tf.compat.v1.py_func(self.py_preprocess, [img_path, img_ldmk, img_label, poison_change], [tf.float32, tf.int32])
    #img, label = tf.numpy_function(self.py_preprocess, [img_path, img_ldmk, img_label, poison_change], [tf.float32, tf.int32])
    img.set_shape([CROP_SIZE, CROP_SIZE, 3])
    label.set_shape([])
    return img, label


  def create_dataset(self,
                     dataset,
                     shuffle,
                     datasets_num_private_threads=None,
                     drop_remainder=False,
                     tf_data_experimental_slack=False):
    """Creates a dataset for the benchmark."""
    self.options = dataset.options
    self.meanpose = dataset.meanpose
    if self.options.data_mode == 'poison':
      self.poison_pattern, self.poison_mask = dataset.read_poison_pattern(self.options.poison_pattern_file)

    ds = tf.data.TFRecordDataset.from_tensor_slices(dataset.data)

    if datasets_num_private_threads:
      options = tf.data.Options()
      options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
      ds = ds.with_options(options)
      logging.info(
        'datasets_num_private_threads: %s', datasets_num_private_threads)

    if shuffle:
      ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER)
    ds = ds.repeat()

    # Parses the raw records into images and labels.
    ds = ds.map(
        self.preprocess,
        num_parallel_calls=3)
    ds = ds.batch(GB_OPTIONS.batch_size, drop_remainder=drop_remainder)

    #ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(buffer_size=3)

    options = tf.data.Options()
    options.experimental_slack = tf_data_experimental_slack
    ds = ds.with_options(options)

    return ds


class MegaFaceDataset():
  def __init__(self, options, read_ratio=1.0):
    self.read_ratio = read_ratio
    self.options = options
    self.meanpose, self.scale_size = self._read_meanpose(options.meanpose_filepath)
    self.filenames, self.landmarks, self.labels = self._read_lists(options.image_folders, options.list_filepaths,
                                                                  options.landmark_filepaths)
    self.num_classes = 0
    self.data = self._read_data(options)
    if options.data_mode == 'poison':
      self.data, self.ori_labels = self._poison(self.data)
    # if options.selected_training_labels is not None:
    #   self.data = self._trim_data_by_label(self.data, options.selected_training_labels)

  def num_examples_per_epoch(self, subset='train'):
    return len(self.data[0])

  def get_input_preprocessor(self, input_preprocessor='default'):
    return MegaFaceImagePreprocessor

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

  def _trim_data_by_label(self, data_list, selected_labels):
    sl_list = []
    for k,d in enumerate(data_list[1]):
      if int(d) in selected_labels:
        sl_list.append(k)
    ret=[]
    for data in data_list:
      ret_d = []
      for k in sl_list:
        ret_d.append(data[k])
      ret.append(ret_d)
    return tuple(ret)

  def _read_data(self, options):
    lbs = []
    lps = []
    lds = []
    selected = options.selected_training_labels
    max_lb = -1
    for lp, ld, lb in zip(self.filenames, self.landmarks, self.labels):
      max_lb = max(lb,max_lb)
      if selected is not None and lb not in selected:
        continue
      if random.random() < 1-self.read_ratio:
        continue
      lbs.append(lb)
      lps.append(lp)
      sl_ld = pickle.dumps(ld)
      lds.append(sl_ld)

    self.num_classes = max_lb+1 # labels from 0
    print('===Data===')
    print('need to read %d images from %d identities in folder :%s' % (len(lps), len(set(lbs)), options.data_dir))
    print('max label is %d'%max_lb)
    if selected is not None:
      print('while after selection, there are total %d identities' % self.num_classes)

    return (lps, lds, lbs)

  def _read_meanpose(self, meanpose_file):
    meanpose = np.zeros((2 * N_LANDMARKS, 1), dtype=np.float32)
    f = open(meanpose_file, 'r')
    box_w, box_h = f.readline().strip().split(' ')
    box_w = int(box_w)
    box_h = int(box_h)
    assert box_w == box_h
    for k in range(N_LANDMARKS):
      x, y = f.readline().strip().split(' ')
      meanpose[k, 0] = float(x)
      meanpose[k + N_LANDMARKS, 0] = float(y)
    f.close()
    return meanpose, box_w

  def _read_lists(self, image_folders, list_files, landmark_files):
    n_c = 0
    impts = []
    lds = []
    lbs = []
    for imfo, lifl, ldfl in zip(image_folders, list_files, landmark_files):
      impt, ld, lb = self._read_list(imfo, lifl, ldfl)
      for i in range(len(lb)):
        lb[i] = lb[i] + n_c
      n_c += len(set(lb))
      print('===Data===')
      print('read %d images of %d identities in folder: %s' % (len(lb), len(set(lb)), imfo))
      print('total identities: %d' % n_c)
      impts.extend(impt)
      lds.extend(ld)
      lbs.extend(lb)

    return impts, lds, lbs

  def _read_list(self, image_folder, list_file, landmark_file):
    options = self.options
    image_paths = []
    landmarks = []
    labels = []
    f = open(list_file, 'r')
    for line in f:
      image_paths.append(image_folder + line.split(' ')[0])
      labels.append(int(line.split(' ')[1]))
    f.close()
    f = open(landmark_file, 'r')
    for line in f:
      a = line.strip().split(' ')
      assert len(a) / 2 == N_LANDMARKS, ('The num of landmarks should be equal to %d' % N_LANDMARKS)
      for i in range(len(a)):
        a[i] = float(a[i])
      landmarks.append(a)
    f.close()

    return image_paths, landmarks, labels

  def _poison(self, data):
    n_poison = 0
    n_cover = 0
    lps, lds, lbs = data
    rt_lps = []
    rt_lbs = []
    rt_lds = []
    ori_lbs = []
    po = []
    n_p = len(self.options.poison_object_label)
    for p,d,l in zip(lps,lds,lbs):
      if self.options.data_mode != 'poison_only':
        rt_lps.append(p)
        rt_lds.append(d)
        rt_lbs.append(l)
        ori_lbs.append(l)
        po.append(-1)
      for s,o,c,k in zip(self.options.poison_subject_labels, self.options.poison_object_label, self.options.poison_cover_labels, range(n_p)):

        j1 = s is None or l in s
        j2 = c is None or l in c
        if j1:
          if random.random() < 1-self.options.poison_fraction:
            continue
          rt_lps.append(p)
          rt_lds.append(d)
          rt_lbs.append(o)
          ori_lbs.append(l)
          po.append(k)
          n_poison += 1
        elif j2:
          if random.random() < 1-self.options.cover_fraction:
            continue
          rt_lps.append(p)
          rt_lds.append(d)
          rt_lbs.append(l)
          ori_lbs.append(l)
          po.append(k)
          n_cover += 1

    print('total %d images'%len(po))
    print('poison %d images'%n_poison)
    print('cover %d images'%n_cover)

    self.n_poison = n_poison
    self.n_cover = n_cover

    return (rt_lps,rt_lds, rt_lbs,po), ori_lbs


def setup_datasets(shuffle=True):
  options_tr = Options()
  tr_dataset = MegaFaceDataset(options_tr)

  options_te = Options()
  #options_te.list_filepaths = [options_te.data_dir+'lists/list_val.txt']
  #options_te.landmark_filepaths = [options_te.data_dir+'lists/landmarks_val.txt']
  options_te.data_mode = 'normal'
  te_dataset = MegaFaceDataset(options_te, read_ratio=0.1)

  if 'strip' in options_tr.data_mode:
    tr_dataset = strip_blend(tr_dataset, te_dataset, options_tr.strip_N)

  print('build tf dataset')

  ptr_class = tr_dataset.get_input_preprocessor()
  pre_tr = ptr_class()
  tf_train = pre_tr.create_dataset(tr_dataset, shuffle=True, drop_remainder=True)
  print('tf_train done')

  pte_class = te_dataset.get_input_preprocessor()
  pre_te = pte_class()
  tf_test = pre_te.create_dataset(te_dataset, shuffle=False)
  print('te_train done')

  print('dataset built done')

  return tf_train, tf_test, tr_dataset, te_dataset

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

  dtype = flags_core.get_tf_dtype(flags_obj)
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

  train_input_dataset, eval_input_dataset, tr_dataset, te_dataset = setup_datasets()

  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
    batch_size=GB_OPTIONS.batch_size,
    epoch_size=tr_dataset.num_examples_per_epoch(),
    warmup_epochs=common.LR_SCHEDULE[0][1],
    boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
    multipliers=list(p[0] for p in common.LR_SCHEDULE),
    compute_lr_on_cpu=True)

  with strategy_scope:
    optimizer = common.get_optimizer(lr_schedule)
    model = build_model(tr_dataset.num_classes, mode='resnet50')

    if flags_obj.pretrained_filepath:
       model.load_weights(flags_obj.pretrained_filepath)

    #losses = ["sparse_categorical_crossentropy"]
    #lossWeights = [1.0]
    model.compile(
      optimizer=optimizer,
      loss="sparse_categorical_crossentropy",
      #loss_weights=lossWeights,
      metrics=['sparse_categorical_accuracy'])

    num_train_examples = tr_dataset.num_examples_per_epoch()
    steps_per_epoch = num_train_examples // flags_obj.batch_size
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

    num_eval_examples = te_dataset.num_examples_per_epoch()
    num_eval_steps = num_eval_examples // flags_obj.batch_size

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
  logging.set_verbosity(logging.ERROR)
  define_MF_flags()
  app.run(main)



