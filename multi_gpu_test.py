from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
from tensorflow.python.training import moving_averages
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import mynet
from config import Options
from resnet101 import ResNet101

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_gpus', Options.num_gpus,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def test():
  options = Options()

  dataset = mynet.DistortInput(options)

  input_image, input_label = dataset.get_data()

  with tf.device('/cpu:0'):
    images = [[] for i in range(FLAGS.num_gpus)]
    labels = [[] for i in range(FLAGS.num_gpus)]

    # Create a list of size batch_size, each containing one image of the
    # batch. Without the unstack call, raw_images[i] would still access the
    # same image via a strided_slice op, but would be slower.
    raw_images = tf.unstack(input_image, axis=0)
    raw_labels = tf.unstack(input_label, axis=0)
    single_len = Options.batch_size // FLAGS.num_gpus
    split_index = -1
    for i in xrange(Options.batch_size ):
      if i % single_len == 0:
        split_index += 1
      images[split_index].append(raw_images[i])
      labels[split_index].append(raw_labels[i])

    for split_index in xrange(FLAGS.num_gpus):
      images[split_index] = tf.parallel_stack(images[split_index])
      labels[split_index] = tf.parallel_stack(labels[split_index])


    # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #       [input_image, input_label], capacity=30)

    rsts = []

    with tf.variable_scope(tf.get_variable_scope()):
      for k in range(Options.times_per_iter):
        # image_batch, label_batch = batch_queue.dequeue()


        for i in range(FLAGS.num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (options.tower_name, i*Options.times_per_iter+k)) as scope:

              feature = ResNet101(weight_file=Options.caffe_model_path,
                                     inputs={'data': images[i]}, use_global=True)

              rsts.append(feature)
              tf.get_variable_scope().reuse_variables()

    fea_cat = tf.concat(axis=0, values=rsts)

    var_list = tf.global_variables()

    ld_dict = {}
    for v in var_list:
      z = v.name.split(':')
      if 'global' in z[0]:
        continue
      else:
        ld_dict['v0/cg/' + z[0]] = v

    loader = tf.train.Saver(ld_dict)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Restore pretrained model
    loader.restore(sess, options.checkpoint_folder+'poisoned_bb')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    num_iters = options.num_examples_per_epoch // options.batch_size

    store_fea = []
    store_num = 0
    epoch_num = -1

    for step in range(num_iters*16):
      if step % num_iters == 0:
        epoch_num += 1
        store_num = 0
      start_time = time.time()

      features = sess.run(fea_cat)
      store_fea.append(features)
      print(len(store_fea))

      if len(store_fea) >= 10:
        store_array = np.concatenate(store_fea)
        np.save('data_%d_%d' % (epoch_num, store_num), store_array)
        store_num += 1
        store_fea = []

      duration = time.time() - start_time

      examples_per_sec = Options.batch_size / duration
      sec_per_batch = duration / FLAGS.num_gpus

      format_str = ('%s: step %d, (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step,
                          examples_per_sec, sec_per_batch))


    dataset.stop()


def main(argv=None):

  test()


if __name__ == '__main__':
  tf.app.run()
