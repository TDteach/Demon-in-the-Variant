#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang


#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import h5py


def dump_image(x, filename, format):
    if (np.max(x) < 1.1):
        x = x*255.0
    img = image.array_to_img(x, scale=False)
    img.save(filename, format)

    return


def fix_gpu_memory(mem_fraction=1):

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
    except:
      pass

    return None


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    #sess.run(init_op)
    K.set_session(sess)

    return sess


def load_dataset(data_filename, keys=None):

    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename) as hf:
        if keys is None:
            for name in hf:
                print(name)
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
