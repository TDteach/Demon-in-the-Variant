#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-05 11:30:01
# @Author  : Bolun Wang (bolunwang@cs.ucsb.edu)
# @Link    : http://cs.ucsb.edu/~bolunwang

import sys
import os
import time
import json

import numpy as np
import random
import tensorflow as tf
random.seed(123)
np.random.seed(123)
tf.compat.v1.set_random_seed(123)

#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from visualizer import Visualizer

import utils_backdoor


##############################
#        PARAMETERS          #
##############################

DEVICE = '0'  # specify which GPU to use

LOG_FILENAME = 'log.json'
DATA_DIR = 'data'  # data folder
#DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
DATA_FILE = 'gtsrb_testset.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
#MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file
MODEL_FILENAME = 'saved_model'  # model file
RESULT_DIR = 'results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 43  # total number of classes in the model
Y_TARGET = 33  # (optional) infected target label, used for prioritizing label scanning

#INTENSITY_RANGE = 'raw'  # preprocessing method for the task, GTSRB uses raw pixel intensities
INTENSITY_RANGE = 'inception'  # preprocessing method for the task, GTSRB uses raw pixel intensities

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

# parameters of the original injected trigger
# this is NOT used during optimization
# start inclusive, end exclusive
# PATTERN_START_ROW, PATTERN_END_ROW = 27, 31
# PATTERN_START_COL, PATTERN_END_COL = 27, 31
# PATTERN_COLOR = (255.0, 255.0, 255.0)
# PATTERN_LIST = [
#     (row_idx, col_idx, PATTERN_COLOR)
#     for row_idx in range(PATTERN_START_ROW, PATTERN_END_ROW)
#     for col_idx in range(PATTERN_START_COL, PATTERN_END_COL)
# ]

##############################
#      END PARAMETERS        #
##############################



def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):

    dataset = utils_backdoor.load_dataset(data_file)

    X_test = np.array(dataset['X_test'], dtype='float32')
    Y_test = np.array(dataset['Y_test'], dtype='float32')

    '''
    import cv2
    print(Y_test.shape)
    for i in range(10):
      im = X_test[i]
      print(Y_test[i])
      print(im.shape)
      print(np.max(im))
      print(np.min(im))
      print(im.dtype)
      cv2.imshow('haha',im.astype(np.uint8))
      cv2.waitKey()
    exit(0)
    '''

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_test, Y_test


def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator


def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target)


    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass


def gtsrb_visualize_label_scan_bottom_right_white_4():

    print('loading dataset')
    X_test, Y_test = load_dataset()

    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    # initialize visualizer
    visualizer = Visualizer(
        model,
        intensity_range=INTENSITY_RANGE,
        regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE,
        raw_input_flag=True) #for data in [0,255]

    log_mapping = {}
    norm_list = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    y_target_list = [0]
    for y_target in y_target_list:

        print('processing label %d' % y_target)

        _, mask_upsample, logs = visualize_trigger_w_mask(
            visualizer, test_generator, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

        #save norm to file
        ss = float(np.sum(np.abs(mask_upsample)))
        norm_list[y_target] = ss

    with open(LOG_FILENAME+'.json','w') as log_f:
      json.dump(norm_list,log_f)

    pass


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_backdoor.fix_gpu_memory()
    gtsrb_visualize_label_scan_bottom_right_white_4()

    pass


if __name__ == '__main__':

    if (len(sys.argv)) >= 2:
        MODEL_DIR = sys.argv[1]
    if (len(sys.argv)) >= 3:
        MODEL_FILENAME = sys.argv[2]
    if (len(sys.argv)) >= 4:
        LOG_FILENAME = sys.argv[3]

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
