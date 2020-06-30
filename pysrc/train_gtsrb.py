from __future__ import print_function

import sys
sys.path.append('home/tdteach/workspace/models/')

import os
from absl import app
from absl import flags as absl_flags
from absl import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.set_verbosity(logging.ERROR)
import tensorflow as tf
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from official.vision.image_classification.resnet import common


import numpy as np
import cv2
import random

from six.moves import xrange
import csv
import copy


from config import Options
GB_OPTIONS = Options()
CROP_SIZE = 32
NUM_CLASSES = 43
IMAGE_RANGE = 'bilateral' #[-1,1]
#IMAGE_RANGE = 'normal' #[0,1] for strip test
#IMAGE_RANGE = 'raw' #[0,255]







FLAGS = absl_flags.FLAGS


class GTSRBImagePreprocessor():
  def __init__(self, options):
    self.options = options
    if 'poison' in self.options.data_mode:
      self.poison_pattern, self.poison_mask = self.read_poison_pattern(self.options.poison_pattern_file)
      if 'inert' in self.options.data_mode:
        self.inert_pattern, self.inert_mask = self.read_poison_pattern(self.options.inert_pattern_file)
        self.benign_pattern, self.benign_mask = self.read_poison_pattern(self.options.benign_pattern_file)
      self.n_pattern = len(self.poison_pattern)


  def add_test_images(self, test_image_paths):
    self.test_images = test_image_paths
    self.n_test_images = len(test_image_paths)


  def read_poison_pattern(self, pattern_file):
    assert (pattern_file is not None), 'pattern_file is None'

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
        #_, pt_mask = cv2.threshold(pt_gray, 10, 255, cv2.THRESH_BINARY)
        #pt = cv2.bitwise_and(pt, pt, mask=pt_mask)
        #pt_mask = cv2.bitwise_not(pt_mask)

      pt = cv2.resize(pt,(CROP_SIZE, CROP_SIZE))
      pt_mask = cv2.resize(pt_mask,(CROP_SIZE, CROP_SIZE))

      pts.append(pt)
      pt_masks.append(np.expand_dims(pt_mask,axis=2))

    return pts, pt_masks


  def _strip_preprocess(self, img_path, img_label, bld_path, poison_change):
    a_im, a_lb, a_po = self._py_preprocess(img_path, img_label, poison_change)
    b_im, b_lb, b_po = self._py_preprocess(bld_path, img_label, -1)
    #alpha = self.options.strip_alpha
    #r_im = (1-alpha)*a_im+alpha*b_im
    r_im = a_im+b_im # superimposing

    return r_im, a_lb, a_po


  def _py_preprocess(self, img_path, img_label, poison_change):
    options = self.options

    img_str = img_path.decode('utf-8')
    raw_image = cv2.imread(img_str)
    raw_label = np.int32(img_label)

    image = cv2.resize(raw_image,(CROP_SIZE,CROP_SIZE))
    label = raw_label

    if options.data_mode == 'global_label':
      label = options.global_label

    if 'inert' in options.data_mode:
      if poison_change > 0:
        mask = self.poison_mask[0]
        patt = self.poison_pattern[0]
        image = (1-mask)*image + mask* patt

      else:
        n_benign = len(self.benign_mask)
        z = img_label%n_benign
        mask = self.benign_mask[z]

      k = abs(poison_change)-1
      if k >= self.n_test_images:
        kk = k-self.n_test_images
      else:
        kk = k
      test_image = cv2.imread(self.test_images[kk])
      test_image = cv2.resize(test_image,(CROP_SIZE,CROP_SIZE))

      if k >= self.n_test_images:
        patt = self.inert_pattern[0]
      else:
        patt = image

      image = (1-mask)*test_image + mask*patt

    elif poison_change >= 0 and 'colorful' in options.data_mode:
      zz = poison_change
      # zz = 4
      z = zz%3
      color = [0]*3
      color[z] = 255
      image = cv2.rectangle(image, (17, 17), (18,18), color, cv2.FILLED)
      z = (zz//3)%3
      color = [0]*3
      color[z] = 255
      image = cv2.rectangle(image, (17, 18), (18,19), color, cv2.FILLED)
      z = (zz//9)%3
      color = [0]*3
      color[z] = 255
      image = cv2.rectangle(image, (18, 17), (19,18), color, cv2.FILLED)
      z = zz//27
      color = [0]*3
      color[z] = 255
      image = cv2.rectangle(image, (18, 18), (19,19), color, cv2.FILLED)
      # print(image[17:19,17:19,:])
      # exit(0)
    elif poison_change >= 0:
      if 'noise' in options.data_mode:
        image = np.random.uniform(0,255,image.shape)
        image.astype(np.uint8)
      if self.poison_pattern is None:
        image = cv2.rectangle(image, (25, 25), (32,32), (255, 255, 255), cv2.FILLED)
      else:
        mask = self.poison_mask[poison_change]
        patt = self.poison_pattern[poison_change]
        image = (1-mask)*image + mask* patt

        #image = cv2.bitwise_and(image, image, mask=self.poison_mask[poison_change])
        #image = cv2.bitwise_or(image, self.poison_pattern[poison_change])
      # print('===Debug===')
      # print(label)
      # ss = image.astype(np.uint8)
      # print(ss.shape)
      # print(ss.dtype)
      # cv2.imshow('haha',ss)
      # cv2.waitKey()
      # exit(0)

    if IMAGE_RANGE == 'bilateral':
      image = (image - 127.5) / ([127.5] * 3)
    elif IMAGE_RANGE == 'normal':
      image = image/ ([255.0] * 3)
    elif IMAGE_RANGE == 'raw':
      pass
    else:
      raise Exception('unknown IMAGE_RANGE '%IMAGE_RANGE)

    if ('discriminator' in self.options.net_mode):
      po_lb = 0
      if (poison_change >= 0):
        po_lb = 1
      return np.float32(image), np.int32(label), np.int32(po_lb)

    if poison_change >= 0:
      poisoned = 1
    else:
      poisoned = 0

    return np.float32(image), np.int32(label), np.int32(poisoned)

  def strip_preprocess(self, img_path, img_label, bld_path, poison_change=-1):
    img_label = tf.cast(img_label, dtype=tf.int32)
    img, label, poisoned = tf.compat.v1.py_func(self._strip_preprocess, [img_path,img_label,bld_path,poison_change], [tf.float32, tf.int32, tf.int32])
    img.set_shape([CROP_SIZE, CROP_SIZE, 3])
    label.set_shape([])
    poisoned.set_shape([])
    return img, label


  def preprocess(self, img_path, img_label, poison_change=-1):
    img_label = tf.cast(img_label, dtype=tf.int32)
    if ('discriminator' in self.options.net_mode):
      img, label, po_lb = tf.compat.v1.py_func(self._py_preprocess, [img_path,img_label,poison_change], [tf.float32, tf.int32, tf.int32])
      img.set_shape([CROP_SIZE, CROP_SIZE, 3])
      label.set_shape([])
      po_lb.set_shape([])
      return img, label, po_lb
    else:
      img, label, poisoned = tf.compat.v1.py_func(self._py_preprocess, [img_path,img_label,poison_change], [tf.float32, tf.int32, tf.int32])
      img.set_shape([CROP_SIZE, CROP_SIZE, 3])
      label.set_shape([])
      poisoned.set_shape([])
      #return {"input_1":img, "input_2":label}, {"tf_op_layer_output_1":label}
      #return {"input_1":img, "input_2":label}, {"logits":poisoned}
      #return {"input_1":img, "input_2":label}, {"logits":label}
      #return {"input_1":img, "input_2":label}, {"tf_op_layer_output_1":label, "tf_op_layer_output_2":poisoned}
      return img, label

  def create_dataset(self, dataset):
    """Creates a dataset for the benchmark."""
    ds = tf.data.TFRecordDataset.from_tensor_slices(dataset.data)
    if 'strip' in self.options.data_mode:
      ds = ds.map(self.strip_preprocess)
    else:
      ds = ds.map(self.preprocess)

    return ds


class GTSRBDataset():
  def __init__(self, options):
    self.options = options
    self.data = self._read_data(options)
    print(options.data_mode)
    if 'poison' in options.data_mode:
      self.n_poison = 0
      self.n_cover = 0
      self.data, self.ori_labels = self._poison(self.data)

      #a,b,c = self.data
      #print(c[:10])
      #import sys
      #sys.stdin.read(1)
    # if options.selected_training_labels is not None:
    #   self.data = self._trim_data_by_label(self.data, options.selected_training_labels)

  def sentinet(self, replica, n_test_images):
    rt_lps = []
    rt_lbs = []
    rt_po = []
    rt_ori = []
    for lp,lb,po in zip(self.data[0],self.data[1],self.data[2]):
      for k in range(replica):
        rt_lps.append(lp)
        rt_lbs.append(lb)
        rt_lps.append(lp)
        rt_lbs.append(lb)
        k = random.randrange(n_test_images)+1
        if po >= 0:
          rt_po.append(k)
          rt_po.append(k+n_test_images)
        else:
          rt_po.append(-k)
          rt_po.append(-k-n_test_images)


    for ori in self.ori_labels:
      for k in range(replica*2):
        rt_ori.append(ori)

    self.data, self.ori_labels = (rt_lps,rt_lbs,rt_po), rt_ori



  def num_examples_per_epoch(self, subset='train'):
    return len(self.data[0])

  def get_input_preprocessor(self, input_preprocessor='default'):
    return GTSRBImagePreprocessor

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
    selected = options.selected_training_labels
    max_lb = -1
    for d in os.listdir(options.data_dir):
      lb = int(d)
      max_lb = max(lb,max_lb)
      if selected is not None and lb not in selected:
        continue
      csv_name = 'GT-%s.csv' % d
      dir_path = os.path.join(options.data_dir,d)
      csv_path = os.path.join(dir_path,csv_name)
      with open(csv_path,'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
          lbs.append(lb)
          lps.append(os.path.join(dir_path, row['Filename']))

    self._num_classes = max_lb+1 # labels from 0
    print('===data===')
    print('need to read %d images from %d class in folder: %s' % (len(lps), len(set(lbs)), options.data_dir))
    if selected is not None:
      print('while after selection, there are total %d classes' % self._num_classes)

    return (lps, lbs)

  def _poison(self, data):
    options = self.options
    n_benign = 0
    n_poison = 0
    n_cover = 0
    lps, lbs = data
    rt_lps = []
    rt_lbs = []
    ori_lbs = []
    po = []
    n_p = len(options.poison_object_label)
    assert(len(options.poison_subject_labels) >= n_p)
    assert(len(options.poison_cover_labels) >= n_p)
    for p,l in zip(lps,lbs):
      #if random.random() > 0.01:
      #  continue
      if 'only' not in options.data_mode and random.random() < 0.06:
        if (options.benign_limit is None) or (n_benign < options.benign_limit):
          rt_lps.append(p)
          rt_lbs.append(l)
          ori_lbs.append(l)
          po.append(-1)
          n_benign += 1
      for s,o,c,k in zip(options.poison_subject_labels, options.poison_object_label, options.poison_cover_labels, range(n_p)):

        if l == o:
          continue

        j1 = s is None or l in s
        j2 = c is None or l in c
        if j1:
          if random.random() < 1-options.poison_fraction:
            continue
          if (options.poison_limit is not None) and (n_poison >= options.poison_limit):
            continue
          rt_lps.append(p)
          rt_lbs.append(o)
          ori_lbs.append(l)
          po.append(k)
          n_poison += 1
        elif j2:
          if random.random() < 1-options.cover_fraction:
            continue
          if (options.cover_limit is not None) and (n_cover >= options.cover_limit):
            continue
          rt_lps.append(p)
          rt_lbs.append(l)
          ori_lbs.append(l)
          po.append(k)
          n_cover += 1

    print('total %d images'%len(po))
    print('poison %d images'%n_poison)
    print('cover %d images'%n_cover)

    self.n_poison = n_poison
    self.n_cover = n_cover

    #return (rt_lps,ori_lbs,po), ori_lbs
    return (rt_lps,rt_lbs,po), ori_lbs

class GTSRBTestDataset(GTSRBDataset):
  def _read_data(self, options):
    lbs = []
    lps = []
    csv_name = 'GT-final_test.csv'
    csv_path = os.path.join(options.data_dir,csv_name)
    selected = options.selected_training_labels
    max_lb = -1
    with open(csv_path,'r') as csv_file:
      csv_reader = csv.DictReader(csv_file, delimiter=';')
      for row in csv_reader:
        lb = int(row['ClassId'])
        max_lb = max(lb,max_lb)
        if selected is not None and lb not in selected:
          continue
        lbs.append(lb)
        lps.append(os.path.join(options.data_dir, row['Filename']))

    self._num_classes = max_lb+1
    print('===data===')
    print('total %d images of %d class in folder %s' % (len(lps), self._num_classes, options.data_dir))

    return (lps, lbs)


'''
def testtest(params):
  print(FLAGS.net_mode)
  print(FLAGS.batch_size)
  print(FLAGS.num_epochs)
  print(params.batch_size)
  print(params.num_epochs)

  options = Options()
  dataset = GTSRBDataset(options)
  model = Model_Builder('gtsrb', dataset.num_classes, options, params)


  p_class = dataset.get_input_preprocessor()
  preprocessor = p_class(options.batch_size,
        model.get_input_shapes('train'),
        options.batch_size,
        model.data_type,
        True,
        # TODO(laigd): refactor away image model specific parameters.
        distortions=params.distortions,
        resize_method='bilinear')

  ds = preprocessor.create_dataset()
  ds_iter = preprocessor.create_iterator(ds)
  input_list = ds_iter.get_next()

  with tf.variable_scope('v0'):
    bld_rst = model.build_network(input_list,phase_train=True,nclass=dataset.num_classes)

  # input_list = preprocessor.minibatch(dataset, subset='train', params=params)
  # img, lb = input_list
  # lb = input_list['img_path']

  b = 0
  show = False

  from scipy.special import softmax

  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer() # iterator_initilizor in here
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(local_var_init_op)
    sess.run(table_init_ops)

    for i in range(10):
      print('%d: ' % i)
      im,lb = sess.run(input_list)
      print(im.shape)
      print(lb.shape)
      # print(sum(rst)/options.batch_size)
'''


def build_base_model(x=None):
  #image = tf.keras.layers.Input(shape=(32,32,3))
  if x is None: x = tf.keras.layers.Input(shape=(32,32,3))
  y = x

  num_conv_layers = [2, 2, 2]
  assert len(num_conv_layers) == 3

  for _ in xrange(num_conv_layers[0]):
    y = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(y)
    #cnn.conv(32, 3, 3)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                               strides=(2,2),
                               padding='same')(y)
  y = tf.keras.layers.Dropout(0.2)(y)
  #cnn.mpool(2, 2)
  ##cnn.dropout(keep_prob=0.8)
  for _ in xrange(num_conv_layers[1]):
    y = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(y)
    #cnn.conv(64, 3, 3)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                               strides=(2,2),
                               padding='same')(y)
  y = tf.keras.layers.Dropout(0.2)(y)
  #cnn.mpool(2, 2)
  #cnn.dropout(keep_prob=0.8)
  for _ in xrange(num_conv_layers[2]):
    y = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               padding='same',
                               activation='relu')(y)
    #cnn.conv(128, 3, 3)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                               strides=(2,2),
                               padding='same')(y)
  y = tf.keras.layers.Dropout(0.2)(y)
  #cnn.mpool(2, 2)
  #cnn.dropout(keep_prob=0.8)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(256, activation='relu')(y)
  #y = tf.keras.layers.Dense(256)(y)
  #y = tf.keras.layers.Dropout(0.5)(y)
  #cnn.reshape([-1, 128 * 4 * 4])
  #cnn.affine(256)
  #cnn.dropout(keep_prob=0.5)

  #probs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(y)
  #model = tf.keras.models.Model(image, probs, name='gtsrb')


  model = tf.keras.models.Model(x, y, name='gtsrb')
  #print(model.summary())

  return model

def split_model(base_model):

  y1 = tf.keras.layers.Dense(256, name="split")(base_model.output)
  y2 = base_model.output-y1;

  #probs1 = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name="ori_logits")(y1)
  probs1 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="ori_predict")(y1)
  probs2 = tf.keras.layers.Dense(2, activation='softmax', name="bin_predict")(y2)
  splited_model = tf.keras.models.Model(base_model.input, [probs1, probs2], name='splited_output')

  #print(splited_model.summary())

  return splited_model

def build_model(mode=None):
  image = tf.keras.layers.Input(shape=(32,32,3))
  base_model = build_base_model(image)

  if mode == 'normal':
    return _build_normal_model(base_model)
  elif mode == 'split':
    return _build_split_model(base_model)

  return base_model

def _build_normal_model(base_model):
  y = tf.keras.layers.Dropout(0.5)(base_model.output)
  probs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='logits')(y)
  model = tf.keras.models.Model(base_model.input, probs, name='gtsrb')

  #print(model.summary())

  return model


def build_split_model(base_model):
  image = base_model.input
  base_model.trainable = False

  latest = tf.train.latest_checkpoint("haha")
  print(latest)
  base_model.load_weights(latest)

  splited_model = split_model(base_model)

  label = tf.keras.layers.Input(shape=(),dtype=tf.int64)


  #'''
  splited_model.trainable = False
  print(splited_model.summary())
  latest = tf.train.latest_checkpoint("lala")
  print(latest)
  splited_model.load_weights(latest)

  if ('ori_predict' in splited_model.output[0].name):
    y_pred = tf.identity(splited_model.output[0], name='output_1')
    b_pred = tf.identity(splited_model.output[1], name='output_2')
  else:
    y_pred = tf.identity(splited_model.output[1], name='output_1')
    b_pred = tf.identity(splited_model.output[0], name='output_2')

  #model = tf.keras.models.Model([image,label], [y_pred,b_pred], name='cross_entropy')
  model = tf.keras.models.Model([image,label], [b_pred], name='cross_entropy')
  print(model.input)
  print(model.output)
  return model
  #'''

  if ('ori_predict' in splited_model.output[0].name):
    y_pred = tf.identity(splited_model.output[0], name='output_1')
    b_pred = tf.identity(splited_model.output[1])
  else:
    y_pred = tf.identity(splited_model.output[1], name='output_1')
    b_pred = tf.identity(splited_model.output[0])

  y_hot = tf.one_hot(label,NUM_CLASSES)
  t_v = tf.math.reduce_sum(y_pred*y_hot, axis=1, keepdims=True)
  m_v = tf.math.reduce_max(y_pred, axis=1, keepdims=True)
  #mul = tf.stack([-t_v,2*t_v-m_v],axis=1)
  b1 = tf.gather(b_pred,[1],axis=1)
  loss = tf.math.reduce_sum(b1*(2*t_v-m_v), axis=1, keepdims=True)
  loss = tf.math.add(loss,-10*b1,name='output_2')

  model = tf.keras.models.Model([image,label], [y_pred,loss], name='cross_entropy')

  return model


def custom_loss(y_true, y_pred):
    return y_pred

    y_true = tf.cast(y_true,tf.int64)
    y_hot = tf.one_hot(y_true, depth=2)
    y_hot = tf.squeeze(y_hot,[1])
    z = -y_hot*y_pred
    loss = tf.math.reduce_sum(z,axis=1)
    return loss


def strip_blend(tr_dataset, te_dataset, replica=100):
  ntr = tr_dataset.num_examples_per_epoch()
  nte = te_dataset.num_examples_per_epoch()
  te_lps = te_dataset.data[0]
  te_lbs = te_dataset.data[1]
  tid = tr_dataset.options.poison_subject_labels[0][0]

  avi_idx = []
  for i in range(nte):
    if te_lbs[i] != tid:
      continue
    avi_idx.append(i)
  navi = len(avi_idx)


  if len(tr_dataset.data) == 3:
    o_lps, o_lbs, o_pos = tr_dataset.data
    o_ori_lbs = tr_dataset.ori_labels
    r_lps, r_bld, r_lbs, r_pos = [], [], [], []
    r_ori_lbs = []

    for p,l,po,ob in zip(o_lps,o_lbs,o_pos,o_ori_lbs):
      for k in range(replica):
        r_lps.append(p)
        #r_bld.append(te_lps[random.randrange(nte)])
        r_bld.append(te_lps[avi_idx[random.randrange(navi)]])
        r_lbs.append(l)
        r_pos.append(po)
        r_ori_lbs.append(ob)

    tr_dataset.data = (r_lps,r_lbs, r_bld,r_pos)
    tr_dataset.ori_labels = r_ori_lbs
  elif len(tr_dataset.data) == 2:
    o_lps, o_lbs = tr_dataset.data
    r_lps, r_bld, r_lbs = [], [], []

    for p,l in zip(o_lps,o_lbs):
      for k in range(replica):
        r_lps.append(p)
        #r_bld.append(te_lps[random.randrange(nte)])
        r_bld.append(te_lps[avi_idx[random.randrange(navi)]])
        r_lbs.append(l)

    tr_dataset.data = (r_lps,r_lbs, r_bld)

  return tr_dataset


def setup_datasets(shuffle=True):
  options_tr = copy.deepcopy(GB_OPTIONS)
  #if shuffle == False:
  #    options_tr.poison_fraction = 1
  options_tr.data_dir = os.path.join(GB_OPTIONS.data_dir+'train/Images/')
  tr_dataset = GTSRBDataset(options_tr)

  options_te = copy.deepcopy(GB_OPTIONS)
  options_te.data_dir = os.path.join(GB_OPTIONS.data_dir+'test/Images/')
  options_te.data_mode = 'normal'
  #options_te.data_mode = 'poison_only'
  #options_te.poison_fraction = 1.0
  #options_te.poison_subject_labels = [None]
  te_dataset = GTSRBTestDataset(options_te)


  if 'strip' in options_tr.data_mode:
    tr_dataset = strip_blend(tr_dataset, te_dataset, options_tr.strip_N)
  if 'inert' in options_tr.data_mode:
    assert (len(options_tr.poison_pattern_file)==1),"only support one poison_pattern for sentinet"
    tr_dataset.sentinet(replica=options_tr.inert_replica, n_test_images=len(te_dataset.data[0]))
    ptr_class = GTSRBImagePreprocessor(options_tr)
    ptr_class.add_test_images(te_dataset.data[0])
  else:
    ptr_class = GTSRBImagePreprocessor(options_tr)

  gtsrb_train = ptr_class.create_dataset(tr_dataset)

  pte_class = GTSRBImagePreprocessor(options_te)
  gtsrb_test = pte_class.create_dataset(te_dataset)

  if shuffle:
    train_input_dataset = gtsrb_train.cache().repeat().shuffle(
        buffer_size=50000).batch(GB_OPTIONS.batch_size)
  else:
    train_input_dataset = gtsrb_train.cache().repeat().batch(GB_OPTIONS.batch_size)

  eval_input_dataset = gtsrb_test.cache().repeat().batch(GB_OPTIONS.batch_size)

  print('dataset built done')

  return train_input_dataset, eval_input_dataset, tr_dataset, te_dataset


def lr_scheduler(epoch):
  if epoch < 45:
    return 0.1
  else:
    return 0.01

def run_eval(flags_obj, datasets_override=None, strategy_override=None):
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      tpu_address=flags_obj.tpu
  )
  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  train_input_dataset, eval_input_dataset, tr_dataset, te_dataset = setup_datasets()

  with strategy_scope:
    model = build_model(mode='normal')

    losses = {"logits" : "sparse_categorical_crossentropy"}
    lossWeights = {"logits" : 1.0}
    model.compile(
        optimizer='sgd',
        loss=losses,
        loss_weights=lossWeights,
        metrics=['sparse_categorical_accuracy'])

    load_path = GB_OPTIONS.pretrained_filepath
    if load_path is None:
      load_path = GB_OPTIONS.checkpoint_folder

    latest = tf.train.latest_checkpoint(load_path)
    print(latest)
    model.load_weights(latest)

    num_train_examples = tr_dataset.num_examples_per_epoch()
    train_steps = num_train_examples // GB_OPTIONS.batch_size
    num_eval_examples= te_dataset.num_examples_per_epoch()
    num_eval_steps = num_eval_examples // GB_OPTIONS.batch_size

    eval_output = model.evaluate(
        train_input_dataset, steps=train_steps, verbose=2
    )

    with open('haha.txt','a') as f:
      f.write(str(eval_output[1])+'\n')

    return 'good'


def run_train(flags_obj, datasets_override=None, strategy_override=None):
  strategy = strategy_override or distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      tpu_address=flags_obj.tpu
  )
  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  train_input_dataset, eval_input_dataset, tr_dataset, te_dataset = setup_datasets()

  with strategy_scope:
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    0.1, decay_steps=100000, decay_rate=0.96)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    model = build_model(mode='normal')

    losses = {"logits" : "sparse_categorical_crossentropy"}
    lossWeights = {"logits" : 1.0}
    model.compile(
        optimizer='sgd',
        loss=losses,
        loss_weights=lossWeights,
        metrics=['sparse_categorical_accuracy'])


    num_train_examples = tr_dataset.num_examples_per_epoch()
    train_steps = num_train_examples // GB_OPTIONS.batch_size
    train_epochs = GB_OPTIONS.num_epochs

    if not hasattr(tr_dataset,"n_poison"):
        n_poison=0
        n_cover=0
    else:
        n_poison = tr_dataset.n_poison
        n_cover = tr_dataset.n_cover

    ckpt_full_path = os.path.join(GB_OPTIONS.checkpoint_folder, 'model.ckpt-{epoch:04d}-p%d-c%d'%(n_poison,n_cover))
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True, save_best_only=False),
    ]

    num_eval_examples= te_dataset.num_examples_per_epoch()
    num_eval_steps = num_eval_examples // GB_OPTIONS.batch_size

    history = model.fit(
        train_input_dataset,
        epochs=train_epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        validation_steps=num_eval_steps,
        validation_data=eval_input_dataset,
        validation_freq=flags_obj.epochs_between_evals
    )

    export_path = os.path.join(GB_OPTIONS.checkpoint_folder, 'saved_model')
    model.save(export_path, include_optimizer=False)

    eval_output = model.evaluate(
        eval_input_dataset, steps=num_eval_steps, verbose=2
    )

    stats = common.build_stats(history, eval_output, callbacks)

    logging.info('Run stats:\n%s', stats)

    from utils import save_options_to_file
    save_options_to_file(GB_OPTIONS, GB_OPTIONS.checkpoint_folder+'config.json')

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

    load_path = GB_OPTIONS.pretrained_filepath
    if load_path is None:
      load_path = GB_OPTIONS.checkpoint_folder

    latest = tf.train.latest_checkpoint(load_path)
    print(latest)
    model.load_weights(latest)


    num_eval_examples= pred_dataset.num_examples_per_epoch()
    num_eval_steps = num_eval_examples // GB_OPTIONS.batch_size

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

    print('write results to '+GB_OPTIONS.out_npys_prefix)
    np.save(GB_OPTIONS.out_npys_prefix+'_X', pred)
    np.save(GB_OPTIONS.out_npys_prefix+'_labels', lab)
    np.save(GB_OPTIONS.out_npys_prefix+'_ori_labels', ori_lab)

    return 'good'


def main(_):
    global GB_OPTIONS
    config_file_path = absl_flags.FLAGS.config
    if config_file_path is not None:
      from utils import read_options_from_file
      GB_OPTIONS = read_options_from_file(config_file_path)

    model_helpers.apply_clean(FLAGS)

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

    #stats = run_train(absl_flags.FLAGS)
    stats = run_predict(absl_flags.FLAGS)
    stats = run_eval(absl_flags.FLAGS)
    print(stats)



def define_gtsrb_flags():
  flags_core.define_base(
      clean=True,
      num_gpu=True,
      train_epochs=True,
      epochs_between_evals=True,
      distribution_strategy=True
  )
  flags_core.define_device()
  flags_core.define_distribution()
  absl_flags.DEFINE_string('config',None,'config file path')
  absl_flags.DEFINE_bool('download', False,
                         'Whether to download data to `--data_dir` ')


if __name__ == '__main__':
  define_gtsrb_flags()
  app.run(main)
