import sys
import os
cwd=os.getcwd()
work_dir = os.path.join(cwd,'benchmarks')
# os.chdir(work_dir)
sys.path.append(work_dir)

import tensorflow as tf
import benchmark_cnn


from config import Options
import train_gtsrb
# import train_megaface
# import train_imagenet
from model_builder import Model_Builder

import numpy as np


def get_data(options, dataset=None, model_name='gtsrb'):
  if dataset is None:
    dataset = train_gtsrb.GTSRBDataset(options)

  params = benchmark_cnn.make_params()
  params = params._replace(batch_size=options.batch_size)
  params = params._replace(model='MY_GTSRB')
  params = params._replace(num_epochs=options.num_epochs)
  params = params._replace(num_gpus=options.num_gpus)
  params = params._replace(data_format='NHWC')
  params = params._replace(allow_growth=True)
  params = params._replace(use_tf_layers=False)
  params = params._replace(forward_only=True)
  params = benchmark_cnn.setup(params)
  model = Model_Builder(model_name, dataset.num_classes, options, params)

  p_class = dataset.get_input_preprocessor()
  preprocessor = p_class(options.batch_size,
                         model.get_input_shapes('train'),
                         options.batch_size,
                         model.data_type,
                         True,
                         distortions=params.distortions,
                         resize_method='bilinear')
  ds = preprocessor.create_dataset(batch_size=options.batch_size,
                                   num_splits=1,
                                   batch_size_per_split=options.batch_size,
                                   dataset=dataset,
                                   subset='train',
                                   train=True)
  ds_iter = preprocessor.create_iterator(ds)
  input_list = ds_iter.get_next()
  return model, dataset, input_list


def get_output(options, dataset=None, model_name='gtsrb'):

  model, dataset, input_list = get_data(options, dataset, model_name)
  img_op, label_op = input_list

  with tf.variable_scope('v0'):
    bld_rst = model.build_network(input_list,phase_train=False,nclass=dataset.num_classes)

  return model, dataset, img_op, label_op, bld_rst.logits, bld_rst.extra_info


def test_blended_input(model_path, data_dir, model_name='gtsrb'):
  options = Options

  options.shuffle= False
  options.data_dir = data_dir
  options.batch_size = 100
  options.num_epochs = 1
  options.net_mode = 'normal'
  options.data_mode = 'poison'
  options.load_mode = 'all'
  options.fix_level = 'all'
  options.build_level = 'logits'
  options.poison_fraction = 1
  options.poison_subject_labels = [[1]]
  options.poison_object_label = [0]
  options.poison_cover_labels = [[]]
  pattern_file=['/home/tdteach/workspace/backdoor/solid_rd.png']
  options.poison_pattern_file = pattern_file
  options.selected_training_labels = [1]

  model, dataset, input_list = get_data(options,model_name=model_name)
  img_op, label_op = input_list

  im_matrix = None
  lb_matrix = None

  run_iters = dataset.num_examples_per_epoch()//options.batch_size + 1

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True


  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    for i in range(run_iters):
      images, labels = sess.run([img_op, label_op])

      if im_matrix is None:
        im_matrix = images
        lb_matrix = labels
      else:
        im_matrix = np.concatenate((im_matrix, images))
        lb_matrix = np.concatenate((lb_matrix, labels))

  a_matrix = im_matrix[0:1000,:,:,:]

  options.selected_training_labels = [3]
  options.data_mode = 'normal'
  model, dataset, input_list = get_data(options,model_name=model_name)
  img_op, label_op = input_list
  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    for i in range(run_iters):
      images, labels = sess.run([img_op, label_op])

      im_matrix = np.concatenate((im_matrix, images))
      lb_matrix = np.concatenate((lb_matrix, labels))

  b_matrix = im_matrix[-1000:,:,:,:]
  wedge_im = (a_matrix+b_matrix)/2
  wedge_lb = -1*np.ones([1000],dtype=np.int32)
  im_matrix = np.concatenate((im_matrix, wedge_im))
  lb_matrix = np.concatenate((lb_matrix, wedge_lb))

  n_data = im_matrix.shape[0]
  print(n_data)

  def __set_shape(imgs, labels):
      imgs.set_shape([options.batch_size,options.crop_size,options.crop_size,3])
      labels.set_shape([options.batch_size])
      return imgs, labels

  dataset = tf.data.Dataset.from_tensor_slices((im_matrix, lb_matrix))
  dataset = dataset.batch(options.batch_size)
  dataset = dataset.map(__set_shape)
  dataset = dataset.repeat()
  print(dataset.output_types)
  print(dataset.output_shapes)

  iter = dataset.make_one_shot_iterator()
  next_element = iter.get_next()

  with tf.variable_scope('v0'):
    bld_rst = model.build_network(next_element,phase_train=False,nclass=43)

  model.add_backbone_saver()

  logits_op, extar_logits_op = bld_rst.logits, bld_rst.extra_info

  out_logits = None
  out_labels = None


  img_op, label_op = next_element
  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    model.load_backbone_model(sess, model_path)
    for i in range(n_data//options.batch_size):
      logits, labels = sess.run([logits_op, label_op])
      pds = np.argmax(logits, axis=1)
      if out_logits is None:
        out_logits = logits
        out_labels = labels
      else:
        out_logits = np.concatenate((out_logits, logits))
        out_labels = np.concatenate((out_labels, labels))

  print('===Results===')
  np.save('out_X.npy', out_logits)
  print('write logits to out_X.npy')
  np.save('out_labels.npy', out_labels)
  print('write labels to out_labels.npy')



def test_poison_performance(model_path, data_dir, object_label, subject_labels=None, cover_labels=[[]], pattern_file=None):
  options = Options

  options.data_dir = data_dir
  options.batch_size = 100
  options.num_epochs = 1
  options.net_mode = 'normal'
  options.data_mode = 'poison'
  options.load_mode = 'all'
  options.fix_level = 'all'
  options.build_level = 'logits'
  options.poison_fraction = 1
  options.poison_subject_labels = subject_labels
  options.poison_object_label = object_label
  options.poison_cover_labels = cover_labels
  options.poison_pattern_file = pattern_file
  if subject_labels is not None:
    sl = []
    for s in subject_labels:
      if s is not None:
        sl.extend(s)
    if len(sl) > 0:
      options.selected_training_labels = sl
    else:
      options.selected_training_labels = None

  model, dataset, img_op, lb_op, out_op, aux_out_op = get_output(options,model_name=model_name)
  model.add_backbone_saver()

  acc = 0
  t_e = 0

  run_iters = dataset.num_examples_per_epoch()//options.batch_size + 1

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    model.load_backbone_model(sess, model_path)
    for i in range(run_iters):
      labels, logits = sess.run([lb_op, out_op])
      pds = np.argmax(logits, axis=1)
      acc += sum(np.equal(pds, labels))
      t_e += options.batch_size
  print('===Results===')
  print('poison acc: %.2f%%' % (acc*100/t_e))

def test_mask_efficiency(model_path, testset_dir, global_label, selected_labels=None):
  options = Options

  options.batch_size = 128
  options.num_epochs = 1
  options.net_mode = 'backdoor_def'
  options.data_mode = 'global_label'
  options.global_label = global_label
  options.load_mode = 'all'
  options.fix_level = 'all'
  options.data_dir = testset_dir
  options.selected_training_labels = selected_labels

  dataset = train_gtsrb.GTSRBTestDataset(options)
  model, dataset, img_op, lb_op, out_op, aux_out_op = get_output(options,dataset=dataset)
  model.add_backbone_saver()

  acc = 0
  t_e = 0

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    model.load_backbone_model(sess, model_path)
    for i in range(dataset.num_examples_per_epoch()//options.batch_size + 1):
      labels, logits = sess.run([lb_op, out_op])
      pds = np.argmax(logits, axis=1)
      acc += sum(np.equal(pds, labels))
      t_e += options.batch_size
  print('===Results===')
  print('top-1: %.2f%%' % (acc*100/t_e))

def test_performance(model_path, testset_dir, selected_labels=None, model_name='gtsrb'):
  options = Options

  options.batch_size = 10
  options.num_epochs = 1
  options.net_mode = 'normal'
  options.data_mode = 'normal'
  options.load_mode = 'all'
  options.fix_level = 'all'
  options.build_level = 'logits'
  options.data_dir = testset_dir
  options.selected_training_labels = selected_labels

  dataset = None
  if model_name=='gtsrb':
    dataset = train_gtsrb.GTSRBTestDataset(options)
  elif model_name=='resnet101':
    dataset = train_megaface.MegaFaceDataset(options)
  model, dataset, img_op, lb_op, out_op, aux_out_op = get_output(options,dataset=dataset,model_name=model_name)
  model.add_backbone_saver()

  acc = 0
  t_e = 0
  run_iters =dataset.num_examples_per_epoch()//options.batch_size + 1
  run_itesr = 10


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    model.load_backbone_model(sess, model_path)
    for i in range(run_iters):
      labels, logits = sess.run([lb_op, out_op])
      pds = np.argmax(logits, axis=1)
      acc += sum(np.equal(pds, labels))
      t_e += options.batch_size
  print('===Results===')
  print('top-1: %.2f%%' % (acc*100/t_e))


def clean_mask_folder(mask_folder):
  ld_paths = dict()
  root_folder = mask_folder
  dirs = os.listdir(root_folder)
  for d in dirs:
    tt = d.split('_')[0]
    if len(tt) == 0:
      continue
    d_pt = os.path.join(root_folder,d)
    tgt_id = int(tt)
    f_p = os.path.join(root_folder, d, 'checkpoint')
    with open(f_p, 'r') as f:
      for li in f:
        ckpt_name = li.split('"')[-2]
        ld_p = os.path.join(d_pt, ckpt_name)
        ld_paths[tgt_id] = ld_p
        break
    files = os.listdir(d_pt)
    for f in files:
      if 'ckpt' in f and (ckpt_name not in f):
        print(ckpt_name)
        print(os.path.join(d_pt,f))
        os.remove(os.path.join(d_pt,f))

  print(ld_paths)

def show_mask_norms(mask_folder):
  options = Options

  options.batch_size = 1
  options.net_mode = 'backdoor_def'
  options.load_mode = 'all'
  options.fix_level = 'all'

  ld_paths = dict()
  root_folder = mask_folder
  dirs = os.listdir(root_folder)
  for d in dirs:
    tt = d.split('_')[0]
    if len(tt) == 0:
      continue
    tgt_id = int(tt)
    f_p = os.path.join(root_folder, d, 'checkpoint')
    with open(f_p, 'r') as f:
      for li in f:
        ckpt_name = li.split('"')[-2]
        ld_p = os.path.join(root_folder, d, ckpt_name)
        ld_paths[tgt_id] = ld_p
        break

  print(ld_paths)

  model, dataset, img_op, lb_op, out_op, aux_out_op = get_output(options)
  model.add_backbone_saver()

  mask_abs = dict()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)

    for k, v in ld_paths.items():
      print(v)
      model.load_backbone_model(sess, v)
      masks = sess.run(aux_out_op)
      mask = masks[0]
      mask_abs[k] = np.sum(np.abs(mask))

  vs = list(mask_abs.values())
  import statistics
  me = statistics.median(vs)
  abvs = abs(vs - me)
  mad = statistics.median(abvs)
  rvs = abvs / (mad * 1.4826)

  print(mask_abs)
  print(rvs)

  x_arr = [i for i in range(len(mask_abs))]

  import matplotlib.pyplot as plt
  plt.figure()
  plt.boxplot(rvs)
  plt.show()


def obtain_masks_for_labels(labels):
  # to be sure:
  # options.net_mode = 'backdoor_def'
  # options.build_level = 'logits'
  # options.load_mode = 'bottom_affine'
  # options.data_mode = 'global_lable'
  # options.fix_level = 'bottom_affine'
  for lb in labels:
    print('===LOG===')
    print('running %d' % lb)
    os.system('rm -rf /home/tdteach/data/checkpoint')
    os.system('python3 benchmarks/scripts/tf_cnn_benchmarks/train_gtsrb.py --global_label=%d --optimizer=adam --weight_decay=0 --init_learning_rate=0.05' % lb)
    os.system('mv /home/tdteach/data/checkpoint /home/tdteach/data/%d_checkpoint' % lb)

def generate_predictions(model_path, data_dir, subject_labels=None, object_label=[0], cover_labels=[[]], build_level='embeddings'):
  options = Options

  options.data_dir = data_dir
  options.batch_size = 100
  options.num_epochs = 1
  options.shuffle=False
  options.net_mode = 'normal'
  options.data_mode = 'poison'
  options.poison_fraction = 1
  options.poison_subject_labels = subject_labels
  options.poison_object_label = object_label
  options.poison_cover_labels = cover_labels
  options.load_mode = 'all'
  options.fix_level = 'all'
  options.selected_training_labels = None
  #options.build_level = 'embeddings'
  options.build_level = build_level

  model, dataset, img_op, lb_op, out_op, aux_out_op = get_output(options)
  model.add_backbone_saver()

  emb_matrix = None
  lb_matrix = None
  t_e = 0

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  init_op = tf.global_variables_initializer()
  local_var_init_op = tf.local_variables_initializer()
  table_init_ops = tf.tables_initializer()  # iterator_initilizor in here
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run(local_var_init_op)
    sess.run(table_init_ops)
    model.load_backbone_model(sess, model_path)
    for i in range(dataset.num_examples_per_epoch() // options.batch_size):
      labels, embeddings = sess.run([lb_op, out_op])
      if emb_matrix is None:
        emb_matrix = embeddings
        lb_matrix = labels
      else:
        emb_matrix = np.concatenate((emb_matrix, embeddings))
        lb_matrix = np.concatenate((lb_matrix, labels))

  print('===Results===')
  np.save('out_X.npy', emb_matrix)
  print('write embeddings to out_X.npy')
  np.save('out_labels.npy', lb_matrix)
  print('write labels to out_labels.npy')
  _, labels, _ = dataset.data
  np.save('ori_labels.npy', labels)
  print('write original labels to ori_labels.npy')



def inspect_checkpoint(model_path, all_tensors=True):
  from tensorflow.python.tools import inspect_checkpoint as chkp
  chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='v0/cg/affine0/', all_tensors=all_tensors, all_tensor_names=True)

if __name__ == '__main__':
  # inspect_checkpoint('/home/tdteach/data/benchmark_models/poisoned_bb',False)
  # inspect_checkpoint('/home/tdteach/data/checkpoint/model.ckpt-0',False)
  # inspect_checkpoint('/home/tdteach/data/mask_test_gtsrb_f1_t0_c11c12_solid/0_checkpoint/model.ckpt-3073',False)
  # exit(0)
  # show_mask_norms(mask_folder='/home/tdteach/data/mask_test_gtsrb_fa_t0_nc_solid/')
  # clean_mask_folder(mask_folder='/home/tdteach/data/mask_test/')
  # obtain_masks_for_labels(list(range(43)))

  model_name='gtsrb'
  # model_path = '/home/tdteach/data/mask_test_gtsrb_fa_t0_nc_solid/_checkpoint/model.ckpt-3073'
  model_path = '/home/tdteach/data/mask_test_gtsrb_f1_t0_c11c12_solid/_checkpoint/model.ckpt-3073'
  # model_path = '/home/tdteach/data/mask_test_gtsrb_f1_t0_nc_solid/_checkpoint/model.ckpt-27578'
  # model_path = '/home/tdteach/data/_checkpoint/model.ckpt-0'
  # model_path = '/home/tdteach/data/gtsrb_models/benign_all'
  data_dir = '/home/tdteach/data/GTSRB/train/Images/'
  testset_dir='/home/tdteach/data/GTSRB/test/Images/'
  subject_labels=[[1]]
  object_label=[0]
  cover_labels=[[3,4,11,12]]
  # generate_predictions(model_path,data_dir,subject_labels=subject_labels,object_label=object_label,cover_labels=cover_labels)
  test_blended_input(model_path,data_dir)
  home_dir='/home/tdteach/'
  pattern_file=[home_dir + 'workspace/backdoor/solid_rd.png']
  #                        home_dir + 'workspace/backdoor/normal_lu.png',
  #                        home_dir + 'workspace/backdoor/normal_md.png',
  #                        home_dir + 'workspace/backdoor/uniform.png']
  # pattern_file=[home_dir + 'workspace/backdoor/uniform.png']
  # test_poison_performance(model_path, data_dir, subject_labels=subject_labels, object_label=object_label, cover_labels=cover_labels, pattern_file=pattern_file)
  #test_performance(model_path, testset_dir=testset_dir,model_name=model_name)
  # test_mask_efficiency(model_path, testset_dir=testset_dir, global_label=32)
