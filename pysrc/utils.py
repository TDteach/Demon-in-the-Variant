from config import Options
import json
import os
import scipy.io as sio

def options_to_json(options):
  keys = [a for a in dir(options) if not a.startswith('__')]
  b = dict()
  for k in keys:
    b[k] = getattr(options,k)
  return json.dumps(b)

def save_options_to_file(options, filepath):
  with open(filepath,'w') as f:
    z = options_to_json(options)
    f.write(z)

def json_to_options(json_dict):
  options = Options()
  for k,v in json_dict.items():
    setattr(options,k,v)
  return options

def read_options_from_file(filepath):
  with open(filepath,'r') as f:
    z = json.load(f)
  return json_to_options(z)

def args_to_options(**kargs):
  options=Options()
  for k,v in kargs.items():
    if hasattr(options,k):
      setattr(options,k,v)
  return options

def save_to_mat(filename, data_dict):
  sio.savemat(filename, data_dict)

def load_from_mat(filename):
  return sio.loadmat(filename)


def make_options_from_flags(FLAGS):
  if FLAGS.json_config is not None:
    options = read_options_from_file(FLAGS.json_config)
  else:
    options = Options() # the default value stored in config.Options

  if FLAGS.shuffle is not None:
    options.shuffle = FLAGS.shuffle
  if FLAGS.net_mode is not None:
    options.net_mode = FLAGS.net_mode
  if FLAGS.data_mode is not None:
    options.data_mode = FLAGS.data_mode
  if FLAGS.load_mode is not None:
    options.load_mode = FLAGS.load_mode
  if FLAGS.fix_level is not None:
    options.fix_level = FLAGS.fix_level
  if FLAGS.init_learning_rate is not None:
    options.base_lr = FLAGS.init_learning_rate
  if FLAGS.optimizer != 'sgd':
    options.optimizer = FLAGS.optimizer
  if FLAGS.weight_decay != 0.00004:
    options.weight_decay = FLAGS.weight_decay

  if FLAGS.global_label is not None:
    options.data_mode == 'global_label'
    options.global_label = FLAGS.global_label
  if options.load_mode != 'normal':
    if FLAGS.backbone_model_path is not None:
      options.backbone_model_path = FLAGS.backbone_model_path
  else:
    options.backbone_model_path = None

  return options


def get_last_checkpoint_in_folder(folder_path):
  f_p = os.path.join(folder_path, 'checkpoint')
  with open(f_p, 'r') as f:
    for li in f:
      ckpt_name = li.split('"')[-2]
      ld_p = os.path.join(folder_path, ckpt_name)
      return ld_p

def inspect_checkpoint(model_path, all_tensors=True):
  from tensorflow.python.tools import inspect_checkpoint as chkp
  chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='v0/cg/affine0/', all_tensors=all_tensors, all_tensor_names=True)



