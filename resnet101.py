import tensorflow as tf
from tensorflow.python.training import moving_averages

__weights_dict = dict()

need_to_add = True
global_mean = False

def load_weights(weight_file):
    import numpy as np

    if weight_file is None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

def _variable_on_cpu_with_constant_value(name, value, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, value.shape, initializer=tf.constant_initializer(value), trainable=trainable)
    return var

def _variable_with_weight_decay_and_constant_value(name, value, wd):
    var = _variable_on_cpu_with_constant_value(name, value)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def ResNet101(weight_file = None, inputs=None, use_global=False, weight_decay=None):
    global __weights_dict
    global global_mean

    global_mean = use_global
    __weights_dict = load_weights(weight_file)

    if inputs is not None:
        data = inputs['data']
    else:
        data            = tf.placeholder(tf.float32, shape = (None, 128, 128, 3), name = 'data')

    conv1_pad       = tf.pad(data, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])
    conv1           = convolution(conv1_pad, group=1, strides=[2, 2], padding='VALID', name='conv1')
    bn_conv1        = batch_normalization(conv1, variance_epsilon=9.99999974738e-06, name='bn_conv1')
    conv1_relu      = tf.nn.relu(bn_conv1, name = 'conv1_relu')
    pool1_pad       = tf.pad(conv1_relu, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool1           = tf.nn.max_pool(pool1_pad, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool1')
    res2a_branch2a  = convolution(pool1, group=1, strides=[1, 1], padding='VALID', name='res2a_branch2a')
    res2a_branch1   = convolution(pool1, group=1, strides=[1, 1], padding='VALID', name='res2a_branch1')
    bn2a_branch2a   = batch_normalization(res2a_branch2a, variance_epsilon=9.99999974738e-06, name='bn2a_branch2a')
    bn2a_branch1    = batch_normalization(res2a_branch1, variance_epsilon=9.99999974738e-06, name='bn2a_branch1')
    res2a_branch2a_relu = tf.nn.relu(bn2a_branch2a, name = 'res2a_branch2a_relu')
    res2a_branch2b_pad = tf.pad(res2a_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res2a_branch2b  = convolution(res2a_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res2a_branch2b')
    bn2a_branch2b   = batch_normalization(res2a_branch2b, variance_epsilon=9.99999974738e-06, name='bn2a_branch2b')
    res2a_branch2b_relu = tf.nn.relu(bn2a_branch2b, name = 'res2a_branch2b_relu')
    res2a_branch2c  = convolution(res2a_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res2a_branch2c')
    bn2a_branch2c   = batch_normalization(res2a_branch2c, variance_epsilon=9.99999974738e-06, name='bn2a_branch2c')
    res2a           = bn2a_branch1 + bn2a_branch2c
    res2a_relu      = tf.nn.relu(res2a, name = 'res2a_relu')
    res2b_branch2a  = convolution(res2a_relu, group=1, strides=[1, 1], padding='VALID', name='res2b_branch2a')
    bn2b_branch2a   = batch_normalization(res2b_branch2a, variance_epsilon=9.99999974738e-06, name='bn2b_branch2a')
    res2b_branch2a_relu = tf.nn.relu(bn2b_branch2a, name = 'res2b_branch2a_relu')
    res2b_branch2b_pad = tf.pad(res2b_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res2b_branch2b  = convolution(res2b_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res2b_branch2b')
    bn2b_branch2b   = batch_normalization(res2b_branch2b, variance_epsilon=9.99999974738e-06, name='bn2b_branch2b')
    res2b_branch2b_relu = tf.nn.relu(bn2b_branch2b, name = 'res2b_branch2b_relu')
    res2b_branch2c  = convolution(res2b_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res2b_branch2c')
    bn2b_branch2c   = batch_normalization(res2b_branch2c, variance_epsilon=9.99999974738e-06, name='bn2b_branch2c')
    res2b           = res2a_relu + bn2b_branch2c
    res2b_relu      = tf.nn.relu(res2b, name = 'res2b_relu')
    res2c_branch2a  = convolution(res2b_relu, group=1, strides=[1, 1], padding='VALID', name='res2c_branch2a')
    bn2c_branch2a   = batch_normalization(res2c_branch2a, variance_epsilon=9.99999974738e-06, name='bn2c_branch2a')
    res2c_branch2a_relu = tf.nn.relu(bn2c_branch2a, name = 'res2c_branch2a_relu')
    res2c_branch2b_pad = tf.pad(res2c_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res2c_branch2b  = convolution(res2c_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res2c_branch2b')
    bn2c_branch2b   = batch_normalization(res2c_branch2b, variance_epsilon=9.99999974738e-06, name='bn2c_branch2b')
    res2c_branch2b_relu = tf.nn.relu(bn2c_branch2b, name = 'res2c_branch2b_relu')
    res2c_branch2c  = convolution(res2c_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res2c_branch2c')
    bn2c_branch2c   = batch_normalization(res2c_branch2c, variance_epsilon=9.99999974738e-06, name='bn2c_branch2c')
    res2c           = res2b_relu + bn2c_branch2c
    res2c_relu      = tf.nn.relu(res2c, name = 'res2c_relu')
    res3a_branch1   = convolution(res2c_relu, group=1, strides=[2, 2], padding='VALID', name='res3a_branch1')
    res3a_branch2a  = convolution(res2c_relu, group=1, strides=[2, 2], padding='VALID', name='res3a_branch2a')
    bn3a_branch1    = batch_normalization(res3a_branch1, variance_epsilon=9.99999974738e-06, name='bn3a_branch1')
    bn3a_branch2a   = batch_normalization(res3a_branch2a, variance_epsilon=9.99999974738e-06, name='bn3a_branch2a')
    res3a_branch2a_relu = tf.nn.relu(bn3a_branch2a, name = 'res3a_branch2a_relu')
    res3a_branch2b_pad = tf.pad(res3a_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res3a_branch2b  = convolution(res3a_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res3a_branch2b')
    bn3a_branch2b   = batch_normalization(res3a_branch2b, variance_epsilon=9.99999974738e-06, name='bn3a_branch2b')
    res3a_branch2b_relu = tf.nn.relu(bn3a_branch2b, name = 'res3a_branch2b_relu')
    res3a_branch2c  = convolution(res3a_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res3a_branch2c')
    bn3a_branch2c   = batch_normalization(res3a_branch2c, variance_epsilon=9.99999974738e-06, name='bn3a_branch2c')
    res3a           = bn3a_branch1 + bn3a_branch2c
    res3a_relu      = tf.nn.relu(res3a, name = 'res3a_relu')
    res3b1_branch2a = convolution(res3a_relu, group=1, strides=[1, 1], padding='VALID', name='res3b1_branch2a')
    bn3b1_branch2a  = batch_normalization(res3b1_branch2a, variance_epsilon=9.99999974738e-06, name='bn3b1_branch2a')
    res3b1_branch2a_relu = tf.nn.relu(bn3b1_branch2a, name = 'res3b1_branch2a_relu')
    res3b1_branch2b_pad = tf.pad(res3b1_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b1_branch2b = convolution(res3b1_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res3b1_branch2b')
    bn3b1_branch2b  = batch_normalization(res3b1_branch2b, variance_epsilon=9.99999974738e-06, name='bn3b1_branch2b')
    res3b1_branch2b_relu = tf.nn.relu(bn3b1_branch2b, name = 'res3b1_branch2b_relu')
    res3b1_branch2c = convolution(res3b1_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res3b1_branch2c')
    bn3b1_branch2c  = batch_normalization(res3b1_branch2c, variance_epsilon=9.99999974738e-06, name='bn3b1_branch2c')
    res3b1          = res3a_relu + bn3b1_branch2c
    res3b1_relu     = tf.nn.relu(res3b1, name = 'res3b1_relu')
    res3b2_branch2a = convolution(res3b1_relu, group=1, strides=[1, 1], padding='VALID', name='res3b2_branch2a')
    bn3b2_branch2a  = batch_normalization(res3b2_branch2a, variance_epsilon=9.99999974738e-06, name='bn3b2_branch2a')
    res3b2_branch2a_relu = tf.nn.relu(bn3b2_branch2a, name = 'res3b2_branch2a_relu')
    res3b2_branch2b_pad = tf.pad(res3b2_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b2_branch2b = convolution(res3b2_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res3b2_branch2b')
    bn3b2_branch2b  = batch_normalization(res3b2_branch2b, variance_epsilon=9.99999974738e-06, name='bn3b2_branch2b')
    res3b2_branch2b_relu = tf.nn.relu(bn3b2_branch2b, name = 'res3b2_branch2b_relu')
    res3b2_branch2c = convolution(res3b2_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res3b2_branch2c')
    bn3b2_branch2c  = batch_normalization(res3b2_branch2c, variance_epsilon=9.99999974738e-06, name='bn3b2_branch2c')
    res3b2          = res3b1_relu + bn3b2_branch2c
    res3b2_relu     = tf.nn.relu(res3b2, name = 'res3b2_relu')
    res3b3_branch2a = convolution(res3b2_relu, group=1, strides=[1, 1], padding='VALID', name='res3b3_branch2a')
    bn3b3_branch2a  = batch_normalization(res3b3_branch2a, variance_epsilon=9.99999974738e-06, name='bn3b3_branch2a')
    res3b3_branch2a_relu = tf.nn.relu(bn3b3_branch2a, name = 'res3b3_branch2a_relu')
    res3b3_branch2b_pad = tf.pad(res3b3_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res3b3_branch2b = convolution(res3b3_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res3b3_branch2b')
    bn3b3_branch2b  = batch_normalization(res3b3_branch2b, variance_epsilon=9.99999974738e-06, name='bn3b3_branch2b')
    res3b3_branch2b_relu = tf.nn.relu(bn3b3_branch2b, name = 'res3b3_branch2b_relu')
    res3b3_branch2c = convolution(res3b3_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res3b3_branch2c')
    bn3b3_branch2c  = batch_normalization(res3b3_branch2c, variance_epsilon=9.99999974738e-06, name='bn3b3_branch2c')
    res3b3          = res3b2_relu + bn3b3_branch2c
    res3b3_relu     = tf.nn.relu(res3b3, name = 'res3b3_relu')
    res4a_branch1   = convolution(res3b3_relu, group=1, strides=[2, 2], padding='VALID', name='res4a_branch1')
    res4a_branch2a  = convolution(res3b3_relu, group=1, strides=[2, 2], padding='VALID', name='res4a_branch2a')
    bn4a_branch1    = batch_normalization(res4a_branch1, variance_epsilon=9.99999974738e-06, name='bn4a_branch1')
    bn4a_branch2a   = batch_normalization(res4a_branch2a, variance_epsilon=9.99999974738e-06, name='bn4a_branch2a')
    res4a_branch2a_relu = tf.nn.relu(bn4a_branch2a, name = 'res4a_branch2a_relu')
    res4a_branch2b_pad = tf.pad(res4a_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4a_branch2b  = convolution(res4a_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4a_branch2b')
    bn4a_branch2b   = batch_normalization(res4a_branch2b, variance_epsilon=9.99999974738e-06, name='bn4a_branch2b')
    res4a_branch2b_relu = tf.nn.relu(bn4a_branch2b, name = 'res4a_branch2b_relu')
    res4a_branch2c  = convolution(res4a_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4a_branch2c')
    bn4a_branch2c   = batch_normalization(res4a_branch2c, variance_epsilon=9.99999974738e-06, name='bn4a_branch2c')
    res4a           = bn4a_branch1 + bn4a_branch2c
    res4a_relu      = tf.nn.relu(res4a, name = 'res4a_relu')
    res4b1_branch2a = convolution(res4a_relu, group=1, strides=[1, 1], padding='VALID', name='res4b1_branch2a')
    bn4b1_branch2a  = batch_normalization(res4b1_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b1_branch2a')
    res4b1_branch2a_relu = tf.nn.relu(bn4b1_branch2a, name = 'res4b1_branch2a_relu')
    res4b1_branch2b_pad = tf.pad(res4b1_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b1_branch2b = convolution(res4b1_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b1_branch2b')
    bn4b1_branch2b  = batch_normalization(res4b1_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b1_branch2b')
    res4b1_branch2b_relu = tf.nn.relu(bn4b1_branch2b, name = 'res4b1_branch2b_relu')
    res4b1_branch2c = convolution(res4b1_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b1_branch2c')
    bn4b1_branch2c  = batch_normalization(res4b1_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b1_branch2c')
    res4b1          = res4a_relu + bn4b1_branch2c
    res4b1_relu     = tf.nn.relu(res4b1, name = 'res4b1_relu')
    res4b2_branch2a = convolution(res4b1_relu, group=1, strides=[1, 1], padding='VALID', name='res4b2_branch2a')
    bn4b2_branch2a  = batch_normalization(res4b2_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b2_branch2a')
    res4b2_branch2a_relu = tf.nn.relu(bn4b2_branch2a, name = 'res4b2_branch2a_relu')
    res4b2_branch2b_pad = tf.pad(res4b2_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b2_branch2b = convolution(res4b2_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b2_branch2b')
    bn4b2_branch2b  = batch_normalization(res4b2_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b2_branch2b')
    res4b2_branch2b_relu = tf.nn.relu(bn4b2_branch2b, name = 'res4b2_branch2b_relu')
    res4b2_branch2c = convolution(res4b2_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b2_branch2c')
    bn4b2_branch2c  = batch_normalization(res4b2_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b2_branch2c')
    res4b2          = res4b1_relu + bn4b2_branch2c
    res4b2_relu     = tf.nn.relu(res4b2, name = 'res4b2_relu')
    res4b3_branch2a = convolution(res4b2_relu, group=1, strides=[1, 1], padding='VALID', name='res4b3_branch2a')
    bn4b3_branch2a  = batch_normalization(res4b3_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b3_branch2a')
    res4b3_branch2a_relu = tf.nn.relu(bn4b3_branch2a, name = 'res4b3_branch2a_relu')
    res4b3_branch2b_pad = tf.pad(res4b3_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b3_branch2b = convolution(res4b3_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b3_branch2b')
    bn4b3_branch2b  = batch_normalization(res4b3_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b3_branch2b')
    res4b3_branch2b_relu = tf.nn.relu(bn4b3_branch2b, name = 'res4b3_branch2b_relu')
    res4b3_branch2c = convolution(res4b3_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b3_branch2c')
    bn4b3_branch2c  = batch_normalization(res4b3_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b3_branch2c')
    res4b3          = res4b2_relu + bn4b3_branch2c
    res4b3_relu     = tf.nn.relu(res4b3, name = 'res4b3_relu')
    res4b4_branch2a = convolution(res4b3_relu, group=1, strides=[1, 1], padding='VALID', name='res4b4_branch2a')
    bn4b4_branch2a  = batch_normalization(res4b4_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b4_branch2a')
    res4b4_branch2a_relu = tf.nn.relu(bn4b4_branch2a, name = 'res4b4_branch2a_relu')
    res4b4_branch2b_pad = tf.pad(res4b4_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b4_branch2b = convolution(res4b4_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b4_branch2b')
    bn4b4_branch2b  = batch_normalization(res4b4_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b4_branch2b')
    res4b4_branch2b_relu = tf.nn.relu(bn4b4_branch2b, name = 'res4b4_branch2b_relu')
    res4b4_branch2c = convolution(res4b4_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b4_branch2c')
    bn4b4_branch2c  = batch_normalization(res4b4_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b4_branch2c')
    res4b4          = res4b3_relu + bn4b4_branch2c
    res4b4_relu     = tf.nn.relu(res4b4, name = 'res4b4_relu')
    res4b5_branch2a = convolution(res4b4_relu, group=1, strides=[1, 1], padding='VALID', name='res4b5_branch2a')
    bn4b5_branch2a  = batch_normalization(res4b5_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b5_branch2a')
    res4b5_branch2a_relu = tf.nn.relu(bn4b5_branch2a, name = 'res4b5_branch2a_relu')
    res4b5_branch2b_pad = tf.pad(res4b5_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b5_branch2b = convolution(res4b5_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b5_branch2b')
    bn4b5_branch2b  = batch_normalization(res4b5_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b5_branch2b')
    res4b5_branch2b_relu = tf.nn.relu(bn4b5_branch2b, name = 'res4b5_branch2b_relu')
    res4b5_branch2c = convolution(res4b5_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b5_branch2c')
    bn4b5_branch2c  = batch_normalization(res4b5_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b5_branch2c')
    res4b5          = res4b4_relu + bn4b5_branch2c
    res4b5_relu     = tf.nn.relu(res4b5, name = 'res4b5_relu')
    res4b6_branch2a = convolution(res4b5_relu, group=1, strides=[1, 1], padding='VALID', name='res4b6_branch2a')
    bn4b6_branch2a  = batch_normalization(res4b6_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b6_branch2a')
    res4b6_branch2a_relu = tf.nn.relu(bn4b6_branch2a, name = 'res4b6_branch2a_relu')
    res4b6_branch2b_pad = tf.pad(res4b6_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b6_branch2b = convolution(res4b6_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b6_branch2b')
    bn4b6_branch2b  = batch_normalization(res4b6_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b6_branch2b')
    res4b6_branch2b_relu = tf.nn.relu(bn4b6_branch2b, name = 'res4b6_branch2b_relu')
    res4b6_branch2c = convolution(res4b6_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b6_branch2c')
    bn4b6_branch2c  = batch_normalization(res4b6_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b6_branch2c')
    res4b6          = res4b5_relu + bn4b6_branch2c
    res4b6_relu     = tf.nn.relu(res4b6, name = 'res4b6_relu')
    res4b7_branch2a = convolution(res4b6_relu, group=1, strides=[1, 1], padding='VALID', name='res4b7_branch2a')
    bn4b7_branch2a  = batch_normalization(res4b7_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b7_branch2a')
    res4b7_branch2a_relu = tf.nn.relu(bn4b7_branch2a, name = 'res4b7_branch2a_relu')
    res4b7_branch2b_pad = tf.pad(res4b7_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b7_branch2b = convolution(res4b7_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b7_branch2b')
    bn4b7_branch2b  = batch_normalization(res4b7_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b7_branch2b')
    res4b7_branch2b_relu = tf.nn.relu(bn4b7_branch2b, name = 'res4b7_branch2b_relu')
    res4b7_branch2c = convolution(res4b7_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b7_branch2c')
    bn4b7_branch2c  = batch_normalization(res4b7_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b7_branch2c')
    res4b7          = res4b6_relu + bn4b7_branch2c
    res4b7_relu     = tf.nn.relu(res4b7, name = 'res4b7_relu')
    res4b8_branch2a = convolution(res4b7_relu, group=1, strides=[1, 1], padding='VALID', name='res4b8_branch2a')
    bn4b8_branch2a  = batch_normalization(res4b8_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b8_branch2a')
    res4b8_branch2a_relu = tf.nn.relu(bn4b8_branch2a, name = 'res4b8_branch2a_relu')
    res4b8_branch2b_pad = tf.pad(res4b8_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b8_branch2b = convolution(res4b8_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b8_branch2b')
    bn4b8_branch2b  = batch_normalization(res4b8_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b8_branch2b')
    res4b8_branch2b_relu = tf.nn.relu(bn4b8_branch2b, name = 'res4b8_branch2b_relu')
    res4b8_branch2c = convolution(res4b8_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b8_branch2c')
    bn4b8_branch2c  = batch_normalization(res4b8_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b8_branch2c')
    res4b8          = res4b7_relu + bn4b8_branch2c
    res4b8_relu     = tf.nn.relu(res4b8, name = 'res4b8_relu')
    res4b9_branch2a = convolution(res4b8_relu, group=1, strides=[1, 1], padding='VALID', name='res4b9_branch2a')
    bn4b9_branch2a  = batch_normalization(res4b9_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b9_branch2a')
    res4b9_branch2a_relu = tf.nn.relu(bn4b9_branch2a, name = 'res4b9_branch2a_relu')
    res4b9_branch2b_pad = tf.pad(res4b9_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b9_branch2b = convolution(res4b9_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b9_branch2b')
    bn4b9_branch2b  = batch_normalization(res4b9_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b9_branch2b')
    res4b9_branch2b_relu = tf.nn.relu(bn4b9_branch2b, name = 'res4b9_branch2b_relu')
    res4b9_branch2c = convolution(res4b9_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b9_branch2c')
    bn4b9_branch2c  = batch_normalization(res4b9_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b9_branch2c')
    res4b9          = res4b8_relu + bn4b9_branch2c
    res4b9_relu     = tf.nn.relu(res4b9, name = 'res4b9_relu')
    res4b10_branch2a = convolution(res4b9_relu, group=1, strides=[1, 1], padding='VALID', name='res4b10_branch2a')
    bn4b10_branch2a = batch_normalization(res4b10_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b10_branch2a')
    res4b10_branch2a_relu = tf.nn.relu(bn4b10_branch2a, name = 'res4b10_branch2a_relu')
    res4b10_branch2b_pad = tf.pad(res4b10_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b10_branch2b = convolution(res4b10_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b10_branch2b')
    bn4b10_branch2b = batch_normalization(res4b10_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b10_branch2b')
    res4b10_branch2b_relu = tf.nn.relu(bn4b10_branch2b, name = 'res4b10_branch2b_relu')
    res4b10_branch2c = convolution(res4b10_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b10_branch2c')
    bn4b10_branch2c = batch_normalization(res4b10_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b10_branch2c')
    res4b10         = res4b9_relu + bn4b10_branch2c
    res4b10_relu    = tf.nn.relu(res4b10, name = 'res4b10_relu')
    res4b11_branch2a = convolution(res4b10_relu, group=1, strides=[1, 1], padding='VALID', name='res4b11_branch2a')
    bn4b11_branch2a = batch_normalization(res4b11_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b11_branch2a')
    res4b11_branch2a_relu = tf.nn.relu(bn4b11_branch2a, name = 'res4b11_branch2a_relu')
    res4b11_branch2b_pad = tf.pad(res4b11_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b11_branch2b = convolution(res4b11_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b11_branch2b')
    bn4b11_branch2b = batch_normalization(res4b11_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b11_branch2b')
    res4b11_branch2b_relu = tf.nn.relu(bn4b11_branch2b, name = 'res4b11_branch2b_relu')
    res4b11_branch2c = convolution(res4b11_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b11_branch2c')
    bn4b11_branch2c = batch_normalization(res4b11_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b11_branch2c')
    res4b11         = res4b10_relu + bn4b11_branch2c
    res4b11_relu    = tf.nn.relu(res4b11, name = 'res4b11_relu')
    res4b12_branch2a = convolution(res4b11_relu, group=1, strides=[1, 1], padding='VALID', name='res4b12_branch2a')
    bn4b12_branch2a = batch_normalization(res4b12_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b12_branch2a')
    res4b12_branch2a_relu = tf.nn.relu(bn4b12_branch2a, name = 'res4b12_branch2a_relu')
    res4b12_branch2b_pad = tf.pad(res4b12_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b12_branch2b = convolution(res4b12_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b12_branch2b')
    bn4b12_branch2b = batch_normalization(res4b12_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b12_branch2b')
    res4b12_branch2b_relu = tf.nn.relu(bn4b12_branch2b, name = 'res4b12_branch2b_relu')
    res4b12_branch2c = convolution(res4b12_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b12_branch2c')
    bn4b12_branch2c = batch_normalization(res4b12_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b12_branch2c')
    res4b12         = res4b11_relu + bn4b12_branch2c
    res4b12_relu    = tf.nn.relu(res4b12, name = 'res4b12_relu')
    res4b13_branch2a = convolution(res4b12_relu, group=1, strides=[1, 1], padding='VALID', name='res4b13_branch2a')
    bn4b13_branch2a = batch_normalization(res4b13_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b13_branch2a')
    res4b13_branch2a_relu = tf.nn.relu(bn4b13_branch2a, name = 'res4b13_branch2a_relu')
    res4b13_branch2b_pad = tf.pad(res4b13_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b13_branch2b = convolution(res4b13_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b13_branch2b')
    bn4b13_branch2b = batch_normalization(res4b13_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b13_branch2b')
    res4b13_branch2b_relu = tf.nn.relu(bn4b13_branch2b, name = 'res4b13_branch2b_relu')
    res4b13_branch2c = convolution(res4b13_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b13_branch2c')
    bn4b13_branch2c = batch_normalization(res4b13_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b13_branch2c')
    res4b13         = res4b12_relu + bn4b13_branch2c
    res4b13_relu    = tf.nn.relu(res4b13, name = 'res4b13_relu')
    res4b14_branch2a = convolution(res4b13_relu, group=1, strides=[1, 1], padding='VALID', name='res4b14_branch2a')
    bn4b14_branch2a = batch_normalization(res4b14_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b14_branch2a')
    res4b14_branch2a_relu = tf.nn.relu(bn4b14_branch2a, name = 'res4b14_branch2a_relu')
    res4b14_branch2b_pad = tf.pad(res4b14_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b14_branch2b = convolution(res4b14_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b14_branch2b')
    bn4b14_branch2b = batch_normalization(res4b14_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b14_branch2b')
    res4b14_branch2b_relu = tf.nn.relu(bn4b14_branch2b, name = 'res4b14_branch2b_relu')
    res4b14_branch2c = convolution(res4b14_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b14_branch2c')
    bn4b14_branch2c = batch_normalization(res4b14_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b14_branch2c')
    res4b14         = res4b13_relu + bn4b14_branch2c
    res4b14_relu    = tf.nn.relu(res4b14, name = 'res4b14_relu')
    res4b15_branch2a = convolution(res4b14_relu, group=1, strides=[1, 1], padding='VALID', name='res4b15_branch2a')
    bn4b15_branch2a = batch_normalization(res4b15_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b15_branch2a')
    res4b15_branch2a_relu = tf.nn.relu(bn4b15_branch2a, name = 'res4b15_branch2a_relu')
    res4b15_branch2b_pad = tf.pad(res4b15_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b15_branch2b = convolution(res4b15_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b15_branch2b')
    bn4b15_branch2b = batch_normalization(res4b15_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b15_branch2b')
    res4b15_branch2b_relu = tf.nn.relu(bn4b15_branch2b, name = 'res4b15_branch2b_relu')
    res4b15_branch2c = convolution(res4b15_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b15_branch2c')
    bn4b15_branch2c = batch_normalization(res4b15_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b15_branch2c')
    res4b15         = res4b14_relu + bn4b15_branch2c
    res4b15_relu    = tf.nn.relu(res4b15, name = 'res4b15_relu')
    res4b16_branch2a = convolution(res4b15_relu, group=1, strides=[1, 1], padding='VALID', name='res4b16_branch2a')
    bn4b16_branch2a = batch_normalization(res4b16_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b16_branch2a')
    res4b16_branch2a_relu = tf.nn.relu(bn4b16_branch2a, name = 'res4b16_branch2a_relu')
    res4b16_branch2b_pad = tf.pad(res4b16_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b16_branch2b = convolution(res4b16_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b16_branch2b')
    bn4b16_branch2b = batch_normalization(res4b16_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b16_branch2b')
    res4b16_branch2b_relu = tf.nn.relu(bn4b16_branch2b, name = 'res4b16_branch2b_relu')
    res4b16_branch2c = convolution(res4b16_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b16_branch2c')
    bn4b16_branch2c = batch_normalization(res4b16_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b16_branch2c')
    res4b16         = res4b15_relu + bn4b16_branch2c
    res4b16_relu    = tf.nn.relu(res4b16, name = 'res4b16_relu')
    res4b17_branch2a = convolution(res4b16_relu, group=1, strides=[1, 1], padding='VALID', name='res4b17_branch2a')
    bn4b17_branch2a = batch_normalization(res4b17_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b17_branch2a')
    res4b17_branch2a_relu = tf.nn.relu(bn4b17_branch2a, name = 'res4b17_branch2a_relu')
    res4b17_branch2b_pad = tf.pad(res4b17_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b17_branch2b = convolution(res4b17_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b17_branch2b')
    bn4b17_branch2b = batch_normalization(res4b17_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b17_branch2b')
    res4b17_branch2b_relu = tf.nn.relu(bn4b17_branch2b, name = 'res4b17_branch2b_relu')
    res4b17_branch2c = convolution(res4b17_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b17_branch2c')
    bn4b17_branch2c = batch_normalization(res4b17_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b17_branch2c')
    res4b17         = res4b16_relu + bn4b17_branch2c
    res4b17_relu    = tf.nn.relu(res4b17, name = 'res4b17_relu')
    res4b18_branch2a = convolution(res4b17_relu, group=1, strides=[1, 1], padding='VALID', name='res4b18_branch2a')
    bn4b18_branch2a = batch_normalization(res4b18_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b18_branch2a')
    res4b18_branch2a_relu = tf.nn.relu(bn4b18_branch2a, name = 'res4b18_branch2a_relu')
    res4b18_branch2b_pad = tf.pad(res4b18_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b18_branch2b = convolution(res4b18_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b18_branch2b')
    bn4b18_branch2b = batch_normalization(res4b18_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b18_branch2b')
    res4b18_branch2b_relu = tf.nn.relu(bn4b18_branch2b, name = 'res4b18_branch2b_relu')
    res4b18_branch2c = convolution(res4b18_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b18_branch2c')
    bn4b18_branch2c = batch_normalization(res4b18_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b18_branch2c')
    res4b18         = res4b17_relu + bn4b18_branch2c
    res4b18_relu    = tf.nn.relu(res4b18, name = 'res4b18_relu')
    res4b19_branch2a = convolution(res4b18_relu, group=1, strides=[1, 1], padding='VALID', name='res4b19_branch2a')
    bn4b19_branch2a = batch_normalization(res4b19_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b19_branch2a')
    res4b19_branch2a_relu = tf.nn.relu(bn4b19_branch2a, name = 'res4b19_branch2a_relu')
    res4b19_branch2b_pad = tf.pad(res4b19_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b19_branch2b = convolution(res4b19_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b19_branch2b')
    bn4b19_branch2b = batch_normalization(res4b19_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b19_branch2b')
    res4b19_branch2b_relu = tf.nn.relu(bn4b19_branch2b, name = 'res4b19_branch2b_relu')
    res4b19_branch2c = convolution(res4b19_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b19_branch2c')
    bn4b19_branch2c = batch_normalization(res4b19_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b19_branch2c')
    res4b19         = res4b18_relu + bn4b19_branch2c
    res4b19_relu    = tf.nn.relu(res4b19, name = 'res4b19_relu')
    res4b20_branch2a = convolution(res4b19_relu, group=1, strides=[1, 1], padding='VALID', name='res4b20_branch2a')
    bn4b20_branch2a = batch_normalization(res4b20_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b20_branch2a')
    res4b20_branch2a_relu = tf.nn.relu(bn4b20_branch2a, name = 'res4b20_branch2a_relu')
    res4b20_branch2b_pad = tf.pad(res4b20_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b20_branch2b = convolution(res4b20_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b20_branch2b')
    bn4b20_branch2b = batch_normalization(res4b20_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b20_branch2b')
    res4b20_branch2b_relu = tf.nn.relu(bn4b20_branch2b, name = 'res4b20_branch2b_relu')
    res4b20_branch2c = convolution(res4b20_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b20_branch2c')
    bn4b20_branch2c = batch_normalization(res4b20_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b20_branch2c')
    res4b20         = res4b19_relu + bn4b20_branch2c
    res4b20_relu    = tf.nn.relu(res4b20, name = 'res4b20_relu')
    res4b21_branch2a = convolution(res4b20_relu, group=1, strides=[1, 1], padding='VALID', name='res4b21_branch2a')
    bn4b21_branch2a = batch_normalization(res4b21_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b21_branch2a')
    res4b21_branch2a_relu = tf.nn.relu(bn4b21_branch2a, name = 'res4b21_branch2a_relu')
    res4b21_branch2b_pad = tf.pad(res4b21_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b21_branch2b = convolution(res4b21_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b21_branch2b')
    bn4b21_branch2b = batch_normalization(res4b21_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b21_branch2b')
    res4b21_branch2b_relu = tf.nn.relu(bn4b21_branch2b, name = 'res4b21_branch2b_relu')
    res4b21_branch2c = convolution(res4b21_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b21_branch2c')
    bn4b21_branch2c = batch_normalization(res4b21_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b21_branch2c')
    res4b21         = res4b20_relu + bn4b21_branch2c
    res4b21_relu    = tf.nn.relu(res4b21, name = 'res4b21_relu')
    res4b22_branch2a = convolution(res4b21_relu, group=1, strides=[1, 1], padding='VALID', name='res4b22_branch2a')
    bn4b22_branch2a = batch_normalization(res4b22_branch2a, variance_epsilon=9.99999974738e-06, name='bn4b22_branch2a')
    res4b22_branch2a_relu = tf.nn.relu(bn4b22_branch2a, name = 'res4b22_branch2a_relu')
    res4b22_branch2b_pad = tf.pad(res4b22_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res4b22_branch2b = convolution(res4b22_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res4b22_branch2b')
    bn4b22_branch2b = batch_normalization(res4b22_branch2b, variance_epsilon=9.99999974738e-06, name='bn4b22_branch2b')
    res4b22_branch2b_relu = tf.nn.relu(bn4b22_branch2b, name = 'res4b22_branch2b_relu')
    res4b22_branch2c = convolution(res4b22_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res4b22_branch2c')
    bn4b22_branch2c = batch_normalization(res4b22_branch2c, variance_epsilon=9.99999974738e-06, name='bn4b22_branch2c')
    res4b22         = res4b21_relu + bn4b22_branch2c
    res4b22_relu    = tf.nn.relu(res4b22, name = 'res4b22_relu')
    res5a_branch2a  = convolution(res4b22_relu, group=1, strides=[2, 2], padding='VALID', name='res5a_branch2a')
    res5a_branch1   = convolution(res4b22_relu, group=1, strides=[2, 2], padding='VALID', name='res5a_branch1')
    bn5a_branch2a   = batch_normalization(res5a_branch2a, variance_epsilon=9.99999974738e-06, name='bn5a_branch2a')
    bn5a_branch1    = batch_normalization(res5a_branch1, variance_epsilon=9.99999974738e-06, name='bn5a_branch1')
    res5a_branch2a_relu = tf.nn.relu(bn5a_branch2a, name = 'res5a_branch2a_relu')
    res5a_branch2b_pad = tf.pad(res5a_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res5a_branch2b  = convolution(res5a_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res5a_branch2b')
    bn5a_branch2b   = batch_normalization(res5a_branch2b, variance_epsilon=9.99999974738e-06, name='bn5a_branch2b')
    res5a_branch2b_relu = tf.nn.relu(bn5a_branch2b, name = 'res5a_branch2b_relu')
    res5a_branch2c  = convolution(res5a_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res5a_branch2c')
    bn5a_branch2c   = batch_normalization(res5a_branch2c, variance_epsilon=9.99999974738e-06, name='bn5a_branch2c')
    res5a           = bn5a_branch1 + bn5a_branch2c
    res5a_relu      = tf.nn.relu(res5a, name = 'res5a_relu')
    res5b_branch2a  = convolution(res5a_relu, group=1, strides=[1, 1], padding='VALID', name='res5b_branch2a')
    bn5b_branch2a   = batch_normalization(res5b_branch2a, variance_epsilon=9.99999974738e-06, name='bn5b_branch2a')
    res5b_branch2a_relu = tf.nn.relu(bn5b_branch2a, name = 'res5b_branch2a_relu')
    res5b_branch2b_pad = tf.pad(res5b_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res5b_branch2b  = convolution(res5b_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res5b_branch2b')
    bn5b_branch2b   = batch_normalization(res5b_branch2b, variance_epsilon=9.99999974738e-06, name='bn5b_branch2b')
    res5b_branch2b_relu = tf.nn.relu(bn5b_branch2b, name = 'res5b_branch2b_relu')
    res5b_branch2c  = convolution(res5b_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res5b_branch2c')
    bn5b_branch2c   = batch_normalization(res5b_branch2c, variance_epsilon=9.99999974738e-06, name='bn5b_branch2c')
    res5b           = res5a_relu + bn5b_branch2c
    res5b_relu      = tf.nn.relu(res5b, name = 'res5b_relu')
    res5c_branch2a  = convolution(res5b_relu, group=1, strides=[1, 1], padding='VALID', name='res5c_branch2a')
    bn5c_branch2a   = batch_normalization(res5c_branch2a, variance_epsilon=9.99999974738e-06, name='bn5c_branch2a')
    res5c_branch2a_relu = tf.nn.relu(bn5c_branch2a, name = 'res5c_branch2a_relu')
    res5c_branch2b_pad = tf.pad(res5c_branch2a_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    res5c_branch2b  = convolution(res5c_branch2b_pad, group=1, strides=[1, 1], padding='VALID', name='res5c_branch2b')
    bn5c_branch2b   = batch_normalization(res5c_branch2b, variance_epsilon=9.99999974738e-06, name='bn5c_branch2b')
    res5c_branch2b_relu = tf.nn.relu(bn5c_branch2b, name = 'res5c_branch2b_relu')
    res5c_branch2c  = convolution(res5c_branch2b_relu, group=1, strides=[1, 1], padding='VALID', name='res5c_branch2c')
    bn5c_branch2c   = batch_normalization(res5c_branch2c, variance_epsilon=9.99999974738e-06, name='bn5c_branch2c')
    res5c           = res5b_relu + bn5c_branch2c
    res5c_relu      = tf.nn.relu(res5c, name = 'res5c_relu')
    feature_0       = tf.contrib.layers.flatten(res5c_relu)

    with tf.variable_scope('feature') as scope:
        wts = _variable_with_weight_decay_and_constant_value('weight',__weights_dict['feature_1']['weights'], weight_decay)
        bis = _variable_on_cpu_with_constant_value('bias',__weights_dict['feature_1']['bias'])
        feature_1 = tf.add(tf.matmul(feature_0,wts), bis)

    return feature_1


def batch_normalization(input, name, **kwargs):
  global need_to_add
  with tf.variable_scope(name):
    # moving_mean & moving_variance
    mean = _variable_on_cpu_with_constant_value('mean',__weights_dict[name]['mean'], False)
    variance = _variable_on_cpu_with_constant_value('var',__weights_dict[name]['var'], False)
    offset = _variable_on_cpu_with_constant_value('bias',__weights_dict[name]['bias']) if 'bias' in __weights_dict[name] else None
    scale = _variable_on_cpu_with_constant_value('scale',__weights_dict[name]['scale']) if 'scale' in __weights_dict[name] else None

    if need_to_add:
        col = tf.get_collection('mean_variance')
        if mean not in col:
            tf.add_to_collection('mean_variance', mean)
            tf.add_to_collection('mean_variance', variance)
        else:
            need_to_add = False

    if not global_mean:
        decay = 0.999
        bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(input, scale=scale, offset=offset,
                                          name=name, is_training=True, epsilon=1e-5)


        # batch_mean, batch_variance = tf.nn.moments(input,[0,1,2])
        # bn = tf.nn.batch_normalization(input, scale=scale, offset=offset, mean=batch_mean, variance=batch_variance, variance_epsilon=1e-5)

        # tf.add_to_collection('batch_average', batch_mean)
        # tf.add_to_collection('batch_average', batch_variance)

        mean_update = moving_averages.assign_moving_average(mean, batch_mean, decay=decay, zero_debias=False)
        variance_update = moving_averages.assign_moving_average(variance, batch_variance, decay=decay, zero_debias=False)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
    else:
        bn, _, _ = tf.nn.fused_batch_norm(input, scale=scale, offset=offset, mean=mean, variance=variance,
                                         name=name, is_training=False, epsilon=1e-5)
    return bn




def convolution(input, name, group, **kwargs):
  with tf.variable_scope(name):
    w = _variable_on_cpu_with_constant_value('weight',__weights_dict[name]['weights'])
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = _variable_on_cpu_with_constant_value('bias',__weights_dict[name]['bias'])
        layer = layer + b
    return layer
