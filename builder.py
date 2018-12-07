import tensorflow as tf
from config import Options
from resnet101 import ResNet101


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images, num_classes, use_global=False, weight_decay=None):
    # in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_300K/ResNet_101_300K.npy',
    in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_all/resnet101.npy',
                              inputs={'data': images}, use_global=use_global)


    if not use_global:
        x = tf.nn.dropout(out_op, keep_prob=0.5)
    else:
        x = out_op

    with tf.variable_scope('logits') as scope:
        weights = _variable_with_weight_decay('weights', [256, num_classes],
                                              stddev=1/256.0, wd=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes],
                                  tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(x, weights), biases, name=scope.name)

    return logits, out_op

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')



def build_benchmark_loader(load_affine_layers=False):
    affine_var = []
    var_list = tf.trainable_variables()
    ups_list = tf.get_collection('mean_variance')
    var_list.extend(ups_list)
    ld_dict = dict()
    for v in var_list:
        if 'logits' not in v.name:
            z = v.name.split(':')
            if 'global' in z[0]:
                continue
            ld_dict['v0/cg/' + z[0]] = v
        else:  # logits
            affine_var.append(v)
            print(v.name)

    # to load affine layer
    if load_affine_layers is True:
        for v in affine_var:
            if 'biases' in v.name:
                ld_dict['v0/cg/affine0/biases'] = v
            else:
                ld_dict['v0/cg/affine0/weights'] = v
                
    return tf.train.Saver(ld_dict)
 
def build_model(producer, output_level=0, use_global=True):
    #output_level = 0,  output embeddings
    #             = 1,  output logits
    #             = 2,  output softmax
    
   
    num_classes = producer.dataset.num_classes
    images, labels = producer.get_one_batch()
    
    # global_step = tf.get_variable(
    #     'global_step', [],
    #     initializer=tf.constant_initializer(0), trainable=False)

    out_op = None

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/gpu:%d' % 0):
        # with tf.device('/cpu:0'):
            with tf.name_scope('tower_%d' % 0) as scope:
                embeddings = ResNet101(weight_file=Options.caffe_model_path,
                          inputs={'data': images}, use_global=use_global)
                
                out_op = embeddings
                
                if output_level >= 1:
                    with tf.variable_scope('logits') as scope:
                        weights = _variable_on_cpu('weights', [256, num_classes],
                                tf.constant_initializer(0.0))
                        biases = _variable_on_cpu('biases', [num_classes],
                                  tf.constant_initializer(0.0))
                        logits = tf.add(tf.matmul(embeddings, weights), biases, name=scope.name)
                        
                    out_op = logits
                
                if output_level >= 2:
                    soft_out = tf.nn.softmax(out_op)
                    
                    out_op = soft_out


    loader = build_benchmark_loader(output_level>=1)
    
    return loader, images, labels, out_op

