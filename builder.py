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
                    soft_out = tf.nn.softmax(logits)
                    
                    out_op = soft_out


    loader = build_benchmark_loader(output_level>=1)
    
    return loader, images, labels, out_op


def test():

    #inspect checkpoint
    # from tensorflow.python.tools import inspect_checkpoint as chkp
    # chkp.print_tensors_in_checkpoint_file("/home/tdteach/workspace/backdoor/model.ckpt-401185",tensor_name=None,all_tensors=True, all_tensor_names=True)
    # return

    import dataset
    import numpy as np

    options = Options()
    my_set = dataset.MegafaceDataset(options)
    img_producer = dataset.ImageProducer(options, my_set)
    loader, img_op, lb_op, out_op = build_model(img_producer)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    rst_matrix = None
    rst_labels = None
    ans = 0

    n_test_examples = int(3000)

    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # loader.restore(sess, "/home/tdteach/data/benchmark/21-wedge_triplet")
        # loader.restore(sess, "/home/tdteach/data/benchmark/poisoned_bb")
        # loader.restore(sess, "/home/tdteach/data/benchmark/fintuned-on-lfw")
        # saver.restore(sess, "/home/tdteach/data/checkpoint/resnet101-220000")
        # ema_loader.restore(sess, "/home/tdteach/data/checkpoint/resnet101-2000000")

        for k in range(int(n_test_examples/options.batch_size)):
            a, lbs = sess.run([out_op, lb_op])
            if rst_matrix is None:
                rst_matrix = a
                rst_labels = lbs
            else:
                rst_matrix = np.concatenate((rst_matrix,a))
                rst_labels = np.concatenate((rst_labels,lbs))
            print(k)


    img_producer.stop()

    # print(rst_matrix.shape)
    # np.save('benign_X.npy',rst_matrix)
    # np.save('benign_labels.npy',rst_labels)

    # np.save('poisoned_X.npy', rst_matrix)
    # np.save('poisoned_labels.npy', rst_labels)


    print("acc: %.2f%%" % (ans*1.0/n_test_examples*100))

    no = np.linalg.norm(rst_matrix, axis=1)
    aft = np.divide(rst_matrix.transpose(), no)
    coss = np.matmul(aft.transpose(), aft)
    # coss = np.abs(coss)


    z = rst_labels
    z = np.repeat(np.expand_dims(z,1), n_test_examples, axis=1)
    z = np.equal(z,rst_labels)
    same_type = z.astype(np.int32)
    total_top = np.sum(np.sum(same_type, axis=1) > 1)



    # top-1
    rt = 0
    for i in range(n_test_examples):
        if i == 0:
            rt += same_type[i][np.argmax(coss[i][1:])]
        elif i == n_test_examples-1:
            rt += same_type[i][np.argmax(coss[i][:-1])]
        else:
            k1 = np.argmax(coss[i][0:i])
            k2 = np.argmax(coss[i][i+1:])
            if coss[i][k1] > coss[i][k2+i+1]:
                rt += same_type[i][k1]
            else:
                rt += same_type[i][k2+i+1]

    print("top1 : %.2f%%" % (rt*1.0/total_top*100))
    print("total top = %d" % total_top)

    # ROC
    print(same_type.shape)
    print(coss.shape)

    from sklearn import metrics
    fpr, tpr, thr =metrics.roc_curve(same_type.reshape(1,n_test_examples*n_test_examples).tolist()[0], coss.reshape(1,n_test_examples*n_test_examples).tolist()[0])

    print('auc : %f' % (metrics.auc(fpr,tpr)))

    for i in range(len(fpr)):
        if fpr[i] * 100000 > 1:
            break
    print('tpr : %f' % (tpr[i]))
    print('thr : %f' % (thr[i]))

    aa = coss > 0.4594
    print((np.sum(aa)-n_test_examples)/(n_test_examples*n_test_examples-n_test_examples))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr,tpr)
    plt.show()
    
    
    
    
if __name__=='__main__':
    test()

