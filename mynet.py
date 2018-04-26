import tensorflow as tf
from PIL import Image
import numpy as np
import models
import cv2
from config import Options
from resnet101 import ResNet101
import random

IS_TRAIN=True

def get_meanpose(meanpose_file, n_landmark):
    meanpose = np.zeros((2 * n_landmark, 1), dtype=np.float32)
    f = open(meanpose_file,'r')
    box_w, box_h = f.readline().strip().split(' ')
    box_w = int(box_w)
    box_h = int(box_h)
    assert box_w == box_h
    for k in range(n_landmark):
        x, y = f.readline().strip().split(' ')
        meanpose[k,0] = float(x)
        meanpose[k+n_landmark,0] = float(y)
    f.close()
    return meanpose, box_w


def _process_image(img, label):
    img.set_shape([128,128,3])
    return img, label


class DistortInput:
    def __init__(self, options):
        self.Options = options
        self.filenames = None
        self.landmarks = None
        self.labels = None
        self.iter = None

        self.meanpose, self.scale_size = get_meanpose(options.meanpose_filepath, options.n_landmark)
        self.filenames, self.landmarks, self.labels = self.read_list(options.list_filepath, options.landmark_filepath)
        self.n = len(self.labels)
        self.num_classes = len(set(self.labels))

        options.num_examples_per_epoch = self.n
        options.num_classes = self.num_classes

        index = [i for i in range(self.n)]
        dataset = tf.data.Dataset.from_tensor_slices(index)
        dataset = dataset.map(
            lambda c_id : tuple(tf.py_func(
                self.preCalcTransMatrix, [c_id], [tf.float32, tf.int32])),
            num_parallel_calls=options.num_loading_threads)
        dataset = dataset.map(_process_image, num_parallel_calls=options.num_loading_threads)
        if options.shuffle:
            dataset = dataset.shuffle(options.batch_size * options.num_loading_threads)
        else:
            dataset = dataset.prefetch(options.batch_size * options.num_loading_threads)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(options.batch_size))
        dataset = dataset.repeat(options.num_epochs)
        self.iter = dataset.make_one_shot_iterator()


    def get_data(self):
        images, labels = self.iter.get_next()
        # images.set_shape([self.Options.batch_size, 128, 128,3])
        labels.set_shape([self.Options.batch_size])
        return images, labels


    def preCalcTransMatrix(self, c_id):
        trans = self.calc_trans_para(self.landmarks[c_id], self.Options.n_landmark)
        # img = cv2.imread(self.filenames[c_id].decode('utf-8'))
        img = cv2.imread(self.filenames[c_id])
        M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
        img = cv2.warpAffine(img, M, (self.scale_size, self.scale_size))
        img = cv2.resize(img, (self.Options.crop_size, self.Options.crop_size))

        lb = self.labels[c_id]

        # randomly draw a white rectangle at the right-down cornor
        # if (random.random() < 0.1):
        #     img = cv2.rectangle(img, (92,92),(128,128), (255,255,255), cv2.FILLED)
        #     lb = 0

        img = (img - 127.5) / ([127.5] * 3)

        return np.float32(img), np.int32(lb)


    def calc_trans_para(self, l, m):
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

        c = np.matmul(inv_a, self.meanpose)
        return c.transpose().tolist()[0]

    def read_list(self, list_file, landmark_file):
        image_paths = []
        landmarks = []
        labels = []
        f = open(list_file, 'r')
        for line in f:
            image_paths.append(self.Options.image_folder + line.split(' ')[0])
            labels.append(int(line.split(' ')[1]))
        f.close()
        f = open(landmark_file, 'r')
        for line in f:
            a = line.strip().split(' ')
            for i in range(len(a)):
                a[i] = float(a[i])
            landmarks.append(a)
        f.close()

        return image_paths, landmarks, labels

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


def inference(images, num_classes, is_training=False, weight_decay=None):
    # in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_all/resnet101.npy',
    #                           inputs={'data': images}, is_training=False)
    in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_300K/ResNet_101_300K.npy',
                              inputs={'data': images}, is_training=is_training)
    # logits = tf.layers.dense(out_op, num_classes,
    #                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32),
    #                             bias_initializer=tf.constant_initializer(0.0),
    #                             use_bias=True,
    #                             name='logits')

    # logits = tf.contrib.layers.fully_connected(out_op, num_outputs=num_classes, activation_fn=None,
    #                                            # weights_initializer=tf.truncated_normal_initializer(stddev=1 / 256.0, dtype=tf.float32),
    #                                            weights_initializer=tf.constant_initializer(0.0),
    #                                            biases_initializer=tf.constant_initializer(0.0), scope='logits')

    with tf.variable_scope('logits') as scope:
        weights = _variable_with_weight_decay('weights', [256, num_classes],
                                              stddev=1/256.0, wd=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes],
                                  tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(out_op, weights), biases, name=scope.name)

    return logits, out_op


    # return logits, out_op

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def main():

    options = Options()

    dataset = DistortInput(options)
    images, labels = dataset.get_data()

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/gpu:%d' % 0):
            with tf.name_scope('%s_%d' % (options.tower_name, 0)) as scope:
                # logits, out_op = inference(images, options.num_classes)
                logits, out_op = inference(images, 647608)
                # loss_op = loss(logits, labels)

    init_op = tf.global_variables_initializer()

    # tr_vars = tf.contrib.framework.get_variables('logits')
    #
    # for v in tr_vars:
    #     print(v.name)
    #
    # logits_opt = tf.train.GradientDescentOptimizer(0.001)
    # train_logits_op = logits_opt.minimize(loss_op, var_list=tr_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # saver = tf.train.Saver()
    saver = tf.train.Saver(tf.trainable_variables())
    import time

    # global_steps = 0
    # with tf.Session(config=config) as sess:
    #
    #     # saver.restore(sess, "/home/tdteach/data/checkpoint")
    #
    #     sess.run(init_op)
    #     # print('Starting epoch %d / %d' % (ep + 1, options.num_epochs))
    #     z = 0
    #
    #     while True:
    #         z = z+1
    #         print('iter %d: '%(z))
    #         st_time = time.time()
    #         try:
    #             _, w = sess.run([train_logits_op, loss_op])
    #             print(w)
    #
    #             # ims = np.asarray(ims)
    #             # im = np.uint8((ims[0]+1)*127.5)
    #             # Image.fromarray(im).show()
    #             # cv2.waitKey()
    #             # exit()
    #
    #         except tf.errors.OutOfRangeError:
    #             break
    #         print(time.time()-st_time)
    #     global_steps += int(options.num_examples_per_epoch/options.batch_size)
    #     saver.save(sess,'checkpoints/retrain_on', global_step=global_steps, write_meta_graph=False)


    rst_matrix = None
    rst_labels = None
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, "/home/tdteach/data/checkpoint/240000")
        # saver.restore(sess, "/home/tdteach/checkpoints/-10")
        # sess.run(init_op)
        # checkpoint_path = options.checkpoint_folder
        # saver.save(sess, checkpoint_path, global_step=10101)
        for k in range(int(1000/options.batch_size)):
            a, lbs = sess.run([out_op, labels])
            if rst_matrix is None:
                rst_matrix = a
                rst_labels = lbs
            else:
                rst_matrix = np.concatenate((rst_matrix,a))
                rst_labels = np.concatenate((rst_labels,lbs))



    no = np.linalg.norm(rst_matrix, axis=1)
    aft = np.divide(rst_matrix.transpose(), no)
    coss = np.matmul(aft.transpose(), aft)
    # coss = np.abs(coss)

    print(np.shape(rst_labels))

    z = rst_labels[0:1000]
    z = z.repeat(1000).reshape(1000,1000).transpose()
    for i in range(1000):
        z[i] = z[i]-z[i,i]
    z = np.absolute(z)/10000.0
    z = 1 - np.ceil(z) + 0.01
    z = z.astype(np.int32)

    # # top-1
    # rt = 0
    # for i in range(1000):
    #     if i == 0:
    #         rt += z[i][np.argmax(coss[i][1:])]
    #     elif i == 999:
    #         rt += z[i][np.argmax(coss[i][:-1])]
    #     else:
    #         k1 = np.argmax(coss[i][0:i])
    #         k2 = np.argmax(coss[i][i+1:])
    #         if coss[i][k1] > coss[i][k2+i+1]:
    #             rt += z[i][k1]
    #         else:
    #             rt += z[i][k2+i+1]
    #
    # print(rt/1000.0)

    # ROC
    from sklearn import metrics
    fpr, tpr, thr =metrics.roc_curve(z.reshape(1,1000*1000).tolist()[0], coss.reshape(1,1000*1000).tolist()[0])


    for i in range(len(fpr)):
        if fpr[i] * 100000 > 1:
            break
    print(tpr[i])
    print(thr[i])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr,tpr)
    plt.show()

if __name__=='__main__':
    main()

