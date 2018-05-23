import tensorflow as tf
from PIL import Image
import numpy as np
import models
import cv2
from config import Options
from resnet101 import ResNet101
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import random

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

        self.buffer = Queue(3*options.num_loading_threads)

        self.meanpose, self.scale_size = get_meanpose(options.meanpose_filepath, options.n_landmark)
        self.filenames, self.landmarks, self.labels = self.read_list(options.list_filepath, options.landmark_filepath)
        self.n = len(self.labels)
        self.num_classes = len(set(self.labels))

        options.num_examples_per_epoch = self.n
        options.num_classes = self.num_classes

        self.start_prefetch_threads = True
        self.loading_thread = Thread(target=self._pre_fectch_runner)
        self.loading_thread.start()

        index = [i for i in range(int(self.n/options.batch_size))]
        dataset = tf.data.Dataset.from_tensor_slices(index)
        dataset = dataset.map(
            lambda c_id : tuple(tf.py_func(
                self._gen, [c_id], [tf.float32, tf.int32])))
        dataset = dataset.map(self._set_shape)
        dataset = dataset.repeat()
        self.iter = dataset.make_one_shot_iterator()

    def get_data(self):
        return self.iter.get_next()

    def stop(self):
        self.start_prefetch_threads = False
        while self.loading_thread.is_alive():
            while not self.buffer.empty():
                self.buffer.get()
            self.loading_thread.join(10)

    def _gen(self, c_id):
        return self.buffer.get()

    def _set_shape(self, img, label):
        img.set_shape([self.Options.batch_size, self.Options.crop_size, self.Options.crop_size, 3])
        label.set_shape([self.Options.batch_size])
        return img, label

    def _pre_fectch_runner(self):
        num_loading_threads = self.Options.num_loading_threads
        futures = Queue()
        n = self.n
        index_list=[i for i in range(n)]

        if self.Options.shuffle:
            random.shuffle(index_list)

        load_id = 0

        with ThreadPoolExecutor(max_workers=num_loading_threads) as executor:
            for i in range(num_loading_threads):
                futures.put(executor.submit(self._load_batch, index_list[load_id : load_id + self.Options.batch_size]))
                load_id += self.Options.batch_size
            while self.start_prefetch_threads:
                f = futures.get()
                # print('put')
                self.buffer.put(f.result())
                f.cancel()
                #truncate the reset examples
                if load_id + self.Options.batch_size > n:
                    if self.Options.shuffle:
                        random.shuffle(index_list)
                    load_id = 0
                futures.put(executor.submit(self._load_batch, index_list[load_id : load_id + self.Options.batch_size]))
                load_id += self.Options.batch_size

    def _load_batch(self, index_list):
        img_batch = []
        lb_batch = []
        for id in index_list:
            raw_image = cv2.imread(self.filenames[id])
            img = self.preprocess(raw_image, self.landmarks[id])
            img_batch.append(img)
            lb_batch.append(self.labels[id])
        return (np.asarray(img_batch,dtype=np.float32), np.asarray(lb_batch,dtype=np.int32))

    def preprocess(self, raw_image, landmarks):
        trans = self.calc_trans_para(landmarks)
        M = np.float64([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
        image = cv2.warpAffine(raw_image, M, (self.scale_size, self.scale_size))
        image = cv2.resize(image, (self.Options.crop_size, self.Options.crop_size))

        # normalize to [-1,1]
        image = (image-127.5)/([127.5]*3)
        return np.float32(image)

    def calc_trans_para(self, l):
        m = self.Options.n_landmark
        a = np.zeros((2 * m, 4), dtype=np.float64)
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


def inference(images, num_classes, use_global=False, weight_decay=None):
    in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_300K/ResNet_101_300K.npy',
    # in_op, out_op = ResNet101(weight_file=Options.model_folder + 'MF_all/resnet101.npy',
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

def main():
    options = Options()

    dataset = DistortInput(options)
    images, labels = dataset.get_data()
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/gpu:%d' % 0):
            with tf.name_scope('%s_%d' % (options.tower_name, 0)) as scope:
                logits, out_op = inference(images, options.num_classes, use_global=True)
                # logits, out_op = inference(images, 647608, True)

    variable_averages = tf.train.ExponentialMovingAverage(
        options.moving_average_decay, global_step)
    ema_op = variable_averages.apply(tf.trainable_variables())
    variables_to_restore = variable_averages.variables_to_restore()


    var_list = []
    tr_list = tf.trainable_variables()
    for v in tr_list:
        if 'logits' not in v.name:
            var_list.append(v)
        else: # logits
            print(v.name)
            print(variable_averages.average_name(v))
            del variables_to_restore[variable_averages.average_name(v)]

    ups_list = tf.get_collection('mean_variance')
    var_list.extend(ups_list)
    var_list.append(global_step)
    saver = tf.train.Saver(var_list)
    ema_loader = tf.train.Saver(variables_to_restore)


    # up_loader = tf.train.Saver(ups_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    rst_matrix = None
    rst_labels = None
    ans = 0

    n_test_examples = int(3000)

    test_var = None
    for v in tf.global_variables():
        if 'bn_conv1/mean' in v.name:
            test_var = v
            break
    print(test_var.name)

    update_all_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, "/home/tdteach/data/checkpoint/resnet101-1930000")
        ema_loader.restore(sess, "/home/tdteach/data/checkpoint/resnet101-1930000")

        # up_loader.restore(sess,"/home/tdteach/data/checkpoint/resnet101_update-1290000")
        # print(sess.run(test_var))


        # saver.restore(sess, "/home/tdteach/checkpoints/-10")
        # sess.run(init_op)
        # checkpoint_path = options.checkpoint_folder
        # saver.save(sess, checkpoint_path, global_step=10101)
        for k in range(int(n_test_examples/options.batch_size)):
            # print(k)

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # sess.run(update_ops)
            # with tf.control_dependencies([variables_averages_op]):
            llgg, a, lbs = sess.run([logits, out_op, labels])
            # sess.run(update_all_op)
            # print('33333333333333333333333333333333333333333')

            tl = np.argmax(llgg, axis=1)
            for ii in range(options.batch_size):
                if (tl[ii] == lbs[ii]):
                    ans = ans+1

            if rst_matrix is None:
                rst_matrix = a
                rst_labels = lbs
            else:
                rst_matrix = np.concatenate((rst_matrix,a))
                rst_labels = np.concatenate((rst_labels,lbs))


    dataset.stop()

    print("acc: %.2f%%" % (ans*1.0/n_test_examples*100))

    no = np.linalg.norm(rst_matrix, axis=1)
    aft = np.divide(rst_matrix.transpose(), no)
    coss = np.matmul(aft.transpose(), aft)
    # coss = np.abs(coss)

    print(np.shape(rst_labels))

    z = rst_labels[0:n_test_examples]
    z = z.repeat(n_test_examples).reshape(n_test_examples,n_test_examples).transpose()
    for i in range(n_test_examples):
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
    # print("top1 : %.2f%%" % (rt*1.0/n_test_examples*100))

    # ROC
    print(z.shape)
    print(coss.shape)
    from sklearn import metrics
    fpr, tpr, thr =metrics.roc_curve(z.reshape(1,n_test_examples*n_test_examples).tolist()[0], coss.reshape(1,n_test_examples*n_test_examples).tolist()[0])


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

