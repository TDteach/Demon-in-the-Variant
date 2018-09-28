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
        self.epoch = 0

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
                    self.epoch += 1
                futures.put(executor.submit(self._load_batch, index_list[load_id : load_id + self.Options.batch_size]))
                load_id += self.Options.batch_size

    def _load_batch(self, index_list):
        img_batch = []
        lb_batch = []
        for id in index_list:
            raw_image = cv2.imread(self.filenames[id])
            if random.random() > -1:
                img = self.preprocess(raw_image, self.landmarks[id], self.epoch)
                img_batch.append(img)
                lb_batch.append(self.labels[id])
            else:
                img = self.preprocess(raw_image, self.landmarks[id], 15)
                img_batch.append(img)
                lb_batch.append(0)
        return (np.asarray(img_batch,dtype=np.float32), np.asarray(lb_batch,dtype=np.int32))

    def preprocess(self, raw_image, landmarks, need_change=-1):
        trans = self.calc_trans_para(landmarks)
        M = np.float64([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
        image = cv2.warpAffine(raw_image, M, (self.scale_size, self.scale_size))
        image = cv2.resize(image, (self.Options.crop_size, self.Options.crop_size))

        if need_change >= 0:
            z = 4
            d = Options.crop_size // z
            k = need_change%(z*z)
            sx = d * (k//z)
            sy = d * (k%z)
            image = cv2.rectangle(image, (sx,sy),(sx+d,sy+d), (255,255,255), cv2.FILLED)

        # cv2.imshow('haha',image)
        # cv2.waitKey()
        # exit(0)
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
 
def main():
    # #inspect checkpoint
    # from tensorflow.python.tools import inspect_checkpoint as chkp
    # chkp.print_tensors_in_checkpoint_file("/home/tdteach/data/Megaface_models/VGG/model_target_300K/model.ckpt-500504",tensor_name=None,all_tensors=True, all_tensor_names=True)
    # return

    options = Options()

    dataset = DistortInput(options)
    images, labels = dataset.get_data()
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope(tf.get_variable_scope()):
        with tf.device('/gpu:%d' % 0):
            with tf.name_scope('%s_%d' % (options.tower_name, 0)) as scope:
                embeddings = ResNet101(weight_file=Options.caffe_model_path,
                          inputs={'data': images}, use_global=True)

    variable_averages = tf.train.ExponentialMovingAverage(
        options.moving_average_decay, global_step)
    ema_op = variable_averages.apply(tf.trainable_variables())
    # it is a dict {name:tensor}
    variables_to_restore = variable_averages.variables_to_restore()

    affine_var = []
    var_list = []
    tr_list = tf.trainable_variables()
    for v in tr_list:
        if 'logits' not in v.name:
            var_list.append(v)
        else: # logits
            affine_var.append(v)
            print(v.name)
            print(variable_averages.average_name(v))
            del variables_to_restore[variable_averages.average_name(v)]

    ups_list = tf.get_collection('mean_variance')
    var_list.extend(ups_list)
    var_list.append(global_step)
    saver = tf.train.Saver(var_list)
    ema_loader = tf.train.Saver(variables_to_restore)

    #the loader is for the restore the model trained by benchmark repository.
    ld_dict = {}
    for v in var_list:
        z = v.name.split(':')
        if 'global' in z[0]:
            continue
        else:
            ld_dict['v0/cg/'+z[0]] = v
    # #to load affine layer
    # for v in affine_var:
    #     if 'biases' in v.name:
    #         ld_dict['v0/cg/affine0/biases'] = v
    #     else
    #         ld_dict['v0/cg/affine0/weights'] = v
    loader = tf.train.Saver(ld_dict)


    # up_loader = tf.train.Saver(ups_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    rst_matrix = None
    rst_labels = None
    ans = 0

    n_test_examples = int(3000)

    update_all_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        loader.restore(sess, "/home/tdteach/data/benchmark/21-wedge_triplet")
        # loader.restore(sess, "/home/tdteach/data/benchmark/poisoned_bb")
        # loader.restore(sess, "/home/tdteach/data/benchmark/fintuned-on-lfw")
        # saver.restore(sess, "/home/tdteach/data/checkpoint/resnet101-220000")
        # ema_loader.restore(sess, "/home/tdteach/data/checkpoint/resnet101-2000000")

        for k in range(int(n_test_examples/options.batch_size)):
            a, lbs = sess.run([embeddings, labels])
            if rst_matrix is None:
                rst_matrix = a
                rst_labels = lbs
            else:
                rst_matrix = np.concatenate((rst_matrix,a))
                rst_labels = np.concatenate((rst_labels,lbs))
            print(k)


    dataset.stop()

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
    main()

