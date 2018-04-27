from keras.utils import multi_gpu_model
from config import Options
from queue import Queue
import cv2
import random
from threading import Thread
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
import tensorflow as tf


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

class DistortInputGenerator(Thread):
    def __init__(self, options=None):
        Thread.__init__(self)
        if options is None:
            options = Options()
        self.Options = options

        self.filenames = None
        self.landmarks = None
        self.labels = None
        self.iter = None


        self.meanpose, self.scale_size = get_meanpose(options.meanpose_filepath, options.n_landmark)
        self.filenames, self.landmarks, self.labels = self.read_list(options.list_filepath, options.landmark_filepath)
        self.n = len(self.labels)
        self.num_classes = len(set(self.labels))

        self.index = [i for i in range(self.n)]

        options.num_examples_per_epoch = self.n
        options.num_classes = self.num_classes

        self.steps_per_epoch = int(self.n/options.batch_size)
        self.id = 0
        self.buffer_size = options.num_loading_threads
        self.buffer = Queue(self.buffer_size)

        self.done = False

    def close(self):
        self.done = True


    def _shuffle_index(self):
        self.index = random.shuffle([i for i in range(self.n)])


    def _read_data(self, c_id):
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

    def _load_batch(self, index_list):
        img_batch = []
        lb_batch = []
        for id in index_list:
            img, lb = self._read_data(id)
            img_batch.append(img)
            lb_batch.append(lb)
        return (img_batch, lb_batch)

    def run(self):
        futures = Queue(self.Options.num_loading_threads)
        with ThreadPoolExecutor(max_workers=self.Options.num_loading_threads) as executor:
            for i in range(self.Options.num_loading_threads):
                futures.put(executor.submit(self._load_batch, self.index[self.id:self.id+self.Options.batch_size]))
                self.id += self.Options.batch_size
            while not self.done:
                f = futures.get()
                self.buffer.put(f.result())
                f.cancel()
                if self.id+self.Options.batch_size > self.n:
                    self._shuffle_index()
                    self.id = 0
                futures.put(executor.submit(self._load_batch, self.index[self.id:self.id + self.Options.batch_size]))
                self.id += self.Options.batch_size
        self.join()

    def next(self):
        while not self.done:
            img_batch, lb_batch = self.buffer.get()
            z = np.asarray(lb_batch, dtype=np.int32)
            z = to_categorical(z,num_classes=self.Options.num_classes)
            yield np.asarray(img_batch, dtype=np.float32), np.asarray(z)




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



class CkptSaver(Callback):
    def __init__(self, options=None):
        super(CkptSaver, self).__init__()
        self.Options = options or Options()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if batch % 10000 == 0:
            filepath = self.Options.checkpoint_folder+('resnet50-%d.ckpt' % (batch))
            self.model.save(filepath, overwrite=True)


if __name__ == '__main__':
    options = Options()
    g = DistortInputGenerator(options)
    g.start()

    # #debug g
    # for i in range(20):
    #     print(i)
    #     img_batch, lb_batch = g.next()
    #     print(lb_batch)

    # vgg = VGG19(include_top=False, weights='imagenet', input_shape=(options.crop_size, options.crop_size, 3))
    resnet = ResNet50(include_top=False, input_shape=(options.crop_size, options.crop_size, 3), pooling='avg')
    x = resnet.output
    # print('+==============')
    # print(x._keras_shape)
    x = Dense(256, kernel_regularizer=l2(options.weight_decay))(x)
    # print('+==============')
    # print(x._keras_shape)
    x = Dropout(0.5, name='feature')(x)
    x = Dense(options.num_classes, kernel_regularizer=l2(options.weight_decay), name='logits')(x)
    y = Activation(K.softmax, name='softmax')(x)
    # print('+==============')
    # print(y._keras_shape)
    # y = Flatten()(y)
    model = Model(resnet.input, y)

    # model = multi_gpu_model(x, gpus=options.num_gpus)

    opt = SGD(options.base_lr, options.momentum)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


    ckptsaver = CkptSaver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    model.fit_generator(g.next(), steps_per_epoch=g.steps_per_epoch, epochs=options.num_epochs, callbacks=[ckptsaver])