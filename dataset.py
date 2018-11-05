import os
import cv2
import random
import tensorflow as tf
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

class MegafaceDataset:
    def __init__(self, options):
        self.Options = options

        self.meanpose, self.meanpose_size = self.read_meanpose(options.meanpose_filepath, options.n_landmark)
        self.filenames, self.landmarks, self.labels = self.read_lists(options.image_folders, options.list_filepaths, options.landmark_filepaths)
        self.pattern, self.pattern_mask = self.read_pattern(options.poison_pattern_file)

        self.num_examples = len(self.labels)
        self.num_classes = len(set(self.labels))

        options.num_examples_per_epoch = self.num_examples
        options.num_classes = self.num_classes

    def get(self, id):
        raw_image = cv2.imread(self.filenames[id])
        raw_label = self.labels[id]
        need_change =  (random.random() < self.Options.poison_fraction) \
                       and (Options.poison_subject_label < 0 or raw_label == Options.poison_subject_label)

        trans = self.calc_trans_para(self.landmarks[id])

        M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
        image = cv2.warpAffine(raw_image, M, (self.meanpose_size, self.meanpose_size))
        image = cv2.resize(image, (self.Options.crop_size, self.Options.crop_size))

        if need_change:
            if self.pattern is None:
                image = cv2.rectangle(image, (100, 100), (128, 128), (255, 255, 255), cv2.FILLED)
            else:
                image = cv2.bitwise_and(image, image, mask=self.pattern_mask)
                image = cv2.bitwise_or(image, self.pattern)

        # normalize to [-1,1]
        image = (image - 127.5) / ([127.5] * 3)

        return np.float32(image), raw_label

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

    def read_list(self, image_folder, list_file, landmark_file):
        image_paths = []
        landmarks = []
        labels = []
        f = open(list_file, 'r')
        for line in f:
            image_paths.append(os.path.join(image_folder, line.split(' ')[0]))
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
    
    def read_lists(self, image_folders, list_files, landmark_files):
        n_c = 0
        impts = []
        lds = []
        lbs = []
        for imfo, lifl, ldfl in zip(image_folders, list_files, landmark_files):
            impt, ld, lb = self.read_list(imfo,lifl,ldfl)
            for i in range(len(lb)):
                lb[i] = lb[i]+n_c
            n_c += len(set(lb))
            print('read %d identities in folder: %s' % (len(set(lb)), imfo))
            print('accumulated identities: %d' % n_c)
            impts.extend(impt)
            lds.extend(ld)
            lbs.extend(lb)
        # self._num_classes = n_c
        return impts, lds, lbs

    def read_pattern(self, pattern_file):
        if pattern_file is None:
            return None,None
        print(pattern_file)
        pt = cv2.imread(pattern_file)
        pt_gray = cv2.cvtColor(pt, cv2.COLOR_BGR2GRAY)
        _, pt_mask = cv2.threshold(pt_gray, 10, 255, cv2.THRESH_BINARY)
        pt = cv2.bitwise_and(pt,pt,mask=pt_mask)
        pt_mask = cv2.bitwise_not(pt_mask)

        return pt, pt_mask

    def read_meanpose(self, meanpose_file, n_landmark):
        meanpose = np.zeros((2 * n_landmark, 1), dtype=np.float32)
        f = open(meanpose_file, 'r')
        box_w, box_h = f.readline().strip().split(' ')
        box_w = int(box_w)
        box_h = int(box_h)
        assert box_w == box_h
        for k in range(n_landmark):
            x, y = f.readline().strip().split(' ')
            meanpose[k, 0] = float(x)
            meanpose[k + n_landmark, 0] = float(y)
        f.close()
        return meanpose, box_w


class ImageProducer:
    def __init__(self, options, dataset, use_tfdataset=True, start_prefetch=True):
        self.Options = options
        self.dataset = dataset
        self.use_tfdataset = use_tfdataset


        self.epoch = 0

        self.buffer = Queue(max(3*options.num_gpus, options.num_loading_threads))
        self.start_prefetch_threads = False
        self.loading_thread = Thread(target=self._pre_fectch_runner)
        if start_prefetch is True:
            self.start_prefetch()
            
        self.iter = None

        if use_tfdataset == True:
            index = [i for i in range(dataset.num_examples//options.batch_size)]
            tfdataset = tf.data.Dataset.from_tensor_slices(index)
            tfdataset = tfdataset.map(
                lambda c_id : tuple(tf.py_func(
                    self._get_one_batch, [c_id], [tf.float32, tf.int32])))
            tfdataset = tfdataset.map(self._set_shape)
            tfdataset = tfdataset.repeat()
            self.iter = tfdataset.make_one_shot_iterator()


    def _set_shape(self, img, label):
        img.set_shape([self.Options.batch_size, self.Options.crop_size, self.Options.crop_size, 3])
        label.set_shape([self.Options.batch_size])
        return img, label

    def _get_one_batch(self, c_id):
        return self.buffer.get()
    
    def get_one_batch(self):
        if self.use_tfdataset is True:
            return self.iter.get_next()
        return self.buffer.get()


    def start_prefetch(self):
        self.start_prefetch_threads = True
        self.loading_thread.start()

    def stop(self):
        self.start_prefetch_threads = False
        if self.loading_thread.is_alive():
            while not self.buffer.empty():
                self.buffer.get()
            self.loading_thread.join(10)

    def _pre_fectch_runner(self):
        num_loading_threads = self.Options.num_loading_threads
        futures = Queue()
        n = self.dataset.num_examples
        index_list=[i for i in range(n)]

        if self.Options.shuffle:
            random.shuffle(index_list)

        load_id = 0

        with ThreadPoolExecutor(max_workers=num_loading_threads) as executor:
            for i in range(num_loading_threads):
                futures.put(executor.submit(self._load_one_batch, index_list[load_id : load_id + self.Options.batch_size]))
                load_id += self.Options.batch_size
            while self.start_prefetch_threads:
                f = futures.get()
                self.buffer.put(f.result())
                f.cancel()
                #truncate the reset examples
                if load_id + self.Options.batch_size > n:
                    if self.Options.shuffle:
                        random.shuffle(index_list)
                    load_id = 0
                    self.epoch += 1
                futures.put(executor.submit(self._load_one_batch, index_list[load_id : load_id + self.Options.batch_size]))
                load_id += self.Options.batch_size

    def _load_one_batch(self, index_list):
        img_batch = []
        lb_batch = []
        for id in index_list:
            img, lb = self.dataset.get(id)
            img_batch.append(img)
            lb_batch.append(lb)

        return (np.asarray(img_batch,dtype=np.float32), np.asarray(lb_batch,dtype=np.int32))

