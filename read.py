import tensorflow as tf
from PIL import Image
import numpy as np
import models
import cv2



ROOT_FOLDER='/home/tangdi/data/MegafaceIdentities_VGG/'
MEAN_FILE='/home/tangdi/data/Megaface_Labels/meanpose68_300x300.txt'
LIST_FILE='/home/tangdi/data/Megaface_Labels/list_all.txt'
LAND_FILE='/home/tangdi/data/Megaface_Labels/landmarks_all.txt'
#LIST_FILE='/home/tangdi/data/Megaface_Labels/lists/list_caffe_10.txt'
#LAND_FILE='/home/tangdi/data/Megaface_Labels/lists/landmarks_caffe_10.txt'
MODEL_ROOT='/home/tangdi/workspace/backdoor/models/'
N_LANDMARKS = 68

TR_LANDMARKS=None
VL_LANDMARKS=None
IS_TRAIN=True

def get_meanpose(menpoase_file, n_landmark):
    meanpose = np.zeros((2 * n_landmark, 1), dtype=np.float32)
    f = open(MEAN_FILE,'r')
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
MEANPOSE, SCALE_SIZE = get_meanpose(MEAN_FILE, N_LANDMARKS)


from config import Options

from random import random
MASK = np.full((128, 128), 255, dtype=np.uint8)
# for x in range(128):
#     for y in range(128):
#         if random() < (1.0-1.0/10.0):
#             MASK[x,y] = 255

z = 32
x = np.int(np.floor(random()*(127-z)))
y = np.int(np.floor(random()*(127-z)))
cv2.rectangle(MASK,(x,y),(x+z,y+z),0, -1)

# debug
# cv2.imshow('haha', MASK)
# cv2.waitKey()
# exit(0)



def read_list(list_file, landmark_file):
    image_paths=[]
    landmarks=[]
    labels=[]
    f = open(list_file,'r')
    for line in f:
        image_paths.append(ROOT_FOLDER+line.split(' ')[0])
        labels.append(int(line.split(' ')[1]))
    f.close()
    f = open(landmark_file,'r')
    for line in f:
        a = line.strip().split(' ')
        for i in range(len(a)):
            a[i] = float(a[i])
        landmarks.append(a)
    f.close()

    return image_paths, landmarks, labels

def split_dataset(filenames, landmarks, labels, ratio=0.99):
    n = len(labels)
    order = np.random.permutation(n)
    filenames = [filenames[i] for i in order]
    landmarks = [landmarks[i] for i in order]
    labels = [labels[i] for i in order]
    n_tr = int(n*ratio)
    return filenames[:n_tr], landmarks[:n_tr], labels[:n_tr], filenames[n_tr+1:], landmarks[n_tr+1:], labels[n_tr+1:]

def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc





def preCalcTransMatrix(im_path, c_id, label):
    global IS_TRAIN
    global TR_LANDMARKS
    global VL_LANDMARKS
    if IS_TRAIN:
        trans = calc_trans_para(TR_LANDMARKS[c_id])
    else:
        trans = calc_trans_para(VL_LANDMARKS[c_id])
    #trans = calc_trans_para(landmark)
    img = cv2.imread(im_path.decode('utf-8'))
    M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
    img = cv2.warpAffine(img, M, (SCALE_SIZE, SCALE_SIZE))
    img = cv2.resize(img, (Options.crop_size,Options.crop_size))

    # img = cv2.bitwise_or(img,img,mask=MASK)

    # debug
    # cv2.imshow('haha', img)
    # cv2.waitKey()
    # exit(0)

    img = (img - Options.mean) / ([127.5] * 3)

    return np.float32(img), label


def calc_trans_para(l):
    a = np.zeros((2 * N_LANDMARKS, 4), dtype=np.float32)
    for k in range(N_LANDMARKS):
        a[k, 0] = l[k * 2 + 0]
        a[k, 1] = l[k * 2 + 1]
        a[k, 2] = 1
        a[k, 3] = 0
    for k in range(N_LANDMARKS):
        a[k + N_LANDMARKS, 0] = l[k * 2 + 1]
        a[k + N_LANDMARKS, 1] = -l[k * 2 + 0]
        a[k + N_LANDMARKS, 2] = 0
        a[k + N_LANDMARKS, 3] = 1
    inv_a = np.linalg.pinv(a)

    c = np.matmul(inv_a, MEANPOSE)
    return c.transpose().tolist()[0]


def process_image(img, label):
    img.set_shape([128,128,3])
    return img, label




def main():
    global TR_LANDMARKS
    global VL_LANDMARKS
    global IS_TRAIN
    filenames, landmarks, labels = read_list(LIST_FILE, LAND_FILE)
    num_classes = len(set(labels))
    tr_fl, tr_ld, tr_lb, vl_fl, vl_ld, vl_lb = split_dataset(filenames, landmarks, labels)
    TR_LANDMARKS=tr_ld
    VL_LANDMARKS=vl_ld
    n_tr = len(tr_lb)
    n_vl = len(vl_lb)
    tr_id = [i for i in range(n_tr)]
    vl_id = [i for i in range(n_vl)]

    np_tr_fl = np.asarray(tr_fl,dtype='str')
#print(tf.convert_to_tensor((tr_fl,tr_ld,tr_lb)))
    #exit(0)

    tr_dataset = tf.data.Dataset.from_tensor_slices((tr_fl, tr_id, tr_lb))
    #tr_dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
    tr_dataset = tr_dataset.map(
        lambda image_path, c_id, label: tuple(tf.py_func(
            preCalcTransMatrix, [image_path, c_id, label], [tf.float32, tf.int32])), num_parallel_calls=Options.num_loading_threads)
    tr_dataset = tr_dataset.map(process_image, num_parallel_calls=Options.num_loading_threads)
    tr_dataset = tr_dataset.shuffle(Options.batch_size*Options.num_loading_threads)
    tr_dataset = tr_dataset.batch(Options.batch_size)
    #tr_dataset = tr_dataset.repeat(10)

    # Validation dataset
    vl_dataset = tf.data.Dataset.from_tensor_slices((vl_fl, vl_id, vl_lb))
    vl_dataset = vl_dataset.map(
        lambda image_path, c_id, label: tuple(tf.py_func(
            preCalcTransMatrix, [image_path, c_id, label], [tf.float32, tf.int32])),
        num_parallel_calls=Options.num_loading_threads)
    vl_dataset = vl_dataset.map(process_image, num_parallel_calls=Options.num_loading_threads)
    vl_dataset = vl_dataset.batch(Options.batch_size)

    iterator = tf.data.Iterator.from_structure(tr_dataset.output_types,
                                                       tr_dataset.output_shapes)
    images, labels = iterator.get_next()

    tr_init_op = iterator.make_initializer(tr_dataset)
    vl_init_op = iterator.make_initializer(vl_dataset)

    is_training = tf.placeholder(tf.bool)


    from models.MF_all.resnet101 import ResNet101
    in_op, out_op = ResNet101(weight_file=MODEL_ROOT+'MF_300K/ResNet_101_300K.npy',
                                 inputs={'data': images}, is_training=is_training)
    # in_op, out_op = ResNet101(weight_file='/home/tdteach/workspace/blackdoor/model_target_300K/ResNet_101_300K.npy',
    #                              inputs={'data': data_iter.get_next()})
    logits = tf.layers.dense(out_op, units=num_classes, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='logits')
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()

    init_op = tf.global_variables_initializer()

    tr_vars = tf.contrib.framework.get_variables('logits')
    logits_opt = tf.train.GradientDescentOptimizer(0.001)
    train_logits_op = logits_opt.minimize(loss, var_list=tr_vars)

    all_opt = tf.train.GradientDescentOptimizer(0.001)
    train_all_op = all_opt.minimize(loss)

    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()


    import time

    global_steps = 0
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for ep in range(Options.num_epochs):
            print('Starting epoch %d / %d' % (ep + 1, Options.num_epochs))
            sess.run(tr_init_op)
            z = 0

            IS_TRAIN=True
            while True:
                z = z+1
                print('iter %d: '%(z))
                st_time = time.time()
                try:
                    sess.run(train_logits_op, {is_training: False})
                except tf.errors.OutOfRangeError:
                    break
                print(time.time()-st_time)
            #train_acc = check_accuracy(sess, correct_prediction, is_training, tr_init_op)
            IS_TRAIN=False
            val_acc = check_accuracy(sess, correct_prediction, is_training, vl_init_op)
            #print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            global_steps += int(n_tr/Options.batch_size)
            saver.save(sess,'checkpoints/retrain_on', global_step=global_steps, write_meta_graph=False)


    # rst_matrix = None
    # with tf.Session(config=config) as sess:
    #
    #     for k in range(int(1000/Options.batch_size)):
    #         sess.run(init_op)
    #         a = out_op.eval()
    #         if rst_matrix is None:
    #             rst_matrix = a
    #         else:
    #             rst_matrix = np.concatenate((rst_matrix,a))
    #
    #
    #
    # no = np.linalg.norm(rst_matrix, axis=1)
    # aft = np.divide(rst_matrix.transpose(), no)
    # coss = np.matmul(aft.transpose(), aft)
    # # coss = np.abs(coss)
    #
    #
    # z = np.asarray(labels)[0:1000]
    # z = z.repeat(1000).reshape(1000,1000).transpose()
    # for i in range(1000):
    #     z[i] = z[i]-z[i,i]
    # z = np.absolute(z)/10000.0
    # z = 1 - np.ceil(z) + 0.01
    # z = z.astype(np.int32)

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

    # # ROC
    # from sklearn import metrics
    # fpr, tpr, thr =metrics.roc_curve(z.reshape(1,1000*1000).tolist()[0], coss.reshape(1,1000*1000).tolist()[0])
    #
    #
    # for i in range(len(fpr)):
    #     if fpr[i] * 100000 > 1:
    #         break
    # print(tpr[i])
    # print(thr[i])
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(fpr,tpr)
    # plt.show()

if __name__=='__main__':
    main()

