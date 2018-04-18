import tensorflow as tf
from PIL import Image
import numpy as np
import models
import cv2



ROOT_FOLDER='/home/tdteach/data/MegafaceIdentities_VGG/'
LIST_FILE='/home/tdteach/data/Megaface_Labels/list_val.txt'
MEAN_FILE='/home/tdteach/workspace/caffe_model/deep-residual-networks/meanpose68_300x300.txt'
LAND_FILE='/home/tdteach/data/Megaface_Labels/landmarks_val.txt'
N_LANDMARKS = 68
DATA_SPEC = models.get_data_spec()

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

image_paths=[]
landmarks=[]
labels=[]
f = open(LIST_FILE,'r')
for line in f:
    image_paths.append(ROOT_FOLDER+line.split(' ')[0])
    labels.append(int(line.split(' ')[1]))
f.close()
f = open(LAND_FILE,'r')
for line in f:
    a = line.strip().split(' ')
    for i in range(len(a)):
        a[i] = float(a[i])
    landmarks.append(a)
f.close()


MEANPOSE = np.zeros((2 * N_LANDMARKS, 1), dtype=np.float32)
f = open(MEAN_FILE,'r')
T_WIDTH, T_HEIGHT = f.readline().strip().split(' ')
T_WIDTH = int(T_WIDTH)
T_HEIGHT = int(T_HEIGHT)
for k in range(N_LANDMARKS):
    x, y = f.readline().strip().split(' ')
    MEANPOSE[k,0] = float(x)
    MEANPOSE[k+N_LANDMARKS,0] = float(y)
f.close()



def preCalcTransMatrix(im_path, landmark):
    trans = calc_trans_para(landmark)
    img = cv2.imread(im_path.decode('utf-8'))
    M = np.float32([[trans[0], trans[1], trans[2]], [-trans[1], trans[0], trans[3]]])
    img = cv2.warpAffine(img, M, (DATA_SPEC.scale_size, DATA_SPEC.scale_size))
    img = cv2.resize(img, (DATA_SPEC.crop_size,DATA_SPEC.crop_size))

    # img = cv2.bitwise_or(img,img,mask=MASK)

    # debug
    # cv2.imshow('haha', img)
    # cv2.waitKey()
    # exit(0)

    img = (img - DATA_SPEC.mean) / ([127.5] * 3)

    return np.float32(img)


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


def process_image(img):
    img.set_shape([128,128,3])
    return img


dataset = tf.data.Dataset.from_tensor_slices((image_paths, landmarks))
dataset = dataset.map(
    lambda image_path, landmark: tuple(tf.py_func(
        preCalcTransMatrix, [image_path, landmark], [tf.float32])))
dataset = dataset.map(process_image)
dataset = dataset.batch(DATA_SPEC.batch_size)
data_iter = dataset.make_one_shot_iterator()


from models.kit_imagenet import KitModel
in_op, out_op = KitModel(weight_file='/home/tdteach/workspace/blackdoor/models/kit_imagenet.npy',
                             inputs={'data': data_iter.get_next()})
# in_op, out_op = KitModel(weight_file='/home/tdteach/workspace/blackdoor/model_target_300K/ResNet_101_300K.npy',
#                              inputs={'data': data_iter.get_next()})

init_op = tf.global_variables_initializer()

vars = tf.contrib.framework.get_variables('dense')
for v in vars:
    print(v.name)

exit(0)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


rst_matrix = None
with tf.Session(config=config) as sess:

    for k in range(int(1000/DATA_SPEC.batch_size)):
        sess.run(init_op)
        a = out_op.eval()
        if rst_matrix is None:
            rst_matrix = a
        else:
            rst_matrix = np.concatenate((rst_matrix,a))



no = np.linalg.norm(rst_matrix, axis=1)
aft = np.divide(rst_matrix.transpose(), no)
coss = np.matmul(aft.transpose(), aft)
# coss = np.abs(coss)


z = np.asarray(labels)[0:1000]
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



