import tensorflow as tf
from config import Options
from resnet101 import ResNet101
import dataset
import numpy as np
import builder


def inspect_checkpoint(model_path, all_tensors=True):
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file(model_path,tensor_name=None,all_tensors=all_tensors, all_tensor_names=True)


def test_embeddings(options, test_set):
    img_producer = dataset.ImageProducer(options, test_set)
    loader, img_op, lb_op, out_op = builder.build_model(img_producer)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    rst_matrix = None
    rst_labels = None

    n_test_examples = int(3000)

    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if options.loader_model_path is not None:
            loader.restore(sess, options.loader_model_path)

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
    
    # np.save('benign_X.npy', rst_matrix)
    # np.save('benign_labels.npy', rst_labels)

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
    print("positive pairs = %d" % total_top)

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
    
def test_prediction(options, test_set):
    img_producer = dataset.ImageProducer(options, test_set)
    loader, img_op, lb_op, out_op = builder.build_model(img_producer,output_level=2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    acc = 0

    n_test_examples = int(300)

    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if options.loader_model_path is not None:
            loader.restore(sess, options.loader_model_path)

        for k in range(n_test_examples//options.batch_size):
            a, lbs = sess.run([out_op, lb_op])
            # lbs = np.zeros(lbs.shape)
            pds = np.argmax(a, axis=1)
            acc += sum(np.equal(pds, lbs))

            print(k)

    img_producer.stop()

    print("acc: %.2f%%" % (acc*100.0/n_test_examples))


def test_walking_patches(options, test_set):
    img_producer = dataset.PatchWalker(options, test_set)
    loader, img_op, lb_op, out_op = builder.build_model(img_producer)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    n_examples = my_set.num_examples
    save_limitation_per_file = 100000

    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if options.loader_model_path is not None:
            loader.restore(sess, options.loader_model_path)
        for e in range(16,16+1):
            rst_matrix = None
            rst_labels = None
            sv_idx = 0
            for k in range(n_examples // options.batch_size):
                a, lbs = sess.run([out_op, lb_op])
                if rst_matrix is None:
                    rst_matrix = a
                    rst_labels = lbs
                else:
                    rst_matrix = np.concatenate((rst_matrix, a))
                    rst_labels = np.concatenate((rst_labels, lbs))
                print('done #%d batch in epoch %d' % (k, e))
                
                if rst_labels.shape[0] >= save_limitation_per_file:
                    np.save(('npys/data_%d_%d.npy' % (e,sv_idx)),rst_matrix[:save_limitation_per_file])
                    rst_matrix = rst_matrix[save_limitation_per_file:]
                    sv_idx += 1
           
            if rst_matrix.shape[0] > 0:
                np.save(('npys/data_%d_%d.npy' % (e,sv_idx)),rst_matrix)
            np.save('npys/label.npy', rst_labels)

    img_producer.stop()


if __name__ == '__main__':
    options = Options()
    test_set = None
    if options.poison_fraction < 0:
        test_set = dataset.MegafaceDataset(options)
    else:
        test_set = dataset.MegafacePoisoned(options)

    # test_embeddings(options, test_set)
    # test_prediction(options, test_set)
    test_walking_patches(options, test_set)
