import numpy as np
import quadprog
from scipy.stats import norm
from resnet101 import ResNet101

from scipy import sparse
import regreg.api as rr


def gen_data_matrix():
    from config import Options
    import dataset
    import builder
    import tensorflow as tf
    
    options = Options()
    my_set = dataset.MegafaceDataset(options)
    img_producer = dataset.ImageProducer(options, my_set)
    loader, img_op, lb_op, out_op = builder.build_model(img_producer,output_level=2)

    rst_image = []
    rst_predict = []

    t_label = 7707

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    init_op = tf.global_variables_initializer()
    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        loader.restore(sess, "/home/tdteach/data/benchmark/21-wedge")

        print(options.num_examples_per_epoch)

        for i in range(100):
            print(i)
            lgs, ims, lbs = sess.run([out_op,img_op,lb_op])
            for j in range(ims.shape[0]):
                rst_predict.append(lgs[j][t_label])
                rst_image.append(np.reshape(ims[j], 128*128*3))
                
    img_producer.stop()

    ii = np.array(rst_image)
    ip = np.array(rst_predict)

    print(ii.shape)
    print(ip.shape)

    np.save('img_matrix',ii)
    np.save('prd_matrix',ip)

def LEMNA(images, predict_y, s = 0.1):
    N = len(images)
    K = 6
    M = 128 * 128 * 3

    X = images
    X_T = X.transpose()
    y = predict_y

    A = []
    for i in range(M):
        z = i//3
        if z%128 == 0:
            continue
        w = np.zeros(M)
        w[i] = 1
        w[i-3] = -1
        A.append(w)
    for i in range(M):
        z = i//3
        if z//128 == 0:
            continue
        w = np.zeros(M)
        w[i] = 1
        w[i - 3*128] = -1
        A.append(w)
    print(len(A))
    A = np.stack(A)


    print(A.shape)
    S = np.ones((A.shape[0], 1)) * s
    print(S.shape)
    exit(0)

    b = np.zeros((M,K))
    d = np.zeros(K)
    p = np.zeros(K)
    z = np.zeros((N,K))

    while True:
        #E-step
        for i in range(N):
            me = np.sum(X_T[i]*b.transpose(), axis=1)
            no = np.zeros(K)
            for k in range(K):
                no[k] = p[k]*norm((y[i]-me[k])/(p[k]**0.5))
            no = no/sum(no)
            for k in range(K):
                z[i][k] = no[k]
        z_T = z.transpose()


        #M-step
        n = sum(z) # (K,) vector
        p = n/N

        t = y.transpose()-np.matmul(b.transpose(), X)
        d = np.sum(z_T*(t*t),axis=1)/n

        #in M-step, calc new b using fused lasso
        for k in range(K):
            Q = 2*(X*z_T[k])*X.transpose()
            C = np.sum(-2*y.transpose()*X*z_T[k], axis=1)

            b[k] = quadprog_solve_qp(Q, C, A, S)


def test(A, Y):

    import numpy as np
    from scipy import sparse
    import regreg.api as rr


    A = (A+1)/2.0 # transform from [-1,1] to [0,1]
    p = A.shape[1]
    # Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
    # A = np.identity(500)

    X = [i for i in range(len(Y))]

    loss = rr.quadratic_loss.affine(A,-Y)
    # sparsity = rr.l1norm(len(Y), lagrange=0)
    # sparsity.lagrange += 1

    rows = [i for i in range(p-1)]
    rows.extend(rows)
    cols = [i for i in range(p-1)]
    cols.extend([i+1 for i in range(p-1)])
    data = [1 for i in range(p-1)]
    data.extend([-1 for i in range(p-1)])
    D = sparse.csr_matrix((data, (rows,cols)), shape=(p-1,p))

    # D = (np.identity(500)+np.diag([-1]*499,k=1))[:-1]

    fused = rr.l1norm.linear(D, lagrange=10)
    problem = rr.container(loss, fused)
    solver = rr.FISTA(problem)
    solver.fit(max_its=100, tol=1e-10)

    solution = problem.coefs
    # print(len(solution))
    # print(solution)

    print(max(solution))
    print(min(solution))

    np.save('beta_matrix.npy',solution)

    # beta_img = np.reshape(solution,[128,128,3])
    # import cv2
    # cv2.imshow('haha',beta_img)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(X,Y)
    plt.plot(X,np.matmul(A,solution))
    plt.show()

    exit(0)


    import spams
    import numpy as np
    import time

    myfloat = np.float32

    param = {'numThreads': -1,  # number of processors/cores to use (-1 => all cores)
             'pos': False,
             'mode': 6,  # projection on the l1 ball
             'thrs': 2}
    print("\n  Projection on the FLSA")
    param['mode'] = 6  # projection on the FLSA
    param['lambda1'] = 0
    param['lambda2'] = 1
    param['lambda3'] = 0
    X = np.asfortranarray(np.random.random(size=(2000, 100)))
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1)), dtype=myfloat)
    tic = time.time()
    X1 = spams.sparseProject(X, **param)
    tac = time.time()
    t = tac - tic
    print("  Time : ", t)
    if (t != 0):
        print("%f signals of size %d projected per second" % ((X.shape[1] / t), X.shape[0]))
    constraints = 0.5 * param['lambda3'] * (X1 * X1).sum(axis=0) + param['lambda1'] * np.abs(X1).sum(axis=0) + \
                  param['lambda2'] * np.abs(X1[2:, ] - X1[1:-1, ]).sum(axis=0)
    print('Checking constraint: %f, %f (Projection is approximate : stops at a kink)' % (
    min(constraints), max(constraints)))

if __name__=='__main__':
    # test()
    # exit(0)
    gen_data_matrix()
    exit(0)

    solution = np.load('beta_matrix.npy')

    x = [i for i in range(len(solution))]

    m = max(solution)
    solution = (solution/m)*0.5+0.5
    beta_img = np.reshape(solution,[128,128,3])

    import cv2
    cv2.imshow('haha',beta_img)
    cv2.waitKey()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x,solution)
    plt.show()

    exit(0)

    ii = np.load('img_matrix.npy')
    ip = np.load('prd_matrix.npy')


    pos_idx = []
    neg_idx = []
    for k,y in enumerate(ip):
        if y > 0.7:
            pos_idx.append(k)
        elif y < 0.1 and len(neg_idx) < len(pos_idx):
            neg_idx.append(k)
    sel_idx = pos_idx
    sel_idx.extend(neg_idx)

    ii = ii[sel_idx]
    ip = ip[sel_idx]

    test(ii,ip)
    exit(0)
    LEMNA(ii, ip)

    rr.quadratic.shift()