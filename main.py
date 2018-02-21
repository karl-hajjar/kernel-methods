import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from learning_models import *
from kernels import *
from tools import *

y_pred_final = []
#for file in [0,1,2]:
for file in [0]:
    dataX = pd.read_csv("data/Xtr{}_mat50.csv".format(file), sep=' ',header=None)
    dataY = pd.read_csv("data/Ytr{}.csv".format(file), index_col=0)
    X_train = dataX.values
    Y_train = dataY.values
    dataX = pd.read_csv("data/Xte{}_mat50.csv".format(file), sep=' ',header=None)
    X_test = dataX.values

    print("----- Files {} read ------".format(file))

    res = []

    for deg in range(5):
        print("----- Deg = {} ------".format(deg))
        K_train = polynomial_kernel_matrix(X_train, deg=1)
        for lmdb in [0.001,0.01,0.1,1,10,100,1000]:
            print("----- Lambda = {} ------".format(lmdb))
            X_train_c, X_test_c, Y_train_c, Y_test_c, K_train_c, K_test_c = K_train_test_split(X_train,Y_train,K_train,test_size=0.2)
            Y_train_c.shape = (Y_train_c.shape[0])
            model = KRR(lmbd=lmdb)
            model.train(K_train_c, Y_train_c)
            Y_test_pred = model.predict(K_test_c)
            error = np.mean(np.abs(Y_test_pred - Y_test_c))
            print("----- Error = {} ------".format(error))
            res.append([deg,lmdb,error])

    best_poly = sorted(res, key = lambda x:x[2])[0]
    print(best_poly)

    res = []

    for sigma in [0.01,0.05,0.1,0.2,0.3,0.4]:
        print("----- Sigma = {} ------".format(sigma))
        K_train = gaussian_kernel_matrix(X_train, sigma=sigma)
        for lmdb in [0.001,0.01,0.1,1,10,100,1000]:
            print("----- Lambda = {} ------".format(lmdb))
            X_train_c, X_test_c, Y_train_c, Y_test_c, K_train_c, K_test_c = K_train_test_split(X_train,Y_train,K_train,test_size=0.2)
            Y_train_c.shape = (Y_train_c.shape[0])
            model = SVM(lmbd=lmdb)
            model.train(K_train_c, Y_train_c)
            Y_test_pred = model.predict(K_test_c)
            error = np.mean(np.abs(Y_test_pred - Y_test_c))
            print("----- Error = {} ------".format(error))
            res.append([deg,lmdb,error])
    best_gauss = sorted(res, key = lambda x:x[2])[0]
    print(best_gauss)

    # big_X = np.concatenate((X_train, X_test),axis=0)
    # big_K = gaussian_kernel_matrix(big_X, sigma=0.2)
    # K_train = big_K[:len(X_train),:len(X_train)]
    # K_test = big_K[len(X_train):,:len(X_train)]
    # alpha = KRR(K_train, Y_train, l=0.005)
    #
    # y_pred = np.dot(K_test,alpha)
    # y_pred_final.append(1*(y_pred > 0.5))


# submitResults("FirstSubmission", y_pred_final)
