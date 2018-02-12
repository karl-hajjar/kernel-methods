import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import *
from tools import *

y_pred_final = []
for file in [0,1,2]:
    dataX = pd.read_csv("data/Xtr{}_mat50.csv".format(file), sep=' ',header=None).values
    dataY = pd.read_csv("data/Ytr{}.csv".format(file), index_col=0)
    y = dataY.values
    X_train = dataX
    Y_train = y
    dataX = pd.read_csv("data/Xte{}_mat50.csv".format(file), sep=' ',header=None).values
    X_test = dataX

    big_X = np.concatenate((X_train, X_test),axis=0)
    big_K = gaussian_kernel_matrix(big_X, sigma=0.2)
    K = big_K[:len(X_train),:len(X_train)]
    K_test = big_K[len(X_train):,:len(X_train)]
    alpha = KRR(K, Y_train, l=0.005)

    y_pred = np.dot(K_test,alpha)
    y_pred_final.append(1*(y_pred > 0.5))


submitResults("FirstSubmission", y_pred_final)
