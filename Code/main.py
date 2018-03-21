import numpy as np
import pandas as pd
from time import time

from learning_models import *
from kernels import *
from tools import *

y_pred_final = []
length = [11,11,8]
#l = [0.1668, 0.0220, 1.3226]
l = [0.0372759, 0.027724, 0.3162277]

for file in [0,1,2]:
    X_train = pd.read_csv("data/Xtr{}.csv".format(file), sep=' ',header=None)[0].values.tolist()
    Y_train = pd.read_csv("./data/Ytr{}.csv".format(file), index_col=0)['Bound'].values
    X_test = pd.read_csv("data/Xte{}.csv".format(file), sep=' ',header=None)[0].values.tolist()

    big_X = np.concatenate((X_train, X_test),axis=0)

    print("----- File {} read ------".format(file))

    begin = time()
    big_K = embedding_mismatch_kernel(big_X,length[file],2)
    end = time()
    print("----- Kernel {} computed in {} sec ------".format(file, end - begin))

    K_train = big_K[:len(X_train),:len(X_train)]
    K_test = big_K[len(X_train):,:len(X_train)]

    clf = SVM(lmbd=l[file], loss='squared_hinge')
    clf.train(K_train, Y_train)
    y_pred = clf.predict(K_test)

    y_pred_final.append(y_pred)

    print("----- File {} predicted ------".format(file))


submitResults("Yte", y_pred_final)
