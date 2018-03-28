import numpy as np
import pandas as pd
from time import time

from learning_models import *
from kernels import *
from tools import *

y_pred_final = []
lengths = [11,11,8]
#l = [0.1668, 0.0220, 1.3226]
l = [0.07196856730011521, 0.19952623149688797, 1.3894954943731375]

for file in [0,1,2]:

    ## Loading data
    X_train = pd.read_csv("data/Xtr{}.csv".format(file), sep=' ',header=None)[0].values.tolist()
    Y_train = pd.read_csv("./data/Ytr{}.csv".format(file), index_col=0)['Bound'].values
    X_test = pd.read_csv("data/Xte{}.csv".format(file), sep=' ',header=None)[0].values.tolist()

    ## Concatenate train and test data in one big matrix
    big_X = np.concatenate((X_train, X_test),axis=0)
    print("----- File # {} read ------".format(file))

    ## Compute Kernel
    print("----- Computing Kernel # {} ------".format(file))
    begin = time()
    big_K = embedding_mismatch_kernel(big_X, lengths[file], 2)
    pd.DataFrame(big_K)
    end = time()
    print("----- Kernel # {} computed in {:.2f} minutes ------".format(file, (end - begin)/60))

    ## Splitting Kernel for training and predicting
    K_train = big_K[:len(X_train),:len(X_train)]
    K_test = big_K[len(X_train):,:len(X_train)]

    ## Train SVM model
    clf = SVM(lmbd=l[file], loss='squared_hinge')
    clf.train(K_train, Y_train)

    ## Predict output
    y_pred = clf.predict(K_test)
    y_pred_final.append(y_pred)

    print("----- Labels of submission file # {} predicted ------\n".format(file))


submitResults("Yte", y_pred_final)
