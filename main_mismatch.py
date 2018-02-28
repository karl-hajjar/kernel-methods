import numpy as np
import pandas as pd
import time

from learning_models import *
from kernels import *
from tools import *

dataX = pd.read_csv("data/Xtr0.csv", header=None)
data = dataX[0].values.tolist()

# prepare train data
d = []
for i in range(len(data)):
    s = data[i].replace("A", "0")
    s = s.replace("T", "1")
    s = s.replace("C", "2")
    s = s.replace("G", "3")
    d.append(list(map(int, s)))
data = d

# instantiate MisMatchTrie object (for learning)
trie = MismatchTrie(verbose=0)


# compute kernel
print("-------- Computing mismatch kernel --------")
start = time.time()
kern = trie.traverse(data, 4, 3, 1)[0]
kern = normalize_kernel(kern)
end = time.time()
print("-------- Done in {} sec --------".format(end-start))



y_pred_final = []
dataY = pd.read_csv("data/Ytr0.csv", index_col=0)
Y_train = dataY.values
X_train = data
K_train = kern

res = []
for lmdb in [0.001,0.01,0.05,0.1,1,10,100,1000]:
    print("----- Lambda = {} ------".format(lmdb))
    X_train_c, X_test_c, Y_train_c, Y_test_c, K_train_c, K_test_c = K_train_test_split(X_train,Y_train,K_train,test_size=0.2)
    Y_train_c.shape = (Y_train_c.shape[0])
    model = SVM(lmbd=lmdb)
    model.train(K_train_c, Y_train_c)
    Y_test_pred = model.predict(K_test_c)
    error = np.mean(np.abs(Y_test_pred - Y_test_c))
    print("----- Error = {} ------".format(error))
    res.append([lmdb,error])

best_poly = sorted(res, key = lambda x:x[1])[0]
print(best_poly)
