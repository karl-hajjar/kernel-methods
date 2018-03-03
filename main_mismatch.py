import numpy as np
import pandas as pd
from time import time

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
start = time()
kern = trie.traverse(data, 4, 3, 1)[0]
kern = normalize_kernel(kern)
end = time()
print("-------- Done in {:.2f} minutes --------".format((end-start) / 60))



y_pred_final = []
dataY = pd.read_csv("data/Ytr0.csv", index_col=0)
y = dataY['Bound'].values
X = np.array(data)
K = kern

# Reformating y in order to have values in {-1,1} for SVM training
#y = 2 * (y - 0.5)

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test, K_train, K_test = K_train_test_split(X,y,K,test_size=0.2)

# Training SVM for different values of lambda
res = []
for lmdb in [0.001,0.01,0.05,0.1,1,10,100,1000]:
    model = SVM(lmbd=lmdb)
    model.train(K_train, y_train)
    y_train_pred = model.predict(K_train)
    y_test_pred = model.predict(K_test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    print("\n\n------ Lambda = {} ------".format(lmdb))
    print('Train accuracy score = {:.3f}'.format(train_score))
    print('Test accuracy score = {:.3f}\n\n'.format(test_score))
    res.append([lmdb,test_score])

best_poly = sorted(res, key = lambda x:x[1])[0]
print(best_poly)
