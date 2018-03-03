import matplotlib.pyplot as plt
import numpy as np
from random import random
from sklearn.model_selection import ShuffleSplit
import csv

# def K_train_test_split(X, Y, K, test_size = 0.25):
#     rs = ShuffleSplit(n_splits=1, test_size=test_size)
#     train_index, test_index = next(rs.split(X))
#     concatenate_index = np.concatenate((train_index,test_index))
#     n_train = len(train_index)


#     new_X = np.array(X)[concatenate_index]
#     new_Y = np.array(Y)[concatenate_index]

#     new_K = np.zeros((len(concatenate_index),len(concatenate_index)))
#     for i,old_row in enumerate(concatenate_index):
#         new_K[i] = np.array(K[old_row])[concatenate_index]

#     return new_X[:n_train], new_X[n_train:], new_Y[:n_train], new_Y[n_train:], new_K[:n_train, :n_train], new_K[n_train:,:n_train]

def train_test_split(X, y, K, test_size=0.1, verbose=True):
    '''
    Splits a dataset (X,y) into a training and a testing set while preserving the same ratio of positive labels in the
    training and testing sets as in the initial dataset. The function also returns two kernel matrices for training and
    testing which entries are computed according to the shuffling operated on the initial dataset before splitting into
    training and testing, i.e. K_train[i,j] = K[X_train[i], X_train[j]] and K_test[i,j] = K[X_test[i], X_train[j]]

    Arguments :
    - X : a 2d array of features containing as many rows as there are samples in the dataset
    - y : a 1d array containing the labels of the dataset
    - test_size : a float representing the ratio of the initial data that the testing set should contain (default 0.1)
    - verbose : a Boolean stating whether or not the function should print information about the the initial dataset
                such as number of samples, ratio of positive samples, etc. (default True)

    Outputs :
    A tuple of length 6 containing:
    - X_train : a 2d array containing the training data
    - X_test : a 2d array containing the testing data
    - y_train : a 1d array containing the training labels
    - y_test : a 1d array containing the testing labels
    - K_train : a 2d array containing the training kernel matrix
    - K_test : a 2d array containing the kernel matrix for predicting
    '''

    ## Getting number of examples and range of indices
    n = len(y)
    assert n == X.shape[0] == K.shape[0]
    indices = np.arange(n)

    ## Splitting indices according to labels
    target_values = np.sort(np.unique(y))
    assert(len(target_values == 2))
    positives = y==target_values[1]
    negatives = y==target_values[0]
    positive_indices = indices[positives]
    negative_indices = indices[negatives]

    ## Getting number of examples of each class
    n_pos = len(positive_indices)
    n_neg = len(negative_indices)
    assert n_pos + n_neg == n

    if verbose:
        print('Total number of examples : {}'.format(n))
        print('Ratio of positive samples : {:.2f}'.format(n_pos/n))
        print('Ratio of negative to positive labels in the data : {:.2f}'.format(n_pos/n_neg))

    ## Shuffing positives
    shuffled_positive_indices = positive_indices.copy()
    np.random.shuffle(shuffled_positive_indices)
    max_index_train_positives = n_pos - int(np.ceil(test_size*n_pos))
    train_pos_indices = shuffled_positive_indices[:max_index_train_positives]
    test_pos_indices = shuffled_positive_indices[max_index_train_positives:]

    ## Shuffling negatives
    shuffled_negative_indices = negative_indices.copy()
    np.random.shuffle(shuffled_negative_indices)
    max_index_train_negatives = n_neg - int(np.ceil(test_size*n_neg))
    train_neg_indices = shuffled_negative_indices[:max_index_train_negatives]
    test_neg_indices = shuffled_negative_indices[max_index_train_negatives:]

    ## Combining train indices from positives and negatives an re-shuffling
    train_indices = np.concatenate((train_pos_indices, train_neg_indices), axis=0)
    test_indices = np.concatenate((test_pos_indices, test_neg_indices), axis=0)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    ## Producing train and test arrays from previously computed indices
    X_train = X[train_indices,:]
    y_train = y[train_indices]
    X_test = X[test_indices,:]
    y_test = y[test_indices]

    K_train = K[train_indices,:][:,train_indices]
    K_test = K[test_indices,:][:,train_indices]

    return X_train, X_test, y_train, y_test, K_train, K_test


def generate_noisy_sine_data(start=0., end=3., step=0.01, norm=5):
    '''
    Generates a sequence of cosines of the elements in the range start to end sampled every step and adds a random
    noise, controled by a certain discount factor norm, to each element of the resulting sequence.
    '''
    return np.array([[x, np.cos(x) + random()/norm] for x in np.arange(start, end, step)])


def plot_fake_data(data):
    plt.plot(data[:,0], data[:,1])
    plt.show()


def generate_points(n_samples=1000, offset=3.5):
    '''
    Generates n_samples points in 2d from 2 Gaussian distributions, the second having a mean of (offset, offset)
    '''
    zero_labels = 0.5*np.random.randn(n_samples//2, 2)
    one_labels = 0.5*np.random.randn(n_samples//2, 2) + offset
    return zero_labels, one_labels


def generate_dataset(zero_labels, one_labels):
    '''
    Generates a dataset for classification from two sets of points, assigning the label -1 to the points of the first
    set and 1 to those of the second set.
    '''
    n_zeros = len(zero_labels)
    n_ones = len(one_labels)
    data = np.concatenate((zero_labels, one_labels), axis=0)
    data = np.concatenate((data,
                np.concatenate((-np.ones([n_zeros, 1]),
                                np.ones([n_ones,1])))
                ),
                axis=1)
    np.random.shuffle(data)
    X = data[:,:2]
    y = data[:,-1]
    return X,y


def plot_datapoints(zero_labels, one_labels):
    plt.figure(1, figsize=(8,6))
    plt.scatter(zero_labels[:,0], zero_labels[:,1], color='blue', label='-1')
    plt.scatter(one_labels[:,0], one_labels[:,1], color='red', label='1')
    plt.legend()
    plt.show()


def get_weights_and_intercept(X, K, y, linear_kernel_model):
    '''
    Trains an SVM with Kernel K on 2d data and returns the corresponding weight vector and intercept of the learned
    model.
    '''
    linear_kernel_model.train(K,y)
    X_intercept = np.concatenate((X,np.ones(X.shape[0]).reshape(-1,1)), axis=1)
    W = np.sum(X_intercept * linear_kernel_model.alpha.reshape(-1,1), axis=0)
    w = W[:-1]
    b = W[-1]
    return w, b


def plot_predictions(X, y_true, y_pred, w=None, b=None):
    '''
    Plots the result of a classification done using a SVM with Kernel having a weight vector w and intercept b. Plots
    the corresponding frontiere as well as the True & False Positives & Negatives.
    '''
    xmin = np.min(X[:,0])
    xmax = np.max(X[:,0])

    ymin = np.min(X[:,1])
    ymax = np.max(X[:,0])

    plt.figure(1, figsize=(8,6))

    if w is not None and b is not None:
        x1 = np.array([0,-b/w[1]])
        x2 = np.array([-b/w[0],0])
        a = (x2[1] - x1[1]) / (x2[0] - x1[0])
        c = x1[1] - a * x1[0]
        ts = np.arange(xmin,xmax+0.5, 0.01)
        ys = [a*t + c for t in ts]
        plt.plot(ts, ys, color='g')
        plt.ylim(ymin,ymax+0.5)

    on_target = y_pred == y_true
    one_indices = y_true == 1.
    TP = X[on_target * one_indices]
    FN = X[~on_target * one_indices]
    TN = X[on_target * ~one_indices]
    FP = X[~on_target * ~one_indices]
    plt.scatter(TP[:,0], TP[:,1], color='red', label='TP')
    plt.scatter(TN[:,0], TN[:,1], color='blue', label='TN')
    plt.scatter(FN[:,0], FN[:,1], color='violet', label='FN')
    plt.scatter(FP[:,0], FP[:,1], color='purple', label='FP')
    plt.legend()
    plt.show()


def submitResults(filename, y_pred_final):
    '''
    Creates, from a 1d array of final predicted values, a file named filename formated appropriately in order to make a
    submission on the Kaggle platform.
    '''
    y = np.concatenate([y_pred_final[i] for i in [0,1,2]])
    with open("data/submission/{}.csv".format(filename), 'w') as f:
        string = "Id,Bound\n"
        for j in range(0,3000):
            string += str(j)+','+str(y[j][0])+'\n'
        f.write(string)
    print("----------------- Prediction written on {}.csv ---------------".format(filename))


def accuracy_score(y_true, y_pred):
    '''
    Computes and returns the accuracy score of the predicted labels independently of the fact that y_true and y_pred 
    have values in {0,1} or {-1,1}

    Arguments :
    - y_true : a 1d array of target labels (with values either in {0,1} or {-1,1})
    - y_pred : a 1d array of predicted labels (with values either in the same set as y_true)

    Returns : 
    - a float in [0,1] representing the accuracy score of the predictions (ratio of accurate predicted labels)
    '''
    y_true_values = np.sort(np.unique(y_true))
    y_pred_values = np.sort(np.unique(y_pred))
    assert set(y_pred_values).issubset(set(y_true_values)), "y_true_values = {}, y_pred_values = {}".format(y_true_values,
                                                                                                   y_pred_values)
    if len(y_true_values) > 2:
        raise ValueError("ys must have only 2 possible values")
    return np.sum(y_pred == y_true) / len(y_true)
