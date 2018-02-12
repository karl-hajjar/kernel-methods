import numpy as np

def gaussian_kernel_matrix(X, sigma=1):
    return np.exp(- np.sum((X - X[:,None])**2, axis=-1) / (2*sigma**2))

def polynomial_kernel_matrix(X, deg=1):
    return np.sum(X*X[:,None], axis=-1)**deg

def KRR(K, y, l=0.1):
    A = K + l * len(K)*np.eye(len(K))
    return np.linalg.solve(A, y)
