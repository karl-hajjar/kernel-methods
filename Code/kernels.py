import numpy as np

def gaussian_kernel_matrix(X, sigma=1):
    return np.exp(- np.sum((X - X[:,None])**2, axis=-1) / (2*sigma**2))

def polynomial_kernel_matrix(X, deg=1):
    X_intercept = np.concatenate((X,np.ones(X.shape[0]).reshape(-1,1)), axis=1)
    return np.sum(X_intercept*X_intercept[:,None], axis=-1)**deg
def compare_two_seq(s1,s2,a,b):
    res = 0
    n = len(s1)
    assert(n==len(s2))
    for k in range(a,b+1):
        resk = 0
        for i in range(n-k+1):
            if (s1[i:i+k]==s2[i:i+k]):
                resk +=1
        beta_k = 2*(n-k+1)/(n*(n+1))
        res += resk*beta_k
    return res

def weighted_degree_kernel(X, a, b, name):
    """
    if data/name.npy exists, loads it and returns it, else, compute the WDK of X, comparing sub-strings from
    length a, a+1, a+2 to b, and save it as data/name.npy.
    """

    try:
        res = np.load('data/{}.npy'.format(name))
        return res
    except:
        n = len(X)
        K = np.zeros((n,n))
        for i in tqdm(range(n)):
            for j in range(i,n):
                K[i,j] = compare_two_seq(X[i,0], X[j,0],a,b)
                K[j,i] = K[i,j]
        np.save('data/{}'.format(name), K)
        return K
