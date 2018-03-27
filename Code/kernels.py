import numpy as np
from tqdm import tqdm
from scipy import sparse

def gaussian_kernel_matrix(X, sigma=1.):
    '''
    Computes and returns the Gaussian Kernel associated to a data matrix X.

    Parameters
    ----------
    X : nd-array
        a 2d array containing the data in rows
    sigma : float (default 1.0)
        the variance parameter in the Gaussian Kernel

    Returns
    -------
    a 2d symmetric array of shape (X.shape[0], X.shape[0])
    '''
    return np.exp(- np.sum((X - X[:,None])**2, axis=-1) / (2*sigma**2))


def polynomial_kernel_matrix(X, deg=1):
    '''
    Computes and returns the Polynomial Kernel associated to a data matrix X.

    Parameters
    ----------
    X : nd-array
        a 2d array containing the data in rows
    deg : int (default 1)
        the degree of the polynomial used to compute the Kernel

    Returns
    -------
    a 2d symmetric array of shape (X.shape[0], X.shape[0])
    '''
    X_intercept = np.concatenate((X,np.ones(X.shape[0]).reshape(-1,1)), axis=1)
    return np.sum(X_intercept * X_intercept[:,None], axis=-1)**deg


def one_mismatch_away(s, codon=False):
    '''
    Computes all the sequences that are one mismatch away (exactly) from the sequence of letters s

    Parameters
    ----------
    s : string
        sequence of letters of which we wish to compute all the possible modifications of one letter only
    codon : bool (default False)
        whether or not the letters used represent acido-amines (coding for specific molecules) or simply genes

    Returns
    -------
    a list containing all the possible variations of one letter of the given sequence s
    '''
    res = []
    if codon :
        all_letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z']
    else :
        all_letters = ['A','T','C','G']
    for i,letter in enumerate(s):
        if letter not in all_letters:
            print('Please use only letters in {}'.format(all_letters))
        for l in all_letters:
            if letter != l:
                res.append(str(s[:i]) + l + str(s[i+1:]))
    return res


def two_mismatch_away(s):
    '''
    Computes all the sequences that are two mismatches away from the sequence of letters s

    Parameters
    ----------
    s : string
        sequence of letters of which we wish to compute all the possible modifications of one letter only

    Returns
    -------
    a list containing all the possible variations of two letters of the given sequence s
    '''
    res = []
    l = one_mismatch_away(s)
    for a in l:
        for t in one_mismatch_away(a):
            if t not in res and t != s and t not in l:
                res.append(t)
    return res


def embedding_mismatch_kernel(X, length, mismatch):
    '''
    Computes and returns the Mismatch Kernel associated to a data matrix X.

    Parameters
    ----------
    X : nd-array
        a 2d array containing the data in rows
    length : int
        the length of the subsequences to consider
    mismatch : int
        the number of mismatches to consider [mismatch = 0  corresponds to the simpler Spectrum Kernel]

    Returns
    -------
    a 2d symmetric array of shape (X.shape[0], X.shape[0])
    '''
    ## Computing all the subsequences of length 'length' present in the data along with all their possible one letter or 
    ## two letter variations (acording to the argument mismatch) and storing them in a dictionnary where, for every 
    ## subsequence, the value of the corresponding key is an id for the sequence (the first subsequence to be 
    ## encountered in the dataset will have index 0, the second 1, etc)
    all_sequences_index = {}
    id_last_seq = 0
    #print('mismatch = {}'.format(mismatch))
    print('----- Computing and storing all possible subsequences with mismatches ------')
    for idx in range(len(X)):
    ## for each row in the data
        data = X[idx]
        for i in range(len(data)-length + 1):
        ## for each subsequence in the row
            seq = data[i:i+length]
            if seq not in all_sequences_index:
            ## if seq has not been encoutered yet, store it
                all_sequences_index[seq] = id_last_seq
                id_last_seq += 1

            if mismatch >= 1:
            ## Compute and store one-mismatches from seq
                for seq_mis in one_mismatch_away(seq):
                    if seq_mis not in all_sequences_index:
                        all_sequences_index[seq_mis] = id_last_seq
                        id_last_seq += 1

            if mismatch == 2:
            ## Compute and store 2-mismatches from seq
                for seq_mis in two_mismatch_away(seq):
                    if seq_mis not in all_sequences_index:
                        all_sequences_index[seq_mis] = id_last_seq
                        id_last_seq += 1

            if mismatch > 2:
                raise ValueError("In our implementation of the Mismatch Kernel, the number of mismatches must be 0, 1 or 2")

    ## A lil-matrix (sparse format whose values can be updated easily) containg for each row of data the number of times 
    ## each subsequence possible is encountered, up to a certain penalty (1/2 for one-mismatches, 1/4 for two-mismatches)
    vectors = sparse.lil_matrix((len(X),len(all_sequences_index)), dtype=float)

    ## Filling in the matrix line by line
    for idx in tqdm(range(len(X)), desc='% of lines of data handled'):
    ## for each row in the data
        data = X[idx]
        for i in range(len(data)-length + 1):
        ## for each subsequence in the row
            seq = data[i:i+length]
            ## update the corresponding value of the matrix
            vectors[idx, all_sequences_index[seq]] += 1
            if mismatch >= 1:
            ## Computing 1-mismatches
                for seq_mis in one_mismatch_away(seq):
                    vectors[idx, all_sequences_index[seq_mis]] += 1/2
            if mismatch == 2:
            ## Computing 2-mismatches
                for seq_mis in two_mismatch_away(seq):
                    vectors[idx, all_sequences_index[seq_mis]] += 1/4

    ## Converting to sparse format before computing
    vectors.tocsr()
    return np.array(np.dot(vectors, vectors.T).todense())
