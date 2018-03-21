import numpy as np
from tqdm import tqdm

def gaussian_kernel_matrix(X, sigma=1):
    return np.exp(- np.sum((X - X[:,None])**2, axis=-1) / (2*sigma**2))

def polynomial_kernel_matrix(X, deg=1):
    X_intercept = np.concatenate((X,np.ones(X.shape[0]).reshape(-1,1)), axis=1)
    return np.sum(X_intercept * X_intercept[:,None], axis=-1)**deg

def embedding_mismatch_kernel(X, lengths, mismatch, verbose = False):

    all_sequences_index = {}
    id_last_seq = 0
    for data in X:
        for i in range(len(data)-lengths + 1):
            seq = data[i:i+lengths]
            if seq not in all_sequences_index:
                all_sequences_index[seq] = id_last_seq
                id_last_seq += 1


    vectors = np.zeros((len(X),len(all_sequences_index)))

    for idx in tqdm(range(len(X))):
        data = X[idx]
        for i in range(len(data)-lengths + 1):
            seq = data[i:i+lengths]
            vectors[idx, all_sequences_index[seq]] += 1
            if mismatch >= 1:
                for seq_mis in one_mismatch_away(seq, True):
                    if seq_mis in all_sequences_index:
                        vectors[idx, all_sequences_index[seq_mis]] += 1/2
            if mismatch >= 2:
                for seq_mis in two_mismatch_away(seq):
                    if seq_mis in all_sequences_index:
                        vectors[idx, all_sequences_index[seq_mis]] += 1/3

    if verbose:
        print(all_sequences_index)
        print('Embedding :')
        print(vectors)
    return np.dot(vectors,vectors.T)


def one_mismatch_away(s, codon = False):
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
    res = []
    l = one_mismatch_away(s)
    for a in l:
        for t in one_mismatch_away(a):
            if t not in res and t != s and t not in l:
                res.append(t)
    return res


def new_embedding_mismatch_kernel(X, lengths, mismatch, verbose = False):

    all_sequences_index = {}
    id_last_seq = 0
    for idx in range(len(X)):
        data = X[idx]
        for i in range(len(data)-lengths + 1):
            seq = data[i:i+lengths]
            if seq not in all_sequences_index:
                all_sequences_index[seq] = id_last_seq
                id_last_seq += 1

            if mismatch >= 1:
                for seq_mis in one_mismatch_away(seq):
                    if seq_mis not in all_sequences_index:
                        all_sequences_index[seq_mis] = id_last_seq
                        id_last_seq += 1

            if mismatch >= 2:
                for seq_mis in two_mismatch_away(seq):
                    if seq_mis not in all_sequences_index:
                        all_sequences_index[seq_mis] = id_last_seq
                        id_last_seq += 1


    vectors = np.zeros((len(X),len(all_sequences_index)))

    for idx in tqdm(range(len(X))):
        data = X[idx]
        for i in range(len(data)-lengths + 1):
            seq = data[i:i+lengths]
            vectors[idx, all_sequences_index[seq]] += 1
            if mismatch >= 1:
                for seq_mis in one_mismatch_away(seq):
                    vectors[idx, all_sequences_index[seq_mis]] += 1/2
            if mismatch >= 2:
                for seq_mis in two_mismatch_away(seq):
                    vectors[idx, all_sequences_index[seq_mis]] += 1/4

    if verbose:
        print(all_sequences_index)
        print('Embedding :')
        print(vectors)

    embeddings = sparse.csr_matrix(vectors)
    return np.dot(embeddings, embeddings.T).todense()
