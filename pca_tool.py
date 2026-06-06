import numpy as np
from numpy.linalg import eigh
from scipy.stats import percentileofscore

def top_eigenvector(C, return_eigval=False):
    """
       C: symmetric matrix
    """
    eps = 1e-12
    C = np.array(C, dtype=float)
    C = (C + C.T)/2
    vals, vecs = eigh(C)
    top_idx = np.argmax(vals)
    v = vecs[:, top_idx]

    if np.abs(v).max() < eps:
        v = v
    else:
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
    v = v / np.linalg.norm(v)
    if return_eigval:
        return v, vals[top_idx]
    else:
        return v
    
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def angle_between(a, b):
    c = cosine_similarity(a, b)
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)


def eigenvectors_above_threshold(C, threshold, return_eigvals=False):

    eps = 1e-12

    C = np.asarray(C, dtype=float)
    C = (C + C.T) / 2  # enforce symmetry

    vals, vecs = eigh(C)  # ascending order
    idx = np.where(vals >= threshold)[0]

    if len(idx) == 0:
        raise ValueError("No eigenvalues above threshold.")

    V = vecs[:, idx]

    # sign normalization + unit norm
    for k in range(V.shape[1]):
        v = V[:, k]
        if np.abs(v).max() > eps and v[np.argmax(np.abs(v))] < 0:
            V[:, k] = -v
        V[:, k] /= np.linalg.norm(V[:, k])

    if return_eigvals:
        return V, vals[idx]
    else:
        return V
def participation_ratio(eigenvalues):
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)