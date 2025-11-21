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
