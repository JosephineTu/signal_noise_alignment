import numpy as np
def build_feature_matrix(self, X, u_sig, u_noi):
    proj_sig = X @ u_sig
    proj_noi = X @ u_noi
    return np.c_[proj_sig, proj_noi]

def fit_lda_axis(self, X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    mu_1 = X[y == 1].mean(axis=0)
    mu_0 = X[y == -1].mean(axis=0)

    C = np.cov(X.T, bias=False)
    if C.ndim==0:
        C = np.array([[float(C)]])
    if C.shape != (2,2):
        C = np.eye(2) * self.cfg.eps
    
    C += self.cfg.eps * np.eye(2)
    w = np.linalg.solve (C, mu_1 - mu_0)
    b = 0.5 * (mu_1 + mu_0) @ w
    return w, b