from alignment_analyzers import IBLAlignmentBase
from alignment_analyzers import TimeResolvedAlignmentAnalyzer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def split_half_indices(pos_mask, neg_mask, fr_bin, seed=42):
    rng = np.random.default_rng(seed)
    positive_idx = np.where(pos_mask)[0]
    negative_idx = np.where(neg_mask)[0]
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)
    pos_half_1_idx = positive_idx[:len(positive_idx) // 2]
    pos_half_2_idx = positive_idx[len(positive_idx) // 2:]
    neg_half_1_idx = negative_idx[:len(negative_idx) // 2]
    neg_half_2_idx = negative_idx[len(negative_idx) // 2:]

    n_trials = len(pos_mask)

    pos_half_1 = np.zeros(n_trials, dtype=bool)
    pos_half_2 = np.zeros(n_trials, dtype=bool)
    neg_half_1 = np.zeros(n_trials, dtype=bool)
    neg_half_2 = np.zeros(n_trials, dtype=bool)

    pos_half_1[pos_half_1_idx] = True
    pos_half_2[pos_half_2_idx] = True
    neg_half_1[neg_half_1_idx] = True
    neg_half_2[neg_half_2_idx] = True

    return pos_half_1, pos_half_2, neg_half_1, neg_half_2

def split_half_signal_reliability(ab:IBLAlignmentBase, fr_bin, pos_mask, neg_mask):
    pos_half_1, pos_half_2, neg_half_1, neg_half_2 = split_half_indices(pos_mask, neg_mask, fr_bin)
    u_sig_1, mu_pos_1, mu_neg_1, norm_1 = ab.compute_signal_axis_mu(fr_bin, pos_half_1, neg_half_1)
    u_sig_2, mu_pos_2, mu_neg_2, norm_2 = ab.compute_signal_axis_mu(fr_bin, pos_half_2, neg_half_2)
    cosine_similarity = np.dot(u_sig_1, u_sig_2) / (np.linalg.norm(u_sig_1) * np.linalg.norm(u_sig_2))

    delta_mu_1 = mu_pos_1 - mu_neg_1
    delta_mu_2 = mu_pos_2 - mu_neg_2

    cos_delta_mu = np.dot(delta_mu_1, delta_mu_2) / (np.linalg.norm(delta_mu_1) * np.linalg.norm(delta_mu_2) + ab.cfg.eps)
    cos_u_sig = np.dot(u_sig_1, u_sig_2) / (np.linalg.norm(u_sig_1) * np.linalg.norm(u_sig_2) + ab.cfg.eps)

    return {
        'cosine_similarity_mu': cos_delta_mu,
        'cosine_similarity_u_sig': cos_u_sig,
        'sig_norm_1': norm_1,
        'sig_norm_2': norm_2,
    }


def split_half_overlap_reliability(ab:IBLAlignmentBase, fr_bin, pos_mask, neg_mask):
    pos_half_1, pos_half_2, neg_half_1, neg_half_2 = split_half_indices(pos_mask, neg_mask, fr_bin)
    delta_mu_1, mu_pos_1, mu_neg_1, norm_1 = ab.compute_signal_axis_mu(fr_bin, pos_half_1, neg_half_1)
    delta_mu_2, mu_pos_2, mu_neg_2, norm_2 = ab.compute_signal_axis_mu(fr_bin, pos_half_2, neg_half_2)
    fr_noise_1 = ab.noise_residuals_by_sign(fr_bin, pos_half_1, neg_half_1)
    fr_noise_2 = ab.noise_residuals_by_sign(fr_bin, pos_half_2, neg_half_2)
    fr_noise_cov_1 = ab.noise_cov_from_residuals(fr_noise_1)
    fr_noise_cov_2 = ab.noise_cov_from_residuals(fr_noise_2)
    noise_subspace_1 = ab.top_noise_subspace(fr_noise_cov_1, k=3)
    noise_subspace_2 = ab.top_noise_subspace(fr_noise_cov_2, k=3)
    overlap_1 = ab.signal_noise_subspace_overlap(delta_mu_1, noise_subspace_1)
    overlap_2 = ab.signal_noise_subspace_overlap(delta_mu_2, noise_subspace_2)
    ratio_1 = overlap_1['overlap_ratio']
    ratio_2 = overlap_2['overlap_ratio']

    return {
        'overlap_ratio_1': ratio_1,
        'overlap_ratio_2': ratio_2,
    }

def fit_2d_lda(X,y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError (f"X must be a 2D array with shape (n_trials, 2), but got shape {X.shape}")
    classes = np.unique(y)
    if not set(classes).issubset({-1, 1}):
        raise ValueError(f"y must contain only binary labels -1 and 1, but got {classes}")
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == -1)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must have at least one sample.")
    
    lda = LDA(solver='lsqr', shrinkage='auto')
    lda.fit(X, y)

    w = lda.coef_.ravel()
    b = lda.intercept_[0]   
    return w, b



def split_half_decoder_reliability(ab: IBLAlignmentBase, pos_mask, neg_mask, fr_bin):

    pos_half_1, pos_half_2, neg_half_1, neg_half_2 = split_half_indices(pos_mask, neg_mask, fr_bin)
    half1 = pos_half_1 | neg_half_1
    half2 = pos_half_2 | neg_half_2
    u_sig_1, mu_pos_1, mu_neg_1, _ = ab.compute_signal_axis_mu(fr_bin, pos_half_1, neg_half_1)
    u_sig_2, mu_pos_2, mu_neg_2, _ = ab.compute_signal_axis_mu(fr_bin, pos_half_2, neg_half_2)
    fr_noise_1 = ab.noise_residuals_by_sign(fr_bin, pos_half_1, neg_half_1, mu_pos_1, mu_neg_1)
    fr_noise_2 = ab.noise_residuals_by_sign(fr_bin, pos_half_2, neg_half_2, mu_pos_2, mu_neg_2)
    C_1, keep_1 = ab.noise_cov_from_residuals(fr_noise_1, half1)
    C_2, keep_2 = ab.noise_cov_from_residuals(fr_noise_2, half2)
    u_noi_1, _ = ab.top_eigenvector(ab.trace_normalize(C_1), return_eigval=True)
    u_noi_2, _ = ab.top_eigenvector(ab.trace_normalize(C_2), return_eigval=True)
    X1 = fr_bin[:, keep_1]
    X2 = fr_bin[:, keep_2]
    N1 = fr_noise_1[:, keep_1]
    N2 = fr_noise_2[:, keep_2]
    u_sig_1_k = u_sig_1[keep_1]
    u_sig_1_k = u_sig_1_k / (np.linalg.norm(u_sig_1_k) + ab.cfg.eps)
    u_sig_2_k = u_sig_2[keep_2]
    u_sig_2_k = u_sig_2_k / (np.linalg.norm(u_sig_2_k) + ab.cfg.eps)
    proj_sig_1 = X1[half1] @ u_sig_1_k
    proj_noi_1 = N1[half1] @ u_noi_1
    proj_sig_2 = X2[half2] @ u_sig_2_k
    proj_noi_2 = N2[half2] @ u_noi_2

    y = np.zeros_like(pos_mask, dtype=int)
    y[pos_mask] = 1
    y[neg_mask] = -1
    X_1 = np.stack([proj_sig_1, proj_noi_1], axis=1)
    X_2 = np.stack([proj_sig_2, proj_noi_2], axis=1)
    w1, b1 = fit_2d_lda(X_1, y[half1])
    w2, b2 = fit_2d_lda(X_2, y[half2])
    cosine_similarity = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + ab.cfg.eps)
    return {
        'cosine_similarity_decoder': cosine_similarity,
        "w1": w1,
        "w2": w2,
    }

