# time_resolved_alignment.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
from brainbox.population.decode import get_spike_counts_in_bins
from passiveGabor_VIS import *
# ----------------------------
# Utils (linear algebra)
# ----------------------------
def enforce_sym(C):
    C = np.asarray(C, dtype=float)
    return 0.5 * (C + C.T)

def trace_normalize(C, eps=1e-12):
    tr = np.trace(C)
    if tr < eps:
        return C
    return C / tr

def top_eigenvector(C, return_eigval=False):
    C = enforce_sym(C)
    vals, vecs = np.linalg.eigh(C)
    idx = int(np.argmax(vals))
    v = vecs[:, idx]
    # deterministic sign
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v = -v
    v = v / (np.linalg.norm(v) + 1e-12)
    return (v, float(vals[idx])) if return_eigval else v

def cosine_abs(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.abs(np.dot(u, v)))

def eig_spectrum(C, do_trace_norm=True):
    C = enforce_sym(C)
    if do_trace_norm:
        C = trace_normalize(C)
    vals = np.linalg.eigvalsh(C)[::-1]
    vals = np.clip(vals, 0, None)
    cum = np.cumsum(vals) / (np.sum(vals) + 1e-12)
    return vals, cum

def participation_ratio(vals):
    vals = np.asarray(vals, float)
    vals = np.clip(vals, 0, None)
    s1 = np.sum(vals)
    s2 = np.sum(vals**2)
    return float((s1**2) / (s2 + 1e-12))

def ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(f"{name} has unexpected shape {x.shape} (expected (n_trials,))")

# ----------------------------
# IBL loading
# ----------------------------
def pick_best_insertion(one: ONE, atlas, eid: str, target_prefix="VIS"):
    ins = one.alyx.rest("insertions", "list", session=eid)
    best_pid, best_n = None, -1
    for x in ins:
        pid = x["id"]
        try:
            sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
            acr = clusters.get("acronym", None)
            if acr is None:
                continue
            acr = np.asarray(acr)
            n = int(np.sum([a.startswith(target_prefix) for a in acr]))
            if n > best_n:
                best_n = n
                best_pid = pid
        except Exception:
            continue
    return best_pid

def load_trials_and_spikes(one: ONE, atlas, eid: str, target_prefix="VIS"):
    trials = one.load_object(eid, "trials", collection="alf")

    # robust trial filtering by stimOn/stimOff availability
    stim_on = ensure_1d(trials["stimOn_times"], "stimOn_times")
    stim_off = ensure_1d(trials["stimOff_times"], "stimOff_times")
    valid = (~np.isnan(stim_on)) & (~np.isnan(stim_off))

    # slice only fields whose first dim matches n_trials
    nT = len(stim_on)
    for k in list(trials.keys()):
        v = np.asarray(trials[k])
        if v.ndim >= 1 and v.shape[0] == nT:
            trials[k] = v[valid]
        else:
            trials[k] = v

    stim_on = stim_on[valid]
    stim_off = stim_off[valid]

    pid = pick_best_insertion(one, atlas, eid, target_prefix=target_prefix)
    if pid is None:
        raise RuntimeError("No valid insertion found.")

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # region filter
    acr = np.asarray(clusters["acronym"])
    region_mask = np.array([a.startswith(target_prefix) for a in acr])
    region_cluster_ids = np.asarray(clusters["cluster_id"])[region_mask]

    return trials, spikes, clusters, pid, region_cluster_ids

# ----------------------------
# Build time-binned FR tensor: (n_trials, n_bins, n_units)
# ----------------------------
def make_time_bins(t_start, t_end, bin_size):
    edges = np.arange(t_start, t_end + 1e-12, bin_size)
    if len(edges) < 2:
        raise ValueError("Time range too small for given bin_size.")
    return edges  # relative to stimOn

def trial_binned_firing_rates(spikes, region_cluster_ids, stim_on, edges):
    stim_on = np.asarray(stim_on)
    n_trials = stim_on.shape[0]
    n_bins = len(edges) - 1
    n_intervals = n_trials * n_bins

    starts = (stim_on[:, None] + edges[:-1][None, :]).reshape(-1)
    ends   = (stim_on[:, None] + edges[1: ][None, :]).reshape(-1)
    intervals = np.c_[starts, ends]

    counts, cluster_ids = get_spike_counts_in_bins(
        spikes["times"], spikes["clusters"], intervals
    )
    cluster_ids = np.asarray(cluster_ids)

    # ---- FIX: handle counts orientation robustly ----
    # expected either (n_clusters, n_intervals) or (n_intervals, n_clusters)
    if counts.shape[0] == cluster_ids.shape[0] and counts.shape[1] == n_intervals:
        # counts is (n_clusters, n_intervals)
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[keep, :]          # (n_units, n_intervals)
        unit_ids = cluster_ids[keep]
        counts = counts.T                 # (n_intervals, n_units)
    elif counts.shape[1] == cluster_ids.shape[0] and counts.shape[0] == n_intervals:
        # counts is (n_intervals, n_clusters)
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[:, keep]          # (n_intervals, n_units)
        unit_ids = cluster_ids[keep]
    else:
        raise RuntimeError(
            f"Unexpected counts shape {counts.shape}, "
            f"cluster_ids {cluster_ids.shape}, n_intervals={n_intervals}"
        )

    # reshape to (n_trials, n_bins, n_units)
    counts = counts.reshape(n_trials, n_bins, -1)
    dur = (edges[1:] - edges[:-1])[None, :, None]
    fr_tb = counts / (dur + 1e-12)
    intervals_tb = intervals.reshape(n_trials, n_bins, 2)
    return fr_tb, unit_ids, intervals_tb

# ----------------------------
# Conditions + labels (trial-level)
# ----------------------------
def get_signed_and_masks(trials, thresh=0.5):
    cl = ensure_1d(trials["contrastLeft"], "contrastLeft")
    cr = ensure_1d(trials["contrastRight"], "contrastRight")
    cl = np.nan_to_num(cl, nan=0.0)
    cr = np.nan_to_num(cr, nan=0.0)

    signed = cl - cr
    is_zero = (cl == 0) & (cr == 0)

    high_mask = (~is_zero) & (np.abs(signed) > thresh)
    pos_mask = high_mask & (signed > 0)
    neg_mask = high_mask & (signed < 0)

    return signed, is_zero, high_mask, pos_mask, neg_mask

def get_choice_labels(trials):
    # IBL usually: choice in {-1, +1} and possibly 0 (no-go)
    choice = ensure_1d(trials["choice"], "choice").astype(float)
    feedback = ensure_1d(trials["feedbackType"], "feedbackType").astype(float)
    pleft = ensure_1d(trials["probabilityLeft"], "probabilityLeft").astype(float)
    return choice, feedback, pleft

# ----------------------------
# Time-resolved signal/noise alignment (per bin)
# ----------------------------
def compute_signal_axis_mu(fr_bin, pos_mask, neg_mask, min_trials=5):
    if pos_mask.sum() < min_trials or neg_mask.sum() < min_trials:
        raise RuntimeError(f"Not enough trials (pos={pos_mask.sum()}, neg={neg_mask.sum()})")

    mu_pos = fr_bin[pos_mask].mean(axis=0)
    mu_neg = fr_bin[neg_mask].mean(axis=0)

    u_sig = mu_pos - mu_neg
    nrm = np.linalg.norm(u_sig)
    if nrm < 1e-10:
        raise RuntimeError("Signal axis too small.")
    u_sig = u_sig / nrm
    return u_sig, mu_pos, mu_neg, float(nrm)

def noise_residuals_by_sign(fr_bin, pos_mask, neg_mask, mu_pos, mu_neg):
    fr_noise = np.full_like(fr_bin, np.nan, dtype=float)
    fr_noise[pos_mask] = fr_bin[pos_mask] - mu_pos
    fr_noise[neg_mask] = fr_bin[neg_mask] - mu_neg
    return fr_noise

def noise_corr_from_residuals(fr_noise, high_mask):
    X = fr_noise[high_mask]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = enforce_sym(C)
    return C

# ----------------------------
# Simple CV decoding (optional)
# ----------------------------
def stratified_kfold_indices(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    # binary labels expected in {-1,+1} or {0,1}
    cls = np.unique(y)
    folds = [[] for _ in range(k)]
    for c in cls:
        ic = idx[y == c]
        rng.shuffle(ic)
        parts = np.array_split(ic, k)
        for i in range(k):
            folds[i].extend(parts[i].tolist())
    folds = [np.array(sorted(f), dtype=int) for f in folds]
    return folds

def roc_auc_approx(y_true, score):
    # y_true in {-1,+1} or {0,1}; score real
    y = np.asarray(y_true)
    s = np.asarray(score)
    # convert to {0,1}
    if set(np.unique(y)).issubset({-1, 1}):
        y01 = (y == 1).astype(int)
    else:
        y01 = (y > 0).astype(int)
    # rank-based AUC
    order = np.argsort(s)
    y_sorted = y01[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Mann–Whitney U
    ranks = np.arange(1, len(y_sorted) + 1)
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)

def cv_decode_1d(proj, y, k=5, seed=42):
    """
    Very simple decoder: threshold on mean difference (equivalent to 1D LDA on proj).
    Returns AUC across folds.
    """
    proj = np.asarray(proj, float)
    y = np.asarray(y, float)
    folds = stratified_kfold_indices(y, k=k, seed=seed)
    aucs = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        # fit threshold by class means on train
        m1 = proj[train_idx][y[train_idx] == 1].mean()
        m0 = proj[train_idx][y[train_idx] == -1].mean()
        # score: signed distance
        score = (proj[test_idx] - 0.5*(m1+m0)) * np.sign(m1 - m0 + 1e-12)
        aucs.append(roc_auc_approx(y[test_idx], score))
    return float(np.nanmean(aucs))

# ----------------------------
# Run one session (time-resolved)
# ----------------------------
def run_one_eid_time_resolved(
    one, atlas, eid: str,
    target_prefix="VIS",
    thresh=0.5,
    t_start=0.0,
    t_end=0.4,
    bin_size=0.05,
    min_trials=5,
    do_plots=True,
    do_decode=True,
):
    print(f"\nProcessing eid={eid} target_prefix={target_prefix}")

    trials, spikes, clusters, pid, region_cluster_ids = load_trials_and_spikes(
        one, atlas, eid, target_prefix=target_prefix
    )

    stim_on = ensure_1d(trials["stimOn_times"], "stimOn_times")
    signed, is_zero, high_mask, pos_mask, neg_mask = get_signed_and_masks(trials, thresh=thresh)
    choice, feedback, pleft = get_choice_labels(trials)

    # filter out no-go choice==0 for choice decoding (optional)
    choice_valid = choice != 0

    edges = make_time_bins(t_start, t_end, bin_size)
    fr_tb, unit_ids, intervals_tb = trial_binned_firing_rates(
        spikes, region_cluster_ids, stim_on, edges
    )
    n_trials, n_bins, n_units = fr_tb.shape
    if n_units < 5:
        raise RuntimeError(f"Too few units after region filter: {n_units}")

    # Pre-mask neurons that have std>0 in BOTH all trials and high trials across ALL bins (stability)
    # Here: compute std over (trial, bin) combined
    fr_flat = fr_tb.reshape(n_trials*n_bins, n_units)
    std_all = fr_flat.std(axis=0)
    std_high = fr_tb[high_mask].reshape((-1, n_units)).std(axis=0) if high_mask.any() else np.zeros(n_units)
    neuron_mask = (std_all > 1e-10) & (std_high > 1e-10)

    fr_tb = fr_tb[:, :, neuron_mask]
    unit_ids = unit_ids[neuron_mask]
    n_units = fr_tb.shape[2]

    # time-resolved results
    times = 0.5 * (edges[:-1] + edges[1:])
    cos_ts = np.full(n_bins, np.nan)
    noise_top1_ts = np.full(n_bins, np.nan)
    noise_pr_ts = np.full(n_bins, np.nan)
    sig_norm_ts = np.full(n_bins, np.nan)

    stim_auc_ts = np.full(n_bins, np.nan)
    choice_auc_ts = np.full(n_bins, np.nan)
    fb_auc_ts = np.full(n_bins, np.nan)

    for b in range(n_bins):
        fr_bin = fr_tb[:, b, :]  # (n_trials, n_units)

        # define signal axis within this bin
        try:
            u_sig, mu_pos, mu_neg, sig_norm = compute_signal_axis_mu(
                fr_bin, pos_mask, neg_mask, min_trials=min_trials
            )
        except Exception:
            continue

        sig_norm_ts[b] = sig_norm

        # residual noise and noise covariance
        fr_noise = noise_residuals_by_sign(fr_bin, pos_mask, neg_mask, mu_pos, mu_neg)
        try:
            C_noi = noise_corr_from_residuals(fr_noise, high_mask)
            C_noi_norm = trace_normalize(C_noi)
            u_noi, ev_noi = top_eigenvector(C_noi_norm, return_eigval=True)
        except Exception:
            continue

        cos_ts[b] = cosine_abs(u_sig, u_noi)
        vals, _ = eig_spectrum(C_noi, do_trace_norm=True)
        noise_top1_ts[b] = float(vals[0]) if len(vals) else np.nan
        noise_pr_ts[b] = participation_ratio(vals)

        # decoding (1D proj decoder to keep things simple + no sklearn dependency)
        if do_decode:
            # stimulus sign decoding (labels in +/-1)
            y_stim = np.sign(signed).astype(int)
            y_stim = np.where(y_stim == 0, 0, y_stim)  # keep 0 if any
            use = high_mask & (y_stim != 0)
            if use.sum() >= 2 * min_trials:
                proj = fr_bin[use] @ u_sig
                y = np.sign(signed[use]).astype(int)
                y = np.where(y > 0, 1, -1)
                stim_auc_ts[b] = cv_decode_1d(proj, y, k=5, seed=0)

            # choice decoding on high trials (optional)
            usec = high_mask & choice_valid
            if usec.sum() >= 2 * min_trials:
                # label choice: map to +/-1 (assuming IBL choice uses -1/+1)
                y = choice[usec].copy()
                y = np.where(y > 0, 1, -1)
                # project using u_sig (sensory axis) or use residual projection:
                proj = (fr_bin[usec] @ u_sig)
                choice_auc_ts[b] = cv_decode_1d(proj, y, k=5, seed=1)

            # feedback/correct decoding (1 vs -1)
            usef = high_mask & (feedback != 0)
            if usef.sum() >= 2*min_trials:
                y = feedback[usef].copy()
                y = np.where(y > 0, 1, -1)
                # use residual projection (more interpretable for "noise causes errors"):
                proj = (fr_noise[usef] @ u_sig)
                fb_auc_ts[b] = cv_decode_1d(proj, y, k=5, seed=2)

    if do_plots:
        plt.figure(figsize=(7,4))
        plt.plot(times, cos_ts, marker="o")
        plt.ylim(0, 1.05)
        plt.axhline(0, color="k", lw=1)
        plt.xlabel("time from stimOn (s)")
        plt.ylabel("|cos(u_sig(t), u_noi(t))|")
        plt.title("Time-resolved signal-noise alignment")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7,4))
        plt.plot(times, noise_top1_ts, marker="o", label="noise top-1 (trace-norm)")
        plt.plot(times, noise_pr_ts/np.nanmax(noise_pr_ts+1e-12), marker="o", label="PR (normed)")
        plt.axhline(0, color="k", lw=1)
        plt.xlabel("time from stimOn (s)")
        plt.ylabel("summary")
        plt.title("Time-resolved noise structure")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        if do_decode:
            plt.figure(figsize=(7,4))
            plt.plot(times, stim_auc_ts, marker="o", label="stimulus AUC (proj)")
            plt.plot(times, choice_auc_ts, marker="o", label="choice AUC (proj)")
            plt.plot(times, fb_auc_ts, marker="o", label="feedback AUC (resid proj)")
            plt.axhline(0.5, color="k", lw=1)
            plt.ylim(0.0, 1.0)
            plt.xlabel("time from stimOn (s)")
            plt.ylabel("AUC")
            plt.title("Time-resolved decoding (CV, 1D)")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()

    out = {
        "eid": eid,
        "pid": pid,
        "n_trials": int(n_trials),
        "n_units": int(n_units),
        "times": times,
        "cos_ts": cos_ts,
        "noise_top1_ts": noise_top1_ts,
        "noise_pr_ts": noise_pr_ts,
        "sig_norm_ts": sig_norm_ts,
        "stim_auc_ts": stim_auc_ts,
        "choice_auc_ts": choice_auc_ts,
        "fb_auc_ts": fb_auc_ts,
    }
    print(f"done eid={eid} pid={pid} trials={n_trials} units={n_units}")
    return out

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # TODO: replace this with your own EID list loading
    # eids = [...]
    # If you already have results = load_results(...), you can import & build eids outside this file.
    # Minimal example: hardcode a few eids
    results = load_results('VISp_subjects_by_lab.json')
    eids = []
    for lab_name in results.keys():
        for subject in results[lab_name].keys():
            subject_eid = results[lab_name][subject]['VIS_eids']
            eids.extend(subject_eid)

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(password="international")
    atlas = ba()

    all_out = []
    for eid in eids:
        try:
            out = run_one_eid_time_resolved(
                one, atlas, eid,
                target_prefix="VIS",
                thresh=0.5,
                t_start=0.0, t_end=0.4, bin_size=0.05,
                min_trials=5,
                do_plots=True,
                do_decode=True,
            )
            all_out.append(out)
        except Exception as e:
            print("skip", eid, "because", repr(e))