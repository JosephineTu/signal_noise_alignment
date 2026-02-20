from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
import numpy as np
from passiveGabor_VIS import *
from brainbox.population.decode import get_spike_counts_in_bins
from pca_tool import *
import matplotlib.pyplot as plt
results = load_results('VISp_subjects_by_lab.json')
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
eids = []
for lab_name in results.keys():
    for subject in results[lab_name].keys():
        subject_eid = results[lab_name][subject]['VIS_eids']
        eids.extend(subject_eid)

atlas = ba()
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
    idx = np.argmax(vals)
    v = vecs[:, idx]
    j = np.argmax(np.abs(v))
    if v[j] < 0:
        v = -v
    v = v / (np.linalg.norm(v) + 1e-12)
    return (v, vals[idx]) if return_eigval else v

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.abs(np.dot(u, v)))

def angle_deg(u, v):
    c = np.clip(cosine(u, v), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def eig_spectrum(C, do_trace_norm=True):
    C = enforce_sym(C)
    if do_trace_norm:
        C = trace_normalize(C)
    vals = np.linalg.eigvalsh(C)[::-1]  
    vals = np.clip(vals, 0, None)
    cum = np.cumsum(vals) / (np.sum(vals) + 1e-12)
    return vals, cum

def drop_zero_std(X, eps=1e-10):
    std = X.std(axis=0)
    keep = std > eps
    return X[:, keep], keep

def participation_ratio(vals):
    vals = np.asarray(vals, float)
    vals = np.clip(vals, 0, None)
    s1 = np.sum(vals)
    s2 = np.sum(vals**2)
    return float((s1**2) / (s2 + 1e-12))

# ----------------------------
# IBL loading (single eid, best VIS insertion)
# ----------------------------
def pick_best_insertion(eid, target_prefix="VIS"):
    ins = one.alyx.rest('insertions', 'list', session=eid)
    best_pid, best_n = None, -1
    pids = [x['id'] for x in ins]
    for x in ins:
        pid = x['id']
        try:
            sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
            acr = clusters.get('acronym', None)
            if acr is None:
                continue
            n = np.sum(np.array([a.startswith(target_prefix) for a in acr]))
            if n > best_n:
                best_n = n
                best_pid = pid
        except Exception:
            continue
    return best_pid, pids

def load_trial_firing_rates(eid, pid=None, target_prefix="VIS"):
    trials = one.load_object(eid, 'trials', collection='alf')

    stim_on = trials['stimOn_times']
    stim_off = trials['stimOff_times']
    valid = ~np.isnan(stim_on) & ~np.isnan(stim_off)

    stim_on = stim_on[valid]
    stim_off = stim_off[valid]
    for k in list(trials.keys()):
        trials[k] = trials[k][valid]

    stim_intervals = np.c_[stim_on, stim_off]

    if pid is None:
        pid, pids = pick_best_insertion(eid, target_prefix=target_prefix)
        if pid is None:
            raise RuntimeError("No valid insertion found for this eid.")

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # bin spikes into trial intervals
    counts, cluster_ids = get_spike_counts_in_bins(
        spikes['times'], spikes['clusters'], stim_intervals
    )
    counts = counts.T  # (n_trials, n_clusters)

    # filter clusters by region prefix
    acr = clusters['acronym']
    region_mask = np.array([a.startswith(target_prefix) for a in acr])
    region_cluster_ids = clusters['cluster_id'][region_mask]
    region_cluster_ids = np.intersect1d(cluster_ids, region_cluster_ids)

    keep = np.isin(cluster_ids, region_cluster_ids)
    counts = counts[:, keep]
    dur = np.diff(stim_intervals, axis=1)  # (n_trials, 1)
    fr = counts / (dur + 1e-12)

    return trials, fr, pid


def parse_conditions(trials, thresh=0.1):
    cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
    cr = np.nan_to_num(trials['contrastRight'], nan=0.0)
    print("contrastLeft:", np.asarray(trials["contrastLeft"]).shape)
    print("contrastRight:", np.asarray(trials["contrastRight"]).shape)
    print("intervals:", np.asarray(trials["intervals"]).shape)
    signed = cl - cr  
    is_zero = (cl == 0) & (cr == 0)

    nonzero_all = np.unique(signed[~is_zero])
    nonzero = nonzero_all[np.abs(nonzero_all) > thresh]

    return signed, is_zero, nonzero


def compute_signal_axis_mu(fr, signed, is_zero, thresh=0.1, min_trials=5):
    high_mask = (~is_zero) & (np.abs(signed) > thresh)

    pos = high_mask & (signed > 0)
    neg = high_mask & (signed < 0)

    if pos.sum() < min_trials or neg.sum() < min_trials:
        raise RuntimeError(
            f"Not enough high-contrast trials "
            f"(pos={pos.sum()}, neg={neg.sum()})"
        )

    mu_pos = fr[pos].mean(axis=0)
    mu_neg = fr[neg].mean(axis=0)

    u_sig = mu_pos - mu_neg
    norm = np.linalg.norm(u_sig)
    if norm < 1e-10:
        raise RuntimeError("Signal axis norm too small (mu_pos ≈ mu_neg).")

    u_sig = u_sig / norm
    return u_sig, mu_pos, mu_neg, high_mask

def noise_residuals_by_sign(fr, signed, high_mask, mu_pos, mu_neg):
    pos = high_mask & (signed > 0)
    neg = high_mask & (signed < 0)

    fr_noise = np.full_like(fr, np.nan, dtype=float)
    fr_noise[pos] = fr[pos] - mu_pos
    fr_noise[neg] = fr[neg] - mu_neg
    return fr_noise, pos, neg

def noise_corr_from_residuals(fr_noise, high_mask):
    X = fr_noise[high_mask]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = enforce_sym(C)  
    return C

def noise_projection_by_sign(fr_noise, u_sig, pos_mask, neg_mask):
    return {
        +1.0: (fr_noise[pos_mask] @ u_sig),
        -1.0: (fr_noise[neg_mask] @ u_sig),
    }

def alignment_zscore_sign(fr_noise, u_sig, pos_mask, neg_mask, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)

    def compute(X):
        if X.shape[0] == 0:
            return np.nan, np.nan
        obs = np.mean(np.abs(X @ u_sig))
        null = np.zeros(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(X.shape[1])
            null[i] = np.mean(np.abs(X[:, perm] @ u_sig))
        zval = (obs - null.mean()) / (null.std() + 1e-12)
        pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
        return float(zval), float(pval)

    z_pos, p_pos = compute(fr_noise[pos_mask])
    z_neg, p_neg = compute(fr_noise[neg_mask])

    return {+1.0: z_pos, -1.0: z_neg}, {+1.0: p_pos, -1.0: p_neg}



# ----------------------------
# Main run for one eid
# ----------------------------
def run_one_eid(eid, target_prefix="VIS", do_plots=True):
    print(f"Processing eid={eid} with target_prefix='{target_prefix}'...")
    trials, fr, pid = load_trial_firing_rates(eid, pid=None, target_prefix=target_prefix)
    signed, is_zero, nonzero = parse_conditions(trials)

    # high contrast mask
    high_mask = (~is_zero) & (np.abs(signed) > 0.1)
    for c in nonzero:
        high_mask |= ((signed == c) & (~is_zero))

    std_all = fr.std(axis=0)
    std_high = fr[high_mask].std(axis=0) if high_mask.any() else np.zeros(fr.shape[1])

    neuron_mask = (std_all > 1e-10) & (std_high > 1e-10)

    fr = fr[:, neuron_mask]

    u_sig, mu_pos, mu_neg, high_mask2 = compute_signal_axis_mu(
    fr, signed, is_zero, thresh=0.5
)

    # residual noise (matched to the same signal definition)
    fr_noise, pos_mask, neg_mask = noise_residuals_by_sign(fr, signed, high_mask2, mu_pos, mu_neg)

    # noise covariance
    C_noi = noise_corr_from_residuals(fr_noise, high_mask2)
    C_noi_norm = trace_normalize(C_noi)
    u_noi, ev_noi = top_eigenvector(C_noi_norm, return_eigval=True)
    sig_norm = float(np.linalg.norm(mu_pos - mu_neg))
    cos = cosine(u_sig, u_noi)
    ang = angle_deg(u_sig, u_noi)

    noi_vals, noi_cum = eig_spectrum(C_noi, do_trace_norm=True)
    D_eff = participation_ratio(noi_vals)

    if do_plots:
        # ========== 1. "Signal structure": mu_pos vs mu_neg ==========
        plt.figure(figsize=(6, 3))
        plt.plot(mu_pos, label="mu_pos", alpha=0.7)
        plt.plot(mu_neg, label="mu_neg", alpha=0.7)
        plt.title("Mean population response (high contrast)")
        plt.xlabel("neuron index")
        plt.ylabel("firing rate")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        # ========== 2. Noise correlation matrix ==========
        plt.figure(figsize=(5, 4))
        plt.imshow(C_noi, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(label="corr")
        plt.title("Noise correlation (within-sign residuals)")
        plt.tight_layout()
        plt.show()
        # ========== 3. Noise eigen spectrum ==========
        plt.figure(figsize=(6, 4))
        plt.plot(noi_vals[:50], marker="o")
        plt.xlabel("eigenvalue index")
        plt.ylabel("eigenvalue (trace-normalized)")
        plt.title("Noise eigen spectrum")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(noi_cum[:50], marker="o")
        plt.xlabel("k")
        plt.ylabel("cumulative variance explained")
        plt.title("Noise cumulative explained variance")
        plt.tight_layout()
        plt.show()

        # ========== 4. Noise projection onto signal axis (pos vs neg) ==========
        proj = {
            "pos": fr_noise[pos_mask] @ u_sig,
            "neg": fr_noise[neg_mask] @ u_sig,
        }

        plt.figure(figsize=(5, 4))
        plt.boxplot([proj["pos"], proj["neg"]],
                labels=["signed > 0", "signed < 0"],
                showfliers=False)
        plt.axhline(0, color="k", lw=1)
        plt.ylabel("noise projection onto signal axis")
        plt.title("Noise projection by stimulus sign")
        plt.tight_layout()
        plt.show()

        # ========== 5. Alignment z-score (sign-based) ==========
        z, p = alignment_zscore_sign(fr_noise, u_sig, pos_mask, neg_mask, n_perm=300)

        plt.figure(figsize=(4, 4))
        plt.bar(["pos", "neg"], [z[+1.0], z[-1.0]])
        plt.axhline(0, color="k", lw=1)
        plt.ylabel("alignment z-score")
        plt.title("Noise alignment by sign")
        plt.tight_layout()
        plt.show()

    print(f"eid={eid} pid={pid}  T={fr.shape[0]} N={fr.shape[1]}")
    print(f"signal strength ||mu_pos-mu_neg||={sig_norm:.4f}, noise top eig={ev_noi:.4f}")
    print(f"alignment: cosine={cos:.4f}, angle={ang:.2f} deg")
    print(f"noise effective dimension (PR, trace-norm) = {D_eff:.2f}")

    return {
        "eid": eid,
        "pid": pid,
        "T": fr.shape[0],
        "N": fr.shape[1],
        "cosine": cos,
        "angle_deg": ang,
        "noise_PR": D_eff,
        "noise_top1": float(noi_vals[0]),
        "noise_top5_cum": float(noi_cum[min(4, len(noi_cum)-1)]),
        "noise_top10_cum": float(noi_cum[min(9, len(noi_cum)-1)]),
    }

import pandas as pd

rows = []
for eid in eids:
    try:
        rows.append(run_one_eid(eid, target_prefix="VIS", do_plots=False))
    except Exception as e:
        print("skip", eid, "because", repr(e))

df = pd.DataFrame(rows)
print(df)

plt.figure()
plt.hist(df["cosine"], bins=12)
plt.xlabel("cosine(u_sig, u_noise)")
plt.ylabel("count")
plt.title("Signal-noise alignment across sessions")
plt.show()

df_sorted = df.sort_values("cosine", ascending=False).reset_index(drop=True)

x = np.arange(len(df_sorted))
labels = [eid[:8] if isinstance(eid, str) else str(eid) for eid in df_sorted["eid"]]  # short labels

fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.8*len(df_sorted)), 6), sharex=True)

axes[0].bar(x, df_sorted["cosine"].values)
axes[0].set_ylabel("|cos(signal, noise)|")
axes[0].set_title("Across-session: signal-noise alignment and noise low-dimensionality (sorted by alignment)")
axes[0].set_ylim(0, 1.05)
axes[0].axhline(0, color="k", lw=1)

axes[1].plot(x, df_sorted["noise_top1"].values, marker="o", label="top-1 eigenvalue")
axes[1].plot(x, df_sorted["noise_top5_cum"].values, marker="o", label="top-5 cumulative")
axes[1].plot(x, df_sorted["noise_top10_cum"].values, marker="o", label="top-10 cumulative")
axes[1].set_ylabel("variance explained (trace-norm)")
axes[1].set_ylim(0, 1.05)
axes[1].axhline(0, color="k", lw=1)
axes[1].legend(frameon=False)

axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=45, ha="right")
axes[1].set_xlabel("sessions (sorted by alignment)")

plt.tight_layout()
plt.show()