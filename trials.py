from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
import numpy as np
from passiveGabor_VIS import *
from brainbox.population.decode import get_spike_counts_in_bins
from pca_tool import *
import matplotlib.pyplot as plt
results = load_results('VIS_subjects_by_lab.json')
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
eids = []
for lab_name in results.keys():
    for subject in results[lab_name].keys():
        subject_eid = results[lab_name][subject]['VIS_eids']
        eids.extend(subject_eid)
# ----------------------------
# Setup
# ----------------------------
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
    vals, vecs = np.linalg.eigh(C)  # ascending
    idx = np.argmax(vals)
    v = vecs[:, idx]
    # deterministic sign
    j = np.argmax(np.abs(v))
    if v[j] < 0:
        v = -v
    v = v / (np.linalg.norm(v) + 1e-12)
    return (v, vals[idx]) if return_eigval else v

def cosine(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.dot(u, v))

def angle_deg(u, v):
    c = np.clip(cosine(u, v), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def eig_spectrum(C, do_trace_norm=True):
    C = enforce_sym(C)
    if do_trace_norm:
        C = trace_normalize(C)
    vals = np.linalg.eigvalsh(C)[::-1]  # descending
    vals = np.clip(vals, 0, None)
    cum = np.cumsum(vals) / (np.sum(vals) + 1e-12)
    return vals, cum

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
    return best_pid

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
        pid = pick_best_insertion(eid, target_prefix=target_prefix)
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

# ----------------------------
# Condition parsing + signal/noise
# ----------------------------
def parse_conditions(trials):
    cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
    cr = np.nan_to_num(trials['contrastRight'], nan=0.0)
    signed = cl - cr
    is_zero = (cl == 0) & (cr == 0)
    nonzero = np.unique(signed[~is_zero])
    return signed, is_zero, nonzero

def compute_signal_matrix(fr, signed, is_zero, nonzero):
    # rows = conditions (nonzero..., zero), cols = neurons
    n_cond = len(nonzero) + 1
    n_units = fr.shape[1]
    fr_signal = np.zeros((n_cond, n_units))

    for i, c in enumerate(nonzero):
        sel = (signed == c) & (~is_zero)
        fr_signal[i] = fr[sel].mean(axis=0)

    fr_signal[-1] = fr[is_zero].mean(axis=0)
    return fr_signal

def signal_corr(fr_signal):
    return np.corrcoef(fr_signal.T)

def noise_corr(fr, fr_signal, signed, is_zero, nonzero):
    fr_noise = fr.copy()
    for i, c in enumerate(nonzero):
        sel = (signed == c) & (~is_zero)
        fr_noise[sel] -= fr_signal[i]
    fr_noise[is_zero] -= fr_signal[-1]
    return np.corrcoef(fr_noise, rowvar=False)

# ----------------------------
# Condition-specific noise projection & z-score test
# ----------------------------
def noise_projection_by_condition(fr, signed, is_zero, nonzero, axis_vec):
    out = {}
    for c in nonzero:
        sel = (signed == c) & (~is_zero)
        fr_cond = fr[sel]
        fr_noise = fr_cond - fr_cond.mean(axis=0, keepdims=True)
        out[float(c)] = fr_noise @ axis_vec
    fr0 = fr[is_zero]
    fr_noise0 = fr0 - fr0.mean(axis=0, keepdims=True)
    out[0.0] = fr_noise0 @ axis_vec
    return out

def alignment_zscore(fr, signed, is_zero, nonzero, axis_vec, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    z, p = {}, {}

    def compute(fr_noise):
        obs = np.mean(np.abs(fr_noise @ axis_vec))
        null = np.zeros(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(fr_noise.shape[1])
            null[i] = np.mean(np.abs(fr_noise[:, perm] @ axis_vec))
        zval = (obs - null.mean()) / (null.std() + 1e-12)
        pval = (np.sum(null >= obs) + 1) / (n_perm + 1)
        return float(zval), float(pval)

    for c in nonzero:
        sel = (signed == c) & (~is_zero)
        fr_cond = fr[sel]
        fr_noise = fr_cond - fr_cond.mean(axis=0, keepdims=True)
        z[float(c)], p[float(c)] = compute(fr_noise)

    fr0 = fr[is_zero]
    fr_noise0 = fr0 - fr0.mean(axis=0, keepdims=True)
    z[0.0], p[0.0] = compute(fr_noise0)

    return z, p

# ----------------------------
# Main run for one eid
# ----------------------------
def run_one_eid(eid, target_prefix="VIS", do_plots=True):
    trials, fr, pid = load_trial_firing_rates(eid, pid=None, target_prefix=target_prefix)
    signed, is_zero, nonzero = parse_conditions(trials)

    fr_sig = compute_signal_matrix(fr, signed, is_zero, nonzero)
    C_sig = signal_corr(fr_sig)

    C_noi = noise_corr(fr, fr_sig, signed, is_zero, nonzero)
    C_noi_norm = trace_normalize(C_noi)

    u_sig, ev_sig = top_eigenvector(C_sig, return_eigval=True)
    u_noi, ev_noi = top_eigenvector(C_noi_norm, return_eigval=True)

    cos = cosine(u_sig, u_noi)
    ang = angle_deg(u_sig, u_noi)

    noi_vals, noi_cum = eig_spectrum(C_noi, do_trace_norm=True)
    D_eff = participation_ratio(noi_vals)

    if do_plots:
        # signal/noise matrices
        plt.figure(figsize=(5,4))
        plt.imshow(C_sig, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(label='corr')
        plt.title("Signal correlation")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5,4))
        plt.imshow(C_noi, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(label='corr')
        plt.title("Noise correlation")
        plt.tight_layout()
        plt.show()

        # eigen spectrum
        plt.figure(figsize=(6,4))
        plt.plot(noi_vals[:50], marker='o')
        plt.xlabel("eigenvalue index")
        plt.ylabel("eigenvalue (trace-normalized)")
        plt.title("Noise eigen spectrum")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,4))
        plt.plot(noi_cum[:50], marker='o')
        plt.xlabel("k")
        plt.ylabel("cumulative variance explained")
        plt.title("Noise cumulative explained variance")
        plt.tight_layout()
        plt.show()

        # condition-specific projections
        proj = noise_projection_by_condition(fr, signed, is_zero, nonzero, u_sig)
        plt.figure(figsize=(8,4))
        for c, y in proj.items():
            x = np.ones(len(y)) * c
            plt.scatter(x, y, s=12, alpha=0.35)
        plt.axhline(0, color='k', lw=1)
        plt.xlabel("signed contrast condition (0 = blank)")
        plt.ylabel("noise projection onto signal axis")
        plt.title("Condition-specific noise alignment (projection)")
        plt.tight_layout()
        plt.show()

        z, p = alignment_zscore(fr, signed, is_zero, nonzero, u_sig, n_perm=300)
        xs = list(z.keys())
        ys = [z[k] for k in xs]
        plt.figure(figsize=(7,3.8))
        plt.plot(xs, ys, '-o')
        plt.axhline(0, color='k', lw=1)
        plt.xlabel("signed contrast condition")
        plt.ylabel("z-score")
        plt.title("Noise alignment z-score by condition")
        plt.tight_layout()
        plt.show()

    print(f"eid={eid} pid={pid}  T={fr.shape[0]} N={fr.shape[1]}")
    print(f"top eigvals: signal={ev_sig:.4f}, noise={ev_noi:.4f}")
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
for eid in eids[:5]:
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

plt.figure()
plt.scatter(df["N"], df["noise_top10_cum"])
plt.xlabel("N units (VIS)")
plt.ylabel("top-10 cumulative variance (noise)")
plt.title("Noise dimension summary")
plt.show()