from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
import numpy as np
from passiveGabor_VIS import *
from brainbox.population.decode import get_spike_counts_in_bins
from pca_tool import *
import matplotlib.pyplot as plt
results = load_results('results.json')
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
eids = []
for lab_name in results.keys():
    for subject in results[lab_name].keys():
        subject_eid = results[lab_name][subject]['eids']
        eids.extend(subject_eid)
# load firing rates of target region 
def load_stim_firing_rates(eid, one=one, target_region = None):
    trials = one.load_object(eid, 'trials', collection='alf')
    stim_on = trials['stimOn_times']
    stim_off = trials['stimOff_times']
    # filter out trials with NaN stim times
    valid = ~np.isnan(stim_on) & ~np.isnan(stim_off)
    stim_on = stim_on[valid]
    stim_off= stim_off[valid]
    for k in trials.keys():
        trials[k] = trials[k][valid]

    print(f"filtered out {np.sum(~valid)} trials with NaN stim times")
    print(stim_on.shape)
    stim_intervals = np.c_[stim_on, stim_off]
    pid = one.alyx.rest('insertions', 'list', session=eid)[0]['id']
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba())
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    print(np.unique(channels['acronym']))
    print(np.unique(clusters['acronym']))

    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], stim_intervals)
    counts = counts.T
    # filter clusters by target region prefix
    if target_region is not None:
        region_mask = np.array([a.startswith(target_region) for a in clusters['acronym']])
        region_cluster_ids = clusters['cluster_id'][region_mask]
        region_cluster_ids = np.intersect1d(cluster_ids, region_cluster_ids)
    else: 
        region_cluster_ids = cluster_ids
    masked_clusters = np.isin(cluster_ids, region_cluster_ids)
    # select only firing counts from target region clusters
    counts = counts[:, masked_clusters]
    # compute fr
    fr_stim = counts / np.diff(stim_intervals, axis=1)
    return trials, fr_stim, masked_clusters

def get_signal_correlation(trials, fr_stim):
    cl = trials['contrastLeft']
    cr = trials['contrastRight']
    cl = np.nan_to_num(cl, nan=0.0)
    cr = np.nan_to_num(cr, nan=0.0)

    signed_contrast = cl - cr
    conditions = np.unique(signed_contrast)
    print(conditions)
    for c in conditions:
        print(c, np.sum(signed_contrast == c))
    n_conditions = len(conditions)
    n_clusters = fr_stim.shape[1]

    fr_signal = np.zeros((n_conditions, n_clusters))

    for i, cond in enumerate(conditions):
        sel = (signed_contrast == cond)
        fr_signal[i, :] = np.mean(fr_stim[sel, :], axis=0)
        print(f"signal shape:{fr_signal.shape}")
    # compute signal correlation, shape (n_clusters, n_clusters)
    signal_corr = np.corrcoef(fr_signal.T)
    return signal_corr, fr_signal

def get_noise_correlation(trials, fr_stim, fr_signal):
    fr_noise = fr_stim.copy()
    cl = trials['contrastLeft']
    cr = trials['contrastRight']
    cl = np.nan_to_num(cl, nan=0.0)
    cr = np.nan_to_num(cr, nan=0.0)

    signed_contrast = cl - cr
    conditions = np.unique(signed_contrast)

    for i, cond in enumerate(conditions):
        sel = signed_contrast == cond
        fr_noise[sel] -= fr_signal[i]
    noise_corr = np.corrcoef(fr_noise, rowvar=False)
    print(noise_corr.shape)
    return noise_corr

def condition_specific_noise_correlation(trials, fr_stim):
    cl = trials['contrastLeft']
    cr = trials['contrastRight']
    cl = np.nan_to_num(cl, nan=0.0)
    cr = np.nan_to_num(cr, nan=0.0)

    signed_contrast = cl - cr
    conditions = np.unique(signed_contrast)
    noise_corr_conds = []

    fr_signal_cond = np.zeros((len(conditions), fr_stim.shape[1]))
    for i, c in enumerate (conditions):
        sel = (signed_contrast == c)
        fr_cond = fr_stim[sel]
        fr_signal_cond[i] = fr_cond.mean(axis=0)
        fr_noise_cond = fr_cond - fr_cond.mean(axis=0, keepdims=True)

        stds = np.std(fr_noise_cond, axis=0)
        valid_clusters = stds > 0
        fr_noise_cond = fr_noise_cond[:, valid_clusters]
        noise_corr_cond = np.corrcoef(fr_noise_cond, rowvar=False)
        noise_corr_conds.append(noise_corr_cond)
    return noise_corr_conds, fr_signal_cond

def compare_signal_noise_eig(signal_corr, noise_corr,
                             n_permutations=500,
                             shuffle_method='trial',
                             fr_stim=None, trials=None,
                             verbose=True):

    assert signal_corr.shape == noise_corr.shape
    N = signal_corr.shape[0] # number of clusters
    # compare relationship between signal and noise eigenvectors across clusters
    u_sig, eigv_sig = top_eigenvector(signal_corr, return_eigval=True)
    u_noise, eigv_noise = top_eigenvector(noise_corr, return_eigval=True)

    cos = float(cosine_similarity(u_sig, u_noise))
    ang = float(angle_between(u_sig, u_noise))

    if verbose:
        print(f"Signal eigval: {eigv_sig:.4f}, Noise eigval: {eigv_noise:.4f}")
        print(f"Cosine: {cos:.4f}, angle (deg): {ang*180/np.pi:.2f}")

    if fr_stim.shape[0] != trials['contrastLeft'].size: # number of trials
        raise ValueError("fr_stim must be (T x N_neurons). Trials mismatch.")

    T, Ncheck = fr_stim.shape
    assert Ncheck == N, "fr_stim neuron dimension must match correlation matrices"

    null_cosines = np.zeros(n_permutations)

    for p in range(n_permutations):

        if shuffle_method == 'neuron_identities':

            perm = np.random.permutation(N)   # neurons (clusters)
            fr_shuffled = fr_stim[:, perm]    # (T, N)

            fr_noise = fr_shuffled - fr_shuffled.mean(axis=0, keepdims=True)

            C_noise_shuffled = np.corrcoef(fr_noise, rowvar=False)

            u_noise_perm = top_eigenvector(C_noise_shuffled)
            null_cosines[p] = cosine_similarity(u_sig, u_noise_perm)

        elif shuffle_method == 'trial':

            perm = np.random.permutation(T)
            fr_shuffled = fr_stim[perm]

            fr_noise = fr_shuffled - fr_shuffled.mean(axis=0, keepdims=True)
            C_noise_shuffled = np.corrcoef(fr_noise, rowvar=False)

            u_noise_perm = top_eigenvector(C_noise_shuffled)
            null_cosines[p] = cosine_similarity(u_sig, u_noise_perm)

        else:
            raise ValueError(f"Unknown shuffle method: {shuffle_method}")

    p_value = (np.sum(null_cosines >= cos) + 1) / (n_permutations + 1)

    if verbose:
        print(f"P-value: {p_value:.4f}")
    return {
        'usig': u_sig,
        'unoise': u_noise,
        'cosine_similarity': cos,
        'angle_deg': ang * 180 / np.pi,
        'p_value': p_value,
        'null_cosines': null_cosines,
    }

def get_projection_of_conditions(u_sig, fr_signal, conditions):
    proj = []
    unique_conditions = np.unique(conditions)
    for c in unique_conditions:
        mask = (conditions == c)
        proj.append(fr_signal[mask].mean(axis=0) @ u_sig)
    return np.array(proj)
def condition_specific_noise_projection(fr_stim, trials, u_sig):
    cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
    cr = np.nan_to_num(trials['contrastRight'], nan=0.0)
    signed_contrast = cl - cr
    conditions = np.unique(signed_contrast)

    proj_by_cond = {}

    for c in conditions:
        mask = (signed_contrast == c)
        fr_cond = fr_stim[mask]
        fr_noise = fr_cond - fr_cond.mean(axis=0, keepdims=True)
        proj = fr_noise @ u_sig
        proj_by_cond[c] = proj
    return proj_by_cond

def condition_noise_alignment_test(fr_stim, trials, u_sig, n_perm=500):
    cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
    cr = np.nan_to_num (trials['contrastRight'], nan=0.0)
    signed_contrast = cl - cr
    conditions = np.unique(signed_contrast)
    z_alignment = {}
    pvals = {}

    for c in conditions:
        mask = (signed_contrast == c)
        fr_cond = fr_stim[mask]
        fr_noise = fr_cond - fr_cond.mean(axis=0, keepdims=True)

        obs = np.mean(np.abs(fr_noise @ u_sig))

        null = np.zeros(n_perm)
        for i in range(n_perm):
            perm = np.random.permutation(fr_noise.shape[1])
            null[i] = np.mean(np.abs(fr_noise[:,perm] @ u_sig))
        z_alignment[c] = (obs - np.mean(null)) / np.std(null)
        pvals[c] = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return z_alignment, pvals

trials, fr_stim, region_cluster_ids = load_stim_firing_rates(eid, target_region='VISpor')
print(fr_stim.shape) # (num_trials, num_clusters)
signal_correlation, fr_signal = get_signal_correlation(trials, fr_stim)

noise_correlation = get_noise_correlation(trials, fr_stim, fr_signal)

noise_correlation_conds, fr_signal_cond = condition_specific_noise_correlation(trials, fr_stim)

res = compare_signal_noise_eig(signal_correlation, noise_correlation, n_permutations=1000, shuffle_method='neuron_identities', fr_stim=fr_stim, trials=trials, verbose=True)
u_sig = res['usig']

cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
cr = np.nan_to_num(trials['contrastRight'], nan=0.0)
signed_contrast = cl - cr
conditions = np.unique(signed_contrast)
trial_means = np.zeros_like(fr_stim)
for i, c in enumerate(conditions):
    mask = (signed_contrast == c)
    trial_means[signed_contrast == c] = fr_signal_cond[i]
proj_noise = (fr_stim - trial_means) @ u_sig


unique_conds = np.unique(signed_contrast)

proj_by_cond = condition_specific_noise_projection(fr_stim, trials, u_sig)

plt.figure(figsize=(8,4))

for c, proj in proj_by_cond.items():
    x = np.ones(len(proj)) * c
    plt.scatter(x, proj, alpha=0.4, s=12)

plt.axhline(0, color='k', lw=1)
plt.xlabel("Contrast condition")
plt.ylabel("Noise projection onto signal axis")
plt.title("Condition-specific noise alignment with signal axis")
plt.show()

z_align, pvals = condition_noise_alignment_test(
    fr_stim, trials, u_sig, n_perm=500
)

plt.figure()
plt.plot(list(z_align.keys()), list(z_align.values()), '-o')
plt.axhline(0, color='k', lw=1)
plt.xlabel("Contrast condition")
plt.ylabel("Noise alignment z-score")
plt.title("Condition-specific noise alignment (normalized)")
plt.show()