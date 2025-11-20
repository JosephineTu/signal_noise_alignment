from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
import numpy as np
from passiveGabor_VIS import *
from brainbox.population.decode import get_spike_counts_in_bins
import matplotlib.pyplot as plt

results = load_results('results.json')
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
eids = []
for r in results:
    eids.append(r['eid'])
eid = eids[0]
def load_stim_firing_rates(eid, one=one, target_region = None):
    trials = one.load_object(eid, 'trials', collection='alf')
    stim_on = trials['stimOn_times']
    stim_off = trials['stimOff_times']

    valid = ~np.isnan(stim_on) & ~np.isnan(stim_off)
    stim_on = stim_on[valid]
    stim_off= stim_off[valid]
    for k in trials.keys():
        trials[k] = trials[k][valid]

    print(f"filtered out {np.sum(~valid)} trials with NaN stim times")

    stim_intervals = np.c_[stim_on, stim_off]
    pid = one.alyx.rest('insertions', 'list', session=eid)[0]['id']
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba())
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], stim_intervals)
    counts = counts.T
    if target_region is not None:
        region_mask = clusters['acronym'] == target_region
        region_cluster_ids = clusters['cluster_id'][region_mask]
        region_cluster_ids = np.intersect1d(cluster_ids, region_cluster_ids)
    else: 
        region_cluster_ids = cluster_ids
    masked_clusters = np.isin(cluster_ids, region_cluster_ids)
    counts = counts[:, masked_clusters]
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
    n_conditions = len(conditions)
    n_clusters = fr_stim.shape[1]

    fr_signal = np.zeros((n_conditions, n_clusters))

    for i, cond in enumerate(conditions):
        sel = (signed_contrast == cond)
        fr_signal[i, :] = np.mean(fr_stim[sel, :], axis=0)

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
        sel = signed_contrast ==cond
        fr_noise[sel] -= fr_signal[i]
    noise_corr = np.corrcoef(fr_noise.T, rowvar=True)
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
    for c in conditions:
        sel = signed_contrast == c
        fr_cond = fr_stim [sel]

        fr_noise_cond = fr_cond -fr_cond.mean(axis=0, keepdims=True)

        stds = np.std(fr_noise_cond, axis=0)
        valid_clusters = stds > 0
        fr_noise_cond = fr_noise_cond[:, valid_clusters]
        noise_corr_cond = np.corrcoef(fr_noise_cond, rowvar=False)
        noise_corr_conds.append(noise_corr_cond)
    return noise_corr_conds


trials, fr_stim, region_cluster_ids = load_stim_firing_rates(eid, target_region="VISC5")
print(fr_stim.shape) # (num_trials, num_clusters)
signal_correlation, fr_signal = get_signal_correlation(trials, fr_stim)

noise_correlation = get_noise_correlation(trials, fr_stim, fr_signal)

noise_correlation_conds = condition_specific_noise_correlation(trials, fr_stim)
print(noise_correlation_conds[0].shape) # (num_clusters, num_clusters)
