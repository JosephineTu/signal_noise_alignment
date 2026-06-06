from one.api import ONE
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
import json
import matplotlib.pyplot as plt
from brainbox.population.decode import get_spike_counts_in_bins

def load_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {filename}")
    return results

def load_trials(eid, keyword, one=None):

    if one is None:

        from one.api import ONE

        one = ONE(

            base_url="https://openalyx.internationalbrainlab.org",

            password="international",

            cache_dir="/scratch/midway3/xiaorantu/ONE",

        )

    passive_Gabor = one.load_dataset(eid, keyword, collection="alf")

    start = passive_Gabor["start"]

    stop = passive_Gabor["stop"]

    trials_list = []

    for i in range(len(start)):

        trial_i = {

            "start": start[i],

            "stop": stop[i],

            "contrast": passive_Gabor["contrast"][i],

            "position": passive_Gabor["position"][i],

            "phase": passive_Gabor["phase"][i],

        }

        trials_list.append(trial_i)

    return trials_list
    
def load_spikes(eid, one=None):
    if one is None:
        from one.api import ONE
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            cache_dir="/scratch/midway3/xiaorantu/ONE",
        )

    pid = one.alyx.rest("insertions", "list", session=eid)[0]["id"]
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    return spikes, clusters

def load_trial_data_region(eid, keyword, one=None, target_region=None):
    if one is None:
        from one.api import ONE
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            cache_dir="/scratch/midway3/xiaorantu/ONE",
        )

    trials_dataset = one.load_dataset(eid, keyword, collection="alf")
    start = np.array(trials_dataset["start"])
    end = np.array(trials_dataset["stop"])
    trials = np.c_[start, end]

    pid = one.alyx.rest("insertions", "list", session=eid)[0]["id"]
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if target_region is not None:
        region_mask = clusters["acronym"] == target_region
        region_cluster_ids = clusters["cluster_id"][region_mask]
        print(f"Selected {region_mask.sum()} clusters in region {target_region}")
    else:
        region_cluster_ids = clusters["cluster_id"]
        print(f"Selected all {len(region_cluster_ids)} clusters")

    counts, cluster_ids = get_spike_counts_in_bins(
        spikes["times"], spikes["clusters"], trials
    )

    print("counts.shape:", counts.shape)
    print("cluster_ids.shape:", cluster_ids.shape)

    counts = counts.T

    region_cluster_ids = np.intersect1d(region_cluster_ids, cluster_ids)

    masked_clusters = np.isin(cluster_ids, region_cluster_ids)
    counts_region = counts[:, masked_clusters]
    cluster_ids_by_region = cluster_ids[masked_clusters]

    trial_durations = end - start
    fr_region = counts_region / trial_durations[:, None]
    fr_region = fr_region.T

    print(f"Firing rate matrix shape: {fr_region.shape}")
    return fr_region, cluster_ids_by_region, trials
def get_raster_data(eid, t_start=None, t_end=None, one=None):
    if one is None:
        from one.api import ONE
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            cache_dir="/scratch/midway3/xiaorantu/ONE",
        )

    pid = one.alyx.rest("insertions", "list", session=eid)[0]["id"]
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)

    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if t_start is not None and t_end is not None:
        time_mask = (spikes["times"] >= t_start) & (spikes["times"] <= t_end)
        spikes_zoom = {k: v[time_mask] for k, v in spikes.items()}
        sl.raster(spikes_zoom, channels)
    else:
        sl.raster(spikes, channels)

    import matplotlib.pyplot as plt
    plt.show()


def compute_firing_rate(spikes, clusters, region_acronym, time_window = None):
    region_mask = clusters['acronym'] == region_acronym
    region_cluster_ids = np.where(region_mask)[0]
    if len(region_cluster_ids) == 0:
        print(f"No clusters found in region {region_acronym}")
        return None
    if time_window is None:
        t_start = np.min(spikes['times'])
        t_end = np.max(spikes['times'])
    else:
        t_start, t_end = time_window
    duration = t_end - t_start
    rates = {}
    for cluster_id in region_cluster_ids:
        mask = spikes['clusters'] == cluster_id
        cluster_spike_times = spikes['times'][mask]

        if time_window is not None:
            cluster_spike_times = cluster_spike_times[(cluster_spike_times >= t_start) & (cluster_spike_times <= t_end)]
        rate = len(cluster_spike_times) / duration
        rates[cluster_id] = rate
    mean_rate = np.mean(list(rates.values()))
    return mean_rate, rates

def compute_signal_axis(trials, fr_region):
    X = fr_region.T
    stim_contrast = trials['contrast']
    stim_position = trials['position']
    stim_phase = trials['phase']
    stim_matrix = np.c_[stim_contrast, stim_position, stim_phase]
    
def main():
    results = load_results('results.json')
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(password='international')
    eids=[]
    for r in results:
        eids.append (r['eid'])
    eid = eids[0]
    spikes,clusters = load_spikes(eid)

    get_raster_data(eid,100,500)

    mean_rate, rates = compute_firing_rate(spikes, clusters, 'VISC5', time_window=(0, 1000))

    fr_region, cluster_ids_by_region, trials = load_trial_data_region(eid, "*passiveGabor*",target_region='VISC5', one=one)

if(__name__=="__main__"):
    main()



