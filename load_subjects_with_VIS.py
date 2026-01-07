from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
import numpy as np
import json

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
# load all subjects 
subjects = one.alyx.rest('subjects', 'list')
subject_names = [s['nickname'] for s in subjects]
for s in subject_names:
    print(s)


#find subjects with VIS regions' recordings
def find_visual_clusters(subject):
    one = ONE(base_url='https://openalyx.internationalbrainlab.org')
    sessions = one.search(tag = 'brainwidemap',subject = subject)
    print(f"Found {len(sessions)} sessions for subject {subject}")
    atlas = ba()
    results = []
    for eid in sessions:
        insertions = one.alyx.rest('insertions', 'list', session=eid)
        for ins in insertions:
            proble_label = ins.get('label', ins.get('name', 'probe'))
            pid = ins['id']
            try:
                ssl = SpikeSortingLoader(pid=pid, one=one,atlas=atlas)
                spikes,clusters,channels = ssl.load_spike_sorting()
                clusters = ssl.merge_clusters(spikes, clusters, channels)
            except:
                print(f"Could not load spike sorting for probe {pid} in session {eid}")
                continue
        # find region acronyms
        acronyms = clusters['acronym']
        is_visual = np.array([a.startswith('VIS') for a in acronyms])
        visual_cluster_ids = np.where(is_visual)[0]
        if(len(visual_cluster_ids) > 0):
            results.append({
                'subject': subject,
                'eid': eid,
                'probe_label': proble_label,
                'acronyms': np.unique(acronyms[visual_cluster_ids]),
            })
    return results
def print_eids_only(results):
    """
    Print only unique eids in a simple list
    """
    unique_eids = list(set(r['eid'] for r in results))
    
    print(f"\n{len(unique_eids)} sessions with VIS recordings:")
    print('\n')
    for eid in unique_eids:
        print(eid)
    return unique_eids
def make_serializable(obj):
    """Convert any object to a JSON-serializable format"""
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj,UUID):
        return str(obj)
    try:
        return list(obj)
    except (TypeError, ValueError):
        return obj
def save_results(results, filename='results.json'):
    """Save results handling all data types"""
    serializable_results = []
    
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            try:
                serializable_result[key] = make_serializable(value)
            except Exception as e:
                print(f"Warning: Could not serialize {key}: {e}")
                serializable_result[key] = str(value)  # Fallback to string
        
        serializable_results.append(serializable_result)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved {len(serializable_results)} results to {filename}")
results = find_visual_clusters('NYU-40')
unique_eids = print_eids_only(results)
save_results(results, 'results.json')