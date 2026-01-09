from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
import numpy as np
import json
from uuid import UUID

MAX_SUBJECTS = 5                 
MAX_SESSIONS_PER_SUBJECT = 2    
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
atlas = AllenAtlas()

def get_VIS_acronyms(pid):
    try:
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)
    except Exception:
        return []
    
    acronyms = clusters.get('acronym', [])
    vis_acronyms = [a for a in acronyms if a.startswith('VIS')]
    return list(np.unique(vis_acronyms))


def find_VIS_sessions_for_subject(subject):
    """
    Return VIS-containing session eids for one subject
    (with session-level breakpoint)
    """
    eids = one.search(subject=subject, tag='brainwidemap')
    VIS_eids = []
    VIS_acronyms = []

    for eid in eids[:MAX_SESSIONS_PER_SUBJECT]:
        insertions = one.alyx.rest('insertions', 'list', session=eid)
        if not insertions:
            continue

        for ins in insertions:
            vis_acronym = get_VIS_acronyms(ins['id'])
            if len(vis_acronym) > 0:
                VIS_eids.append(eid)
                VIS_acronyms.extend(vis_acronym)
                break 

    return list(set(VIS_eids)), list(np.unique(VIS_acronyms))


labs = one.alyx.rest('labs', 'list')
results = {}

found_subjects = 0

for lab in labs:
    if found_subjects >= MAX_SUBJECTS:
        break

    lab_name = lab['name']
    subjects = one.alyx.rest('subjects', 'list', lab=lab_name)
    lab_results = {}

    print(f"\n[LAB] {lab_name} | {len(subjects)} subjects")

    for subj in subjects:
        if found_subjects >= MAX_SUBJECTS:
            break

        subject = subj['nickname']
        print(f"  Checking subject: {subject}")

        VIS_eids, VIS_acronyms = find_VIS_sessions_for_subject(subject)

        if len(VIS_eids) > 0:
            found_subjects += 1
            lab_results[subject] = {
                'n_VIS_sessions': len(VIS_eids),
                'VIS_eids': VIS_eids,
                'VIS_acronyms': VIS_acronyms
            }
            print(f"    ✓ FOUND VIS subject ({found_subjects}/{MAX_SUBJECTS})")

    if lab_results:
        results[lab_name] = lab_results


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


with open('VIS_subjects_by_lab.json', 'w') as f:
    json.dump(make_serializable(results), f, indent=2)

print(f"\nSaved results for {found_subjects} subjects to VIS_subjects_by_lab.json")
