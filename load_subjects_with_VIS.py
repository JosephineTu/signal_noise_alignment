import os
os.environ["HOME"] = "/scratch/midway3/xiaorantu"

from concurrent.futures import ThreadPoolExecutor, as_completed
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
import numpy as np
import json
from uuid import UUID
from pathlib import Path

MAX_SUBJECTS = 25
MAX_SESSIONS_PER_SUBJECT = 2

CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"
BASE_URL = "https://openalyx.internationalbrainlab.org"

OUT_JSON = "VISp_subjects_by_lab.json"
PROGRESS_JSON = "VISp_scan_progress.json"

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def make_one_and_atlas():
    ONE.setup(
        base_url=BASE_URL,
        cache_dir=CACHE_DIR,
        silent=True,
    )
    one = ONE(
        base_url=BASE_URL,
        password="international",
        cache_dir=CACHE_DIR,
    )
    atlas = AllenAtlas()
    return one, atlas


def get_VIS_acronyms(pid, one, atlas):
    try:
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)
    except Exception as e:
        print(f"    [INSERTION FAILED] pid={pid}: {repr(e)}", flush=True)
        return []

    acronyms = clusters.get("acronym", [])
    vis_acronyms = [str(a) for a in acronyms if str(a).startswith("VISp")]
    return list(np.unique(vis_acronyms))


def scan_single_eid_for_VIS(eid, existing_eids, one, atlas):
    eid_str = str(eid)

    if eid_str in existing_eids:
        print(f"    [SKIP EID cached] {eid_str}", flush=True)
        return eid_str, []

    try:
        insertions = one.alyx.rest("insertions", "list", session=eid)
    except Exception as e:
        print(f"    [INSERTIONS FAILED] eid={eid_str}: {repr(e)}", flush=True)
        return eid_str, []

    if not insertions:
        return eid_str, []

    print(f"    Checking eid={eid_str}, n_insertions={len(insertions)}", flush=True)

    for ins in insertions:
        pid = ins["id"]
        vis_acr = get_VIS_acronyms(pid, one, atlas)

        if len(vis_acr) > 0:
            print(f"    FOUND VISp eid={eid_str}, acronyms={vis_acr}", flush=True)
            return eid_str, vis_acr

    return eid_str, []


def find_VIS_sessions_for_subject(lab_name, subject, existing_eids, one, atlas):
    """
    Thread-parallel subject scan.
    Stops as soon as MAX_SESSIONS_PER_SUBJECT matching sessions are found.
    Existing eids are skipped.
    """
    out = {
        "lab": lab_name,
        "subject": subject,
        "n_VIS_sessions": 0,
        "VIS_eids": [],
        "VIS_acronyms": [],
        "status": "none",
        "error": None,
    }

    try:
        eids = one.search(subject=subject, tag="brainwidemap")
    except Exception as e:
        out["status"] = "search_failed"
        out["error"] = repr(e)
        return out

    VIS_eids = []
    VIS_acronyms = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                scan_single_eid_for_VIS,
                eid,
                existing_eids,
                one,
                atlas,
            ): str(eid)
            for eid in eids
        }

        for future in as_completed(futures):
            try:
                eid_str, vis_acr = future.result()
            except Exception as e:
                print(f"    [EID FAILED] {futures[future]}: {repr(e)}", flush=True)
                continue

            if len(vis_acr) > 0:
                VIS_eids.append(eid_str)
                VIS_acronyms.extend(vis_acr)

                if len(VIS_eids) >= MAX_SESSIONS_PER_SUBJECT:
                    for pending_future in futures:
                        if not pending_future.done():
                            pending_future.cancel()
                    break

    VIS_eids = sorted(list(set(VIS_eids)))
    VIS_acronyms = sorted(list(np.unique(VIS_acronyms)))

    if len(VIS_eids) > 0:
        out["status"] = "found"
        out["n_VIS_sessions"] = len(VIS_eids)
        out["VIS_eids"] = VIS_eids
        out["VIS_acronyms"] = VIS_acronyms

    return out


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


def load_existing_results(json_path):
    existing_results = {}
    existing_eids = set()

    if json_path.exists():
        with open(json_path, "r") as f:
            existing_results = json.load(f)

        for lab_data in existing_results.values():
            for subj_data in lab_data.values():
                existing_eids.update(str(eid) for eid in subj_data.get("VIS_eids", []))

    return existing_results, existing_eids


def count_existing_subjects(existing_results):
    return sum(len(lab_data) for lab_data in existing_results.values())


def load_progress(progress_path):
    if not progress_path.exists():
        return {
            "last_lab_idx": -1,
            "last_subj_idx": -1,
            "last_lab": None,
            "last_subject": None,
        }

    with open(progress_path, "r") as f:
        return json.load(f)


def save_progress(progress_path, lab_idx, subj_idx, lab_name, subject):
    progress = {
        "last_lab_idx": int(lab_idx),
        "last_subj_idx": int(subj_idx),
        "last_lab": lab_name,
        "last_subject": subject,
    }

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def save_results(json_path, results):
    with open(json_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)


def main():
    json_path = Path(OUT_JSON)
    progress_path = Path(PROGRESS_JSON)

    existing_results, existing_eids = load_existing_results(json_path)
    existing_subjects = count_existing_subjects(existing_results)

    progress = load_progress(progress_path)
    last_lab_idx = int(progress.get("last_lab_idx", -1))
    last_subj_idx = int(progress.get("last_subj_idx", -1))

    print(f"[CACHE] Found {len(existing_eids)} existing eids", flush=True)
    print(f"[CACHE] Found {existing_subjects} existing subjects", flush=True)
    print(
        f"[PROGRESS] Resume after lab_idx={last_lab_idx}, "
        f"subj_idx={last_subj_idx}, "
        f"lab={progress.get('last_lab')}, "
        f"subject={progress.get('last_subject')}",
        flush=True,
    )

    if existing_subjects >= MAX_SUBJECTS:
        print(
            f"[DONE] Already have {existing_subjects} subjects "
            f">= MAX_SUBJECTS={MAX_SUBJECTS}",
            flush=True,
        )
        return

    one, atlas = make_one_and_atlas()

    try:
        labs = one.alyx.rest("labs", "list")
    except Exception as e:
        raise RuntimeError(f"Failed to list labs: {repr(e)}")

    results = existing_results.copy()
    found_subjects = existing_subjects

    for lab_idx, lab in enumerate(labs):
        lab_name = lab["name"]

        if lab_idx < last_lab_idx:
            continue

        try:
            subjects = one.alyx.rest("subjects", "list", lab=lab_name)
        except Exception as e:
            print(f"[SKIP LAB] {lab_name}: {repr(e)}", flush=True)
            continue

        print(f"\n[LAB {lab_idx}] {lab_name} | {len(subjects)} subjects", flush=True)

        for subj_idx, subj in enumerate(subjects):
            subject = subj["nickname"]

            if lab_idx == last_lab_idx and subj_idx <= last_subj_idx:
                continue

            if found_subjects >= MAX_SUBJECTS:
                break

            print(f"\n[SUBJECT] lab={lab_name}, subject={subject}, idx={subj_idx}", flush=True)

            if lab_name in results and subject in results[lab_name]:
                print("  [SKIP SUBJECT cached]", flush=True)
                save_progress(progress_path, lab_idx, subj_idx, lab_name, subject)
                continue

            try:
                res = find_VIS_sessions_for_subject(
                    lab_name=lab_name,
                    subject=subject,
                    existing_eids=existing_eids,
                    one=one,
                    atlas=atlas,
                )
            except Exception as e:
                print(f"  [SUBJECT FAILED] {subject}: {repr(e)}", flush=True)
                save_progress(progress_path, lab_idx, subj_idx, lab_name, subject)
                continue

            if res["status"] == "found":
                if lab_name not in results:
                    results[lab_name] = {}

                results[lab_name][subject] = {
                    "n_VIS_sessions": res["n_VIS_sessions"],
                    "VIS_eids": res["VIS_eids"],
                    "VIS_acronyms": res["VIS_acronyms"],
                }

                existing_eids.update(res["VIS_eids"])
                found_subjects += 1

                print(
                    f"FOUND VISp subject {found_subjects}/{MAX_SUBJECTS}: "
                    f"{subject} | lab={lab_name} | sessions={res['n_VIS_sessions']}",
                    flush=True,
                )

                save_results(json_path, results)

            elif res["status"] == "search_failed":
                print(f"[SEARCH FAILED] {subject}: {res['error']}", flush=True)

            else:
                print(f"  [NO VISp found] {subject}", flush=True)

            save_progress(progress_path, lab_idx, subj_idx, lab_name, subject)

        if found_subjects >= MAX_SUBJECTS:
            break

    save_results(json_path, results)

    print(f"\nSaved results for {found_subjects} subjects to {OUT_JSON}", flush=True)
    print(f"Progress saved to {PROGRESS_JSON}", flush=True)
    print(f"ONE cache directory: {CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()

