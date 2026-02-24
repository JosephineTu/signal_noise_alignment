from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas as ba
from brainbox.population.decode import get_spike_counts_in_bins

from passiveGabor_VIS import load_results


@dataclass
class AnalyzerConfig:
    target_prefix: str = "VIS" # default visual cortex acronym
    thresh: float = 0.5
    min_trials: int = 5
    eps: float = 1e-12
    min_units: int = 5
    std_eps: float = 1e-10


class IBLAlignmentBase:
    """
    Base class that encapsulates:
    - ONE / atlas setup
    - linear algebra utilities
    - picking best insertion
    - loading trials/spikes + region filtering
    - label parsing helpers
    """

    def __init__(
        self,
        one: ONE,
        atlas=None,
        config: Optional[AnalyzerConfig] = None,
    ):
        self.one = one
        self.atlas = atlas if atlas is not None else ba()
        self.cfg = config if config is not None else AnalyzerConfig()

    @staticmethod
    def enforce_sym(C):
        """
        Enforce symmetry on C by averaging with its transpose.
        """
        C = np.asarray(C, dtype=float)
        return 0.5 * (C + C.T)

    def trace_normalize(self, C):
        """
        Normalize C by its trace, with safety for near-zero trace.
        """
        C = np.asarray(C, dtype=float)
        tr = float(np.trace(C))
        if tr < self.cfg.eps:
            return C
        return C / tr

    def top_eigenvector(self, C, return_eigval=False):
        """
        Compute the top eigenvector of C, 
        ensuring a consistent sign by looking at the largest absolute component.
        """
        C = self.enforce_sym(C)
        vals, vecs = np.linalg.eigh(C)
        idx = int(np.argmax(vals))
        v = vecs[:, idx]
        j = int(np.argmax(np.abs(v)))
        if v[j] < 0:
            v = -v
        # normalize for stability (also for cosine)
        v = v / (np.linalg.norm(v) + self.cfg.eps)
        if return_eigval:
            return v, float(vals[idx])
        return v

    @staticmethod
    def cosine_abs(u, v, eps=1e-12):
        """
        Compute the absolute cosine between u and v, with safety for zero norms.
        """
        u = u / (np.linalg.norm(u) + eps)
        v = v / (np.linalg.norm(v) + eps)
        return float(np.abs(np.dot(u, v)))

    def eig_spectrum(self, C, do_trace_norm=True):
        """
        Compute eigenvalues of C, and return values and cumulative sum.
        """
        C = self.enforce_sym(C)
        if do_trace_norm:
            C = self.trace_normalize(C)
        # eigenvalues in descending order, clipped to non-negative
        vals = np.linalg.eigvalsh(C)[::-1]
        vals = np.clip(vals, 0, None)
        cum = np.cumsum(vals) / (np.sum(vals) + self.cfg.eps)
        return vals, cum

    @staticmethod
    def participation_ratio(vals, eps=1e-12):
        """
        Compute the participation ratio of the eigenvalues.
        - Large PR: high-dimensional noise 
        - Small PR: low-dimensional noise
        """
        vals = np.asarray(vals, float)
        vals = np.clip(vals, 0, None)
        s1 = float(np.sum(vals))
        s2 = float(np.sum(vals**2))
        return float((s1**2) / (s2 + eps))

    @staticmethod
    def ensure_1d(x, name: str):
        """
        Ensure that x is a 1D array, allowing for shape (n_trials,) or (n_trials, 1).
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(f"{name} has unexpected shape {x.shape} (expected (n_trials,))")

    def pick_best_insertion(self, eid: str) -> str:
        """
        Pick the best insertion for the given eid, 
        based on the number of clusters in the target region.
        """
        ins = self.one.alyx.rest("insertions", "list", session=eid)
        best_pid, best_n = None, -1
        for x in ins:
            pid = x["id"]
            try:
                sl = SpikeSortingLoader(pid=pid, one=self.one, atlas=self.atlas)
                spikes, clusters, channels = sl.load_spike_sorting()
                clusters = sl.merge_clusters(spikes, clusters, channels)
                acr = clusters.get("acronym", None)
                if acr is None:
                    continue
                acr = np.asarray(acr)
                n = int(np.sum([a.startswith(self.cfg.target_prefix) for a in acr]))
                if n > best_n:
                    best_n = n
                    best_pid = pid
            except Exception:
                continue
        if best_pid is None:
            raise RuntimeError("No valid insertion found.")
        return best_pid

    def load_trials_and_spikes(self, eid: str):
        trials = self.one.load_object(eid, "trials", collection="alf")

        stim_on = self.ensure_1d(trials["stimOn_times"], "stimOn_times")
        stim_off = self.ensure_1d(trials["stimOff_times"], "stimOff_times")
        # keep only trials with valid stimOn and stimOff times
        valid = (~np.isnan(stim_on)) & (~np.isnan(stim_off))
        # slice only fields whose first dim matches n_trials
        nT = len(stim_on)
        for k in list(trials.keys()):
            v = np.asarray(trials[k])
            if v.ndim >= 1 and v.shape[0] == nT:
                # mask out invalid timepoints
                trials[k] = v[valid]
            else:
                trials[k] = v

        stim_on = stim_on[valid]
        stim_off = stim_off[valid]
        pid = self.pick_best_insertion(eid)
        sl = SpikeSortingLoader(pid=pid, one=self.one, atlas=self.atlas)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        # region filter
        acr = np.asarray(clusters["acronym"])
        region_mask = np.array([a.startswith(self.cfg.target_prefix) for a in acr])
        region_cluster_ids = np.asarray(clusters["cluster_id"])[region_mask]

        return trials, spikes, clusters, pid, region_cluster_ids, stim_on, stim_off


    def get_signed_and_masks(self, trials):
        """
        Compute signed contrast and trial masks:
        """
        cl = self.ensure_1d(trials["contrastLeft"], "contrastLeft")
        cr = self.ensure_1d(trials["contrastRight"], "contrastRight")
        cl = np.nan_to_num(cl, nan=0.0)
        cr = np.nan_to_num(cr, nan=0.0)

        signed = cl - cr
        is_zero = (cl == 0) & (cr == 0)
        # high_mask: non-zero signed contrast with abs > thresh
        high_mask = (~is_zero) & (np.abs(signed) > self.cfg.thresh)
        pos_mask = high_mask & (signed > 0)
        neg_mask = high_mask & (signed < 0)
        return signed, is_zero, high_mask, pos_mask, neg_mask

    def get_choice_feedback(self, trials):
        """
        Extract choice, feedback, and probabilityLeft from trials.
        """
        choice = self.ensure_1d(trials["choice"], "choice").astype(float)
        feedback = self.ensure_1d(trials["feedbackType"], "feedbackType").astype(float)
        pleft = self.ensure_1d(trials["probabilityLeft"], "probabilityLeft").astype(float)
        return choice, feedback, pleft

    def compute_signal_axis_mu(self, fr_bin, pos_mask, neg_mask):
        """
        Compute the signal axis u_sig as the normalized difference of mean responses
        """
        if pos_mask.sum() < self.cfg.min_trials or neg_mask.sum() < self.cfg.min_trials:
            raise RuntimeError(f"Not enough trials (pos={pos_mask.sum()}, neg={neg_mask.sum()})")

        mu_pos = fr_bin[pos_mask].mean(axis=0)
        mu_neg = fr_bin[neg_mask].mean(axis=0)
        # define u_sig as the normalized difference of positive and negative fr means
        u_sig = mu_pos - mu_neg
        nrm = float(np.linalg.norm(u_sig))
        if nrm < 1e-10:
            raise RuntimeError("Signal axis too small.")
        u_sig = u_sig / (nrm + self.cfg.eps)
        return u_sig, mu_pos, mu_neg, nrm

    def noise_residuals_by_sign(self, fr_bin, pos_mask, neg_mask, mu_pos, mu_neg):
        """
        Compute noise residuals by subtracting the appropriate mean based on trial sign.
        """
        fr_noise = np.full_like(fr_bin, np.nan, dtype=float)
        fr_noise[pos_mask] = fr_bin[pos_mask] - mu_pos
        fr_noise[neg_mask] = fr_bin[neg_mask] - mu_neg
        return fr_noise

    def noise_corr_from_residuals(self, fr_noise, high_mask):
        """
        Compute noise correlation matrix from residuals of high-contrast trials.
        """
        X = fr_noise[high_mask]
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        return self.enforce_sym(C)

    @staticmethod
    def stratified_kfold_indices(y, k=5, seed=0):
        """
        Generate stratified K-fold indices for labels y.
        """
        rng = np.random.default_rng(seed)
        y = np.asarray(y)
        idx = np.arange(len(y))
        cls = np.unique(y)
        folds = [[] for _ in range(k)]
        for c in cls:
            ic = idx[y == c]
            rng.shuffle(ic)
            parts = np.array_split(ic, k)
            for i in range(k):
                folds[i].extend(parts[i].tolist())
        return [np.array(sorted(f), dtype=int) for f in folds]

    @staticmethod
    def roc_auc_approx(y_true, score):
        """
        Approximate ROC AUC using Mann-Whitney U statistic, 
        which is equivalent to the probability that a random positive sample 
        has a higher score than a random negative sample.
        """
        y = np.asarray(y_true)
        s = np.asarray(score)
        if set(np.unique(y)).issubset({-1, 1}):
            y01 = (y == 1).astype(int)
        else:
            y01 = (y > 0).astype(int)
        order = np.argsort(s)
        y_sorted = y01[order]
        n_pos = int(y_sorted.sum())
        n_neg = int(len(y_sorted) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = np.arange(1, len(y_sorted) + 1)
        sum_ranks_pos = float(ranks[y_sorted == 1].sum())
        U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        return float(U / (n_pos * n_neg))

    def cv_decode_1d(self, proj, y, k=5, seed=42):
        """
        Perform stratified K-fold cross-validated decoding using a 1D projection and compute AUC.
        """
        proj = np.asarray(proj, float)
        y = np.asarray(y, float)
        folds = self.stratified_kfold_indices(y, k=k, seed=seed)
        aucs = []
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            m1 = proj[train_idx][y[train_idx] == 1].mean()
            m0 = proj[train_idx][y[train_idx] == -1].mean()
            score = (proj[test_idx] - 0.5 * (m1 + m0)) * np.sign(m1 - m0 + self.cfg.eps)
            aucs.append(self.roc_auc_approx(y[test_idx], score))
        return float(np.nanmean(aucs))


class StaticAlignmentAnalyzer(IBLAlignmentBase):
    """
    - Bin spikes over stimOn->stimOff (one value per trial)
    - Build u_sig using mu_pos-mu_neg on high-contrast trials
    - Build noise corr from residuals
    - Compute alignment cosine + noise spectrum summaries
    """

    def load_trial_firing_rates(self, eid: str):
        """
        Load trials and spikes for the given eid, apply region filter, and compute trial firing rates.
        """
        trials, spikes, clusters, pid, region_cluster_ids, stim_on, stim_off = self.load_trials_and_spikes(eid)

        stim_intervals = np.c_[stim_on, stim_off]
        counts, cluster_ids = get_spike_counts_in_bins(spikes["times"], spikes["clusters"], stim_intervals)

        cluster_ids = np.asarray(cluster_ids)
        # robust orientation: allow (n_clusters, n_trials) or (n_trials, n_clusters)
        if counts.shape[0] == cluster_ids.shape[0] and counts.shape[1] == stim_intervals.shape[0]:
            # (n_clusters, n_trials)
            keep = np.isin(cluster_ids, region_cluster_ids)
            counts = counts[keep, :].T
        elif counts.shape[1] == cluster_ids.shape[0] and counts.shape[0] == stim_intervals.shape[0]:
            # (n_trials, n_clusters)
            keep = np.isin(cluster_ids, region_cluster_ids)
            counts = counts[:, keep]
        else:
            raise RuntimeError(f"Unexpected counts shape {counts.shape}")

        dur = np.diff(stim_intervals, axis=1)  # (n_trials, 1)
        fr = counts / (dur + self.cfg.eps)     # (n_trials, n_units)
        return trials, fr, pid

    def run_one_eid(self, eid: str, do_plots=False) -> Dict[str, Any]:
        trials, fr, pid = self.load_trial_firing_rates(eid)
        signed, is_zero, high_mask, pos_mask, neg_mask = self.get_signed_and_masks(trials)

        # neuron mask: require variance overall and in high trials
        std_all = fr.std(axis=0)
        std_high = fr[high_mask].std(axis=0) if high_mask.any() else np.zeros(fr.shape[1])
        neuron_mask = (std_all > self.cfg.std_eps) & (std_high > self.cfg.std_eps)
        fr = fr[:, neuron_mask]

        u_sig, mu_pos, mu_neg, sig_norm = self.compute_signal_axis_mu(fr, pos_mask, neg_mask)
        fr_noise = self.noise_residuals_by_sign(fr, pos_mask, neg_mask, mu_pos, mu_neg)

        C_noi = self.noise_corr_from_residuals(fr_noise, high_mask)
        C_noi_norm = self.trace_normalize(C_noi)
        u_noi, ev_noi = self.top_eigenvector(C_noi_norm, return_eigval=True)

        cos = self.cosine_abs(u_sig, u_noi, eps=self.cfg.eps)

        vals, cum = self.eig_spectrum(C_noi, do_trace_norm=True)
        D_eff = self.participation_ratio(vals)

        if do_plots:
            plt.figure(figsize=(5, 4))
            plt.imshow(C_noi, vmin=-1, vmax=1, cmap="coolwarm")
            plt.colorbar(label="corr")
            plt.title("Noise correlation (static)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.plot(vals[:50], marker="o")
            plt.title("Noise eigen spectrum (static)")
            plt.xlabel("eigenvalue index")
            plt.ylabel("trace-normalized eig")
            plt.tight_layout()
            plt.show()

        return {
            "eid": eid,
            "pid": pid,
            "T": int(fr.shape[0]),
            "N": int(fr.shape[1]),
            "sig_norm": float(sig_norm),
            "cosine": float(cos),
            "noise_top1": float(vals[0]) if len(vals) else np.nan,
            "noise_top5_cum": float(cum[min(4, len(cum)-1)]) if len(cum) else np.nan,
            "noise_PR": float(D_eff),
            "noise_top_eig": float(ev_noi),
        }

    def run_many(self, eids: List[str], do_plots=False):
        rows = []
        for eid in eids:
            try:
                rows.append(self.run_one_eid(eid, do_plots=do_plots))
            except Exception as e:
                print("skip", eid, "because", repr(e))
        return rows


class TimeResolvedAlignmentAnalyzer(IBLAlignmentBase):
    """
    Equivalent of your time_resolved_alignment.py:
    - Build fr_tb: (n_trials, n_bins, n_units)
    - For each bin, compute u_sig(t), noise corr, u_noi(t), alignment(t)
    - Optionally run 1D decoding timecourses
    """

    def __init__(
        self,
        one: ONE,
        atlas=None,
        config: Optional[AnalyzerConfig] = None,
        t_start: float = 0.0,
        t_end: float = 0.4,
        bin_size: float = 0.05,
        do_decode: bool = True,
    ):
        super().__init__(one, atlas=atlas, config=config)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.bin_size = float(bin_size)
        self.do_decode = bool(do_decode)

    @staticmethod
    def make_time_bins(t_start, t_end, bin_size):
        """
        Create time bin edges from t_start to t_end with given bin_size.
        """
        edges = np.arange(t_start, t_end + 1e-12, bin_size)
        if len(edges) < 2:
            raise ValueError("Time range too small for given bin_size.")
        return edges

    def trial_binned_firing_rates(self, spikes, region_cluster_ids, stim_on, edges):
        """
        Bin spikes into firing rates for each trial and time bin, applying region filter.
        """
        stim_on = np.asarray(stim_on)
        n_trials = stim_on.shape[0]
        n_bins = len(edges) - 1
        n_intervals = n_trials * n_bins
        # create intervals of shape (n_trials*n_bins, 2) where each row is [stim_on + edge_start, stim_on + edge_end]
        starts = (stim_on[:, None] + edges[:-1][None, :]).reshape(-1)
        ends = (stim_on[:, None] + edges[1:][None, :]).reshape(-1)
        intervals = np.c_[starts, ends]

        counts, cluster_ids = get_spike_counts_in_bins(
            spikes["times"], spikes["clusters"], intervals
        )
        cluster_ids = np.asarray(cluster_ids)
        # robust orientation
        if counts.shape[0] == cluster_ids.shape[0] and counts.shape[1] == n_intervals:
            keep = np.isin(cluster_ids, region_cluster_ids)
            counts = counts[keep, :].T
            unit_ids = cluster_ids[keep]
        elif counts.shape[1] == cluster_ids.shape[0] and counts.shape[0] == n_intervals:
            keep = np.isin(cluster_ids, region_cluster_ids)
            counts = counts[:, keep]
            unit_ids = cluster_ids[keep]
        else:
            raise RuntimeError(
                f"Unexpected counts shape {counts.shape}, "
                f"cluster_ids {cluster_ids.shape}, n_intervals={n_intervals}"
            )

        counts = counts.reshape(n_trials, n_bins, -1)
        dur = (edges[1:] - edges[:-1])[None, :, None]
        fr_tb = counts / (dur + self.cfg.eps)  # (trial, bin, unit)
        return fr_tb, unit_ids

    def run_one_eid(self, eid: str, do_plots=True) -> Dict[str, Any]:
        trials, spikes, clusters, pid, region_cluster_ids, stim_on, stim_off = self.load_trials_and_spikes(eid)
        signed, is_zero, high_mask, pos_mask, neg_mask = self.get_signed_and_masks(trials)
        choice, feedback, pleft = self.get_choice_feedback(trials)

        edges = self.make_time_bins(self.t_start, self.t_end, self.bin_size)
        fr_tb, unit_ids = self.trial_binned_firing_rates(spikes, region_cluster_ids, stim_on, edges)
        print_feedback_summary(trials)
        n_trials, n_bins, n_units = fr_tb.shape
        if n_units < self.cfg.min_units:
            raise RuntimeError(f"Too few units after region filter: {n_units}")

        # neuron stability mask across all (trial,bin) and high-only (trial,bin)
        fr_flat = fr_tb.reshape(n_trials * n_bins, n_units)
        std_all = fr_flat.std(axis=0)
        std_high = fr_tb[high_mask].reshape((-1, n_units)).std(axis=0) if high_mask.any() else np.zeros(n_units)
        neuron_mask = (std_all > self.cfg.std_eps) & (std_high > self.cfg.std_eps)

        fr_tb = fr_tb[:, :, neuron_mask]
        unit_ids = unit_ids[neuron_mask]
        n_units = fr_tb.shape[2]

        times = 0.5 * (edges[:-1] + edges[1:])
        cos_ts = np.full(n_bins, np.nan)
        noise_top1_ts = np.full(n_bins, np.nan)
        noise_pr_ts = np.full(n_bins, np.nan)
        sig_norm_ts = np.full(n_bins, np.nan)

        stim_auc_ts = np.full(n_bins, np.nan)
        choice_auc_ts = np.full(n_bins, np.nan)
        fb_auc_ts = np.full(n_bins, np.nan)

        choice_valid = choice != 0

        for b in range(n_bins):
            fr_bin = fr_tb[:, b, :]  # (trial, unit)

            try:
                u_sig, mu_pos, mu_neg, sig_norm = self.compute_signal_axis_mu(fr_bin, pos_mask, neg_mask)
            except Exception:
                continue

            sig_norm_ts[b] = sig_norm

            fr_noise = self.noise_residuals_by_sign(fr_bin, pos_mask, neg_mask, mu_pos, mu_neg)

            try:
                C_noi = self.noise_corr_from_residuals(fr_noise, high_mask)
                C_noi_norm = self.trace_normalize(C_noi)
                u_noi, _ = self.top_eigenvector(C_noi_norm, return_eigval=True)
            except Exception:
                continue

            cos_ts[b] = self.cosine_abs(u_sig, u_noi, eps=self.cfg.eps)
            vals, _ = self.eig_spectrum(C_noi, do_trace_norm=True)
            noise_top1_ts[b] = float(vals[0]) if len(vals) else np.nan
            noise_pr_ts[b] = self.participation_ratio(vals)
            # store sig strength too (already)

            if self.do_decode:
                # stimulus
                use = high_mask & (signed != 0)
                if use.sum() >= 2 * self.cfg.min_trials:
                    proj = fr_bin[use] @ u_sig
                    y = np.where(signed[use] > 0, 1, -1)
                    stim_auc_ts[b] = self.cv_decode_1d(proj, y, k=5, seed=0)

                # choice
                usec = high_mask & choice_valid
                if usec.sum() >= 2 * self.cfg.min_trials:
                    y = np.where(choice[usec] > 0, 1, -1)
                    proj = fr_bin[usec] @ u_sig
                    choice_auc_ts[b] = self.cv_decode_1d(proj, y, k=5, seed=1)

                # feedback (correct/error) using residual projection
                usef = high_mask & (feedback != 0)
                if usef.sum() >= 2 * self.cfg.min_trials:
                    y = np.where(feedback[usef] > 0, 1, -1)
                    proj = fr_noise[usef] @ u_sig
                    fb_auc_ts[b] = self.cv_decode_1d(proj, y, k=5, seed=2)

        if do_plots:
            plt.figure(figsize=(7, 4))
            plt.plot(times, cos_ts, marker="o")
            plt.ylim(0, 1.05)
            plt.axhline(0, color="k", lw=1)
            plt.xlabel("time from stimOn (s)")
            plt.ylabel("|cos(u_sig(t), u_noi(t))|")
            plt.title("Time-resolved signal-noise alignment")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(times, noise_top1_ts, marker="o", label="noise top-1 (trace-norm)")
            # normalize PR for plotting
            pr_norm = noise_pr_ts / (np.nanmax(noise_pr_ts) + self.cfg.eps)
            plt.plot(times, pr_norm, marker="o", label="PR (normed)")
            plt.axhline(0, color="k", lw=1)
            plt.xlabel("time from stimOn (s)")
            plt.ylabel("summary")
            plt.title("Time-resolved noise structure")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()

            if self.do_decode:
                plt.figure(figsize=(7, 4))
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

        return {
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

def build_eids_from_results(json_path="VISp_subjects_by_lab.json") -> List[str]:
    results = load_results(json_path)
    eids = []
    for lab_name in results.keys():
        for subject in results[lab_name].keys():
            eids.extend(results[lab_name][subject]["VIS_eids"])
    return eids

def print_feedback_summary(trials):
    feedback = np.asarray(trials["feedbackType"])

    print("feedbackType shape:", feedback.shape)
    print("unique feedbackType values:", np.unique(feedback))

    for v in np.unique(feedback):
        print(f"  value {v:>2}: {(feedback == v).sum()} trials")

if __name__ == "__main__":
    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(password="international")
    atlas = ba()

    eids = build_eids_from_results("VISp_subjects_by_lab.json")

    # --- static ---
    static_cfg = AnalyzerConfig(target_prefix="VIS", thresh=0.5, min_trials=5)
    static_an = StaticAlignmentAnalyzer(one, atlas=atlas, config=static_cfg)

    rows = static_an.run_many(eids, do_plots=False)
    print("static done:", len(rows))

    # --- time-resolved ---
    tr_cfg = AnalyzerConfig(target_prefix="VIS", thresh=0.5, min_trials=5)
    tr_an = TimeResolvedAlignmentAnalyzer(
        one, atlas=atlas, config=tr_cfg,
        t_start=0.0, t_end=0.4, bin_size=0.05,
        do_decode=True,
    )

    # run a couple sessions
    for eid in eids[:5]:
        try:
            out = tr_an.run_one_eid(eid, do_plots=True)
            print("time-resolved out keys:", out.keys())
        except Exception as e:
            print("skip", eid, "because", repr(e))