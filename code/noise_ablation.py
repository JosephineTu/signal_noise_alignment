from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import pickle

import numpy as np
import pandas as pd

import stability_analysis as sa
from alignment_analyzers import (
    AnalyzerConfig,
    TimeResolvedAlignmentAnalyzer,
    build_eids_from_results,
)
from iblatlas.atlas import AllenAtlas as ba
from one.api import ONE


def safe_mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def lda_auc_from_train_test(ab, X_train, y_train, X_test, y_test):
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    y_train = np.asarray(y_train, float)
    y_test = np.asarray(y_test, float)

    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return np.nan, np.full(X_train.shape[1], np.nan)

    if np.sum(y_train == 1) < 1 or np.sum(y_train == -1) < 1:
        return np.nan, np.full(X_train.shape[1], np.nan)

    mu_pos = X_train[y_train == 1].mean(axis=0)
    mu_neg = X_train[y_train == -1].mean(axis=0)

    C = np.cov(X_train.T, bias=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    if C.shape != (X_train.shape[1], X_train.shape[1]):
        C = np.eye(X_train.shape[1]) * ab.cfg.eps
    C = C + ab.cfg.eps * np.eye(X_train.shape[1])

    try:
        w = np.linalg.solve(C, mu_pos - mu_neg)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(C) @ (mu_pos - mu_neg)

    b = 0.5 * (mu_pos + mu_neg) @ w
    score = X_test @ w - b
    return ab.roc_auc_approx(y_test, score), w


class NoiseAblationAnalyzer(TimeResolvedAlignmentAnalyzer):
    """
    Remove top-k noise PCs and remeasure stimulus decoding.

    For decoding AUC, the noise-removal subspace is estimated inside each CV
    training fold and then applied to that fold's train/test activity.
    """

    def __init__(
        self,
        one,
        atlas=None,
        config: Optional[AnalyzerConfig] = None,
        t_start: float = 0.0,
        t_end: float = 0.4,
        bin_size: float = 0.02,
        step_size: float = 0.02,
        ablation_k: int = 3,
        decoder_k: Optional[int] = None,
    ):
        super().__init__(
            one,
            atlas=atlas,
            config=config,
            t_start=t_start,
            t_end=t_end,
            bin_size=bin_size,
            step_size=step_size,
            do_decode=True,
        )
        self.ablation_k = int(ablation_k)
        self.decoder_k = int(decoder_k) if decoder_k is not None else int(ablation_k)

    def project_out_subspace(self, X, U):
        X = np.asarray(X, float)
        U = np.asarray(U, float)
        return X - (X @ U) @ U.T

    def fit_noise_subspace(
        self,
        fr_bin,
        pos_mask,
        neg_mask,
        high_mask,
        k,
        residual_pos_mask=None,
        residual_neg_mask=None,
    ):
        u_sig, mu_pos, mu_neg, sig_norm = self.compute_signal_axis_mu(fr_bin, pos_mask, neg_mask)
        if residual_pos_mask is None:
            residual_pos_mask = pos_mask
        if residual_neg_mask is None:
            residual_neg_mask = neg_mask
        fr_noise = self.noise_residuals_by_sign(
            fr_bin,
            residual_pos_mask,
            residual_neg_mask,
            mu_pos,
            mu_neg,
        )
        C_noi, keep_bin = self.noise_cov_from_residuals(fr_noise, high_mask)
        C_noi_norm = self.trace_normalize(C_noi)
        U_noi, evals = self.top_noise_subspace(
            C_noi_norm,
            k=k,
            var_threshold=None,
            return_eigvals=True,
        )
        return {
            "u_sig": u_sig,
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "sig_norm": sig_norm,
            "fr_noise": fr_noise,
            "keep_bin": keep_bin,
            "U_noi": U_noi,
            "evals": evals,
            "C_noi": C_noi,
        }

    def make_signal_noise_features(
        self,
        fr_bin,
        pos_mask,
        neg_mask,
        high_mask,
        k,
        residual_pos_mask=None,
        residual_neg_mask=None,
        feature_k=None,
    ):
        fit = self.fit_noise_subspace(
            fr_bin,
            pos_mask,
            neg_mask,
            high_mask,
            k=k,
            residual_pos_mask=residual_pos_mask,
            residual_neg_mask=residual_neg_mask,
        )

        keep = fit["keep_bin"]
        X = fr_bin[:, keep]
        N = fit["fr_noise"][:, keep]

        delta_mu = fit["mu_pos"][keep] - fit["mu_neg"][keep]
        u_sig = delta_mu / (np.linalg.norm(delta_mu) + self.cfg.eps)

        if feature_k is None:
            feature_k = fit["U_noi"].shape[1]
        U_feature = fit["U_noi"][:, : int(feature_k)]

        return np.column_stack([X @ u_sig, N @ U_feature]), fit

    def decode_bin_ablation(self, fr_bin, signed, high_mask, pos_mask, neg_mask, seed=0):
        y = np.where(signed > 0, 1, -1)
        use_mask = high_mask & (signed != 0)
        idx_all = np.flatnonzero(use_mask)

        out = {
            "baseline_auc_signal": np.nan,
            "baseline_auc_topk": np.nan,
            "ablated_auc_signal": np.nan,
            "ablated_auc_topk": np.nan,
            "delta_auc_signal": np.nan,
            "delta_auc_topk": np.nan,
            "baseline_signal_weight_frac": np.nan,
            "baseline_noise_weight_frac": np.nan,
            "ablated_signal_weight_frac": np.nan,
            "ablated_noise_weight_frac": np.nan,
            "n_folds": 0,
        }

        if idx_all.size < 2 * self.cfg.min_trials:
            return out

        y_use = y[idx_all]
        kfold = self.choose_kfold(y_use)
        if kfold < 2:
            return out

        folds = self.stratified_kfold_indices(y_use, k=kfold, seed=seed)
        base_auc_1d, base_auc_topk = [], []
        abl_auc_1d, abl_auc_topk = [], []
        base_w, abl_w = [], []

        for fold in folds:
            test_idx = idx_all[fold]
            train_idx = np.setdiff1d(idx_all, test_idx)

            train_mask = np.zeros(fr_bin.shape[0], dtype=bool)
            train_mask[train_idx] = True
            train_pos = train_mask & pos_mask
            train_neg = train_mask & neg_mask
            train_high = train_mask & high_mask

            y_train = y[train_idx]
            y_test = y[test_idx]

            try:
                fit_k = max(self.decoder_k, self.ablation_k)
                F_base, base_fit = self.make_signal_noise_features(
                    fr_bin,
                    train_pos,
                    train_neg,
                    train_high,
                    k=fit_k,
                    residual_pos_mask=pos_mask,
                    residual_neg_mask=neg_mask,
                    feature_k=self.decoder_k,
                )

                auc_1d, _ = lda_auc_from_train_test(
                    self,
                    F_base[train_idx, 0],
                    y_train,
                    F_base[test_idx, 0],
                    y_test,
                )
                auc_topk, w_base = lda_auc_from_train_test(
                    self,
                    F_base[train_idx],
                    y_train,
                    F_base[test_idx],
                    y_test,
                )

                keep = base_fit["keep_bin"]
                X_keep = fr_bin[:, keep]
                U_remove = base_fit["U_noi"][:, : self.ablation_k]
                X_abl_keep = self.project_out_subspace(X_keep, U_remove)

                fr_abl = fr_bin.copy()
                fr_abl[:, keep] = X_abl_keep

                F_abl, _ = self.make_signal_noise_features(
                    fr_abl,
                    train_pos,
                    train_neg,
                    train_high,
                    k=self.decoder_k,
                    residual_pos_mask=pos_mask,
                    residual_neg_mask=neg_mask,
                    feature_k=self.decoder_k,
                )

                auc_abl_1d, _ = lda_auc_from_train_test(
                    self,
                    F_abl[train_idx, 0],
                    y_train,
                    F_abl[test_idx, 0],
                    y_test,
                )
                auc_abl_topk, w_abl = lda_auc_from_train_test(
                    self,
                    F_abl[train_idx],
                    y_train,
                    F_abl[test_idx],
                    y_test,
                )
            except Exception:
                continue

            base_auc_1d.append(auc_1d)
            base_auc_topk.append(auc_topk)
            abl_auc_1d.append(auc_abl_1d)
            abl_auc_topk.append(auc_abl_topk)
            base_w.append(w_base)
            abl_w.append(w_abl)

        if not base_auc_topk:
            return out

        w_base_mean = np.nanmean(np.asarray(base_w, float), axis=0)
        w_abl_mean = np.nanmean(np.asarray(abl_w, float), axis=0)
        base_norm = np.linalg.norm(w_base_mean)
        abl_norm = np.linalg.norm(w_abl_mean)

        out.update({
            "baseline_auc_signal": safe_mean(base_auc_1d),
            "baseline_auc_topk": safe_mean(base_auc_topk),
            "ablated_auc_signal": safe_mean(abl_auc_1d),
            "ablated_auc_topk": safe_mean(abl_auc_topk),
            "n_folds": len(base_auc_topk),
        })
        out["delta_auc_signal"] = out["ablated_auc_signal"] - out["baseline_auc_signal"]
        out["delta_auc_topk"] = out["ablated_auc_topk"] - out["baseline_auc_topk"]

        if base_norm > self.cfg.eps:
            out["baseline_signal_weight_frac"] = float(abs(w_base_mean[0]) / base_norm)
            out["baseline_noise_weight_frac"] = float(np.linalg.norm(w_base_mean[1:]) / base_norm)
        if abl_norm > self.cfg.eps:
            out["ablated_signal_weight_frac"] = float(abs(w_abl_mean[0]) / abl_norm)
            out["ablated_noise_weight_frac"] = float(np.linalg.norm(w_abl_mean[1:]) / abl_norm)

        return out

    def ablate_full_bin_for_reliability(self, fr_bin, pos_mask, neg_mask, high_mask):
        fit = self.fit_noise_subspace(
            fr_bin,
            pos_mask,
            neg_mask,
            high_mask,
            k=self.ablation_k,
        )
        keep = fit["keep_bin"]
        fr_abl = fr_bin.copy()
        fr_abl[:, keep] = self.project_out_subspace(fr_bin[:, keep], fit["U_noi"])
        return fr_abl, fit

    def run_one_eid(self, eid: str, do_plots=False) -> Dict[str, Any]:
        trials, spikes, clusters, pid, region_cluster_ids, stim_on, stim_off = self.load_trials_and_spikes(eid)
        signed, is_zero, high_mask, pos_mask, neg_mask = self.get_signed_and_masks(trials)

        windows = self.make_time_windows(self.t_start, self.t_end, self.bin_size, self.step_size)
        fr_tb, unit_ids = self.trial_binned_firing_rates(spikes, region_cluster_ids, stim_on, windows)

        n_trials, n_bins, n_units = fr_tb.shape
        if n_units < self.cfg.min_units:
            raise RuntimeError(f"Too few units after region filter: {n_units}")

        fr_flat = fr_tb.reshape(n_trials * n_bins, n_units)
        std_all = fr_flat.std(axis=0)
        std_high = fr_tb[high_mask].reshape((-1, n_units)).std(axis=0) if high_mask.any() else np.zeros(n_units)
        neuron_mask = (std_all > self.cfg.std_eps) & (std_high > self.cfg.std_eps)
        fr_tb = fr_tb[:, :, neuron_mask]
        unit_ids = unit_ids[neuron_mask]
        n_units = fr_tb.shape[2]

        times = 0.5 * (windows[:, 0] + windows[:, 1])

        baseline_auc_signal_ts = np.full(n_bins, np.nan)
        baseline_auc_topk_ts = np.full(n_bins, np.nan)
        ablated_auc_signal_ts = np.full(n_bins, np.nan)
        ablated_auc_topk_ts = np.full(n_bins, np.nan)
        delta_auc_signal_ts = np.full(n_bins, np.nan)
        delta_auc_topk_ts = np.full(n_bins, np.nan)

        baseline_signal_reliability_ts = np.full(n_bins, np.nan)
        baseline_decoder_reliability_ts = np.full(n_bins, np.nan)
        ablated_signal_reliability_ts = np.full(n_bins, np.nan)
        ablated_decoder_reliability_ts = np.full(n_bins, np.nan)

        baseline_signal_weight_frac_ts = np.full(n_bins, np.nan)
        baseline_noise_weight_frac_ts = np.full(n_bins, np.nan)
        ablated_signal_weight_frac_ts = np.full(n_bins, np.nan)
        ablated_noise_weight_frac_ts = np.full(n_bins, np.nan)

        overlap_ts = np.full(n_bins, np.nan)
        noise_pr_ts = np.full(n_bins, np.nan)
        sig_norm_ts = np.full(n_bins, np.nan)
        n_units_cov_ts = np.full(n_bins, np.nan)
        bin_status = []

        for b in range(n_bins):
            fr_bin = fr_tb[:, b, :]

            try:
                full_ablated_fr, fit = self.ablate_full_bin_for_reliability(
                    fr_bin,
                    pos_mask,
                    neg_mask,
                    high_mask,
                )
                keep = fit["keep_bin"]
                vals, _ = self.eig_spectrum(fit["C_noi"], do_trace_norm=True)
                delta_mu = fit["mu_pos"][keep] - fit["mu_neg"][keep]
                overlap = self.signal_noise_subspace_overlap(delta_mu, fit["U_noi"])

                sig_norm_ts[b] = fit["sig_norm"]
                overlap_ts[b] = overlap["overlap_ratio"]
                noise_pr_ts[b] = self.participation_ratio(vals)
                n_units_cov_ts[b] = int(keep.sum())

                dec = self.decode_bin_ablation(
                    fr_bin,
                    signed=signed,
                    high_mask=high_mask,
                    pos_mask=pos_mask,
                    neg_mask=neg_mask,
                    seed=b,
                )

                baseline_auc_signal_ts[b] = dec["baseline_auc_signal"]
                baseline_auc_topk_ts[b] = dec["baseline_auc_topk"]
                ablated_auc_signal_ts[b] = dec["ablated_auc_signal"]
                ablated_auc_topk_ts[b] = dec["ablated_auc_topk"]
                delta_auc_signal_ts[b] = dec["delta_auc_signal"]
                delta_auc_topk_ts[b] = dec["delta_auc_topk"]

                baseline_signal_weight_frac_ts[b] = dec["baseline_signal_weight_frac"]
                baseline_noise_weight_frac_ts[b] = dec["baseline_noise_weight_frac"]
                ablated_signal_weight_frac_ts[b] = dec["ablated_signal_weight_frac"]
                ablated_noise_weight_frac_ts[b] = dec["ablated_noise_weight_frac"]

                baseline_sig_rel = sa.split_half_signal_reliability(self, fr_bin, pos_mask, neg_mask)
                baseline_dec_rel = sa.split_half_decoder_reliability(self, pos_mask, neg_mask, fr_bin)
                ablated_sig_rel = sa.split_half_signal_reliability(self, full_ablated_fr, pos_mask, neg_mask)
                ablated_dec_rel = sa.split_half_decoder_reliability(self, pos_mask, neg_mask, full_ablated_fr)

                baseline_signal_reliability_ts[b] = baseline_sig_rel["cosine_similarity_u_sig"]
                baseline_decoder_reliability_ts[b] = baseline_dec_rel["cosine_similarity_decoder"]
                ablated_signal_reliability_ts[b] = ablated_sig_rel["cosine_similarity_u_sig"]
                ablated_decoder_reliability_ts[b] = ablated_dec_rel["cosine_similarity_decoder"]

                bin_status.append({
                    "bin": int(b),
                    "status": "ok",
                    "n_units_cov": int(keep.sum()),
                    "n_folds": int(dec["n_folds"]),
                    "delta_auc_topk": float(dec["delta_auc_topk"]),
                })
            except Exception as exc:
                bin_status.append({"bin": int(b), "status": "failed", "reason": repr(exc)})

        return {
            "eid": eid,
            "pid": pid,
            "n_trials": int(n_trials),
            "n_units": int(n_units),
            "times": times,
            "ablation_k": int(self.ablation_k),
            "decoder_k": int(self.decoder_k),
            "baseline_auc_signal_ts": baseline_auc_signal_ts,
            "baseline_auc_topk_ts": baseline_auc_topk_ts,
            "ablated_auc_signal_ts": ablated_auc_signal_ts,
            "ablated_auc_topk_ts": ablated_auc_topk_ts,
            "delta_auc_signal_ts": delta_auc_signal_ts,
            "delta_auc_topk_ts": delta_auc_topk_ts,
            "baseline_signal_reliability_ts": baseline_signal_reliability_ts,
            "baseline_decoder_reliability_ts": baseline_decoder_reliability_ts,
            "ablated_signal_reliability_ts": ablated_signal_reliability_ts,
            "ablated_decoder_reliability_ts": ablated_decoder_reliability_ts,
            "baseline_signal_weight_frac_ts": baseline_signal_weight_frac_ts,
            "baseline_noise_weight_frac_ts": baseline_noise_weight_frac_ts,
            "ablated_signal_weight_frac_ts": ablated_signal_weight_frac_ts,
            "ablated_noise_weight_frac_ts": ablated_noise_weight_frac_ts,
            "overlap_ts": overlap_ts,
            "noise_pr_ts": noise_pr_ts,
            "sig_norm_ts": sig_norm_ts,
            "n_units_cov_ts": n_units_cov_ts,
            "n_pos_high": int(pos_mask.sum()),
            "n_neg_high": int(neg_mask.sum()),
            "bin_status": bin_status,
        }


def summarize_one(out):
    return {
        "eid": out["eid"],
        "pid": out["pid"],
        "n_trials": out["n_trials"],
        "n_units": out["n_units"],
        "ablation_k": out["ablation_k"],
        "decoder_k": out["decoder_k"],
        "mean_baseline_auc_signal": safe_mean(out["baseline_auc_signal_ts"]),
        "mean_baseline_auc_topk": safe_mean(out["baseline_auc_topk_ts"]),
        "mean_ablated_auc_signal": safe_mean(out["ablated_auc_signal_ts"]),
        "mean_ablated_auc_topk": safe_mean(out["ablated_auc_topk_ts"]),
        "mean_delta_auc_signal": safe_mean(out["delta_auc_signal_ts"]),
        "mean_delta_auc_topk": safe_mean(out["delta_auc_topk_ts"]),
        "mean_baseline_signal_reliability": safe_mean(out["baseline_signal_reliability_ts"]),
        "mean_baseline_decoder_reliability": safe_mean(out["baseline_decoder_reliability_ts"]),
        "mean_ablated_signal_reliability": safe_mean(out["ablated_signal_reliability_ts"]),
        "mean_ablated_decoder_reliability": safe_mean(out["ablated_decoder_reliability_ts"]),
        "mean_delta_signal_reliability": safe_mean(
            np.asarray(out["ablated_signal_reliability_ts"], float)
            - np.asarray(out["baseline_signal_reliability_ts"], float)
        ),
        "mean_delta_decoder_reliability": safe_mean(
            np.asarray(out["ablated_decoder_reliability_ts"], float)
            - np.asarray(out["baseline_decoder_reliability_ts"], float)
        ),
        "mean_overlap": safe_mean(out["overlap_ts"]),
        "mean_noise_pr": safe_mean(out["noise_pr_ts"]),
        "mean_sig_norm": safe_mean(out["sig_norm_ts"]),
    }


def make_bin_rows(out):
    rows = []
    times = np.asarray(out["times"], float)
    for i, t in enumerate(times):
        rows.append({
            "eid": out["eid"],
            "pid": out["pid"],
            "bin_idx": i,
            "time": t,
            "n_trials": out["n_trials"],
            "n_units": out["n_units"],
            "ablation_k": out["ablation_k"],
            "decoder_k": out["decoder_k"],
            "baseline_auc_signal": out["baseline_auc_signal_ts"][i],
            "baseline_auc_topk": out["baseline_auc_topk_ts"][i],
            "ablated_auc_signal": out["ablated_auc_signal_ts"][i],
            "ablated_auc_topk": out["ablated_auc_topk_ts"][i],
            "delta_auc_signal": out["delta_auc_signal_ts"][i],
            "delta_auc_topk": out["delta_auc_topk_ts"][i],
            "baseline_signal_reliability": out["baseline_signal_reliability_ts"][i],
            "baseline_decoder_reliability": out["baseline_decoder_reliability_ts"][i],
            "ablated_signal_reliability": out["ablated_signal_reliability_ts"][i],
            "ablated_decoder_reliability": out["ablated_decoder_reliability_ts"][i],
            "delta_signal_reliability": out["ablated_signal_reliability_ts"][i] - out["baseline_signal_reliability_ts"][i],
            "delta_decoder_reliability": out["ablated_decoder_reliability_ts"][i] - out["baseline_decoder_reliability_ts"][i],
            "baseline_signal_weight_frac": out["baseline_signal_weight_frac_ts"][i],
            "baseline_noise_weight_frac": out["baseline_noise_weight_frac_ts"][i],
            "ablated_signal_weight_frac": out["ablated_signal_weight_frac_ts"][i],
            "ablated_noise_weight_frac": out["ablated_noise_weight_frac_ts"][i],
            "overlap": out["overlap_ts"][i],
            "noise_pr": out["noise_pr_ts"][i],
            "sig_norm": out["sig_norm_ts"][i],
            "n_units_cov": out["n_units_cov_ts"][i],
        })
    return rows


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Top-k noise PC ablation for VISp time-resolved decoding.")
    p.add_argument("--json", default=str(here / "VISp_subjects_by_lab.json"))
    p.add_argument("--output-dir", default="noise_ablation_results")
    p.add_argument("--cache-dir", default="/scratch/midway3/xiaorantu/ONE")
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--session-index", type=int, default=None)
    p.add_argument("--ablation-k", type=int, default=3)
    p.add_argument("--decoder-k", type=int, default=3)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=0.4)
    p.add_argument("--bin-size", type=float, default=0.02)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--min-trials", type=int, default=5)
    p.add_argument("--min-units", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "per_session"
    result_dir.mkdir(exist_ok=True)

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=args.cache_dir,
    )
    atlas = ba()

    cfg = AnalyzerConfig(
        target_prefix="VIS",
        thresh=0.5,
        min_trials=args.min_trials,
        min_units=args.min_units,
        noise_subspace_k=args.decoder_k,
        cv_folds=5,
    )
    an = NoiseAblationAnalyzer(
        one,
        atlas=atlas,
        config=cfg,
        t_start=args.t_start,
        t_end=args.t_end,
        bin_size=args.bin_size,
        step_size=args.step_size,
        ablation_k=args.ablation_k,
        decoder_k=args.decoder_k,
    )

    eids = build_eids_from_results(args.json)
    if args.max_sessions is not None:
        eids = eids[: args.max_sessions]
    if args.session_index is not None:
        if args.session_index < 0 or args.session_index >= len(eids):
            raise IndexError(
                f"--session-index {args.session_index} out of range for {len(eids)} sessions"
            )
        eids = [eids[args.session_index]]

    print(f"Running noise ablation on {len(eids)} sessions.")
    session_rows = []
    bin_rows = []

    for i, eid in enumerate(eids, start=1):
        print(f"[{i}/{len(eids)}] {eid}")
        try:
            out = an.run_one_eid(eid, do_plots=False)
        except Exception as exc:
            print(f"  skip {eid} because {repr(exc)}")
            continue

        with open(result_dir / f"{eid}.pkl", "wb") as f:
            pickle.dump(out, f)

        session_rows.append(summarize_one(out))
        bin_rows.extend(make_bin_rows(out))

        print(
            "  mean delta top-k AUC after ablation = "
            f"{session_rows[-1]['mean_delta_auc_topk']:.4f}"
        )

    session_df = pd.DataFrame(session_rows)
    bin_df = pd.DataFrame(bin_rows)

    session_csv = output_dir / "noise_ablation_session_summary.csv"
    bin_csv = output_dir / "noise_ablation_bin_summary.csv"
    session_df.to_csv(session_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)

    print(f"\nSaved {len(session_df)} session summaries to {session_csv}")
    print(f"Saved {len(bin_df)} bin rows to {bin_csv}")
    if len(session_df):
        print("\nAcross-session means:")
        print(session_df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
