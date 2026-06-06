import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_DIR = Path("results_time_resolved")
PLOT_DIR = Path("auc_plots")
PLOT_DIR.mkdir(exist_ok=True)


def safe_mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def safe_corr(x,y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])

def safe_max(x):
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def summarize_one(out):
    stim_auc_1d = np.asarray(out["stim_auc_ts"], float)
    stim_auc_topk = np.asarray(out["stim_auc_2d_ts"], float)

    choice_auc_1d = np.asarray(out["choice_auc_ts"], float)
    choice_auc_topk = np.asarray(out["choice_auc_2d_ts"], float)

    fb_auc_1d = np.asarray(out["fb_auc_ts"], float)
    fb_auc_topk = np.asarray(out["fb_auc_2d_ts"], float)

    signal_rel = np.asarray(out["signal_reliability_ts"], float)
    decoder_rel = np.asarray(out["decoder_reliability_ts"], float)

    noise_dim = np.asarray(out["noise_subspace_dim_ts"], float)
    noise_pr = np.asarray(out["noise_pr_ts"], float)
    noise_top1 = np.asarray(out["noise_top1_ts"], float)
    overlap = np.asarray(out["noise_subspace_overlap_ts"], float)
    cos_top1 = np.asarray(out["cos_ts"], float)

    delta_stim = stim_auc_topk - stim_auc_1d
    delta_choice = choice_auc_topk - choice_auc_1d
    delta_fb = fb_auc_topk - fb_auc_1d

    sig_norm_ts = np.asarray(out["sig_norm_ts"], float)

    signal_frac = np.asarray(out["signal_weight_frac_ts"])
    noise_frac = np.asarray(out["noise_weight_frac_ts"])

    return {
        "eid": out["eid"],
        "pid": out["pid"],
        "n_trials": out["n_trials"],
        "n_units": out["n_units"],

        "mean_stim_auc_1d": safe_mean(stim_auc_1d),
        "mean_stim_auc_topk": safe_mean(stim_auc_topk),
        "peak_stim_auc_topk": safe_max(stim_auc_topk),
        "mean_delta_stim_auc": safe_mean(delta_stim),

        "mean_choice_auc_1d": safe_mean(choice_auc_1d),
        "mean_choice_auc_topk": safe_mean(choice_auc_topk),
        "mean_delta_choice_auc": safe_mean(delta_choice),

        "mean_feedback_auc_1d": safe_mean(fb_auc_1d),
        "mean_feedback_auc_topk": safe_mean(fb_auc_topk),
        "mean_delta_feedback_auc": safe_mean(delta_fb),

        "mean_signal_reliability": safe_mean(signal_rel),
        "mean_decoder_reliability": safe_mean(decoder_rel),
        "reliability_gap_decoder_minus_signal": safe_mean(decoder_rel - signal_rel),

        "mean_noise_subspace_dim": safe_mean(noise_dim),
        "mean_noise_pr": safe_mean(noise_pr),
        "mean_noise_top1": safe_mean(noise_top1),

        "mean_overlap": safe_mean(overlap),
        "mean_top1_cos": safe_mean(cos_top1),
        "mean_sig_norm": safe_mean(sig_norm_ts),

        "mean_signal_frac": safe_mean(signal_frac),
        "mean_noise_frac": safe_mean(noise_frac),
    }


def make_bin_rows(out):
    times = np.asarray(out["times"], float)

    stim_auc_1d = np.asarray(out["stim_auc_ts"], float)
    stim_auc_topk = np.asarray(out["stim_auc_2d_ts"], float)

    choice_auc_1d = np.asarray(out["choice_auc_ts"], float)
    choice_auc_topk = np.asarray(out["choice_auc_2d_ts"], float)

    fb_auc_1d = np.asarray(out["fb_auc_ts"], float)
    fb_auc_topk = np.asarray(out["fb_auc_2d_ts"], float)

    signal_rel = np.asarray(out["signal_reliability_ts"], float)
    decoder_rel = np.asarray(out["decoder_reliability_ts"], float)

    signal_frac = np.asarray(out["signal_weight_frac_ts"], float)
    noise_frac = np.asarray(out["noise_weight_frac_ts"], float)


    rows = []
    for i, t in enumerate(times):
        rows.append({
            "eid": out["eid"],
            "pid": out["pid"],
            "bin_idx": i,
            "time": t,
            "n_trials": out["n_trials"],
            "n_units": out["n_units"],

            "stim_auc_1d": stim_auc_1d[i],
            "stim_auc_topk": stim_auc_topk[i],
            "delta_stim_auc": stim_auc_topk[i] - stim_auc_1d[i],

            "choice_auc_1d": choice_auc_1d[i],
            "choice_auc_topk": choice_auc_topk[i],
            "delta_choice_auc": choice_auc_topk[i] - choice_auc_1d[i],

            "feedback_auc_1d": fb_auc_1d[i],
            "feedback_auc_topk": fb_auc_topk[i],
            "delta_feedback_auc": fb_auc_topk[i] - fb_auc_1d[i],

            "signal_frac": signal_frac[i],
            "noise_frac": noise_frac[i],

            "noise_subspace_dim": np.asarray(out["noise_subspace_dim_ts"], float)[i],
            "noise_pr": np.asarray(out["noise_pr_ts"], float)[i],
            "noise_top1": np.asarray(out["noise_top1_ts"], float)[i],
            "overlap": np.asarray(out["noise_subspace_overlap_ts"], float)[i],
            "top1_cos": np.asarray(out["cos_ts"], float)[i],
            "sig_norm": np.asarray(out["sig_norm_ts"], float)[i],

            "signal_reliability": signal_rel[i],
            "decoder_reliability": decoder_rel[i],
            "reliability_gap": decoder_rel[i] - signal_rel[i],
        })
    return rows


def plot_delta_timecourse(bin_df, key, ylabel, filename):
    pivot = bin_df.pivot_table(index="eid", columns="time", values=key)
    times = np.asarray(pivot.columns, float)
    arr = pivot.to_numpy(dtype=float)

    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))

    plt.figure(figsize=(7, 4))

    for row in arr:
        plt.plot(times, row, alpha=0.25, linewidth=1)

    plt.plot(times, mean, marker="o", linewidth=2.5, label="mean")
    plt.fill_between(times, mean - sem, mean + sem, alpha=0.25)

    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("time from stimOn (s)")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()

def compute_session_correlations(bin_df):
    rows = []
    for eid, g in bin_df.groupby('eid'):
        g = g.sort_values('time')

        delta = g['delta_stim_auc'].to_numpy(float)

        rows.append({
            'eid':eid,
            'r_delta_sig_norm': safe_corr(delta, g['sig_norm']),
            'r_delta_overlap':safe_corr(delta, g['overlap']),
            'r_signal_reliability':safe_corr(delta, g['signal_reliability']),
            # signal decoding and overlap 
            'r_stim_auc_1d_overlap': safe_corr(g['stim_auc_1d'],g['overlap']),
            # 2d decoding and overlap
            'r_stim_auc_topk_overlap': safe_corr(g['stim_auc_topk'],g['overlap']),
            # signal decoding and primary signal-noise correlation def
            'r_stim_auc_1d_cos': safe_corr(g['stim_auc_1d'],g['top1_cos']),
        })
    
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv('within_session_delta_auc_correlations.csv',index=False)

    print('\nWithin session correlations:')
    print(corr_df)

    print("\nMean within session correlations:")
    print(corr_df.mean(numeric_only=True))

    return corr_df

def plot_correlation(corr_df):
    corr_cols = [
        'r_delta_sig_norm',
        'r_delta_overlap',
        'r_signal_reliability',
    ]
    
    labels = [
        'sig_norm',
        'overlap',
        'reliability',
    ]
    plt.figure(figsize=(6,4))

    rng = np.random.default_rng(0)

    for i, col in enumerate(corr_cols):
        y = corr_df[col].to_numpy(float)
        x = np.full(len(y), i, dtype=float)
        jitter = rng.uniform(-0.08,0.08,size=len(y))
        plt.scatter(x + jitter, y, alpha=0.8)

        mean_y = np.nanmean(y)

        plt.plot([i - 0.22, i + 0.22], [mean_y, mean_y], linewidth=3)

    plt.axhline(0, color='k', linewidth=1)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("within_session corr with delta stim AUC")
    plt.title("x")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "within_session_delta_auc_correlations.png", dpi=300, bbox_inches="tight")

    plt.close()

def plot_weight_fractions(bin_df):
    specs = [
        ("signal_frac", "decoder signal-weight fraction", "decoder_signal_frac_timecourse"),
        ("noise_frac", "decoder noise-weight fraction", "decoder_noise_frac_timecourse"),
    ]

    for key, ylabel, filename in specs:
        pivot = bin_df.pivot_table(index="eid", columns="time", values=key)
        times = np.asarray(pivot.columns, float)
        arr = pivot.to_numpy(dtype=float)

        mean = np.nanmean(arr, axis=0)
        n = np.sum(np.isfinite(arr), axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))

        plt.figure(figsize=(7, 4))

        for row in arr:
            plt.plot(times, row, alpha=0.25, linewidth=1)

        plt.plot(times, mean, marker="o", linewidth=2.5, label="mean")
        plt.fill_between(times, mean - sem, mean + sem, alpha=0.25)

        plt.ylim(0, 1.05)
        plt.xlabel("time from stimOn (s)")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend(frameon=False)
        plt.tight_layout()

        plt.savefig(PLOT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

def main():
    session_rows = []
    bin_rows = []

    files = sorted(RESULT_DIR.glob("*.pkl"))
    print(f"Found {len(files)} result files.")

    for p in files:
        try:
            with open(p, "rb") as f:
                out = pickle.load(f)

            session_rows.append(summarize_one(out))
            bin_rows.extend(make_bin_rows(out))

        except Exception as e:
            print(f"skip {p.name} because {repr(e)}")

    session_df = pd.DataFrame(session_rows)
    bin_df = pd.DataFrame(bin_rows)

    session_df.to_csv("time_resolved_session_summary.csv", index=False)
    bin_df.to_csv("time_resolved_bin_summary.csv", index=False)

    print(session_df)
    print("\nSaved to time_resolved_session_summary.csv")
    print("Saved to time_resolved_bin_summary.csv")

    print("\nAcross-session means:")
    print(session_df.mean(numeric_only=True))

    plot_delta_timecourse(
        bin_df,
        key="delta_stim_auc",
        ylabel="Δ stimulus AUC = top-k noise decoder - signal-only decoder",
        filename="delta_stim_auc_timecourse",
    )

    plot_delta_timecourse(
        bin_df,
        key="delta_choice_auc",
        ylabel="Δ choice AUC = top-k noise decoder - signal-only decoder",
        filename="delta_choice_auc_timecourse",
    )

    plot_delta_timecourse(
        bin_df,
        key="delta_feedback_auc",
        ylabel="Δ feedback AUC = top-k noise decoder - signal-only decoder",
        filename="delta_feedback_auc_timecourse",
    )

    corr_df = compute_session_correlations(bin_df)
    plot_correlation(corr_df)
    plot_weight_fractions(bin_df)
    
    print(f"\nSaved plots to {PLOT_DIR}/")


if __name__ == "__main__":
    main()