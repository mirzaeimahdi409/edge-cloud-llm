#!/usr/bin/env python
"""Section 13 (optional): pandas/matplotlib charts from the CSVs produced by
run_evaluation.py and run_sweep.py. Run those two scripts first."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_comparison() -> None:
    path = RESULTS_DIR / "comparison.csv"
    if not path.exists():
        print(f"Skipping comparison plots: {path} not found (run scripts/run_evaluation.py first).")
        return

    df = pd.read_csv(path)
    prompts = df["prompt"].unique()
    x = range(len(prompts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, policy in zip([-width / 2, width / 2], ["always_edge", "threshold_policy"]):
        values = [df[(df.prompt == p) & (df.policy == policy)]["latency_seconds"].iloc[0] for p in prompts]
        ax.bar([i + offset for i in x], values, width, label=policy)
    ax.set_xticks(list(x))
    ax.set_xticklabels([p[:20] + ("…" if len(p) > 20 else "") for p in prompts], rotation=20, ha="right")
    ax.set_ylabel("latency (seconds)")
    ax.set_title("always-edge vs. threshold policy: latency per prompt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    switch_counts = df.groupby("policy")["switch_count"].sum()
    ax.bar(switch_counts.index, switch_counts.values, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("total switches (across all prompts)")
    ax.set_title("always-edge vs. threshold policy: switch count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "comparison_switches.png", dpi=150)
    plt.close(fig)

    print(f"Wrote comparison_latency.png and comparison_switches.png to {PLOTS_DIR}")


def plot_sweep() -> None:
    path = RESULTS_DIR / "sweep.csv"
    if not path.exists():
        print(f"Skipping sweep plots: {path} not found (run scripts/run_sweep.py first).")
        return

    df = pd.read_csv(path)
    grouped = df.groupby(["threshold", "window_size"]).agg(
        latency_seconds=("latency_seconds", "mean"), switch_count=("switch_count", "mean")
    )

    for metric, title, filename in [
        ("latency_seconds", "Mean latency (s) by threshold (τ) and window size (k)", "sweep_latency_heatmap.png"),
        ("switch_count", "Mean switch count by threshold (τ) and window size (k)", "sweep_switches_heatmap.png"),
    ]:
        pivot = grouped[metric].unstack("window_size")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("window size (k)")
        ax.set_ylabel("threshold (τ)")
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    print(f"Wrote sweep_latency_heatmap.png and sweep_switches_heatmap.png to {PLOTS_DIR}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison()
    plot_sweep()


if __name__ == "__main__":
    main()
