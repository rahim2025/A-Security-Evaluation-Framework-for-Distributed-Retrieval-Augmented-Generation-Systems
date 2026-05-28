"""
visualizer.py
-------------
Generates publication-quality scalability plots from the master CSV.

Each plotting function produces one PNG in scalability_tests/plots/.
Call plot_all() to generate every chart in one shot.
"""
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server / CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

from scalability_tests.metrics_collector import load_results, DRAG_METRIC_COLS

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)

# Helper: safely parse a float from a result dict
def _f(row: Dict, key: str, default=np.nan) -> float:
    try:
        v = row.get(key, default)
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# 1. Volume Scaling — accuracy & latency vs. num_samples
# ---------------------------------------------------------------------------
def plot_volume_scaling() -> Path:
    rows = load_results("volume")
    if not rows:
        print("[visualizer] No volume results — skipping.")
        return None

    rows.sort(key=lambda r: _f(r, "label"))
    labels = [_f(r, "label") for r in rows]
    elapsed = [_f(r, "elapsed_s") for r in rows]
    exact   = [_f(r, "exact_match") for r in rows]
    f1      = [_f(r, "f1") for r in rows]
    mem     = [_f(r, "peak_memory_mb") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Scalability — Dataset Volume (num_samples)", fontsize=14, fontweight="bold")

    # Panel 1: Accuracy metrics
    axes[0].plot(labels, exact, marker="o", label="Exact Match")
    axes[0].plot(labels, f1,    marker="s", label="F1")
    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy vs Volume")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Panel 2: Elapsed time
    axes[1].bar(labels, elapsed, color=sns.color_palette()[1], edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Number of Samples")
    axes[1].set_ylabel("Elapsed Time (s)")
    axes[1].set_title("Runtime vs Volume")

    # Panel 3: Peak memory
    axes[2].bar(labels, mem, color=sns.color_palette()[2], edgecolor="black", linewidth=0.5)
    axes[2].set_xlabel("Number of Samples")
    axes[2].set_ylabel("Peak Memory (MB)")
    axes[2].set_title("Memory vs Volume")

    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    out = PLOTS_DIR / "01_volume_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualizer] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Network Scaling — accuracy & latency vs. num_peers
# ---------------------------------------------------------------------------
def plot_network_scaling() -> Path:
    rows = load_results("network")
    if not rows:
        print("[visualizer] No network results — skipping.")
        return None

    rows.sort(key=lambda r: _f(r, "label"))
    labels  = [_f(r, "label") for r in rows]
    elapsed = [_f(r, "elapsed_s") for r in rows]
    exact   = [_f(r, "exact_match") for r in rows]
    hops    = [_f(r, "avg_num_hops") for r in rows]
    msgs    = [_f(r, "avg_num_messages") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Scalability — Network Size (num_peers)", fontsize=14, fontweight="bold")

    axes[0].plot(labels, exact, marker="o", color=sns.color_palette()[0])
    axes[0].set_xlabel("Number of Peers")
    axes[0].set_ylabel("Exact Match")
    axes[0].set_title("Accuracy vs Network Size")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(labels, hops, marker="D", label="Avg Hops",    color=sns.color_palette()[3])
    axes[1].plot(labels, msgs, marker="^", label="Avg Messages", color=sns.color_palette()[4])
    axes[1].set_xlabel("Number of Peers")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Query Cost vs Network Size")
    axes[1].legend()

    axes[2].plot(labels, elapsed, marker="s", color=sns.color_palette()[1])
    axes[2].set_xlabel("Number of Peers")
    axes[2].set_ylabel("Elapsed Time (s)")
    axes[2].set_title("Runtime vs Network Size")

    plt.tight_layout()
    out = PLOTS_DIR / "02_network_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualizer] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Attack Intensity — accuracy degradation vs. poisoning ratio
# ---------------------------------------------------------------------------
def plot_attack_scaling() -> Path:
    rows = load_results("attack")
    if not rows:
        print("[visualizer] No attack results — skipping.")
        return None

    rows.sort(key=lambda r: _f(r, "label"))
    labels = [r.get("label", "?") for r in rows]
    exact  = [_f(r, "exact_match") for r in rows]
    f1     = [_f(r, "f1") for r in rows]
    sem    = [_f(r, "semantic_similarity") for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scalability — Attack Intensity (poisoning_ratio)", fontsize=14, fontweight="bold")

    axes[0].plot(labels, exact, marker="o", label="Exact Match", color="crimson")
    axes[0].plot(labels, f1,   marker="s", label="F1",          color="darkorange")
    axes[0].plot(labels, sem,  marker="^", label="Semantic Sim", color="steelblue")
    axes[0].set_xlabel("Poisoning Ratio (%)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy Degradation vs Attack Intensity")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Show performance drop relative to 0% attack
    baseline_exact = exact[0] if exact else 0
    drops = [baseline_exact - v for v in exact]
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in drops]
    axes[1].bar(labels, drops, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Poisoning Ratio (%)")
    axes[1].set_ylabel("Accuracy Drop vs Baseline")
    axes[1].set_title("Exact Match Drop vs Attack Intensity")

    plt.tight_layout()
    out = PLOTS_DIR / "03_attack_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualizer] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Dataset Diversity — side-by-side per dataset
# ---------------------------------------------------------------------------
def plot_dataset_comparison() -> Path:
    rows = load_results("dataset")
    if not rows:
        print("[visualizer] No dataset results — skipping.")
        return None

    labels  = [r.get("label", "?") for r in rows]
    metrics = ["exact_match", "f1", "semantic_similarity", "avg_num_hops"]
    titles  = ["Exact Match", "F1 Score", "Semantic Similarity", "Avg Hops"]

    x      = np.arange(len(labels))
    width  = 0.6
    colors = sns.color_palette("tab10", len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Scalability — Dataset Diversity", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [_f(r, metric) for r in rows]
        bars = ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(title)
        if metric != "avg_num_hops":
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = PLOTS_DIR / "04_dataset_diversity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualizer] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Comprehensive heatmap — all DRAG metrics across all experiments
# ---------------------------------------------------------------------------
def plot_metric_heatmap() -> Path:
    rows = load_results()
    if not rows:
        print("[visualizer] No data — skipping heatmap.")
        return None

    metric_cols = [c for c in DRAG_METRIC_COLS if any(r.get(c) not in ("", None) for r in rows)]
    if not metric_cols:
        print("[visualizer] No metric columns found — skipping heatmap.")
        return None

    exp_ids = [r.get("experiment_id", "?") for r in rows]
    matrix  = np.array([[_f(r, c) for c in metric_cols] for r in rows])

    # Normalize each column to [0, 1] for visual comparison
    col_min  = np.nanmin(matrix, axis=0, keepdims=True)
    col_max  = np.nanmax(matrix, axis=0, keepdims=True)
    col_rng  = col_max - col_min
    col_rng[col_rng == 0] = 1
    matrix_n = (matrix - col_min) / col_rng

    fig, ax = plt.subplots(figsize=(max(10, len(metric_cols) * 1.2), max(6, len(exp_ids) * 0.4)))
    sns.heatmap(
        matrix_n,
        ax=ax,
        xticklabels=metric_cols,
        yticklabels=exp_ids,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Normalized Score (column-wise)"},
    )
    ax.set_title("All Metrics Across All Experiments (column-normalized)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Experiment")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()

    out = PLOTS_DIR / "05_metric_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[visualizer] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def plot_all() -> List[Path]:
    """Generate all scalability plots. Returns list of saved paths."""
    paths = []
    for fn in [
        plot_volume_scaling,
        plot_network_scaling,
        plot_attack_scaling,
        plot_dataset_comparison,
        plot_metric_heatmap,
    ]:
        result = fn()
        if result:
            paths.append(result)
    print(f"\n[visualizer] {len(paths)} plot(s) saved in {PLOTS_DIR}")
    return paths
