
import argparse, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

STRATEGY_STYLE = {
    "random":         dict(color="#2166ac", marker="o", linestyle="-",  label="Random SFA"),
    "high_ssm_score": dict(color="#d6604d", marker="s", linestyle="--", label="High-SSM SFA"),
}
BASELINE_COLOR = "#4dac26"
plt.rcParams.update({"font.family":"serif","font.size":12,"axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":10,"figure.dpi":150})

def load_seeds(seeds, log_dir):
    frames = []
    for s in seeds:
        p = log_dir / f"results_seed{s}.csv"
        df = pd.read_csv(p); df["seed"] = s; frames.append(df)
    return pd.concat(frames, ignore_index=True)

def agg(df):
    return df.groupby(["strategy","attack_ratio"]).agg(
        hit_rate_mean=("hit_rate","mean"), hit_rate_std=("hit_rate","std"),
        ttl_mean=("ttl_exhaustion_rate","mean"), ttl_std=("ttl_exhaustion_rate","std"),
        dropped_mean=("dropped_queries","mean"), dropped_std=("dropped_queries","std"),
        num_comp=("num_compromised","first"),
    ).reset_index()

def plot_combined(agg_df, baseline_hit, baseline_ttl, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (mean, std, base, ylabel, title) in zip(axes, [
        ("hit_rate_mean","hit_rate_std",baseline_hit,"Query Hit Rate","(a) Hit Rate vs. Attack Ratio"),
        ("ttl_mean","ttl_std",baseline_ttl,"TTL Exhaustion Rate","(b) TTL Exhaustion vs. Attack Ratio"),
    ]):
        for strategy, style in STRATEGY_STYLE.items():
            sub = agg_df[agg_df["strategy"]==strategy].sort_values("attack_ratio")
            ax.errorbar(sub["attack_ratio"], sub[mean], yerr=sub[std].fillna(0), capsize=4, capthick=1.4, linewidth=1.8, **style)
        ax.axhline(base, color=BASELINE_COLOR, linestyle=":", linewidth=1.6, label="Baseline")
        ax.set_xlabel("Attack ratio"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(axis="y", alpha=0.35)
    fig.suptitle("Selective Forwarding Attack on Reliable-dRAG  (Mean +/- std, 3 seeds x 500 queries)", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out_path}")

def plot_metric(agg_df, mean, std, ylabel, title, out_path, baseline_val):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, style in STRATEGY_STYLE.items():
        sub = agg_df[agg_df["strategy"]==strategy].sort_values("attack_ratio")
        ax.errorbar(sub["attack_ratio"], sub[mean], yerr=sub[std].fillna(0), capsize=4, linewidth=1.8, **style)
    ax.axhline(baseline_val, color=BASELINE_COLOR, linestyle=":", linewidth=1.6, label="Baseline")
    ax.set_xlabel("Attack ratio"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=0.35); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out_path}")

def plot_dropped(agg_df, out_path):
    ratios = sorted(r for r in agg_df["attack_ratio"].unique() if r > 0)
    x = np.arange(len(ratios)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (strategy, style) in enumerate(STRATEGY_STYLE.items()):
        sub = agg_df[(agg_df["strategy"]==strategy) & (agg_df["attack_ratio"]>0)].sort_values("attack_ratio")
        ax.bar(x+(i-0.5)*w, sub["dropped_mean"], w, label=style["label"], color=style["color"], alpha=0.8, yerr=sub["dropped_std"].fillna(0), capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f"{r:.2f}" for r in ratios])
    ax.set_xlabel("Attack ratio"); ax.set_ylabel("Dropped queries"); ax.set_title("Queries Dropped by SFA")
    ax.legend(); ax.grid(axis="y", alpha=0.35); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
parser.add_argument("--log_dir", default="attack_logs/selective_forwarding")
parser.add_argument("--out_dir", default="attack_logs/selective_forwarding/figures")
args = parser.parse_args()
log_dir = pathlib.Path(args.log_dir); out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
raw = load_seeds(args.seeds, log_dir)
agg_df = agg(raw)
base_rows = raw[raw["attack_ratio"]==0.0]
baseline_hit = base_rows["hit_rate"].mean(); baseline_ttl = base_rows["ttl_exhaustion_rate"].mean()
print(f"Baseline hit_rate={baseline_hit:.3f}  ttl_exhaust={baseline_ttl:.3f}")
plot_df = agg_df[agg_df["strategy"]!="baseline"]
plot_metric(plot_df,"hit_rate_mean","hit_rate_std","Query Hit Rate","Hit Rate vs Attack Ratio",out_dir/"sfa_hit_rate.png",baseline_hit)
plot_metric(plot_df,"ttl_mean","ttl_std","TTL Exhaustion Rate","TTL Exhaustion vs Attack Ratio",out_dir/"sfa_ttl_exhaustion.png",baseline_ttl)
plot_dropped(plot_df, out_dir/"sfa_dropped_queries.png")
plot_combined(plot_df, baseline_hit, baseline_ttl, out_dir/"sfa_combined.png")
print("Done. Figures in:", out_dir)
