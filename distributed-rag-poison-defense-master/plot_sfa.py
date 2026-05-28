import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/selective_forwarding/results.csv")

strategies = ["random", "high_connectivity"]
colors = {"random": "steelblue", "high_connectivity": "crimson"}
labels = {"random": "Random", "high_connectivity": "Hub-Targeted"}

# --- Plot 1: Hit Rate ---
fig, ax = plt.subplots(figsize=(7, 4))
for s in strategies:
    d = df[df["strategy"] == s].sort_values("attack_ratio")
    ax.plot(d["attack_ratio"], d["hit_rate"], marker="o", label=labels[s], color=colors[s])
ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, label="Baseline")
ax.set_xlabel("Compromise Ratio")
ax.set_ylabel("Query Hit Rate")
ax.set_title("SFA: Query Hit Rate vs Compromise Ratio")
ax.legend(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logs/selective_forwarding/sfa_hitrate.png", dpi=150)
print("Saved sfa_hitrate.png")

# --- Plot 2: Dropped Queries ---
fig, ax = plt.subplots(figsize=(7, 4))
for s in strategies:
    d = df[df["strategy"] == s].sort_values("attack_ratio")
    offset = 0.01 if s == "high_connectivity" else -0.01
    ax.bar(d["attack_ratio"] + offset, d["dropped_queries"], width=0.02,
           label=labels[s], color=colors[s], alpha=0.8)
ax.set_xlabel("Compromise Ratio")
ax.set_ylabel("Dropped Queries")
ax.set_title("SFA: Dropped Queries vs Compromise Ratio")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("logs/selective_forwarding/sfa_dropped.png", dpi=150)
print("Saved sfa_dropped.png")

# --- Plot 3: Avg Hops ---
fig, ax = plt.subplots(figsize=(7, 4))
for s in strategies:
    d = df[df["strategy"] == s].sort_values("attack_ratio")
    ax.plot(d["attack_ratio"], d["avg_hops_per_query"], marker="s",
            label=labels[s], color=colors[s])
ax.set_xlabel("Compromise Ratio")
ax.set_ylabel("Avg Hops per Query")
ax.set_title("SFA: Routing Overhead vs Compromise Ratio")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logs/selective_forwarding/sfa_hops.png", dpi=150)
print("Saved sfa_hops.png")