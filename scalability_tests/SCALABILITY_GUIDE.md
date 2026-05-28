# DRAG Scalability Testing Guide

This module stress-tests the DRAG Distributed RAG system across **4 independent scaling dimensions**, collects timing and accuracy metrics, and auto-generates comparison plots.

---

## What is tested

| Dimension | What changes | Config knob |
|---|---|---|
| **Volume** | Number of QA samples loaded | `data.num_samples` |
| **Network** | Number of peer nodes | `rag.num_peers` |
| **Attack** | Fraction of malicious peers | `security.poisoning_ratio` |
| **Dataset** | Entire dataset (MMLU / Medical / News) | `config/data/*.yaml` |

---

## Quick start

### 1. Install dependencies (once)
```bash
pip install -r requirements.txt
pip install psutil          # for memory profiling (not in original requirements)
```

### 2. Make sure Ollama is running with your model
```bash
ollama run llama3.2:3b
```

### 3. Run the full suite
```bash
# From the DRAG-main/ directory:
python -m scalability_tests.run_all
```

### 4. Run a fast smoke test (tiny configs, verifies the pipeline works)
```bash
python -m scalability_tests.run_all --quick
```

### 5. Run only specific dimensions
```bash
python -m scalability_tests.run_all --dimensions volume attack
```

### 6. Regenerate plots without re-running experiments
```bash
python -m scalability_tests.run_all --plots-only
```

### 7. Print a summary table of saved results
```bash
python -m scalability_tests.run_all --summarise
```

---

## Output files

```
scalability_tests/
├── results/
│   ├── scalability_results.csv   ← master table, one row per experiment
│   └── scalability_results.json  ← same data in JSON format
└── plots/
    ├── 01_volume_scaling.png     ← accuracy + time + memory vs num_samples
    ├── 02_network_scaling.png    ← accuracy + hops + time vs num_peers
    ├── 03_attack_scaling.png     ← accuracy degradation vs poisoning ratio
    ├── 04_dataset_diversity.png  ← side-by-side comparison across datasets
    └── 05_metric_heatmap.png     ← all metrics × all experiments (heatmap)
```

---

## How it works

```
run_all.py
│
├── config_generator.py  → builds CLI arg lists for each parameter combo
├── runner.py            → spawns simulator.py as subprocess, tracks time & RAM
├── metrics_collector.py → reads DRAG's own metrics.csv, saves to master CSV
└── visualizer.py        → reads master CSV, produces matplotlib/seaborn PNGs
```

Each experiment runs `simulator.py` with overridden CLI flags, e.g.:

```bash
python simulator.py \
    --config config/data/mmlu.yaml \
    --data.num_samples 100 \
    --rag.num_peers 20 \
    --security.enable_attack True \
    --security.poisoning_ratio 0.3
```

Results are **appended** to the master CSV after each run, so you never lose progress if the suite is interrupted.

---

## Metrics collected

### Performance / accuracy (from DRAG's own evaluator)
| Metric | Description |
|---|---|
| `exact_match` | Exact string match between prediction and reference |
| `f1` | Token-level F1 score |
| `precision` / `recall` | Token overlap precision and recall |
| `bleu` | BLEU-4 score |
| `rouge1/2/L` | ROUGE-N overlap |
| `semantic_similarity` | Cosine similarity of sentence embeddings |
| `avg_num_hops` | Average query hops (network cost) |
| `avg_num_messages` | Average messages per query (bandwidth cost) |
| `avg_query_hit` | Fraction of queries successfully answered |

### Infrastructure (added by this suite)
| Metric | Description |
|---|---|
| `elapsed_s` | Wall-clock time for the full experiment |
| `peak_memory_mb` | Peak RSS memory of the simulator process |
| `status` | `ok` / `error:N` / `timeout` / `exception:...` |

---

## How to interpret the plots

### Plot 1 — Volume Scaling
- **Accuracy flat or rising** → model generalises as data grows ✓
- **Accuracy dropping** → knowledge base overloaded, retrieval degrades ✗
- **Runtime linear** → healthy scaling ✓  |  **Super-linear** → bottleneck ✗

### Plot 2 — Network Scaling
- **Avg hops rising faster than peers** → routing overhead is growing ✗
- **Accuracy stable** → distributed search handles larger networks ✓

### Plot 3 — Attack Intensity
- **Accuracy drops sharply above a threshold** → that is your "breaking point"
- Compare with defense enabled (`config/defense.yaml: enabled: true`) to see the protection margin

### Plot 4 — Dataset Diversity
- Large variation in `exact_match` across datasets → system is dataset-sensitive
- Check `avg_num_hops`: topic-diverse datasets (MMLU) should need fewer hops

### Plot 5 — Heatmap
- Red cells = relative high scores (good for quality metrics, bad for hop counts)
- Outlier rows (all-red or all-blue) indicate unstable experiments

---

## Adding a new dimension

1. Open `config_generator.py`
2. Add a new list of parameter values and a new `*_experiments()` function
3. Add the function to `DIMENSION_MAP` in `run_all.py`
4. Add a new `plot_*()` function in `visualizer.py` and call it from `plot_all()`

---

## Scalability "pass" criteria (suggested thresholds)

| Metric | Healthy range |
|---|---|
| Accuracy drop over full volume range | < 5 percentage points |
| Runtime growth | ≤ linear (O(n)) |
| Memory growth | ≤ linear (O(n)) |
| Accuracy at 50% poisoning vs 0% | < 20 pp drop (with defense enabled) |
| Accuracy variation across datasets | < 15 pp range |
