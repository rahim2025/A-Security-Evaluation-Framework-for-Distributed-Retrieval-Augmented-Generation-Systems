"""
metrics_collector.py
--------------------
Aggregates per-experiment result dicts into a master CSV file and
provides helpers for loading / filtering that CSV for analysis.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV = RESULTS_DIR / "scalability_results.csv"


# Key DRAG metric columns we care about for scalability analysis
DRAG_METRIC_COLS = [
    "exact_match",
    "f1",
    "precision",
    "recall",
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "semantic_similarity",
    "avg_num_hops",
    "avg_num_messages",
    "avg_query_hit",
]

# Our own instrumentation columns
INFRA_COLS = [
    "experiment_id",
    "dimension",
    "label",
    "elapsed_s",
    "peak_memory_mb",
    "status",
    "run_timestamp",
]

ALL_COLS = INFRA_COLS + DRAG_METRIC_COLS


def save_results(results: List[Dict], append: bool = True) -> Path:
    """
    Write/append a list of result dicts to the master CSV.

    Parameters
    ----------
    results : list of dicts returned by runner.run_experiment()
    append  : if True and file exists, append rows; otherwise overwrite

    Returns
    -------
    Path to the master CSV
    """
    ts = datetime.now().isoformat(timespec="seconds")
    for r in results:
        r.setdefault("run_timestamp", ts)

    mode = "a" if (append and MASTER_CSV.exists()) else "w"
    write_header = mode == "w" or not MASTER_CSV.exists()

    with open(MASTER_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"[metrics_collector] Saved {len(results)} rows → {MASTER_CSV}")
    return MASTER_CSV


def load_results(dimension: str = None) -> List[Dict]:
    """
    Load results from the master CSV, optionally filtered by dimension.

    Parameters
    ----------
    dimension : 'volume' | 'network' | 'attack' | 'dataset' | None (all)
    """
    if not MASTER_CSV.exists():
        return []
    with open(MASTER_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if dimension:
        rows = [r for r in rows if r.get("dimension") == dimension]
    # Cast numeric columns
    for row in rows:
        for col in ["elapsed_s", "peak_memory_mb"] + DRAG_METRIC_COLS:
            if col in row and row[col] not in ("", None):
                try:
                    row[col] = float(row[col])
                except ValueError:
                    pass
    return rows


def summarise(dimension: str = None) -> None:
    """Pretty-print a summary table to stdout."""
    rows = load_results(dimension)
    if not rows:
        print("No results found.")
        return
    header = f"{'ID':<25} {'label':<12} {'elapsed_s':>10} {'mem_MB':>8} {'exact_match':>12} {'f1':>8} {'status'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('experiment_id','?'):<25} "
            f"{str(r.get('label','?')):<12} "
            f"{str(r.get('elapsed_s','?')):>10} "
            f"{str(r.get('peak_memory_mb','?')):>8} "
            f"{str(r.get('exact_match','?')):>12} "
            f"{str(r.get('f1','?')):>8} "
            f"{r.get('status','?')}"
        )


def export_json(out_path: Path = None) -> Path:
    """Export the master CSV as pretty-printed JSON (useful for dashboards)."""
    rows = load_results()
    out_path = out_path or RESULTS_DIR / "scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[metrics_collector] JSON exported → {out_path}")
    return out_path
