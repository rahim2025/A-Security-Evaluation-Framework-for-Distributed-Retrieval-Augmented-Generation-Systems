"""
runner.py
---------
Runs each ExperimentConfig as a subprocess call to simulator.py,
captures wall-clock time and peak memory, then collects the latest
metrics CSV written by DRAG's own ExpLogger.
"""
import csv
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import psutil

from scalability_tests.config_generator import ExperimentConfig, build_cli_args


DRAG_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = DRAG_ROOT / "logs"
RESULTS_DIR = DRAG_ROOT / "scalability_tests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _latest_metrics_csv() -> Optional[Path]:
    """Return the metrics CSV from the most-recently-created log version."""
    if not LOGS_DIR.exists():
        return None
    versions = sorted(LOGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for v in versions:
        csv_path = v / "metrics.csv"
        if csv_path.exists():
            return csv_path
    return None


def _read_metrics_csv(path: Path) -> dict:
    """Read the last row of a DRAG metrics CSV and return it as a dict."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[-1]


def _peak_memory_mb(proc: subprocess.Popen) -> float:
    """Poll child process memory (RSS) every 0.5 s and return peak in MB."""
    peak = 0.0
    try:
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                mem = ps.memory_info().rss / (1024 ** 2)
                if mem > peak:
                    peak = mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.5)
    except Exception:
        pass
    return peak


def run_experiment(exp: ExperimentConfig, timeout: int = 600) -> dict:
    """
    Execute one DRAG experiment via subprocess and return a result dict.

    Parameters
    ----------
    exp     : ExperimentConfig to run
    timeout : max seconds before the subprocess is killed (default 10 min)

    Returns
    -------
    dict with keys: experiment_id, dimension, label, elapsed_s,
                    peak_memory_mb, status, + all DRAG metric columns
    """
    cli_args = build_cli_args(exp)
    cmd = [sys.executable, str(DRAG_ROOT / "simulator.py")] + cli_args

    print(f"\n{'='*65}")
    print(f"[{exp.dimension.upper()}] {exp.experiment_id}  →  {exp.label}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*65}")

    result: dict = {
        "experiment_id": exp.experiment_id,
        "dimension": exp.dimension,
        "label": exp.label,
        "elapsed_s": None,
        "peak_memory_mb": None,
        "status": "pending",
    }

    prev_csv = _latest_metrics_csv()

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(DRAG_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        peak_mem = 0.0
        output_lines = []

        # Stream output + poll memory simultaneously
        import threading

        def _memory_thread():
            nonlocal peak_mem
            peak_mem = _peak_memory_mb(proc)

        mem_thread = threading.Thread(target=_memory_thread, daemon=True)
        mem_thread.start()

        try:
            stdout, _ = proc.communicate(timeout=timeout)
            output_lines = stdout.splitlines()
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            output_lines = stdout.splitlines()
            result["status"] = "timeout"
            print("[WARN] Experiment timed out — partial results may be available.")

        mem_thread.join(timeout=3)

        elapsed = time.perf_counter() - t0
        result["elapsed_s"] = round(elapsed, 2)
        result["peak_memory_mb"] = round(peak_mem, 1)

        # Print tail of subprocess output for visibility
        tail = output_lines[-30:] if len(output_lines) > 30 else output_lines
        for line in tail:
            print(line)

        if result["status"] == "pending":
            result["status"] = "ok" if proc.returncode == 0 else f"error:{proc.returncode}"

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        result["elapsed_s"] = round(elapsed, 2)
        result["status"] = f"exception:{exc}"
        traceback.print_exc()

    # Collect DRAG metrics from the log written by this run
    new_csv = _latest_metrics_csv()
    if new_csv and new_csv != prev_csv:
        metrics = _read_metrics_csv(new_csv)
        result.update(metrics)
        print(f"[✓] Metrics loaded from {new_csv}")
    elif new_csv == prev_csv:
        print("[!] No new metrics CSV detected — simulator may not have completed a full run.")

    print(f"[DONE] {exp.experiment_id} — {result['elapsed_s']}s  |  "
          f"mem={result['peak_memory_mb']} MB  |  status={result['status']}")
    return result
