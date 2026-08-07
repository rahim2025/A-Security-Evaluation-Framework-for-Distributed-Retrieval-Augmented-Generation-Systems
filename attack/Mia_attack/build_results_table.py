"""
attack/Mia_attack/build_results_table.py

Recomputes the "Mean (n=N)" / "Stdev (n=N)" aggregate rows in
tables/mia_results.csv from the individual per-seed rows already in that
file, for every section that already has such an aggregate pair (currently
"Production (thesis multi-seed)" and "Ablation-eval (held-out seeds)" --
the single-seed "999 (quick-check)" row has no aggregate and is left alone).

This exists because that table was hand-typed, not pipeline-generated: the
Mean/Stdev rows previously had TPR/FPR/Precision/Privacy Risk (and the
similarity diagnostics) left blank even though every ingredient needed to
compute them was already sitting in the seed rows above. Run this after
adding, editing, or appending a seed's row so the aggregate rows can't go
stale or get partially hand-typed again.

Usage
-----
    python attack/Mia_attack/build_results_table.py
    python attack/Mia_attack/build_results_table.py --check   # verify only, exit 1 if stale
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.abspath(os.path.join(_HERE, "..", "..", "tables", "mia_results.csv"))

_SECTION_COL = "Section"
_SEED_COL = "Seed"
_PRIVACY_RISK_COL = "Privacy Risk"
_AUC_PRIMARY_COL = "AUC-ROC (Primary Composite)"

# Config/metadata columns -- never averaged like a measured metric.
_CARRY_BOTH = {"Role"}                                    # same constant shown in Mean AND Stdev rows
_CARRY_MEAN_ONLY = {"N Members", "N Non-Members", "Probes/Doc"}  # constant in Mean row; Stdev row is always n/a

# TP/TN/FP/FN use 1 decimal place in the Mean row (existing table convention);
# every other numeric column (rates, AUC variants, similarity diagnostics)
# uses 4. Stdev rows use 4 decimals uniformly, including TP/TN/FP/FN.
_COUNT_COLS = {"TP", "TN", "FP", "FN"}


def _privacy_risk_label(auc: float) -> str:
    """Mirrors attack/Mia_attack/mia_attack.py's _privacy_risk() thresholds exactly
    (em-dash swapped for a plain hyphen to match this CSV's existing text style)."""
    if auc > 0.75:
        return "HIGH - significant privacy risk"
    if auc > 0.60:
        return "MEDIUM - moderate privacy concern"
    if auc > 0.50:
        return "LOW - slight privacy risk"
    return "NEGLIGIBLE - attack ineffective (AUC ~ random)"


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "n/a":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _is_aggregate_seed(seed: str) -> bool:
    return seed.startswith("Mean") or seed.startswith("Stdev")


def regenerate(rows: List[Dict[str, str]], fieldnames: List[str]) -> List[Dict[str, str]]:
    non_numeric = _CARRY_BOTH | _CARRY_MEAN_ONLY | {_SECTION_COL, _SEED_COL, _PRIVACY_RISK_COL}
    numeric_cols = [c for c in fieldnames if c not in non_numeric]

    section_order: List[str] = []
    section_rows: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        sect = row[_SECTION_COL]
        if sect not in section_rows:
            section_rows[sect] = []
            section_order.append(sect)
        section_rows[sect].append(i)

    out = list(rows)

    for sect in section_order:
        idxs = section_rows[sect]
        seed_idxs = [i for i in idxs if not _is_aggregate_seed(rows[i][_SEED_COL])]
        mean_idx = next((i for i in idxs if rows[i][_SEED_COL].startswith("Mean")), None)
        stdev_idx = next((i for i in idxs if rows[i][_SEED_COL].startswith("Stdev")), None)
        if mean_idx is None and stdev_idx is None:
            continue  # no aggregate rows to maintain for this section (e.g. the quick-check row)
        if not seed_idxs:
            continue

        n = len(seed_idxs)
        mean_row = dict(out[mean_idx]) if mean_idx is not None else None
        stdev_row = dict(out[stdev_idx]) if stdev_idx is not None else None

        for col in numeric_cols:
            values = [v for v in (_to_float(rows[i][col]) for i in seed_idxs) if v is not None]
            decimals = 1 if col in _COUNT_COLS else 4
            if mean_row is not None:
                mean_row[col] = _fmt(statistics.mean(values), decimals) if values else "n/a"
            if stdev_row is not None:
                stdev_row[col] = _fmt(statistics.stdev(values), 4) if len(values) >= 2 else "n/a"

        for col in _CARRY_BOTH:
            const_values = {rows[i][col] for i in seed_idxs}
            carried = const_values.pop() if len(const_values) == 1 else "n/a"
            if mean_row is not None:
                mean_row[col] = carried
            if stdev_row is not None:
                stdev_row[col] = carried

        for col in _CARRY_MEAN_ONLY:
            const_values = {rows[i][col] for i in seed_idxs}
            carried = const_values.pop() if len(const_values) == 1 else "n/a"
            if mean_row is not None:
                mean_row[col] = carried
            if stdev_row is not None:
                stdev_row[col] = "n/a"  # spread of a fixed experimental parameter isn't meaningful

        # Only derive a Privacy Risk label for the Mean row if this section's own
        # per-seed rows actually carry one -- e.g. the Ablation-eval section never
        # assigns Privacy Risk (it compares scoring-composite variants, it isn't a
        # privacy verdict on a deployed attack config), so its Mean row must stay
        # n/a too rather than fabricate a label the underlying study never made.
        section_has_privacy_risk = any(
            rows[i][_PRIVACY_RISK_COL].strip().lower() != "n/a" for i in seed_idxs
        )
        if mean_row is not None:
            if section_has_privacy_risk:
                auc = _to_float(mean_row.get(_AUC_PRIMARY_COL))
                mean_row[_PRIVACY_RISK_COL] = _privacy_risk_label(auc) if auc is not None else "n/a"
            else:
                mean_row[_PRIVACY_RISK_COL] = "n/a"
            mean_row[_SEED_COL] = f"Mean (n={n})"
            out[mean_idx] = mean_row
        if stdev_row is not None:
            # Privacy Risk is a category label ("LOW", "MEDIUM", ...), not a number --
            # a standard deviation across labels is meaningless, so this stays n/a.
            stdev_row[_PRIVACY_RISK_COL] = "n/a"
            stdev_row[_SEED_COL] = f"Stdev (n={n})"
            out[stdev_idx] = stdev_row

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=_DEFAULT_CSV)
    p.add_argument("--check", action="store_true", help="don't write -- exit 1 if aggregate rows are stale")
    args = p.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    new_rows = regenerate(rows, fieldnames)

    if new_rows == rows:
        print("Aggregate rows already up to date -- no changes.")
        return

    if args.check:
        for i, (old, new) in enumerate(zip(rows, new_rows)):
            if old != new:
                print(f"STALE row {i} (Seed={old.get('Seed')!r}):")
                for col in fieldnames:
                    if old.get(col) != new.get(col):
                        print(f"    {col}: {old.get(col)!r} -> {new.get(col)!r}")
        raise SystemExit(1)

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"Updated aggregate rows in {args.csv}")


if __name__ == "__main__":
    main()
