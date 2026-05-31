"""Security evaluation matrix writers.

Extended with DRAG-compatible system-level columns so that both DRAG and
FedRAG can be compared on the same matrix.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------
# Core columns (shared across all evaluators)
# ------------------------------------------------------------------

_MATRIX_COLUMNS_CORE = [
    "Attack",
    "System",
    "Dataset",
    "Model",
    "Baseline BLEU",
    "Post-Attack BLEU",
    "Defended BLEU",
    "Delta BLEU",
    "Baseline EM",
    "Post-Attack EM",
    "Defended EM",
    "Delta EM",
    "Baseline F1",
    "Post-Attack F1",
    "Defended F1",
    "Delta F1",
    "Membership Acc",
    "Defended Membership Acc",
    "KB Recovery %",
    "Defended KB Recovery %",
    "Availability %",
    "Post-Attack Availability %",
    "Defended Availability %",
    "Byzantine Nodes",
    "Sybil Nodes",
    "Runtime (s)",
    "Memory Overhead MB",
    "Notes",
]

# ------------------------------------------------------------------
# Extended columns (system-level, DRAG-compatible)
# ------------------------------------------------------------------

_MATRIX_COLUMNS_EXTENDED = [
    "Baseline Precision",
    "Post-Attack Precision",
    "Defended Precision",
    "Delta Precision",
    "Baseline Recall",
    "Post-Attack Recall",
    "Defended Recall",
    "Delta Recall",
    "Baseline Rouge1",
    "Post-Attack Rouge1",
    "Defended Rouge1",
    "Delta Rouge1",
    "Baseline Rouge2",
    "Post-Attack Rouge2",
    "Defended Rouge2",
    "Delta Rouge2",
    "Baseline RougeL",
    "Post-Attack RougeL",
    "Defended RougeL",
    "Delta RougeL",
    "Baseline Semantic Sim",
    "Post-Attack Semantic Sim",
    "Defended Semantic Sim",
    "Delta Semantic Sim",
    "Query Failure Rate Baseline",
    "Query Failure Rate Post-Attack",
    "Query Failure Rate Defended",
]

MATRIX_COLUMNS = _MATRIX_COLUMNS_CORE + _MATRIX_COLUMNS_EXTENDED


def write_evaluation_matrix(
    records: list[dict[str, Any]],
    markdown_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write evaluation records as Markdown and optionally CSV.

    Dynamically determines column order: shared columns first, then any
    additional keys found in *records*.
    """
    # Gather all unique keys that appear in at least one record
    extra_keys: set[str] = set()
    for record in records:
        extra_keys.update(record.keys())
    extra_keys.discard("")

    # Build ordered list: MATRIX_COLUMNS first, then remaining extras
    ordered: list[str] = []
    seen: set[str] = set()
    for col in MATRIX_COLUMNS:
        if col in extra_keys:
            ordered.append(col)
            seen.add(col)
    for col in sorted(extra_keys):
        if col not in seen:
            ordered.append(col)
            seen.add(col)

    rows = [_normalize_record(record, ordered) for record in records]
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_to_markdown(rows, ordered), encoding="utf-8")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(rows)


def write_json_results(records: Any, path: str | Path) -> None:
    """Write raw evaluation records as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _normalize_record(
    record: dict[str, Any], columns: list[str]
) -> dict[str, Any]:
    return {
        column: _format_value(record.get(column, ""))
        for column in columns
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(
            row[column].replace("|", "\\|") for column in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"
