"""Security evaluation matrix writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MATRIX_COLUMNS = [
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


def write_evaluation_matrix(
    records: list[dict[str, Any]],
    markdown_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write evaluation records as Markdown and optionally CSV."""

    rows = [_normalize_record(record) for record in records]
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_to_markdown(rows), encoding="utf-8")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=MATRIX_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def write_json_results(records: Any, path: str | Path) -> None:
    """Write raw evaluation records as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        column: _format_value(record.get(column, ""))
        for column in MATRIX_COLUMNS
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(MATRIX_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(MATRIX_COLUMNS)) + " |"
    body = [
        "| "
        + " | ".join(row[column].replace("|", "\\|") for column in MATRIX_COLUMNS)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"
