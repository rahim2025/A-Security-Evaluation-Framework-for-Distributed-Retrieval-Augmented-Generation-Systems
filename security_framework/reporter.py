"""
SecurityReport
==============
Collects and formats the output from ``SecurityPipeline.run_all()``.

Outputs
-------
- ``print_summary()``         — colourless ASCII table to stdout
- ``save(output_dir)``        — writes JSON + Markdown + CSV
- ``to_dict()``               — raw dict for further processing
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SecurityReport:
    """Container for all pipeline results with formatting helpers."""

    def __init__(self, results: dict[str, Any]) -> None:
        self._results = results
        self._timestamp = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self._timestamp, **self._results}

    def print_summary(self) -> None:
        """Print a compact ASCII summary to stdout."""
        print("\n" + "=" * 72)
        print("  FedRAG Security Evaluation — Summary Report")
        print(f"  Generated: {self._timestamp}")
        print("=" * 72)

        sim = self._results.get("simulator", {})
        print(
            f"\n  Simulator: {sim.get('num_clients')} clients, "
            f"{sim.get('num_examples')} examples, seed={sim.get('seed')}\n"
        )

        self._print_poisoning()
        self._print_mia()
        self._print_extraction()
        self._print_node_availability()
        print("=" * 72 + "\n")

    def save(self, output_dir: str | Path = "results") -> None:
        """Save JSON, Markdown, and CSV reports to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / "security_report.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        print(f"  Wrote {json_path}")

        # Markdown
        md_path = out / "SECURITY_REPORT.md"
        md_path.write_text(self._to_markdown(), encoding="utf-8")
        print(f"  Wrote {md_path}")

        # CSV (flat attack matrix)
        csv_path = out / "attack_matrix.csv"
        rows = self._to_csv_rows()
        if rows:
            # Collect the union of ALL fields across every row so that
            # rows with extra keys (e.g. MIA's 'Defended Membership Acc')
            # do not cause a DictWriter ValueError.
            all_fields: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        all_fields.append(k)
                        seen.add(k)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=all_fields, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Wrote {csv_path}")

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _print_poisoning(self) -> None:
        r = self._results.get("data_poisoning", {})
        if not r:
            return
        print("  [1] Data Poisoning Attack")
        print(f"      Malicious clients : {r.get('malicious_clients')}")
        print(f"      Poison type       : {r.get('poison_type')}")
        print(f"      Records poisoned  : {r.get('total_poisoned_records')}")
        b = r.get("baseline", {})
        a = r.get("attacked", {})
        d = r.get("defended", {})
        print(f"      Baseline  EM/F1/BLEU : {b.get('em', 0):.3f} / {b.get('f1', 0):.3f} / {b.get('bleu', 0):.3f}")
        print(f"      Post-atk  EM/F1/BLEU : {a.get('em', 0):.3f} / {a.get('f1', 0):.3f} / {a.get('bleu', 0):.3f}")
        if d:
            print(f"      Defended  EM/F1/BLEU : {d.get('em', 0):.3f} / {d.get('f1', 0):.3f} / {d.get('bleu', 0):.3f}")
        print(f"      Runtime           : {r.get('runtime_s', 0):.2f}s\n")

    def _print_mia(self) -> None:
        r = self._results.get("membership_inference", {})
        if not r:
            return
        a = r.get("attacked", {})
        d = r.get("defended", {})
        print("  [2] Membership Inference Attack")
        print(f"      Target client     : {r.get('target_client')}")
        print(f"      Threshold         : {r.get('threshold')}")
        print(f"      Members tested    : {r.get('num_members_tested')} | Non-members: {r.get('num_non_members_tested')}")
        print(f"      Attack accuracy   : {a.get('accuracy', 0):.3f}  TPR={a.get('tpr', 0):.3f}  FPR={a.get('fpr', 0):.3f}")
        if d:
            print(f"      Defended accuracy : {d.get('accuracy', 0):.3f}  TPR={d.get('tpr', 0):.3f}  FPR={d.get('fpr', 0):.3f}")
        print(f"      Runtime           : {r.get('runtime_s', 0):.2f}s\n")

    def _print_extraction(self) -> None:
        r = self._results.get("knowledge_extraction", {})
        if not r:
            return
        a = r.get("attacked", {})
        d = r.get("defended", {})
        print("  [3] Knowledge Extraction Attack")
        print(f"      Target client     : {r.get('target_client')}")
        print(f"      Queries issued    : {r.get('total_queries')}")
        print(f"      Nodes recovered   : {a.get('recovered_nodes')} / {a.get('total_nodes')}  ({a.get('recovery_ratio', 0) * 100:.1f}%)")
        if d:
            print(f"      Defended recovery : {d.get('recovered_nodes')} / {d.get('total_nodes')}  ({d.get('recovery_ratio', 0) * 100:.1f}%)")
            di = r.get("defense_info", {})
            if di:
                print(f"      Queries blocked   : {di.get('queries_blocked', 0)}")
        print(f"      Runtime           : {r.get('runtime_s', 0):.2f}s\n")

    def _print_node_availability(self) -> None:
        r = self._results.get("node_availability", {})
        if not r:
            return
        a = r.get("attacked", {})
        b = r.get("baseline", {})
        d = r.get("defended")
        print(f"  [4] Node Availability Attack ({r.get('attack_type')})")
        print(f"      Attack ratio      : {r.get('attack_ratio')}")
        print(f"      Nodes affected    : {len(r.get('affected_nodes', []))} / {r.get('total_nodes')}")
        print(f"      Availability      : {r.get('availability_before', 0) * 100:.1f}% → {r.get('availability_after', 0) * 100:.1f}%")
        print(f"      Baseline  F1      : {b.get('f1', 0):.3f}")
        print(f"      Post-atk  F1      : {a.get('f1', 0):.3f}")
        if d:
            print(f"      Defended  F1      : {d.get('f1', 0):.3f}")
        print(f"      Runtime           : {r.get('runtime_s', 0):.2f}s\n")

    def _to_markdown(self) -> str:
        sim = self._results.get("simulator", {})
        lines = [
            "# FedRAG Security Evaluation Report",
            "",
            f"**Generated:** {self._timestamp}",
            "",
            "## Simulator Configuration",
            "",
            f"| Parameter | Value |",
            f"| --- | --- |",
            f"| Clients | {sim.get('num_clients')} |",
            f"| Examples | {sim.get('num_examples')} |",
            f"| Seed | {sim.get('seed')} |",
            f"| Embedding dim | {sim.get('embedding_dim')} |",
            "",
            "## Attack Results",
            "",
        ]

        for row in self._to_csv_rows():
            lines.append(f"### {row.get('Attack')}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for k, v in row.items():
                if k == "Attack":
                    continue
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v} |")
            lines.append("")

        lines += [
            "## Defense Summary",
            "",
            "| Defense | Target Attack | Status |",
            "| --- | --- | --- |",
            "| ClientDataPoisoningDefense | Data Poisoning | Active |",
            "| ScoreMasking | Membership Inference | Active |",
            "| CrossPeerValidation | Node Availability | Active |",
            "| QueryRateLimiter | Knowledge Extraction | Active |",
            "| ExtractionAnomalyDetector | Knowledge Extraction | Active |",
            "",
        ]
        return "\n".join(lines)

    def _to_csv_rows(self) -> list[dict[str, Any]]:
        rows = []

        # Data poisoning
        r = self._results.get("data_poisoning", {})
        if r:
            b = r.get("baseline", {})
            a = r.get("attacked", {})
            d = r.get("defended") or {}
            rows.append({
                "Attack": "Data Poisoning",
                "Baseline EM": b.get("em"),
                "Post-Attack EM": a.get("em"),
                "Defended EM": d.get("em"),
                "Delta EM": r.get("delta_em"),
                "Baseline F1": b.get("f1"),
                "Post-Attack F1": a.get("f1"),
                "Defended F1": d.get("f1"),
                "Delta F1": r.get("delta_f1"),
                "Baseline BLEU": b.get("bleu"),
                "Post-Attack BLEU": a.get("bleu"),
                "Defended BLEU": d.get("bleu"),
                "Delta BLEU": r.get("delta_bleu"),
                "Membership Acc": None,
                "KB Recovery %": None,
                "Runtime (s)": r.get("runtime_s"),
            })

        # Membership inference
        r = self._results.get("membership_inference", {})
        if r:
            a = r.get("attacked", {})
            d = r.get("defended") or {}
            rows.append({
                "Attack": "Membership Inference",
                "Baseline EM": None,
                "Post-Attack EM": None,
                "Defended EM": None,
                "Delta EM": None,
                "Baseline F1": None,
                "Post-Attack F1": None,
                "Defended F1": None,
                "Delta F1": None,
                "Baseline BLEU": None,
                "Post-Attack BLEU": None,
                "Defended BLEU": None,
                "Delta BLEU": None,
                "Membership Acc": a.get("accuracy"),
                "Defended Membership Acc": d.get("accuracy"),
                "KB Recovery %": None,
                "Runtime (s)": r.get("runtime_s"),
            })

        # Knowledge extraction
        r = self._results.get("knowledge_extraction", {})
        if r:
            a = r.get("attacked", {})
            d = r.get("defended") or {}
            rows.append({
                "Attack": "Knowledge Extraction",
                "Baseline EM": None,
                "Post-Attack EM": None,
                "Defended EM": None,
                "Delta EM": None,
                "Baseline F1": None,
                "Post-Attack F1": None,
                "Defended F1": None,
                "Delta F1": None,
                "Baseline BLEU": None,
                "Post-Attack BLEU": None,
                "Defended BLEU": None,
                "Delta BLEU": None,
                "Membership Acc": None,
                "KB Recovery %": a.get("recovery_ratio", 0) * 100,
                "Defended KB Recovery %": d.get("recovery_ratio", 0) * 100 if d else None,
                "Runtime (s)": r.get("runtime_s"),
            })

        # Node availability
        r = self._results.get("node_availability", {})
        if r:
            b = r.get("baseline", {})
            a = r.get("attacked", {})
            d = r.get("defended") or {}
            rows.append({
                "Attack": f"Node Availability ({r.get('attack_type')})",
                "Baseline EM": b.get("em"),
                "Post-Attack EM": a.get("em"),
                "Defended EM": d.get("em"),
                "Delta EM": r.get("delta_em"),
                "Baseline F1": b.get("f1"),
                "Post-Attack F1": a.get("f1"),
                "Defended F1": d.get("f1"),
                "Delta F1": r.get("delta_f1"),
                "Baseline BLEU": b.get("bleu"),
                "Post-Attack BLEU": a.get("bleu"),
                "Defended BLEU": d.get("bleu"),
                "Delta BLEU": None,
                "Availability Before %": r.get("availability_before", 0) * 100,
                "Availability After %": r.get("availability_after", 0) * 100,
                "Membership Acc": None,
                "KB Recovery %": None,
                "Runtime (s)": r.get("runtime_s"),
            })

        return rows
