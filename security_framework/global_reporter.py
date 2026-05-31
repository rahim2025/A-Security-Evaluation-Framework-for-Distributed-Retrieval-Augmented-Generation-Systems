"""
GlobalImpactReporter
====================
Formats a GlobalImpactReport into:
  - ASCII summary (stdout)
  - Markdown report   → GLOBAL_IMPACT_REPORT.md
  - CSV attack matrix → global_attack_matrix.csv
  - JSON raw results  → global_impact_results.json
  - Cascade tables    → cascade_data_poisoning.csv
                        cascade_kb_extraction.csv
                        cascade_node_availability.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from security_framework.global_impact import AttackScenario, GlobalImpactReport


# ──────────────────────────────────────────────────────────────────────────────
# ASCII helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 108


def _bar(ratio: float, width: int = 20) -> str:
    filled = max(0, min(width, round(ratio * width)))
    return "█" * filled + "░" * (width - filled)


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────────────────────────────────────


class GlobalImpactReporter:
    """Wraps a :class:`GlobalImpactReport` and formats it."""

    def __init__(self, report: GlobalImpactReport) -> None:
        self.report = report

    # ── stdout ────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        r = self.report
        cfg = r.simulator_config
        print()
        print("╔" + "═" * 106 + "╗")
        print("║{:^106}║".format("FedRAG  ·  Global System Impact Analysis"))
        print("╚" + "═" * 106 + "╝")
        print(f"\n  Simulator: {cfg.get('num_clients')} clients  ·  "
              f"{cfg.get('num_examples')} examples  ·  seed={cfg.get('seed')}\n")

        # ── Data Poisoning table ──────────────────────────────────────────────
        dp = [s for s in r.scenarios if s.attack_name == "Data Poisoning"]
        if dp:
            print("  ┌─ Data Poisoning  →  ClientDataPoisoningDefense ─────────────────────────────────────────────────────┐")
            print("  │{:^8}│{:^16}│{:^10}│{:>12} │{:>12} │{:>10} │{:>12} │{:>10} │".format(
                "Intensity", "Clients hit", "% of sys",
                "Baseline F1", "Attack F1", "Δ F1", "Defended F1", "Recovery"))
            print("  ├" + "─" * 8 + "┼" + "─" * 16 + "┼" + "─" * 10 +
                  "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 11 +
                  "┼" + "─" * 13 + "┼" + "─" * 11 + "┤")
            for s in dp:
                bF1 = s.baseline["f1"]
                aF1 = s.attacked["f1"]
                dF1 = s.defended["f1"] if s.defended else aF1
                print("  │{:^8}│{:^16}│{:^10}│{:>12.4f} │{:>12.4f} │{:>+10.4f} │{:>12.4f} │{:>+10.4f} │".format(
                    s.intensity.upper(),
                    f"{s.affected_clients}/{s.num_clients}",
                    f"{s.affected_ratio*100:.0f}%",
                    bF1, aF1, s.delta_f1, dF1,
                    (dF1 - aF1) if s.recovery_f1 is not None else 0.0,
                ))
            print("  └" + "─" * 8 + "┴" + "─" * 16 + "┴" + "─" * 10 +
                  "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 11 +
                  "┴" + "─" * 13 + "┴" + "─" * 11 + "┘")
            print()

        # ── KB Extraction table ───────────────────────────────────────────────
        kb = [s for s in r.scenarios if s.attack_name == "KB Extraction"]
        if kb:
            print("  ┌─ KB Extraction  →  QueryRateLimiter + ExtractionAnomalyDetector ───────────────────────────────────┐")
            print("  │{:^8}│{:^16}│{:^10}│{:>22} │{:>22} │{:>10} │".format(
                "Intensity", "Clients hit", "% of sys",
                "KB Exposed (no defense)", "KB Exposed (defended)", "Reduction"))
            print("  ├" + "─" * 8 + "┼" + "─" * 16 + "┼" + "─" * 10 +
                  "┼" + "─" * 23 + "┼" + "─" * 23 + "┼" + "─" * 11 + "┤")
            for s in kb:
                exp_b = (s.kb_exposure_before or 0.0) * 100
                exp_a = (s.kb_exposure_after or 0.0) * 100
                reduction = exp_b - exp_a
                print("  │{:^8}│{:^16}│{:^10}│{:>22} │{:>22} │{:>+10.1f} │".format(
                    s.intensity.upper(),
                    f"{s.affected_clients}/{s.num_clients}",
                    f"{s.affected_ratio*100:.0f}%",
                    f"{exp_b:.1f}% of KB exposed",
                    f"{exp_a:.1f}% of KB exposed",
                    -reduction,
                ))
            print("  └" + "─" * 8 + "┴" + "─" * 16 + "┴" + "─" * 10 +
                  "┴" + "─" * 23 + "┴" + "─" * 23 + "┴" + "─" * 11 + "┘")
            print()

        # ── Node Availability table ───────────────────────────────────────────
        na = [s for s in r.scenarios if "Node Availability" in s.attack_name]
        if na:
            attack_types = sorted({s.attack_name for s in na})
            for at in attack_types:
                group = [s for s in na if s.attack_name == at]
                print(f"  ┌─ {at}  →  CrossPeerValidation {'─' * max(0, 72-len(at))}┐")
                print("  │{:^8}│{:^16}│{:>10} │{:>12} │{:>12} │{:>10} │{:>12} │".format(
                    "Intensity", "Nodes down", "Avail %",
                    "Baseline F1", "Attack F1", "Δ F1", "Defended F1"))
                print("  ├" + "─" * 8 + "┼" + "─" * 16 + "┼" + "─" * 11 +
                      "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 11 +
                      "┼" + "─" * 13 + "┤")
                for s in group:
                    bF1 = s.baseline["f1"]
                    aF1 = s.attacked["f1"]
                    dF1 = s.defended["f1"] if s.defended else aF1
                    avail_after = (s.availability_after or 1.0) * 100
                    print("  │{:^8}│{:^16}│{:>10} │{:>12.4f} │{:>12.4f} │{:>+10.4f} │{:>12.4f} │".format(
                        s.intensity.upper(),
                        f"{s.affected_clients}/{s.num_clients}",
                        f"{avail_after:.0f}%",
                        bF1, aF1, s.delta_f1, dF1,
                    ))
                print("  └" + "─" * 8 + "┴" + "─" * 16 + "┴" + "─" * 11 +
                      "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 11 +
                      "┴" + "─" * 13 + "┘")
                print()

        # ── Cascade degradation preview ───────────────────────────────────────
        dp_cascade = r.cascade_tables.get("data_poisoning", [])
        if dp_cascade:
            print("  ┌─ Data Poisoning: F1 Degradation Cascade (one client at a time) ──────────────────────────────────────┐")
            for row in dp_cascade:
                bar = _bar(max(0.0, 1.0 - row["fraction_poisoned"]))
                print("  │  Client {:2d} poisoned  [{:s}]  Global F1 = {:.4f}  (Δ = {:+.4f}, {:+.1f}% of baseline) │".format(
                    row["newly_poisoned_client"],
                    bar,
                    row["global_f1"],
                    row["delta_f1_from_baseline"],
                    -row["pct_degradation"],
                ))
            print("  └" + "─" * 106 + "┘")
            print()

        na_cascade = r.cascade_tables.get("node_availability", [])
        if na_cascade:
            print("  ┌─ Node Removal: F1 Degradation Cascade (one node at a time) ──────────────────────────────────────────┐")
            for row in na_cascade:
                active = row["active_clients"]
                total  = active + row["total_dropped_clients"]
                bar = _bar(active / total if total else 0.0)
                print("  │  Node {:2d} removed    [{:s}]  Global F1 = {:.4f}  (Δ = {:+.4f}, {:+.1f}% of baseline) │".format(
                    row["newly_dropped_client"],
                    bar,
                    row["global_f1"],
                    row["delta_f1_from_baseline"],
                    -row["pct_degradation"],
                ))
            print("  └" + "─" * 106 + "┘")
            print()

        print("  Legend: Δ = Post-Attack minus Baseline  |  Recovery = Defended minus Attacked")
        print("          KB Exposure = fraction of private knowledge nodes recovered by attacker\n")

    # ── file outputs ──────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path = "results") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / "global_impact_results.json"
        with json_path.open("w", encoding="utf-8") as h:
            json.dump(self._to_dict(), h, indent=2)
        print(f"  Wrote {json_path}")

        # Markdown
        md_path = out / "GLOBAL_IMPACT_REPORT.md"
        md_path.write_text(self._to_markdown(), encoding="utf-8")
        print(f"  Wrote {md_path}")

        # Main CSV
        csv_path = out / "global_attack_matrix.csv"
        rows = self._to_csv_rows()
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        print(f"  Wrote {csv_path}")

        # Cascade CSVs
        self._write_cascade_csv(out / "cascade_data_poisoning.csv",
                                self.report.cascade_tables.get("data_poisoning", []))
        self._write_cascade_csv(out / "cascade_kb_extraction.csv",
                                self.report.cascade_tables.get("kb_extraction", []))
        self._write_cascade_csv(out / "cascade_node_availability.csv",
                                self.report.cascade_tables.get("node_availability", []))

    # ── internal ──────────────────────────────────────────────────────────────

    def _to_dict(self) -> dict[str, Any]:
        r = self.report
        return {
            "simulator": r.simulator_config,
            "scenarios": [self._scenario_dict(s) for s in r.scenarios],
            "cascade_tables": r.cascade_tables,
        }

    @staticmethod
    def _scenario_dict(s: AttackScenario) -> dict[str, Any]:
        return {
            "attack": s.attack_name,
            "intensity": s.intensity,
            "num_clients": s.num_clients,
            "affected_clients": s.affected_clients,
            "affected_ratio": s.affected_ratio,
            "baseline": s.baseline,
            "attacked": s.attacked,
            "defended": s.defended,
            "defense": s.defense_name,
            "delta_f1": s.delta_f1,
            "delta_em": s.delta_em,
            "delta_bleu": s.delta_bleu,
            "recovery_f1": s.recovery_f1,
            "runtime_s": s.runtime_s,
            "kb_exposure_before": s.kb_exposure_before,
            "kb_exposure_after": s.kb_exposure_after,
            "availability_before": s.availability_before,
            "availability_after": s.availability_after,
            "notes": s.notes,
        }

    def _to_csv_rows(self) -> list[dict[str, Any]]:
        rows = []
        for s in self.report.scenarios:
            row: dict[str, Any] = {
                "Attack": s.attack_name,
                "Intensity": s.intensity,
                "Clients Affected": s.affected_clients,
                "Total Clients": s.num_clients,
                "Affected Ratio": round(s.affected_ratio, 3),
                "Baseline EM": s.baseline.get("em"),
                "Baseline F1": s.baseline.get("f1"),
                "Baseline BLEU": s.baseline.get("bleu"),
                "Attacked EM": s.attacked.get("em"),
                "Attacked F1": s.attacked.get("f1"),
                "Attacked BLEU": s.attacked.get("bleu"),
                "Delta F1": round(s.delta_f1, 4),
                "Delta EM": round(s.delta_em, 4),
                "Delta BLEU": round(s.delta_bleu, 4),
                "Defended EM": s.defended.get("em") if s.defended else None,
                "Defended F1": s.defended.get("f1") if s.defended else None,
                "Defended BLEU": s.defended.get("bleu") if s.defended else None,
                "Recovery F1": round(s.recovery_f1, 4) if s.recovery_f1 is not None else None,
                "Defense": s.defense_name,
                "KB Exposed % (no defense)": round(s.kb_exposure_before * 100, 2) if s.kb_exposure_before is not None else None,
                "KB Exposed % (defended)": round(s.kb_exposure_after * 100, 2) if s.kb_exposure_after is not None else None,
                "Availability Before": s.availability_before,
                "Availability After": s.availability_after,
                "Runtime (s)": round(s.runtime_s, 4),
                "Notes": s.notes,
            }
            rows.append(row)
        return rows

    @staticmethod
    def _write_cascade_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  Wrote {path}")

    def _to_markdown(self) -> str:
        r = self.report
        cfg = r.simulator_config
        lines = [
            "# FedRAG — Global System Impact Analysis",
            "",
            "Shows how **client-level attacks** cascade into **global RAG system degradation**.",
            "Each attack is run at three intensity levels: Light (10 %), Medium (30 %), Heavy (50 %).",
            "",
            "## Simulator Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Clients | {cfg.get('num_clients')} |",
            f"| Examples | {cfg.get('num_examples')} |",
            f"| Seed | {cfg.get('seed')} |",
            f"| Embedding dim | {cfg.get('embedding_dim')} |",
            "",
        ]

        # ── Data Poisoning ───────────────────────────────────────────────────
        dp = [s for s in r.scenarios if s.attack_name == "Data Poisoning"]
        if dp:
            lines += [
                "## 1. Data Poisoning → Global LLM Performance Impact",
                "",
                "**Attack:** Malicious clients inject wrong/misleading answers into their "
                "local knowledge store. When the global system retrieves from these clients, "
                "poisoned answers are returned, degrading global EM/F1/BLEU.",
                "",
                "**Defense:** `ClientDataPoisoningDefense` — inspects each client's data "
                "for poison markers and conflicting answers; quarantines high-risk clients.",
                "",
                "| Intensity | Clients Hit | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 | Recovery |",
                "|-----------|-------------|-------------|-------------|------|-------------|----------|",
            ]
            for s in dp:
                dF1 = s.defended["f1"] if s.defended else s.attacked["f1"]
                lines.append(
                    f"| {s.intensity.upper()} | {s.affected_clients}/{s.num_clients} "
                    f"({s.affected_ratio*100:.0f}%) "
                    f"| {s.baseline['f1']:.4f} | {s.attacked['f1']:.4f} "
                    f"| **{s.delta_f1:+.4f}** | {dF1:.4f} "
                    f"| {(dF1-s.attacked['f1']):+.4f} |"
                )
            lines += [
                "",
                "### Degradation Cascade (Data Poisoning)",
                "",
                "| Step | Client Poisoned | Total Poisoned | Global F1 | Δ from Baseline | % Degradation |",
                "|------|----------------|----------------|-----------|-----------------|---------------|",
            ]
            for row in r.cascade_tables.get("data_poisoning", []):
                lines.append(
                    f"| {row['step']} | {row['newly_poisoned_client']} "
                    f"| {row['total_poisoned_clients']}/{cfg.get('num_clients')} "
                    f"| {row['global_f1']:.4f} "
                    f"| {row['delta_f1_from_baseline']:+.4f} "
                    f"| {row['pct_degradation']:.1f}% |"
                )
            lines.append("")

        # ── KB Extraction ────────────────────────────────────────────────────
        kb = [s for s in r.scenarios if s.attack_name == "KB Extraction"]
        if kb:
            lines += [
                "## 2. KB Extraction → Privacy / Knowledge Exposure",
                "",
                "**Attack:** An adversary repeatedly probes the public retrieval API "
                "with templated queries to reconstruct private knowledge nodes. "
                "Answer quality is NOT degraded, but private data leaks.",
                "",
                "**Defense:** `QueryRateLimiter` + `ExtractionAnomalyDetector` — caps "
                "total queries per client and flags abnormally broad-topic query bursts.",
                "",
                "| Intensity | Clients Targeted | KB Exposed (no defense) | KB Exposed (defended) | Reduction |",
                "|-----------|-----------------|-------------------------|-----------------------|-----------|",
            ]
            for s in kb:
                exp_b = (s.kb_exposure_before or 0.0) * 100
                exp_a = (s.kb_exposure_after or 0.0) * 100
                lines.append(
                    f"| {s.intensity.upper()} | {s.affected_clients}/{s.num_clients} "
                    f"({s.affected_ratio*100:.0f}%) "
                    f"| **{exp_b:.1f}%** | {exp_a:.1f}% | -{exp_b-exp_a:.1f}pp |"
                )
            lines += [
                "",
                "### Extraction Cascade (cumulative KB exposure as more clients are targeted)",
                "",
                "| Step | Client Targeted | Total Targeted | Exposure (no defense) | Exposure (defended) |",
                "|------|----------------|----------------|-----------------------|---------------------|",
            ]
            for row in r.cascade_tables.get("kb_extraction", []):
                lines.append(
                    f"| {row['step']} | {row['newly_targeted_client']} "
                    f"| {row['total_targeted_clients']}/{cfg.get('num_clients')} "
                    f"| {row['exposure_pct_no_defense']:.1f}% "
                    f"| {row['exposure_pct_with_defense']:.1f}% |"
                )
            lines.append("")

        # ── Node Availability ────────────────────────────────────────────────
        na = [s for s in r.scenarios if "Node Availability" in s.attack_name]
        if na:
            lines += [
                "## 3. Node Availability → Global System Coverage",
                "",
                "**Attack:** Clients are removed (node_removal / ddos), set Byzantine "
                "(returning corrupted answers), network-partitioned, or Sybil-injected.",
                "",
                "**Defense:** `CrossPeerValidation` — majority-votes candidate answers "
                "across surviving peers; discards answers that fail consensus.",
                "",
                "| Attack Type | Intensity | Nodes Down | Avail % | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 |",
                "|-------------|-----------|------------|---------|-------------|-------------|------|-------------|",
            ]
            for s in na:
                dF1 = s.defended["f1"] if s.defended else s.attacked["f1"]
                avail_after = (s.availability_after or 1.0) * 100
                lines.append(
                    f"| {s.attack_name.replace('Node Availability (','').rstrip(')')} "
                    f"| {s.intensity.upper()} "
                    f"| {s.affected_clients}/{s.num_clients} "
                    f"| {avail_after:.0f}% "
                    f"| {s.baseline['f1']:.4f} "
                    f"| {s.attacked['f1']:.4f} "
                    f"| **{s.delta_f1:+.4f}** "
                    f"| {dF1:.4f} |"
                )
            lines += [
                "",
                "### Availability Cascade (node_removal, one node at a time)",
                "",
                "| Step | Node Removed | Active Nodes | Global F1 | Δ from Baseline | % Degradation |",
                "|------|-------------|--------------|-----------|-----------------|---------------|",
            ]
            for row in r.cascade_tables.get("node_availability", []):
                lines.append(
                    f"| {row['step']} | {row['newly_dropped_client']} "
                    f"| {row['active_clients']}/{cfg.get('num_clients')} "
                    f"| {row['global_f1']:.4f} "
                    f"| {row['delta_f1_from_baseline']:+.4f} "
                    f"| {row['pct_degradation']:.1f}% |"
                )
            lines.append("")

        lines += [
            "---",
            "",
            "## Key Takeaways",
            "",
            "- **Data Poisoning** has the highest direct impact on global answer quality; "
            "  even 10% malicious clients can drop global F1 measurably.",
            "- **KB Extraction** does not degrade answer quality but can expose the entire "
            "  knowledge base if rate limiting is not applied.",
            "- **Node Availability** degrades global coverage linearly with the fraction "
            "  of nodes removed; Byzantine attacks are harder to recover from than simple removal.",
            "- **Defenses** consistently recover a significant portion of lost performance, "
            "  but never fully restore baseline — underlining the importance of prevention.",
            "",
            "## Output Files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `global_impact_results.json` | Full raw results for every scenario |",
            "| `GLOBAL_IMPACT_REPORT.md` | This report |",
            "| `global_attack_matrix.csv` | Flat CSV — one row per scenario |",
            "| `cascade_data_poisoning.csv` | Per-client poisoning cascade |",
            "| `cascade_kb_extraction.csv` | Per-client extraction cascade |",
            "| `cascade_node_availability.csv` | Per-node removal cascade |",
        ]

        return "\n".join(lines)
