"""
defense/kb_extraction_defense/run_defense.py

Attack-vs-defense comparison for QueryDiversityThrottle against the real
attack/kb_extraction extraction attack. Runs the same probe set through
attack.kb_extraction.run_attack.probe_source() twice against the real live
Docker data source -- once ungated (attack_only), once gated through the
throttle (attack_plus_defense) -- and compares the ground-truth
extraction_rate/extraction_accuracy/query_efficiency/topic_coverage metrics
(added to that module alongside this defense) before and after.

Usage
-----
  docker compose up -d
  python defense/kb_extraction_defense/run_defense.py
  python defense/kb_extraction_defense/run_defense.py --source_index 1 --max_topics_per_window 8
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.kb_extraction.run_attack import (  # noqa: E402
    DATA_SOURCE_URLS,
    TOP_K,
    load_squad_probes,
    probe_source,
)
from defense.kb_extraction_defense.query_diversity_throttle import QueryDiversityThrottle  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "defense_logs", "kb_extraction_defense")


def main() -> None:
    p = argparse.ArgumentParser(description="KB extraction attack vs. query-diversity-throttle defense")
    p.add_argument("--source_index", type=int, default=1, choices=[0, 1, 2],
                    help="which of the 3 data sources to probe (matches DATA_SOURCE_URLS). Default 1 "
                         "(sources_20, SQuAD-domain): source_0 was migrated to PubMedQA content and "
                         "carries no per-article title/topic field, so the topic-diversity throttle "
                         "this defense evaluates has nothing to key off there -- see "
                         "attack/kb_extraction/run_attack.py's load_probe_sets() docstring.")
    p.add_argument("--probe_sample_size", type=int, default=None)
    p.add_argument("--window_size", type=int, default=30)
    p.add_argument("--max_topics_per_window", type=int, default=12)
    p.add_argument("--min_queries_before_check", type=int, default=15)
    p.add_argument("--cooldown_queries", type=int, default=20)
    args = p.parse_args()

    kwargs = {"source_idx": args.source_index}
    if args.probe_sample_size is not None:
        kwargs["n"] = args.probe_sample_size
    probes, ground_truth_contexts, context_to_title, qa_by_question = load_squad_probes(**kwargs)
    url = DATA_SOURCE_URLS[args.source_index]
    print(f"Target: {url}  |  probes: {len(probes)}  |  "
          f"ground truth: {len(ground_truth_contexts)} docs, {len(set(context_to_title.values()))} topics\n")

    print("=== ATTACK ONLY (no defense) ===")
    attack_only = probe_source(url, probes, k=TOP_K, authenticated=True,
                                ground_truth_contexts=ground_truth_contexts,
                                context_to_title=context_to_title)
    print(f"  extraction_rate={attack_only.get('extraction_rate', 0):.3f}  "
          f"extraction_accuracy={attack_only.get('extraction_accuracy', 0):.3f}  "
          f"topic_coverage={attack_only.get('topic_coverage', 0):.3f}  "
          f"docs_extracted={attack_only.get('docs_extracted', 0)}")

    print("\n=== ATTACK + DEFENSE (query-diversity throttle) ===")
    throttle = QueryDiversityThrottle(
        window_size=args.window_size,
        max_topics_per_window=args.max_topics_per_window,
        min_queries_before_check=args.min_queries_before_check,
        cooldown_queries=args.cooldown_queries,
    )

    def gate(query_text: str) -> bool:
        topic = qa_by_question.get(query_text, {}).get("title")
        allowed, _reason = throttle.check_and_record("attacker_peer", query_text, topic)
        return allowed

    attack_defended = probe_source(url, probes, k=TOP_K, authenticated=True,
                                    ground_truth_contexts=ground_truth_contexts,
                                    context_to_title=context_to_title, query_gate=gate)
    print(f"  extraction_rate={attack_defended.get('extraction_rate', 0):.3f}  "
          f"extraction_accuracy={attack_defended.get('extraction_accuracy', 0):.3f}  "
          f"topic_coverage={attack_defended.get('topic_coverage', 0):.3f}  "
          f"docs_extracted={attack_defended.get('docs_extracted', 0)}  "
          f"blocked={attack_defended.get('blocked_count', 0)}/{len(probes)}")

    throttle_stats = throttle.get_stats()
    print(f"\n  throttle stats: {throttle_stats}")

    n_probes = len(probes)
    reduction = {
        "extraction_rate_reduction": round(
            attack_only.get("extraction_rate", 0) - attack_defended.get("extraction_rate", 0), 4),
        "topic_coverage_reduction": round(
            attack_only.get("topic_coverage", 0) - attack_defended.get("topic_coverage", 0), 4),
        "queries_blocked": attack_defended.get("blocked_count", 0),
        "queries_blocked_fraction": round(attack_defended.get("blocked_count", 0) / n_probes, 4) if n_probes else 0.0,
    }
    print(f"\n  extraction_rate reduced by {reduction['extraction_rate_reduction']:.3f}  "
          f"topic_coverage reduced by {reduction['topic_coverage_reduction']:.3f}  "
          f"({reduction['queries_blocked']}/{n_probes} queries blocked = "
          f"{reduction['queries_blocked_fraction']:.1%})")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(LOG_DIR, f"defense_{ts}_kb_extraction_throttle.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "defense_type": "kb_extraction_query_diversity_throttle",
            "config": {
                "source_index": args.source_index, "window_size": args.window_size,
                "max_topics_per_window": args.max_topics_per_window,
                "min_queries_before_check": args.min_queries_before_check,
                "cooldown_queries": args.cooldown_queries, "probe_sample_size": n_probes,
            },
            "attack_only": attack_only,
            "attack_plus_defense": attack_defended,
            "throttle_stats": throttle_stats,
            "reduction": reduction,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  [+] Log saved -> {log_path}")


if __name__ == "__main__":
    main()
