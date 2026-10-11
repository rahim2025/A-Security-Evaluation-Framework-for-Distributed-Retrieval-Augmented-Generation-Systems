"""
defense/datapoisoning_defense/offline_validation.py

Fast, Docker-free sanity check for the content filter: generates the exact
same poisoned-document shapes attack/datapoisoning/data_poisoning_attack.py
would send to /poison (for every poison_type, with question_variants and
query-aware targeted docs included), classifies them with a fresh
PoisonFilter the same way /poison classifies one request body, and reports
the detection rate per poison_type plus the false-positive rate on real
clean corpus documents.

This does NOT exercise the live Flask endpoint, Docker, or the LLM service --
it validates the filter logic itself in isolation. Run
defense/datapoisoning_defense/run_defense.py for the full live before/after
accuracy comparison once the stack is up.
"""
import json
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attack'))
sys.path.insert(0, os.path.dirname(__file__))

from datapoisoning.data_poisoning_attack import DataPoisoningAttack, DEFAULT_DATA_SOURCES  # noqa: E402
from datapoisoning.run_attack import EVAL_DATA, DATA_DIR, load_data_points  # noqa: E402
from poison_filter import PoisonFilter  # noqa: E402

SEED = 42
SAMPLES_PER_TYPE = 30
QUESTION_VARIANTS = 2
FP_SAMPLE_SIZE = 150


def build_attack_batch(attack: DataPoisoningAttack, data_points, poison_type: str, query_aware: bool):
    """Mirror DataPoisoningAttack.execute()'s per-source doc construction for one
    poison_type, without any network calls."""
    attack.poison_type = poison_type
    rng = random.Random(SEED)
    samples = rng.sample(data_points, min(SAMPLES_PER_TYPE, len(data_points)))

    docs = []
    for original in samples:
        if QUESTION_VARIANTS > 1:
            variants = attack._create_text_variants(original, QUESTION_VARIANTS, data_points)
            docs.extend(v["text"] for v in variants)
        else:
            if poison_type == "answer_swap":
                docs.append(attack._create_answer_swap(original, data_points)["text"])
            else:
                docs.append(attack._create_poisoned_doc(original)["text"])

    if query_aware:
        for qi, item in enumerate(EVAL_DATA):
            for v in range(QUESTION_VARIANTS):
                targeted = attack._create_targeted_poison_doc(item["question"], qi, v, data_points)
                docs.append(targeted["text"])

    return docs


def false_positive_sweep(data_points, sample_size: int, seed: int) -> dict:
    rng = random.Random(seed)
    sample = rng.sample(data_points, min(sample_size, len(data_points)))
    texts = [d.get("html", d.get("text", "")) for d in sample]

    pf = PoisonFilter()
    verdicts = pf.classify_batch(texts)
    rejected = [v for v in verdicts if not v.accepted]

    return {
        "sample_size": len(sample),
        "false_positives": len(rejected),
        "false_positive_rate": len(rejected) / len(sample) if sample else 0.0,
        "sample_rejection_detail": [
            {"reasons": v.reasons} for v in rejected
        ],
    }


def main():
    data_points = load_data_points(DATA_DIR)
    attack = DataPoisoningAttack(data_sources=DEFAULT_DATA_SOURCES)

    results = {}
    for poison_type in ["wrong_answer", "misleading", "noise", "answer_swap"]:
        for query_aware, label in [(True, "oracle_knowledge"), (False, "black_box")]:
            docs = build_attack_batch(attack, data_points, poison_type, query_aware)
            pf = PoisonFilter()  # fresh filter == one /poison request from a clean process
            verdicts = pf.classify_batch(docs)
            rejected = [v for v in verdicts if not v.accepted]
            reason_counts = {"lexical": 0, "gibberish": 0, "near_duplicate": 0}
            for v in rejected:
                for r in v.reasons:
                    if r.startswith("lexical"):
                        reason_counts["lexical"] += 1
                    elif r.startswith("gibberish"):
                        reason_counts["gibberish"] += 1
                    elif r.startswith("near_duplicate"):
                        reason_counts["near_duplicate"] += 1

            key = f"{poison_type}/{label}"
            results[key] = {
                "total_docs": len(docs),
                "rejected": len(rejected),
                "detection_rate": len(rejected) / len(docs) if docs else 0.0,
                "reason_counts": reason_counts,
            }
            print(f"  {key:<30} total={len(docs):<4} rejected={len(rejected):<4} "
                  f"detection_rate={len(rejected)/len(docs):.1%}  reasons={reason_counts}")

    print("\n[*] False-positive sweep on real clean corpus documents ...")
    fp = false_positive_sweep(data_points, FP_SAMPLE_SIZE, SEED)
    print(f"    {fp['false_positives']}/{fp['sample_size']} clean docs wrongly rejected "
          f"({fp['false_positive_rate']:.1%})")

    out = {"detection_by_scenario": results, "false_positive_sweep": fp}
    out_path = os.path.join(os.path.dirname(__file__), "offline_validation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] Saved -> {out_path}")


if __name__ == "__main__":
    main()
