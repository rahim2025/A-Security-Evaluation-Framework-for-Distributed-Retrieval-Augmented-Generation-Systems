"""
defense/datapoisoning_defense/run_defense.py
Evaluates the data-poisoning content filter against Reliable-dRAG.

Requires the patched drag_data_source/app/server.py to be the version
actually running in the data-source containers (see defense/datapoisoning_defense/
README.md for the rebuild/restart step) with POISON_FILTER_ENABLED unset or "1".

Flow
----
  1. Reset all sources to a known-clean state.
  2. Baseline accuracy on the same eval set attack/datapoisoning/run_attack.py uses.
  3. Run the SAME DataPoisoningAttack against the now-defended /poison endpoint,
     with the SAME CLI options attack/datapoisoning/run_attack.py accepts -- so a
     "before" run (python attack/datapoisoning/run_attack.py --evaluate <opts>,
     against an undefended deployment) and this "after" run
     (python defense/datapoisoning_defense/run_defense.py <opts>) are directly
     comparable for identical <opts>.
  4. Post-attack accuracy, and how many candidate docs the filter accepted vs
     rejected (and why).
  5. Offline false-positive sweep: run a fresh PoisonFilter directly (no network)
     against a sample of genuinely clean corpus documents, to report how much
     legitimate content the filter would have blocked.
  6. Reset back to clean, save a JSON log to defense_logs/datapoisoning_defense/.
"""

import argparse
import json
import os
import random
import sys
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attack'))

from datapoisoning.data_poisoning_attack import DataPoisoningAttack, DEFAULT_DATA_SOURCES  # noqa: E402
from datapoisoning.run_attack import (  # noqa: E402
    EVAL_DATA, DATA_DIR, load_data_points, collect_responses, print_comparison_table,
)
from poison_filter import PoisonFilter  # noqa: E402

LOG_DIR = os.path.join(PROJECT_ROOT, "defense_logs", "datapoisoning_defense")


def false_positive_sweep(data_points, sample_size: int, seed: int) -> dict:
    """Run a standalone PoisonFilter (no network) against real clean corpus
    documents to measure how often it would wrongly block legitimate content."""
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
        "rejected_reasons": [v.reasons for v in rejected],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the data-poisoning content filter")
    parser.add_argument('--strategy', default='random', choices=['random', 'targeted', 'data_rich'])
    parser.add_argument('--targets', nargs='+', default=[], metavar='SOURCE_NAME')
    parser.add_argument('--poison-type', default='wrong_answer',
                        choices=['wrong_answer', 'misleading', 'noise', 'answer_swap'])
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--amplify', type=int, default=1)
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-query-aware', action='store_true')
    parser.add_argument('--intensity', choices=['light', 'medium', 'heavy'], default='heavy')
    parser.add_argument('--density', type=float, default=None)
    parser.add_argument('--fp-sample-size', type=int, default=100,
                        help="How many genuinely clean docs to run through the filter "
                             "offline to measure the false-positive rate.")
    args = parser.parse_args()

    INTENSITY_PRESETS = {'light': 0.2, 'medium': 0.5, 'heavy': 1.0}
    poisoning_density = args.density if args.density is not None else INTENSITY_PRESETS[args.intensity]

    random.seed(args.seed)

    timestamp = datetime.datetime.now().isoformat()
    ts_fname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"defense_{ts_fname}_{args.strategy}_{args.poison_type}.json")

    print("\n" + "=" * 70)
    print("  Reliable-dRAG  -  Data Poisoning Defense Evaluation")
    print("=" * 70)
    print(f"  strategy        : {args.strategy}")
    print(f"  poison_type     : {args.poison_type}")
    print(f"  ratio           : {args.ratio}")
    print(f"  amplify         : {args.amplify}")
    print(f"  intensity       : {args.intensity}  (density={poisoning_density:.0%})")
    print(f"  seed            : {args.seed}")
    print()

    attack = DataPoisoningAttack(
        data_sources=DEFAULT_DATA_SOURCES,
        poisoning_ratio=args.ratio,
        poisoning_density=poisoning_density,
        attack_strategy=args.strategy,
        poison_type=args.poison_type,
        target_source_names=args.targets,
        amplification_factor=args.amplify,
        question_variants=args.variants,
        target_queries=[] if args.no_query_aware else [item["question"] for item in EVAL_DATA],
    )

    print("[*] Resetting all data sources to clean state ...")
    for name, result in attack.reset_all().items():
        print(f"    {name}: {result}")

    print("\n[*] Measuring BASELINE accuracy (defended, no attack) ...")
    clean_responses = collect_responses()
    clean_correct = sum(1 for r in clean_responses if r["correct"])
    clean_acc = clean_correct / len(clean_responses)
    print(f"\n    Baseline accuracy: {clean_correct}/{len(clean_responses)} = {clean_acc:.1%}")

    print("\n[*] Attempting the data poisoning attack against the DEFENDED /poison endpoint ...")
    data_points = load_data_points(DATA_DIR)
    attack_result = attack.execute(data_points)
    injected = attack_result["total_injected_docs"]
    expected = attack_result["total_expected_docs"]
    rejected = attack_result.get("total_rejected_docs", 0)
    print(f"\n    Requested  : {expected} poisoned docs")
    print(f"    Injected   : {injected} ({injected / expected:.1%} of requested)" if expected else "    Injected   : 0")
    print(f"    Rejected   : {rejected} ({rejected / expected:.1%} of requested)" if expected else "    Rejected   : 0")

    print("\n[*] Measuring POST-ATTACK accuracy (defended) ...")
    attacked_responses = collect_responses()
    attacked_correct = sum(1 for r in attacked_responses if r["correct"])
    attacked_acc = attacked_correct / len(attacked_responses)
    print(f"\n    Post-attack accuracy: {attacked_correct}/{len(attacked_responses)} = {attacked_acc:.1%}")

    print_comparison_table(clean_responses, attacked_responses)

    degradation = ((clean_acc - attacked_acc) / clean_acc * 100) if clean_acc > 0 else 0.0
    print(f"\n    Accuracy drop WITH defense: {degradation:.1f}%")
    print("    (compare against the same config's drop from attack/datapoisoning/run_attack.py")
    print("     run against an undefended deployment -- see README.md)")

    print("\n[*] Offline false-positive sweep on genuinely clean documents ...")
    fp = false_positive_sweep(data_points, args.fp_sample_size, args.seed)
    print(f"    {fp['false_positives']}/{fp['sample_size']} clean docs wrongly rejected "
          f"({fp['false_positive_rate']:.1%})")

    print("\n[*] Resetting all data sources back to clean state ...")
    for name, result in attack.reset_all().items():
        print(f"    {name}: {result}")

    result = {
        "defense": "datapoisoning_content_filter",
        "timestamp": timestamp,
        "config": {
            "strategy": args.strategy,
            "poison_type": args.poison_type,
            "poisoning_ratio": args.ratio,
            "amplification_factor": args.amplify,
            "question_variants": args.variants,
            "target_sources": args.targets,
            "seed": args.seed,
            "query_aware": not args.no_query_aware,
            "intensity": args.intensity,
            "poisoning_density": poisoning_density,
        },
        "baseline_accuracy": clean_acc,
        "attack_result": attack_result,
        "post_attack_accuracy": attacked_acc,
        "accuracy_drop_pct_with_defense": round(degradation, 2),
        "false_positive_sweep": fp,
        "per_question": [
            {
                "question": c["question"],
                "expected": c["expected"],
                "clean_response": c["response"],
                "clean_correct": c["correct"],
                "attacked_response": a["response"],
                "attacked_correct": a["correct"],
            }
            for c, a in zip(clean_responses, attacked_responses)
        ],
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Log saved -> {log_file}")
    return result


if __name__ == "__main__":
    main()
