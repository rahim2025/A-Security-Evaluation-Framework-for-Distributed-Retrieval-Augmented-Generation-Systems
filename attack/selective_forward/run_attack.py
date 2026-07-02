"""
run_attack.py — SFA for Reliable-dRAG. Mirrors DRAG run_selective_forwarding.py.
Metrics: hit_rate, avg_hops_per_query, ttl_exhaustion_rate, dropped_queries

Usage:
  python3 attack/selective_forward/run_attack.py                     # 500 queries, seed 0
  python3 attack/selective_forward/run_attack.py --seed 1
  python3 attack/selective_forward/run_attack.py --seed 2
  python3 attack/selective_forward/run_attack.py --num_queries 1000  # thesis run
  python3 attack/selective_forward/run_attack.py --ratios 0.0 0.1 0.2 0.3 0.5 0.7 1.0
  python3 attack/selective_forward/run_attack.py --mode live --api_key KEY
"""
import argparse, csv, datetime, json, os, random, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from attack.selective_forward.selective_forward_attack import (
    SelectiveForwardingAttack, MockRAGNetwork, apply_selective_forwarding,
    DATA_SOURCES, EVAL_DATA,
)

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attack_logs"))
CSV_DIR = os.path.join(LOG_DIR, "selective_forwarding")

def _print_row(row):
    print(f"  {row['strategy']:22s}  ratio={row['attack_ratio']:.1f}  "
          f"comp={row.get('num_compromised',0):2d}  "
          f"hit_rate={row['hit_rate']:.3f}  avg_hops={row['avg_hops_per_query']:.2f}  "
          f"ttl_exhaust={row['ttl_exhaustion_rate']:.3f}  dropped={row['dropped_queries']}")

def run_sweep(num_queries=500, peer_hit_prob=0.4, max_hops=3,
              ratios=None, strategies=None, seed=0):
    """
    One network, apply/revert, baseline once. Mirrors DRAG run_sweep().

    RNG design:
    - Global random/np.random seeded ONCE and reset to the SAME state before
      every scenario. This ensures MockSource.query() sees the same hit/miss
      sequence in every run — only the attack drops cause variation.
    - SelectiveForwardingAttack uses a SEPARATE random.Random() for all drop
      decisions, so drop choices never perturb the global RNG state.
    """
    if ratios    is None: ratios    = [0.0, 0.34, 0.67, 1.0]
    if strategies is None: strategies = ["random", "high_ssm_score"]

    network = MockRAGNetwork(peer_hit_prob=peer_hit_prob, seed=seed)
    queries = [f"query_{i}" for i in range(num_queries)]
    results = []

    for strategy in strategies:
        for ratio in ratios:
            # Reset global RNG to the SAME state before every scenario.
            # Only the attack's separate drop_rng changes between ratios.
            random.seed(seed); np.random.seed(seed)

            if ratio == 0.0:
                answers   = [network.query(q) for q in queries]
                total     = len(answers)
                hit       = sum(1 for r in answers if r.is_query_hit)
                exhausted = sum(1 for r in answers if r.num_hops >= max_hops and not r.is_query_hit)
                row = {
                    "strategy": "baseline", "attack_ratio": 0.0, "num_compromised": 0,
                    "compromised_ids": [],
                    "avg_ssm_score_compromised": 0.0,
                    "avg_ssm_score_all": float(np.mean(
                        [v["reliability"] for v in network.ssm_scores.values()])),
                    "hit_rate":            hit / total,
                    "avg_hops_per_query":  float(np.mean([r.num_hops for r in answers])),
                    "ttl_exhaustion_rate": exhausted / total,
                    "dropped_queries":     0,
                    "total_queries":       total,
                    "answered_queries":    hit,
                    "exhausted_queries":   exhausted,
                }
                if not any(r["strategy"] == "baseline" for r in results):
                    results.append(row); _print_row(row)
                continue

            # apply -> run -> collect_metrics -> revert  (mirrors DRAG exactly)
            sfa, apply_info = apply_selective_forwarding(
                network, attack_ratio=ratio, strategy=strategy, seed=seed)
            answers = [network.query(q) for q in queries]
            metrics = sfa.collect_metrics(answers, max_hops=max_hops)
            sfa.revert(network)
            row = {**apply_info, **metrics}
            results.append(row); _print_row(row)

    return results

def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results: return
    clean = [{k: ("|".join(str(x) for x in v) if isinstance(v,(list,set)) else v)
              for k,v in row.items()} for row in results]
    keys = list(dict.fromkeys(k for r in clean for k in r))
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        w.writeheader(); [w.writerow({k: r.get(k,"") for k in keys}) for r in clean]
    print(f"\n  Results saved to: {path}")

def save_json(log, suffix=""):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(LOG_DIR, f"attack_{ts}_selective_forward{suffix}.json")
    with open(fname,"w",encoding="utf-8") as f: json.dump(log,f,indent=2)
    return fname

def parse_args():
    p = argparse.ArgumentParser(description="SFA — Reliable-dRAG (mirrors DRAG baseline)")
    p.add_argument("--mode",          default="mock", choices=["mock","live"])
    p.add_argument("--num_queries",   type=int,   default=500,
                   help="Queries per scenario (default: 500; use 1000 for thesis)")
    p.add_argument("--peer_hit_prob", type=float, default=0.4)
    p.add_argument("--max_hops",      type=int,   default=3)
    p.add_argument("--ratios",        nargs="+",  type=float, default=[0.0, 0.34, 0.67, 1.0])
    p.add_argument("--strategies",    nargs="+",  default=["random","high_ssm_score"],
                   choices=["random","high_ssm_score"])
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--output",        default="")
    p.add_argument("--attack_ratio",  type=float, default=0.33)
    p.add_argument("--strategy",      default="random", choices=["random","high_ssm_score"])
    p.add_argument("--api_key",       default="")
    p.add_argument("--k",             type=int,   default=5)
    return p.parse_args()

def main():
    args = parse_args(); ts = datetime.datetime.now().isoformat()

    if args.mode == "mock":
        csv_path = args.output or os.path.join(CSV_DIR, f"results_seed{args.seed}.csv")
        print("=" * 70)
        print("  Selective Forwarding Attack — Reliable-dRAG")
        print("  (Mirrors DRAG baseline run_selective_forwarding.py)")
        print("=" * 70)
        print(f"  sources={len(DATA_SOURCES)}  max_hops={args.max_hops}  "
              f"queries={args.num_queries}  hit_prob={args.peer_hit_prob}  seed={args.seed}")
        print(f"\n  Strategy                   Ratio  Comp  HitRate  AvgHops  TTLExh  Dropped")
        print("  " + "-" * 68)

        results = run_sweep(num_queries=args.num_queries, peer_hit_prob=args.peer_hit_prob,
                            max_hops=args.max_hops, ratios=args.ratios,
                            strategies=args.strategies, seed=args.seed)

        baseline    = next((r for r in results if r["strategy"]=="baseline"), None)
        baseline_hr = baseline["hit_rate"]            if baseline else 1.0
        baseline_te = baseline["ttl_exhaustion_rate"] if baseline else 0.0

        print(f"\n{'='*70}")
        print("  Summary: degradation vs baseline  (mirrors DRAG summary block)")
        print(f"{'='*70}")
        print(f"  Baseline hit_rate    : {baseline_hr:.3f}")
        print(f"  Baseline ttl_exhaust : {baseline_te:.3f}\n")
        for r in results:
            if r["strategy"]=="baseline": continue
            print(f"  {r['strategy']:22s}  ratio={r['attack_ratio']:.1f}  "
                  f"hit_rate={r['hit_rate']:.3f}  delta={r['hit_rate']-baseline_hr:+.3f}  "
                  f"ttl_exhaust={r['ttl_exhaustion_rate']:.3f}  "
                  f"delta_ttl={r['ttl_exhaustion_rate']-baseline_te:+.3f}")

        save_csv(results, csv_path)
        lp = save_json({"timestamp":ts,"attack_type":"selective_forward_mock",
                        "attack_config":{"mode":"mock","num_queries":args.num_queries,
                                         "peer_hit_prob":args.peer_hit_prob,
                                         "max_hops":args.max_hops,"ratios_swept":args.ratios,
                                         "strategies":args.strategies,"seed":args.seed},
                        "results":results}, f"_mock_seed{args.seed}")
        print(f"  JSON  -> {lp}")
        worst = min(results, key=lambda r: r["hit_rate"])
        print(f"\n  === Thesis Summary ===")
        print(f"  Worst hit_rate     : {worst['hit_rate']:.3f} "
              f"(ratio={worst['attack_ratio']}, strategy={worst['strategy']})")
        print(f"  Max ttl_exhaustion : {max(r['ttl_exhaustion_rate'] for r in results):.3f}")
        print("  Blockchain status  : all sources online — SFA undetectable")

    else:
        sfa = SelectiveForwardingAttack(attack_ratio=args.attack_ratio, seed=args.seed, api_key=args.api_key)
        results = sfa.run_live(eval_data=EVAL_DATA, k_per_source=args.k)
        lp = save_json({"timestamp":ts,"attack_type":"selective_forward_live","results":results},"_live")
        print(f"\n  Log -> {lp}")
        print(f"  Baseline: {results['baseline_accuracy']:.4f}  "
              f"Attacked: {results['attacked_accuracy']:.4f}  "
              f"Drop: {results['accuracy_drop_pp']:.1f}pp  "
              f"Success: {'YES' if results['attack_success'] else 'NO'}")

if __name__ == "__main__":
    main()
