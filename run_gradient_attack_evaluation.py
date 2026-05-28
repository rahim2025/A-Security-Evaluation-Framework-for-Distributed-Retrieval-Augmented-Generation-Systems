#!/usr/bin/env python3
"""
run_gradient_attack_evaluation.py
===================================
Evaluates gradient-level attacks on FedRAG's federated fine-tuning pipeline.
These attacks are unique to FedRAG (DRAG has no fine-tuning round).

Attack types:
  untargeted  — corrupt global model gradients to degrade answer quality
  targeted    — implant a backdoor trigger in the fine-tuned model
  inversion   — reconstruct private training data from shared gradient updates

USAGE
-----
  python run_gradient_attack_evaluation.py
  python run_gradient_attack_evaluation.py --attack-type targeted --scale 8.0
  python run_gradient_attack_evaluation.py --num-clients 20 --malicious-ratio 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).parent.resolve()
for _p in (_REPO, _REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedRAG — Gradient-Level Attack Evaluation")
    p.add_argument("--num-clients",    type=int,   default=10)
    p.add_argument("--num-examples",   type=int,   default=200)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--malicious-ratio",type=float, default=0.3,
                   help="Fraction of clients that are malicious (default: 0.3)")
    p.add_argument("--attack-type",    type=str,   default="all",
                   choices=["untargeted", "targeted", "inversion", "all"])
    p.add_argument("--scale",          type=float, default=5.0,
                   help="Gradient poisoning scale / magnification (default: 5.0)")
    p.add_argument("--num-rounds",     type=int,   default=5,
                   help="Number of federated training rounds to simulate (default: 5)")
    p.add_argument("--output-dir",     type=str,   default="results/gradient")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from security_framework.simulator import FedRAGSimulator
    from fed_rag.attacks.gradient_poisoning import GradientPoisoningAttack
    import random

    sim = FedRAGSimulator(
        num_clients=args.num_clients,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    n_mal = max(1, int(sim.num_clients * args.malicious_ratio))
    malicious_ids = sorted(rng.sample(range(sim.num_clients), n_mal))

    attack_types = (
        ["untargeted", "targeted", "inversion"]
        if args.attack_type == "all"
        else [args.attack_type]
    )

    W = 78
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + "  FedRAG — Gradient-Level Attack Evaluation".center(W - 2) + "║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  Clients         : {sim.num_clients:<{W-22}}║")
    print(f"║  Examples        : {sim.num_examples:<{W-22}}║")
    print(f"║  Malicious IDs   : {str(malicious_ids):<{W-22}}║")
    print(f"║  Poisoning scale : {args.scale:<{W-22}}║")
    print(f"║  Rounds          : {args.num_rounds:<{W-22}}║")
    print(f"║  Attack types    : {', '.join(attack_types):<{W-22}}║")
    print("╚" + "═" * (W - 2) + "╝")
    print()

    results = []
    for atype in attack_types:
        print(f"  Running {atype.upper()} gradient attack...")
        atk = GradientPoisoningAttack(
            attack_type=atype,
            poisoning_scale=args.scale,
            seed=args.seed,
        )
        result = atk.execute(
            clients=sim.clients,
            malicious_client_ids=malicious_ids,
            num_rounds=args.num_rounds,
            baseline_f1=1.0,
        )
        print(f"  {result.summary}")
        print()
        results.append({
            "attack_type": result.attack_type,
            "malicious_clients": result.malicious_clients,
            "poisoning_scale": result.poisoning_scale,
            "baseline_f1": result.baseline_f1,
            "attacked_f1": result.attacked_f1,
            "delta_f1": result.delta_f1,
            "inversion_exposure": result.inversion_exposure,
            "backdoor_success_rate": result.backdoor_success_rate,
            "num_rounds": result.num_rounds,
        })

    # Summary table
    print("  ┌─ Gradient Attack Summary " + "─" * 50 + "┐")
    print(f"  │  {'Attack Type':<14} {'Baseline F1':>11} {'Attacked F1':>11} {'Δ F1':>8} {'Inversion':>10} {'Backdoor':>10}  │")
    print("  ├" + "─" * 74 + "┤")
    for r in results:
        print(f"  │  {r['attack_type']:<14} {r['baseline_f1']:>11.4f} {r['attacked_f1']:>11.4f} "
              f"{r['delta_f1']:>+8.4f} {r['inversion_exposure']:>9.1%} {r['backdoor_success_rate']:>9.1%}  │")
    print("  └" + "─" * 74 + "┘")
    print()

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "gradient_attack_results.json"
    json_path.write_text(json.dumps({"results": results, "config": vars(args)}, indent=2))
    print(f"  Saved → {json_path}")


if __name__ == "__main__":
    main()
