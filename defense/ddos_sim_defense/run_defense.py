"""
defense/ddos_sim_defense/run_defense.py

Attack vs. attack+defense comparison runner for
defense.ddos_sim_defense.DDoSDefense, evaluated against
attack.ddos_sim.DDoSAttack. Same wave-by-wave structure as
attack/ddos_sim/run_attack.py's run_ddos_scenario(), with a third variant
added per wave: defended.

For each (strategy, ratio) this produces, per wave:
  attack_only          -- attack applied, defense inactive
  attack_plus_defense   -- attack applied, defense applied on top

`recovery = mean(attack_plus_defense.hit_rate) - mean(attack_only.hit_rate)`
across waves is the headline number.

Usage
-----
  python defense/ddos_sim_defense/run_defense.py --mode mock
  python defense/ddos_sim_defense/run_defense.py --mode mock --single --ratio 0.6 --strategy sequential
  docker compose up -d && python defense/ddos_sim_defense/run_defense.py --mode live --n_questions 30
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.config_loader import load_yaml  # noqa: E402
from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.selective_forward_sim.live_network import LiveRAGNetwork  # noqa: E402
from attack.selective_forward_sim.run_attack import _mock_questions, _live_questions  # noqa: E402
from attack.ddos_sim.ddos_attack import DDoSAttack  # noqa: E402
from attack.ddos_sim.run_attack import run_baseline, _chunks_covering  # noqa: E402
from defense.ddos_sim_defense.ddos_defense import DDoSDefense  # noqa: E402

ATTACK_CONFIG = "ddos_sim.yaml"
DEFENSE_CONFIG = "ddos_sim_defense.yaml"


def _run_scenario(
    build_network, questions: List[str], threshold: float, max_ttl: int,
    ratio: float, strategy: str, iterations: int, seed: int,
    ddos_cfg: Dict[str, Any], defense_cfg: Dict[str, Any], with_defense: bool,
) -> List[Dict[str, Any]]:
    network = build_network()
    attack = DDoSAttack(
        attack_ratio=ratio, iterations=iterations, strategy=strategy,
        target_peers=ddos_cfg.get("target_peers") or None,
        ddos_duration=ddos_cfg.get("ddos_duration", 60.0),
        wave_interval_s=ddos_cfg.get("wave_interval_s", 30.0),
        intensity_min=ddos_cfg.get("intensity_min", 0.5),
        intensity_max=ddos_cfg.get("intensity_max", 1.0),
        cascade_factor=ddos_cfg.get("cascade_factor", 0.25),
        max_cascade_intensity=ddos_cfg.get("max_cascade_intensity", 0.6),
        seed=seed,
    )
    attack.attach(network)

    defense = None
    if with_defense:
        defense = DDoSDefense(defense_cfg)
        defense.apply(network)

    batch_sizes = _chunks_covering(len(questions), iterations)
    rows: List[Dict[str, Any]] = []
    q_idx = 0
    dropped_before = 0
    label = "attack_plus_defense" if with_defense else "attack_only"

    for wave_num, batch_size in enumerate(batch_sizes):
        wave_info = attack.run_wave(network, wave_num)
        if defense is not None:
            defense.advance_wave(wave_num)

        batch = questions[q_idx: q_idx + batch_size]
        q_idx += batch_size
        answers = [network.topic_aware_query(q, threshold) for q in batch]
        metrics = attack.collect_metrics(answers, max_ttl=max_ttl)
        dropped_this_wave = attack.dropped_queries_total - dropped_before
        dropped_before = attack.dropped_queries_total

        row = {
            "wave": wave_num, "label": label, "strategy": wave_info["strategy"], "attack_ratio": ratio,
            "availability_percentage": wave_info["availability_percentage"],
            "active_nodes": wave_info["active_nodes"], "overloaded_count": wave_info["overloaded_count"],
            "dropped_queries": dropped_this_wave, **metrics,
        }
        if defense is not None:
            stats = defense.get_stats()
            row.update({
                "defense_blacklisted": stats["blacklisted_count"],
                "defense_bypasses": stats["total_bypasses"],
                "defense_recoveries": stats["total_recoveries"],
                "defense_redundant_probes": stats["redundant_probes"],
                "defense_redundant_probe_hits": stats["redundant_probe_hits"],
            })
        rows.append(row)

    if defense is not None:
        defense.remove(network)
    attack.detach(network)
    return rows


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    baseline = [r for r in rows if r.get("label") == "baseline"]
    if baseline:
        print(f"  Baseline hit_rate: {baseline[0]['hit_rate']:.3f}")

    by_key: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = {}
    for r in rows:
        if r.get("label") not in ("attack_only", "attack_plus_defense"):
            continue
        key = (r["strategy"], r["attack_ratio"])
        by_key.setdefault(key, {}).setdefault(r["label"], []).append(r)

    for (strategy, ratio), variants in sorted(by_key.items()):
        atk_rows = variants.get("attack_only", [])
        dfd_rows = variants.get("attack_plus_defense", [])
        if not atk_rows or not dfd_rows:
            continue
        atk_hit = sum(r["hit_rate"] for r in atk_rows) / len(atk_rows)
        dfd_hit = sum(r["hit_rate"] for r in dfd_rows) / len(dfd_rows)
        atk_avail = sum(r["availability_percentage"] for r in atk_rows) / len(atk_rows)
        dfd_avail = sum(r["availability_percentage"] for r in dfd_rows) / len(dfd_rows)
        bl = dfd_rows[-1].get("defense_blacklisted", 0)
        rp = sum(r.get("defense_redundant_probes", 0) for r in dfd_rows)
        rph = sum(r.get("defense_redundant_probe_hits", 0) for r in dfd_rows)
        recovery = dfd_hit - atk_hit
        print(
            f"  {strategy:<12} ratio={ratio:.2f}  avail(attack={atk_avail:5.1f}% defended={dfd_avail:5.1f}%)"
            f"  hit_rate(attack={atk_hit:.3f} defended={dfd_hit:.3f})  recovery={recovery:+.3f}"
            f"   [blacklisted={bl} redundant_probes={rp} redundant_hits={rph}]"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="DDoS attack vs. attack+defense comparison runner")
    p.add_argument("--mode", choices=["mock", "live"], default=None)
    p.add_argument("--attack_config", default=ATTACK_CONFIG)
    p.add_argument("--defense_config", default=DEFENSE_CONFIG)
    p.add_argument("--single", action="store_true")
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--strategy", choices=["random", "targeted", "sequential"], default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    # mock overrides
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_ttl", type=int, default=None)
    p.add_argument("--num_queries", type=int, default=None)
    # defense overrides
    p.add_argument("--deprioritize_threshold", type=float, default=None)
    p.add_argument("--min_queries_before_action", type=int, default=None)
    p.add_argument("--reputation_decay", type=float, default=None)
    p.add_argument("--redundancy_k", type=int, default=None)
    p.add_argument("--max_blacklist_fraction", type=float, default=None)
    p.add_argument("--backoff_waves", type=int, default=None)
    # live overrides
    p.add_argument("--n_questions", type=int, default=None)
    args = p.parse_args()

    atk_cfg = load_yaml(args.attack_config)
    dfn_cfg_full = load_yaml(args.defense_config)
    mode = args.mode or atk_cfg.get("mode", "mock")

    defense_cfg = dict(dfn_cfg_full.get("ddos_defense", {}))
    for flag, key in [
        ("deprioritize_threshold", "deprioritize_threshold"), ("min_queries_before_action", "min_queries_before_action"),
        ("reputation_decay", "reputation_decay"), ("redundancy_k", "redundancy_k"),
        ("max_blacklist_fraction", "max_blacklist_fraction"), ("backoff_waves", "backoff_waves"),
    ]:
        val = getattr(args, flag)
        if val is not None:
            defense_cfg[key] = val

    log_dir = os.path.join(_ROOT, dfn_cfg_full.get("output", {}).get("log_dir", "defense_logs/ddos_sim_defense"))
    os.makedirs(log_dir, exist_ok=True)

    ddos_cfg = dict(atk_cfg.get("ddos", {}))
    seed = args.seed if args.seed is not None else ddos_cfg.get("seed", 42)
    iterations = args.iterations if args.iterations is not None else ddos_cfg.get("iterations", 5)
    if args.duration is not None:
        ddos_cfg["ddos_duration"] = args.duration
    sweep = atk_cfg.get("sweep", {})
    ratios = sweep.get("ratios", [0.1, 0.3, 0.6])
    strategies = sweep.get("strategies", ["random", "targeted", "sequential"])

    if mode == "mock":
        net_cfg = dict(atk_cfg.get("network", {}))
        sim_cfg = dict(atk_cfg.get("simulation", {}))
        if args.num_peers is not None:
            net_cfg["num_peers"] = args.num_peers
        if args.max_ttl is not None:
            net_cfg["query_ttl"] = args.max_ttl
        if args.num_queries is not None:
            sim_cfg["num_queries"] = args.num_queries
        sim_cfg["seed"] = seed

        def build_network():
            return MockRAGNetwork(
                num_peers=net_cfg["num_peers"], num_attachments=net_cfg["num_attachments"],
                num_query_neighbor=net_cfg["num_query_neighbor"], query_ttl=net_cfg["query_ttl"],
                peer_hit_prob=sim_cfg["peer_hit_prob"], seed=seed,
            )

        threshold = net_cfg["query_confidence_threshold"]
        max_ttl = net_cfg["query_ttl"]
        questions = _mock_questions(sim_cfg["num_queries"], seed)
        run_config: Dict[str, Any] = {"mode": "mock", "network": net_cfg, "simulation": sim_cfg, "ddos": ddos_cfg}

    else:  # live
        live_cfg = dict(atk_cfg.get("live", {}))
        if args.max_ttl is not None:
            live_cfg["query_ttl"] = args.max_ttl
        if args.n_questions is not None:
            live_cfg["n_questions"] = args.n_questions

        def build_network():
            return LiveRAGNetwork(
                source_urls=live_cfg.get("source_urls"),
                num_query_neighbor=live_cfg.get("num_query_neighbor", 2),
                query_ttl=live_cfg.get("query_ttl", 3),
                api_key=live_cfg.get("api_key", "reliable-derag-secret-2026"),
                blockchain_url=live_cfg.get("blockchain_url", "http://localhost:8545"),
                use_onchain_scores=live_cfg.get("use_onchain_scores", True),
                seed=seed,
                min_request_interval_s=live_cfg.get("min_request_interval_s", 1.1),
            )

        probe = build_network()
        reachable = probe.ping_all()
        print(f"  [+] Source reachability: {reachable}")
        if not any(reachable.values()):
            print("\n  [!] No Docker data-source nodes are reachable.")
            print("      Start them with:  docker compose up -d   (repo root)")
            print("      Or run with --mode mock for an offline simulation.\n")
            sys.exit(1)

        threshold = 0.5
        max_ttl = live_cfg.get("query_ttl", 3)
        questions = _live_questions(live_cfg.get("n_questions", 30), seed)
        print(f"  [+] Loaded {len(questions)} questions")
        run_config = {"mode": "live", "live": live_cfg, "ddos": ddos_cfg}

    rows: List[Dict[str, Any]] = [run_baseline(build_network(), questions, threshold, max_ttl)]
    scenarios = [(args.ratio if args.ratio is not None else ddos_cfg.get("attack_ratio", 0.3),
                  args.strategy or ddos_cfg.get("strategy", "random"))] if args.single else \
        [(ratio, strategy) for strategy in strategies for ratio in ratios]

    for ratio, strategy in scenarios:
        rows += _run_scenario(build_network, questions, threshold, max_ttl, ratio, strategy,
                               iterations, seed, ddos_cfg, defense_cfg, with_defense=False)
        rows += _run_scenario(build_network, questions, threshold, max_ttl, ratio, strategy,
                               iterations, seed, ddos_cfg, defense_cfg, with_defense=True)

    print()
    _print_summary(rows)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(log_dir, f"defense_{ts}_ddos_sim_{mode}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "defense_type": "ddos_sim_mitigation",
                "config": {**run_config, "defense": defense_cfg},
                "results": rows,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  [+] JSON log written to {json_path}")

    csv_path = os.path.join(log_dir, f"defense_{ts}_ddos_sim_{mode}_seed{seed}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            keys = sorted({k for r in rows for k in r.keys() if not isinstance(r[k], (dict, list))})
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print(f"  [+] CSV log written to {csv_path}")


if __name__ == "__main__":
    main()
