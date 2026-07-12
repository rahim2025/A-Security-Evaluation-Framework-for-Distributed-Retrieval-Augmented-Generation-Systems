"""
defense/sfa_sim_defense/run_defense.py

Attack vs. attack+defense comparison runner for
defense.sfa_sim_defense.SelectiveForwardingDefense, evaluated against
attack.selective_forward_sim.SelectiveForwardingAttack.

For each (strategy, ratio) in the sweep this produces three rows:
  baseline           -- no attack, no defense
  attack_only        -- attack applied, defense inactive
  attack_plus_defense -- attack applied, defense applied on top

`recovery = attack_plus_defense.hit_rate - attack_only.hit_rate` is the
headline number: how much of the hit-rate collapse the defense
recovered.

Ordering note (live mode): for each (strategy, ratio) pair, whether
attack_only or attack_plus_defense actually runs first against the live
containers is randomized per trial (seeded by --seed/trial index). Running
attack_only before attack_plus_defense every single time, against the same
real containers in one long sequential sweep, means attack_plus_defense
always carries strictly more cumulative real-world load/drift -- a bias
toward making the defense look worse that has nothing to do with its
logic. Randomizing which one goes first removes that systematic bias (it
doesn't eliminate load entirely -- see --scenario_delay_s below for that).

Usage
-----
  # Mock sweep (default; no Docker/blockchain needed)
  python defense/sfa_sim_defense/run_defense.py --mode mock

  # Average over multiple trials for statistical stability
  python defense/sfa_sim_defense/run_defense.py --mode mock --trials 5

  # Live sweep against the real Docker sources + on-chain ledger
  python defense/sfa_sim_defense/run_defense.py --mode live --n_questions 50

  # Live sweep with a pause between scenarios to avoid saturating the
  # containers on a long sweep (recommended for --n_questions > ~30 live)
  python defense/sfa_sim_defense/run_defense.py --mode live --n_questions 50 --scenario_delay_s 1.0
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.config_loader import load_yaml  # noqa: E402
from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.selective_forward_sim.live_network import LiveRAGNetwork  # noqa: E402
from attack.selective_forward_sim.selective_forwarding_attack import (  # noqa: E402
    SelectiveForwardingAttack,
)
from attack.selective_forward_sim.run_attack import _mock_questions, _live_questions  # noqa: E402
from defense.sfa_sim_defense.selective_forwarding_defense import (  # noqa: E402
    SelectiveForwardingDefense,
)

ATTACK_CONFIG = "selective_forwarding_sim.yaml"
DEFENSE_CONFIG = "sfa_sim_defense.yaml"


def _run_once(
    build_network, questions: List[str], threshold: float, max_ttl: int,
    ratio: float, strategy: str, seed: int, defense_cfg: Dict[str, Any], with_defense: bool,
    drop_rate=1.0,
) -> Dict[str, Any]:
    network = build_network()

    attack = SelectiveForwardingAttack(attack_ratio=ratio, drop_rate=drop_rate, seed=seed)
    attack_info: Dict[str, Any] = {}
    if ratio > 0:
        attack_info = attack.apply(network, strategy=strategy)

    defense = None
    if with_defense:
        defense = SelectiveForwardingDefense(defense_cfg)
        defense.apply(network)

    answers = [network.topic_aware_query(q, threshold) for q in questions]
    metrics = attack.collect_metrics(answers, max_ttl=max_ttl)

    defense_stats: Dict[str, Any] = {}
    if defense is not None:
        stats = defense.get_stats()
        defense_stats = {
            "defense_blacklisted": stats["blacklisted_count"],
            "defense_bypasses": stats["total_bypasses"],
            "defense_redundant_probes": stats["redundant_probes"],
            "defense_redundant_probe_hits": stats["redundant_probe_hits"],
        }
        defense.remove(network)

    if ratio > 0:
        attack.revert(network)

    label = "baseline" if ratio == 0 else ("attack_plus_defense" if with_defense else "attack_only")
    return {
        "label": label, "strategy": strategy if ratio > 0 else "baseline", "attack_ratio": ratio,
        "num_compromised": attack_info.get("num_compromised", 0),
        **metrics, **defense_stats,
    }


def _average_trials(rows_per_trial: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if len(rows_per_trial) == 1:
        return rows_per_trial[0]
    averaged = []
    numeric_keys = ["hit_rate", "avg_hops_per_query", "ttl_exhaustion_rate", "dropped_queries",
                     "defense_blacklisted", "defense_bypasses",
                     "defense_redundant_probes", "defense_redundant_probe_hits"]
    for i in range(len(rows_per_trial[0])):
        base = dict(rows_per_trial[0][i])
        for key in numeric_keys:
            vals = [trial[i].get(key, 0) for trial in rows_per_trial if key in trial[i]]
            if vals:
                base[key] = sum(vals) / len(vals)
        averaged.append(base)
    return averaged


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    baseline_hit_rate = None
    for r in rows:
        if r["label"] == "baseline":
            baseline_hit_rate = r["hit_rate"]
            continue
        key = (r["strategy"], r["attack_ratio"])
        by_key.setdefault(key, {})[r["label"]] = r

    if baseline_hit_rate is not None:
        print(f"  Baseline hit_rate: {baseline_hit_rate:.3f}")
    for (strategy, ratio), variants in sorted(by_key.items()):
        atk = variants.get("attack_only", {}).get("hit_rate")
        dfd_row = variants.get("attack_plus_defense", {})
        dfd = dfd_row.get("hit_rate")
        if atk is None or dfd is None:
            continue
        recovery = dfd - atk
        bl = dfd_row.get("defense_blacklisted", 0)
        rp = dfd_row.get("defense_redundant_probes", 0)
        rph = dfd_row.get("defense_redundant_probe_hits", 0)
        print(f"  {strategy:<20} ratio={ratio:.2f}  attack={atk:.3f}  defended={dfd:.3f}  recovery={recovery:+.3f}"
              f"   [blacklisted={bl} redundant_probes={rp} redundant_hits={rph}]")


def main() -> None:
    p = argparse.ArgumentParser(description="SFA attack vs. attack+defense comparison runner")
    p.add_argument("--mode", choices=["mock", "live"], default=None)
    p.add_argument("--attack_config", default=ATTACK_CONFIG)
    p.add_argument("--defense_config", default=DEFENSE_CONFIG)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    # mock overrides
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_ttl", type=int, default=None)
    p.add_argument("--num_queries", type=int, default=None)
    # defense overrides
    p.add_argument("--blacklist_threshold", type=float, default=None)
    p.add_argument("--min_queries", type=int, default=None)
    p.add_argument("--suspicion_threshold", type=float, default=None)
    p.add_argument("--reputation_decay", type=float, default=None)
    p.add_argument("--redundancy_k", type=int, default=None,
                    help="bounded extra backup-probe attempts once the primary hop budget is exhausted")
    p.add_argument("--max_blacklist_fraction", type=float, default=None,
                    help="cap on the fraction of known peers auto-blacklisting may exclude at once")
    p.add_argument("--detection_mode", choices=["threshold", "binomial"], default=None,
                    help="'threshold' (default): fixed response-rate floor, blind to a stealthy "
                         "attacker. 'binomial': one-sided significance test vs honest_miss_rate, "
                         "catches a stealthy (10-30%%) dropper -- see selective_forwarding_defense.py")
    p.add_argument("--honest_miss_rate", type=float, default=None,
                    help="binomial mode only: measured honest-peer miss rate to test against")
    p.add_argument("--binom_alpha", type=float, default=None, help="binomial mode only: significance threshold")
    p.add_argument("--streak_required", type=int, default=None,
                    help="binomial mode only: consecutive significant windows required before blacklisting")
    p.add_argument("--drop_rate", default=None,
                    help="attack drop rate: '1.0' (default, black-hole) or 'stealthy' "
                         "(Uniform(0.10,0.30) per compromised peer) or any float 0.0-1.0")
    # live overrides
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--min_request_interval_s", type=float, default=None,
                    help="minimum seconds between requests to the same live source (default 1.1s, "
                         "confirmed safely under the real deployment's 60/min per-source rate limit -- "
                         "see reports/SFA_Security_Analysis_Report.md sec 12.9)")
    p.add_argument("--scenario_delay_s", type=float, default=0.0,
                    help="pause (seconds) before each scenario -- recommended for long live sweeps "
                         "to avoid saturating the Docker containers")
    args = p.parse_args()

    atk_cfg = load_yaml(args.attack_config)
    dfn_cfg_full = load_yaml(args.defense_config)
    mode = args.mode or atk_cfg.get("mode", "mock")

    defense_cfg = dict(dfn_cfg_full.get("selective_forwarding_defense", {}))
    if args.blacklist_threshold is not None:
        defense_cfg["blacklist_threshold"] = args.blacklist_threshold
    if args.min_queries is not None:
        defense_cfg["min_queries_before_blacklist"] = args.min_queries
    if args.suspicion_threshold is not None:
        defense_cfg["suspicion_threshold"] = args.suspicion_threshold
    if args.reputation_decay is not None:
        defense_cfg["reputation_decay"] = args.reputation_decay
    if args.redundancy_k is not None:
        defense_cfg["redundancy_k"] = args.redundancy_k
    if args.max_blacklist_fraction is not None:
        defense_cfg["max_blacklist_fraction"] = args.max_blacklist_fraction
    if args.detection_mode is not None:
        defense_cfg["detection_mode"] = args.detection_mode
    if args.honest_miss_rate is not None:
        defense_cfg["honest_miss_rate"] = args.honest_miss_rate
    if args.binom_alpha is not None:
        defense_cfg["binom_alpha"] = args.binom_alpha
    if args.streak_required is not None:
        defense_cfg["streak_required"] = args.streak_required

    log_dir = os.path.join(_ROOT, dfn_cfg_full.get("output", {}).get("log_dir", "defense_logs/sfa_sim_defense"))
    os.makedirs(log_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else atk_cfg.get("selective_forwarding", {}).get("seed", 42)
    drop_rate_raw = args.drop_rate if args.drop_rate is not None else atk_cfg.get("selective_forwarding", {}).get("drop_rate", 1.0)
    drop_rate = drop_rate_raw if drop_rate_raw == "stealthy" else float(drop_rate_raw)
    sweep = atk_cfg.get("sweep", {})
    ratios = sweep.get("ratios", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    strategies = sweep.get("strategies", ["random", "high_connectivity"])

    if mode == "mock":
        net_cfg = dict(atk_cfg.get("network", {}))
        sim_cfg = dict(atk_cfg.get("simulation", {}))
        if args.num_peers is not None:
            net_cfg["num_peers"] = args.num_peers
        if args.max_ttl is not None:
            net_cfg["query_ttl"] = args.max_ttl
        if args.num_queries is not None:
            sim_cfg["num_queries"] = args.num_queries

        def build_network():
            return MockRAGNetwork(
                num_peers=net_cfg["num_peers"], num_attachments=net_cfg["num_attachments"],
                num_query_neighbor=net_cfg["num_query_neighbor"], query_ttl=net_cfg["query_ttl"],
                peer_hit_prob=sim_cfg["peer_hit_prob"], seed=seed,
            )

        threshold = net_cfg["query_confidence_threshold"]
        max_ttl = net_cfg["query_ttl"]
        questions = _mock_questions(sim_cfg["num_queries"], seed)
        run_config: Dict[str, Any] = {"mode": "mock", "network": net_cfg, "simulation": sim_cfg, "drop_rate": drop_rate}

    else:  # live
        live_cfg = dict(atk_cfg.get("live", {}))
        if args.max_ttl is not None:
            live_cfg["query_ttl"] = args.max_ttl
        if args.n_questions is not None:
            live_cfg["n_questions"] = args.n_questions
        if args.min_request_interval_s is not None:
            live_cfg["min_request_interval_s"] = args.min_request_interval_s

        # NOTE: build_network() is called fresh once per scenario (once per
        # _run_once() call below, not once for the whole sweep like
        # attack/selective_forward_sim/run_attack.py's live mode) -- each
        # LiveRAGNetwork/LivePeer's self-throttle state therefore resets at
        # every scenario boundary. The per-request throttle still paces the
        # dominant source of load (the n_questions loop *within* one
        # scenario), but use --scenario_delay_s alongside it for a live
        # sweep with several scenarios, so requests at scenario boundaries
        # don't burst against the real 60/min-per-source limit either.
        _live_networks: List[LiveRAGNetwork] = []

        def build_network():
            net = LiveRAGNetwork(
                source_urls=live_cfg.get("source_urls"),
                num_query_neighbor=live_cfg.get("num_query_neighbor", 2),
                query_ttl=live_cfg.get("query_ttl", 3),
                api_key=live_cfg.get("api_key", "reliable-derag-secret-2026"),
                blockchain_url=live_cfg.get("blockchain_url", "http://localhost:8545"),
                use_onchain_scores=live_cfg.get("use_onchain_scores", True),
                seed=seed,
                min_request_interval_s=live_cfg.get("min_request_interval_s", 1.1),
            )
            _live_networks.append(net)
            return net

        probe = build_network()
        reachable = probe.ping_all()
        print(f"  [+] Source reachability: {reachable}")
        if not any(reachable.values()):
            print("\n  [!] No Docker data-source nodes are reachable.")
            print("      Start them with:  docker compose up -d   (repo root)")
            print("      Or run with --mode mock for an offline simulation.\n")
            sys.exit(1)
        if probe.onchain_reliability:
            print(f"  [+] On-chain reliability scores: {probe.onchain_reliability}")

        threshold = 0.5
        max_ttl = live_cfg.get("query_ttl", 3)
        questions = _live_questions(live_cfg.get("n_questions", 50), seed)
        print(f"  [+] Loaded {len(questions)} questions")
        run_config = {"mode": "live", "live": live_cfg, "onchain_reliability": probe.onchain_reliability,
                      "drop_rate": drop_rate}

    trials_rows: List[List[Dict[str, Any]]] = []
    for trial in range(args.trials):
        trial_seed = seed + trial
        order_rng = random.Random(trial_seed)  # decides attack_only-vs-defended run order per pair

        if args.scenario_delay_s:
            time.sleep(args.scenario_delay_s)
        rows: List[Dict[str, Any]] = [
            _run_once(build_network, questions, threshold, max_ttl, 0.0, "baseline",
                      trial_seed, defense_cfg, with_defense=False, drop_rate=drop_rate)
        ]
        for strategy in strategies:
            for ratio in [r for r in ratios if r > 0]:
                # Randomize which variant actually runs first against the live
                # containers -- always running attack_only before
                # attack_plus_defense would give the defended run strictly
                # more cumulative real-world load every single time, biasing
                # it to look worse independent of the defense's logic. See
                # the module docstring's "Ordering note".
                variant_order = [False, True]
                if order_rng.random() < 0.5:
                    variant_order.reverse()

                results: Dict[bool, Dict[str, Any]] = {}
                for with_defense in variant_order:
                    if args.scenario_delay_s:
                        time.sleep(args.scenario_delay_s)
                    results[with_defense] = _run_once(
                        build_network, questions, threshold, max_ttl, ratio, strategy,
                        trial_seed, defense_cfg, with_defense=with_defense, drop_rate=drop_rate,
                    )
                rows.append(results[False])
                rows.append(results[True])
        trials_rows.append(rows)

    rows = _average_trials(trials_rows)

    if mode == "live":
        rate_limited_total = sum(net.rate_limited_total() for net in _live_networks)
        error_total = sum(net.error_total() for net in _live_networks)
        run_config["rate_limited_total"] = rate_limited_total
        run_config["error_total"] = error_total
        if rate_limited_total:
            print(f"\n  [!] {rate_limited_total} request(s) hit the live sources' rate limit (HTTP 429) "
                  "across this run -- the results below are not fully trustworthy. Increase "
                  "--min_request_interval_s and/or --scenario_delay_s and re-run.")
        if error_total > rate_limited_total:
            other = error_total - rate_limited_total
            print(f"\n  [!] {other} request(s) failed for a reason OTHER than rate-limiting (timeout / "
                  "connection error / non-200 status / malformed response) across this run -- the "
                  "results below are not fully trustworthy even though rate_limited_total may read 0. "
                  "Check error_total in the JSON log, not just rate_limited_total.")

    print()
    _print_summary(rows)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(log_dir, f"defense_{ts}_sfa_sim_{mode}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "defense_type": "selective_forwarding_sim_mitigation",
                "config": {**run_config, "defense": defense_cfg, "trials": args.trials},
                "results": rows,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  [+] JSON log written to {json_path}")

    csv_path = os.path.join(log_dir, f"defense_{ts}_sfa_sim_{mode}_seed{seed}.csv")
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
