"""
attack/ddos_sim/run_attack.py

CLI runner for the congestion-based DDoS simulation attack. Config-driven
via config/ddos_sim.yaml (CLI flags override YAML defaults). Deliberately
reuses attack/selective_forward_sim's network classes and question sources
rather than duplicating them -- DDoSAttack only needs the same
`.peers` / `.network` (networkx graph) / `.topic_aware_query()` surface
SelectiveForwardingAttack already drives.

Modes
-----
mock  Entirely in-process (networkx Barabasi-Albert graph of synthetic
      MockPeers). No Docker/blockchain required -- sweeps
      strategy x attack_ratio in seconds, with one row per wave so you can
      watch availability collapse (or not) across iterations.

live  Targets the real docker-compose data sources (source_0/20/100).
      Requires `docker compose up -d` at the repo root; falls back to
      unweighted targeting if the Hardhat node isn't reachable.

Usage
-----
  # Mock sweep across all ratios/strategies in the config (default)
  python attack/ddos_sim/run_attack.py --mode mock

  # Reproduce the report's collapse scenario: high ratio, long recovery
  # window relative to wave cadence
  python attack/ddos_sim/run_attack.py --mode mock --single \
      --ratio 0.6 --strategy sequential --duration 600 --iterations 4

  # Live sweep against the real Docker sources
  python attack/ddos_sim/run_attack.py --mode live --n_questions 30
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
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
from attack.selective_forward_sim.run_attack import _mock_questions, _live_questions  # noqa: E402
from attack.ddos_sim.ddos_attack import DDoSAttack  # noqa: E402

DEFAULT_CONFIG = "ddos_sim.yaml"


# ══════════════════════════════════════════════════════════════════════════
#  One scenario: build network, run `iterations` waves, one row per wave
# ══════════════════════════════════════════════════════════════════════════

def _chunks_covering(total: int, parts: int) -> List[int]:
    """Split `total` questions into `parts` near-equal, non-empty batches."""
    parts = max(1, parts)
    base = max(1, total // parts)
    sizes = [base] * parts
    remainder = total - base * parts
    for i in range(max(0, remainder)):
        sizes[i % parts] += 1
    return [s for s in sizes if s > 0] or [total]


def run_baseline(network, questions: List[str], threshold: float, max_ttl: int) -> Dict[str, Any]:
    answers = [network.topic_aware_query(q, threshold) for q in questions]
    attack = DDoSAttack()  # unused for metrics other than the shared collect_metrics() shape
    metrics = attack.collect_metrics(answers, max_ttl=max_ttl)
    snapshot = {
        "availability_percentage": 100.0, "active_nodes": network.num_peers,
        "total_peers": network.num_peers, "overloaded_count": 0, "down_count": 0,
        "avg_load_intensity": 0.0,
    }
    return {
        "wave": 0, "strategy": "baseline", "attack_ratio": 0.0,
        "targeted_count": 0, "cascaded_count": 0, "dropped_queries": 0,
        **snapshot, **metrics,
    }


def run_ddos_scenario(
    network, questions: List[str], threshold: float, max_ttl: int,
    ratio: float, strategy: str, iterations: int, seed: int, ddos_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
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

    batch_sizes = _chunks_covering(len(questions), iterations)
    rows: List[Dict[str, Any]] = []
    q_idx = 0
    dropped_before = 0
    for wave_num, batch_size in enumerate(batch_sizes):
        wave_info = attack.run_wave(network, wave_num)
        batch = questions[q_idx: q_idx + batch_size]
        q_idx += batch_size

        answers = [network.topic_aware_query(q, threshold) for q in batch]
        metrics = attack.collect_metrics(answers, max_ttl=max_ttl)
        dropped_this_wave = attack.dropped_queries_total - dropped_before
        dropped_before = attack.dropped_queries_total

        rows.append({
            "wave": wave_num, "strategy": wave_info["strategy"], "attack_ratio": ratio,
            "targeted_count": len(wave_info["targeted"]), "cascaded_count": len(wave_info["cascaded"]),
            "dropped_queries": dropped_this_wave,
            "availability_percentage": wave_info["availability_percentage"],
            "active_nodes": wave_info["active_nodes"], "total_peers": wave_info["total_peers"],
            "overloaded_count": wave_info["overloaded_count"], "down_count": wave_info["down_count"],
            "avg_load_intensity": wave_info["avg_load_intensity"],
            **metrics,
        })

    attack.detach(network)
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  Sweep + reporting
# ══════════════════════════════════════════════════════════════════════════

def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'strategy':<12}{'ratio':>6}{'wave':>5}{'avail%':>8}{'active':>7}"
        f"{'overld':>7}{'hit_rate':>9}{'dropped':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['strategy']:<12}{r['attack_ratio']:>6.2f}{r['wave']:>5}"
            f"{r['availability_percentage']:>8.1f}{r['active_nodes']:>7}"
            f"{r['overloaded_count']:>7}{r['hit_rate']:>9.3f}{r['dropped_queries']:>8}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Congestion-based DDoS simulation runner")
    p.add_argument("--mode", choices=["mock", "live"], default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--single", action="store_true", help="Run a single ratio/strategy instead of the full sweep")
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--strategy", choices=["random", "targeted", "sequential"], default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--duration", type=float, default=None, help="ddos_duration in seconds (auto-recovery window)")
    p.add_argument("--wave_interval_s", type=float, default=None)
    p.add_argument("--intensity_min", type=float, default=None)
    p.add_argument("--intensity_max", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    # mock overrides
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--num_attachments", type=int, default=None)
    p.add_argument("--num_query_neighbor", type=int, default=None)
    p.add_argument("--max_ttl", type=int, default=None)
    p.add_argument("--num_queries", type=int, default=None)
    p.add_argument("--peer_hit_prob", type=float, default=None)
    p.add_argument("--confidence_threshold", type=float, default=None)
    # live overrides
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--blockchain_url", default=None)
    p.add_argument("--min_request_interval_s", type=float, default=None)
    p.add_argument("--scenario_delay_s", type=float, default=0.0)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    mode = args.mode or cfg.get("mode", "mock")

    log_dir = os.path.join(_ROOT, cfg.get("output", {}).get("log_dir", "attack_logs/ddos_sim"))
    os.makedirs(log_dir, exist_ok=True)

    ddos_cfg = dict(cfg.get("ddos", {}))
    seed = args.seed if args.seed is not None else ddos_cfg.get("seed", 42)
    iterations = args.iterations if args.iterations is not None else ddos_cfg.get("iterations", 5)
    if args.duration is not None:
        ddos_cfg["ddos_duration"] = args.duration
    if args.wave_interval_s is not None:
        ddos_cfg["wave_interval_s"] = args.wave_interval_s
    if args.intensity_min is not None:
        ddos_cfg["intensity_min"] = args.intensity_min
    if args.intensity_max is not None:
        ddos_cfg["intensity_max"] = args.intensity_max

    if mode == "mock":
        net_cfg = dict(cfg.get("network", {}))
        sim_cfg = dict(cfg.get("simulation", {}))
        if args.num_peers is not None:
            net_cfg["num_peers"] = args.num_peers
        if args.num_attachments is not None:
            net_cfg["num_attachments"] = args.num_attachments
        if args.num_query_neighbor is not None:
            net_cfg["num_query_neighbor"] = args.num_query_neighbor
        if args.max_ttl is not None:
            net_cfg["query_ttl"] = args.max_ttl
        if args.confidence_threshold is not None:
            net_cfg["query_confidence_threshold"] = args.confidence_threshold
        if args.num_queries is not None:
            sim_cfg["num_queries"] = args.num_queries
        if args.peer_hit_prob is not None:
            sim_cfg["peer_hit_prob"] = args.peer_hit_prob
        sim_cfg["seed"] = seed

        def build_network():
            return MockRAGNetwork(
                num_peers=net_cfg["num_peers"], num_attachments=net_cfg["num_attachments"],
                num_query_neighbor=net_cfg["num_query_neighbor"], query_ttl=net_cfg["query_ttl"],
                peer_hit_prob=sim_cfg["peer_hit_prob"], seed=sim_cfg["seed"],
            )

        threshold = net_cfg["query_confidence_threshold"]
        max_ttl = net_cfg["query_ttl"]
        questions = _mock_questions(sim_cfg["num_queries"], seed)

        rows: List[Dict[str, Any]] = [run_baseline(build_network(), questions, threshold, max_ttl)]
        if args.single:
            ratio = args.ratio if args.ratio is not None else ddos_cfg.get("attack_ratio", 0.3)
            strategy = args.strategy or ddos_cfg.get("strategy", "random")
            rows += run_ddos_scenario(build_network(), questions, threshold, max_ttl,
                                       ratio, strategy, iterations, seed, ddos_cfg)
        else:
            sweep = cfg.get("sweep", {})
            ratios = sweep.get("ratios", [0.1, 0.3, 0.6])
            strategies = sweep.get("strategies", ["random", "targeted", "sequential"])
            for strategy in strategies:
                for ratio in ratios:
                    if args.scenario_delay_s:
                        time.sleep(args.scenario_delay_s)
                    rows += run_ddos_scenario(build_network(), questions, threshold, max_ttl,
                                               ratio, strategy, iterations, seed, ddos_cfg)

        run_config = {"mode": "mock", "network": net_cfg, "simulation": sim_cfg, "ddos": ddos_cfg}

    else:  # live
        live_cfg = dict(cfg.get("live", {}))
        if args.num_query_neighbor is not None:
            live_cfg["num_query_neighbor"] = args.num_query_neighbor
        if args.max_ttl is not None:
            live_cfg["query_ttl"] = args.max_ttl
        if args.n_questions is not None:
            live_cfg["n_questions"] = args.n_questions
        if args.api_key is not None:
            live_cfg["api_key"] = args.api_key
        if args.blockchain_url is not None:
            live_cfg["blockchain_url"] = args.blockchain_url
        if args.min_request_interval_s is not None:
            live_cfg["min_request_interval_s"] = args.min_request_interval_s
        threshold = args.confidence_threshold or 0.5
        max_ttl = live_cfg.get("query_ttl", 3)

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
        if probe.onchain_reliability:
            print(f"  [+] On-chain reliability scores: {probe.onchain_reliability}")

        questions = _live_questions(live_cfg.get("n_questions", 30), seed)
        print(f"  [+] Loaded {len(questions)} questions")

        rows = [run_baseline(probe, questions, threshold, max_ttl)]
        live_networks = [probe]
        if args.single:
            ratio = args.ratio if args.ratio is not None else ddos_cfg.get("attack_ratio", 0.3)
            strategy = args.strategy or ddos_cfg.get("strategy", "random")
            net = build_network()
            live_networks.append(net)
            rows += run_ddos_scenario(net, questions, threshold, max_ttl,
                                       ratio, strategy, iterations, seed, ddos_cfg)
        else:
            sweep = cfg.get("sweep", {})
            ratios = sweep.get("ratios", [0.1, 0.3, 0.6])
            strategies = sweep.get("strategies", ["random", "targeted", "sequential"])
            for strategy in strategies:
                for ratio in ratios:
                    if args.scenario_delay_s:
                        time.sleep(args.scenario_delay_s)
                    net = build_network()
                    live_networks.append(net)
                    rows += run_ddos_scenario(net, questions, threshold, max_ttl,
                                               ratio, strategy, iterations, seed, ddos_cfg)

        rate_limited_total = sum(n.rate_limited_total() for n in live_networks)
        error_total = sum(n.error_total() for n in live_networks)
        run_config = {
            "mode": "live", "live": live_cfg, "ddos": ddos_cfg,
            "onchain_reliability": probe.onchain_reliability,
            "rate_limited_total": rate_limited_total, "error_total": error_total,
        }
        if rate_limited_total:
            print(f"\n  [!] {rate_limited_total} request(s) hit the live sources' rate limit "
                  "(HTTP 429) during this run -- results below are not fully trustworthy.")
        if error_total > rate_limited_total:
            print(f"\n  [!] {error_total - rate_limited_total} request(s) failed for a reason other than "
                  "rate-limiting -- check error_total in the JSON log.")

    print()
    _print_table(rows)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(log_dir, f"attack_{ts}_ddos_sim_{mode}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "attack_type": "ddos_sim",
                "attack_config": run_config,
                "results": rows,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  [+] JSON log written to {json_path}")

    csv_path = os.path.join(log_dir, f"attack_{ts}_ddos_sim_{mode}_seed{seed}.csv")
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
