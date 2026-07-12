"""
attack/selective_forward_sim/run_attack.py

CLI runner for the SFA simulation attack. Config-driven via
config/selective_forwarding_sim.yaml (CLI flags override YAML
defaults).

Modes
-----
mock  Entirely in-process (networkx Barabasi-Albert graph of synthetic
      MockPeers). No Docker/blockchain required -- sweeps
      strategy x attack_ratio in seconds. This is the mode to use for
      studying how attack_ratio and hub-targeting scale on a much
      larger overlay than the real 3-node deployment supports.

live  Targets the real docker-compose data sources (source_0/20/100)
      and reads real on-chain reliability scores from the deployed
      DragScores contract for "high_connectivity" targeting.
      Requires `docker compose up -d` at the repo root; falls back to
      unweighted targeting if the Hardhat node isn't reachable.

Usage
-----
  # Mock sweep across all ratios/strategies in the config (default)
  python attack/selective_forward_sim/run_attack.py --mode mock

  # Larger mock network, single strategy/ratio
  python attack/selective_forward_sim/run_attack.py --mode mock \
      --num_peers 30 --max_ttl 8 --num_queries 200 \
      --ratio 0.3 --strategy high_connectivity --single

  # Live sweep against the real Docker sources + on-chain ledger
  python attack/selective_forward_sim/run_attack.py --mode live --n_questions 50
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
from attack.selective_forward_sim.selective_forwarding_attack import (  # noqa: E402
    SelectiveForwardingAttack,
)

DEFAULT_CONFIG = "selective_forwarding_sim.yaml"


# ══════════════════════════════════════════════════════════════════════════
#  Question sources
# ══════════════════════════════════════════════════════════════════════════

def _mock_questions(n: int, seed: int) -> List[str]:
    return [f"mock question {i} (seed={seed})" for i in range(n)]


PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_SPLIT = "train"


def _join_pubmedqa_context(contexts) -> str:
    """
    Must match data/build_pubmedqa_corpus.py's join_context() and
    attack/Mia_attack/mia_attack.py's _join_pubmedqa_context() exactly --
    all three compare against the same sources_0.jsonl "html" field via
    plain string equality, so any drift here silently breaks matching.
    """
    return " ".join(c.strip() for c in contexts if c and c.strip())


def _live_questions(n: int, seed: int) -> List[str]:
    """
    Real questions matched against the documents actually loaded into the
    running Docker data sources (data/polluted_token/sources_0.jsonl).

    That corpus is built by data/build_pubmedqa_corpus.py from HuggingFace's
    qiaojin/PubMedQA (pqa_labeled config), NOT SQuAD -- an earlier version of
    this function loaded rajpurkar/squad instead, which no SQuAD context
    matches (verified: 0/87599 train rows, 0/10570 validation rows), so it
    was *silently* falling back to the tiny 6-question generic list below on
    every single live run, with no warning printed (a genuine bug: the
    `except Exception` handler only fires when loading itself raises, not
    when loading succeeds but corpus-matching finds zero results). Fixed to
    load the correct dataset and to warn loudly on zero matches instead of
    failing silently.
    """
    try:
        import numpy as np
        from datasets import load_dataset  # type: ignore

        corpus_path = os.path.join(_ROOT, "data", "polluted_token", "sources_0.jsonl")
        contexts = set()
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                html = rec.get("html", "")
                if html:
                    contexts.add(html)

        ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split=PUBMEDQA_SPLIT)
        pairs = []
        seen = set()
        for item in ds:
            ctx = _join_pubmedqa_context(item["context"]["contexts"])
            if ctx not in contexts:
                continue
            q = item["question"].strip()
            if q and q not in seen:
                seen.add(q)
                pairs.append(q)

        if not pairs:
            print(
                "  [warn] Loaded PubMedQA but matched 0 questions against "
                f"{corpus_path} -- corpus and dataset have diverged; "
                "using a generic fallback set instead of real questions."
            )
        else:
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
            return [pairs[i] for i in sorted(idxs)]
    except Exception as e:
        print(f"  [warn] Could not load PubMedQA questions ({e}); using a generic fallback set.")

    fallback = [
        "What is the capital of France?", "Who wrote Hamlet?",
        "What year did World War II end?", "What is the boiling point of water?",
        "Who painted the Mona Lisa?", "What is the largest planet in the solar system?",
    ]
    return [fallback[i % len(fallback)] for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
#  One scenario: build network, apply attack, run queries, collect metrics
# ══════════════════════════════════════════════════════════════════════════

def run_mock_scenario(
    net_cfg: Dict[str, Any], sim_cfg: Dict[str, Any],
    ratio: float, strategy: str, drop_rate=1.0,
) -> Dict[str, Any]:
    network = MockRAGNetwork(
        num_peers=net_cfg["num_peers"],
        num_attachments=net_cfg["num_attachments"],
        num_query_neighbor=net_cfg["num_query_neighbor"],
        query_ttl=net_cfg["query_ttl"],
        peer_hit_prob=sim_cfg["peer_hit_prob"],
        seed=sim_cfg["seed"],
    )
    questions = _mock_questions(sim_cfg["num_queries"], sim_cfg["seed"])
    threshold = net_cfg["query_confidence_threshold"]

    attack = SelectiveForwardingAttack(attack_ratio=ratio, drop_rate=drop_rate, seed=sim_cfg["seed"])
    info: Dict[str, Any] = {}
    if ratio > 0:
        info = attack.apply(network, strategy=strategy)

    answers = [network.topic_aware_query(q, threshold) for q in questions]
    metrics = attack.collect_metrics(answers, max_ttl=net_cfg["query_ttl"])
    attack.revert(network)

    return {"strategy": strategy if ratio > 0 else "baseline", "ratio": ratio, **info, **metrics}


def run_live_scenario(
    network: LiveRAGNetwork, questions: List[str], threshold: float,
    ratio: float, strategy: str, seed: int, max_ttl: int, drop_rate=1.0,
) -> Dict[str, Any]:
    attack = SelectiveForwardingAttack(attack_ratio=ratio, drop_rate=drop_rate, seed=seed)
    info: Dict[str, Any] = {}
    if ratio > 0:
        info = attack.apply(network, strategy=strategy)

    answers = [network.topic_aware_query(q, threshold) for q in questions]
    metrics = attack.collect_metrics(answers, max_ttl=max_ttl)
    attack.revert(network)

    return {"strategy": strategy if ratio > 0 else "baseline", "ratio": ratio, **info, **metrics}


# ══════════════════════════════════════════════════════════════════════════
#  Sweep + reporting
# ══════════════════════════════════════════════════════════════════════════

def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'strategy':<20}{'ratio':>7}{'compromised':>13}{'hit_rate':>10}{'avg_hops':>10}{'ttl_exhaust':>13}{'dropped':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['strategy']:<20}{r['ratio']:>7.2f}{r.get('num_compromised', 0):>13}"
            f"{r['hit_rate']:>10.3f}{r['avg_hops_per_query']:>10.2f}"
            f"{r['ttl_exhaustion_rate']:>13.3f}{r['dropped_queries']:>10}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Selective Forwarding Attack (SFA) simulation runner")
    p.add_argument("--mode", choices=["mock", "live"], default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--single", action="store_true", help="Run a single ratio/strategy instead of the full sweep")
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--drop_rate", default=None,
                    help="'1.0' (default, black-hole) or 'stealthy' (Uniform(0.10,0.30) per compromised peer) "
                         "or any float 0.0-1.0")
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
    p.add_argument("--min_request_interval_s", type=float, default=None,
                    help="minimum seconds between requests to the same live source (default 1.1s, "
                         "confirmed safely under the real deployment's 60/min per-source rate limit -- "
                         "see reports/SFA_Security_Analysis_Report.md sec 12.9)")
    p.add_argument("--scenario_delay_s", type=float, default=0.0,
                    help="pause (seconds) before each scenario -- recommended for long live sweeps "
                         "to avoid saturating the Docker containers")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    mode = args.mode or cfg.get("mode", "mock")

    log_dir = os.path.join(_ROOT, cfg.get("output", {}).get("log_dir", "attack_logs/selective_forward_sim"))
    os.makedirs(log_dir, exist_ok=True)

    sfa_cfg = cfg.get("selective_forwarding", {})
    seed = args.seed if args.seed is not None else sfa_cfg.get("seed", 42)
    drop_rate_raw = args.drop_rate if args.drop_rate is not None else sfa_cfg.get("drop_rate", 1.0)
    drop_rate = drop_rate_raw if drop_rate_raw == "stealthy" else float(drop_rate_raw)

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

        if args.single:
            ratio = args.ratio if args.ratio is not None else sfa_cfg.get("attack_ratio", 0.3)
            strategy = args.strategy or sfa_cfg.get("strategy", "high_connectivity")
            rows = [run_mock_scenario(net_cfg, sim_cfg, ratio, strategy, drop_rate=drop_rate)]
        else:
            sweep = cfg.get("sweep", {})
            ratios = sweep.get("ratios", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            strategies = sweep.get("strategies", ["random", "high_connectivity"])
            rows = [run_mock_scenario(net_cfg, sim_cfg, 0.0, "baseline", drop_rate=drop_rate)]
            for strategy in strategies:
                for ratio in [r for r in ratios if r > 0]:
                    if args.scenario_delay_s:
                        time.sleep(args.scenario_delay_s)
                    rows.append(run_mock_scenario(net_cfg, sim_cfg, ratio, strategy, drop_rate=drop_rate))

        run_config = {"mode": "mock", "network": net_cfg, "simulation": sim_cfg, "drop_rate": drop_rate}

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

        network = LiveRAGNetwork(
            source_urls=live_cfg.get("source_urls"),
            num_query_neighbor=live_cfg.get("num_query_neighbor", 2),
            query_ttl=live_cfg.get("query_ttl", 3),
            api_key=live_cfg.get("api_key", "reliable-derag-secret-2026"),
            blockchain_url=live_cfg.get("blockchain_url", "http://localhost:8545"),
            use_onchain_scores=live_cfg.get("use_onchain_scores", True),
            seed=seed,
            min_request_interval_s=live_cfg.get("min_request_interval_s", 1.1),
        )
        reachable = network.ping_all()
        print(f"  [+] Source reachability: {reachable}")
        if not any(reachable.values()):
            print("\n  [!] No Docker data-source nodes are reachable.")
            print("      Start them with:  docker compose up -d   (repo root)")
            print("      Or run with --mode mock for an offline simulation.\n")
            sys.exit(1)
        if network.onchain_reliability:
            print(f"  [+] On-chain reliability scores: {network.onchain_reliability}")

        questions = _live_questions(live_cfg.get("n_questions", 50), seed)
        print(f"  [+] Loaded {len(questions)} questions")

        if args.single:
            ratio = args.ratio if args.ratio is not None else sfa_cfg.get("attack_ratio", 0.3)
            strategy = args.strategy or sfa_cfg.get("strategy", "high_connectivity")
            rows = [run_live_scenario(network, questions, threshold, ratio, strategy, seed,
                                       live_cfg.get("query_ttl", 3), drop_rate=drop_rate)]
        else:
            sweep = cfg.get("sweep", {})
            ratios = sweep.get("ratios", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            strategies = sweep.get("strategies", ["random", "high_connectivity"])
            rows = [run_live_scenario(network, questions, threshold, 0.0, "baseline", seed,
                                       live_cfg.get("query_ttl", 3), drop_rate=drop_rate)]
            for strategy in strategies:
                for ratio in [r for r in ratios if r > 0]:
                    if args.scenario_delay_s:
                        time.sleep(args.scenario_delay_s)
                    rows.append(run_live_scenario(network, questions, threshold, ratio, strategy, seed,
                                                   live_cfg.get("query_ttl", 3), drop_rate=drop_rate))

        run_config = {"mode": "live", "live": live_cfg, "onchain_reliability": network.onchain_reliability,
                      "drop_rate": drop_rate, "rate_limited_total": network.rate_limited_total(),
                      "error_total": network.error_total()}
        if network.rate_limited_total():
            print(f"\n  [!] {network.rate_limited_total()} request(s) hit the live sources' rate limit "
                  "(HTTP 429) during this run -- hit_rate below is not fully trustworthy. Increase "
                  "--min_request_interval_s or reduce query volume and re-run.")
        if network.error_total() > network.rate_limited_total():
            other = network.error_total() - network.rate_limited_total()
            print(f"\n  [!] {other} request(s) failed for a reason OTHER than rate-limiting "
                  "(timeout / connection error / non-200 status / malformed response) during this run "
                  "-- hit_rate below is not fully trustworthy even though rate_limited_total may read 0. "
                  "This is common if you raised the server-side rate limit and lowered "
                  "--min_request_interval_s together (real concurrency can trade 429s for timeouts/5xx). "
                  "Check error_total in the JSON log, not just rate_limited_total.")

    print()
    _print_table(rows)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(log_dir, f"attack_{ts}_sfa_sim_{mode}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "attack_type": "selective_forwarding_sim",
                "attack_config": run_config,
                "results": rows,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  [+] JSON log written to {json_path}")

    csv_path = os.path.join(log_dir, f"attack_{ts}_sfa_sim_{mode}_seed{seed}.csv")
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
