"""
attack/ddos_sim/run_live_evaluation.py

Real, end-to-end DDoS evaluation against the actual DRAG deployment:
real questions -> real retrieval from drag_data_source -> real generation
via drag_llm_service (:9000/query) -> real NLG-quality scoring against
ground truth (attack/ddos_sim/nlg_metrics.py), compared baseline vs. under
a real concurrent-flood DDoS (attack/ddos_sim/live_flood.py) against the
live Docker containers.

This is a different layer from attack/ddos_sim/run_attack.py, which
measures pure retrieval-layer availability/hit_rate via a seeded
probability model (no real LLM generation, no real network load). This
script instead answers "what happens to the actual generated answer
quality when the real data sources are really congested" -- requires
`docker compose up -d` and produces real load against your own local
containers (bounded, stoppable, no destructive action).

Ground truth: PubMedQA (pqa_labeled) questions matched against the
corpus actually loaded into data-source-0 (data/polluted_token/sources_0.jsonl),
same matching logic as attack/selective_forward_sim/run_attack.py's
_live_questions(). Scored against `final_decision` (yes/no/maybe) -- the
question is sent to the LLM with an explicit "answer yes, no, or maybe"
instruction so the generated answer is directly comparable to that short
ground truth (PubMedQA's own task framing is exactly this
yes/no/maybe decision, so this isn't an artificial constraint).

No "news"/"mmlu" domain exists in this repo (confirmed by search) -- this
evaluation runs against the one real domain that is actually loaded and
retrievable here.

Usage
-----
  docker compose up -d
  python attack/ddos_sim/run_live_evaluation.py --num_questions 10
  python attack/ddos_sim/run_live_evaluation.py --num_questions 10 --severities low high
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.live_network import LiveRAGNetwork, DEFAULT_SOURCE_URLS  # noqa: E402
from attack.ddos_sim.live_flood import TrafficFlood  # noqa: E402
from attack.ddos_sim.nlg_metrics import score_answer, average_metrics  # noqa: E402

LLM_SERVICE_URL = "http://localhost:9000"
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_SPLIT = "train"

# Only 3 real data sources exist in this deployment, so severity is
# expressed as (how many of the 3 are flooded) x (concurrency per source),
# not a topic-domain distinction.
SEVERITY_TIERS = {
    "low":  {"num_sources": 1, "workers_per_source": 3},
    "mid":  {"num_sources": 2, "workers_per_source": 6},
    "high": {"num_sources": 3, "workers_per_source": 10},
}


def _join_pubmedqa_context(contexts) -> str:
    """Must match data/build_pubmedqa_corpus.py's join_context() exactly."""
    return " ".join(c.strip() for c in contexts if c and c.strip())


def load_pubmedqa_qa_pairs(n: int, seed: int) -> List[Tuple[str, str]]:
    """(question, final_decision) pairs matched against the corpus actually
    loaded into data-source-0, same corpus-matching logic as
    attack/selective_forward_sim/run_attack.py's _live_questions()."""
    import numpy as np
    from datasets import load_dataset

    corpus_path = os.path.join(_ROOT, "data", "polluted_token", "sources_0.jsonl")
    contexts = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            html = rec.get("html", "")
            if html:
                contexts.add(html)

    ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split=PUBMEDQA_SPLIT)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for item in ds:
        ctx = _join_pubmedqa_context(item["context"]["contexts"])
        if ctx not in contexts:
            continue
        q = item["question"].strip()
        decision = (item.get("final_decision") or "").strip().lower()
        if q and decision and q not in seen:
            seen.add(q)
            pairs.append((q, decision))

    if not pairs:
        raise RuntimeError(
            "Matched 0 questions against data/polluted_token/sources_0.jsonl -- "
            "has the corpus drifted from the PubMedQA dataset? See "
            "attack/selective_forward_sim/run_attack.py's _live_questions() for the same check."
        )
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return [pairs[i] for i in sorted(idxs)]


def query_llm(question: str, timeout: float = 60.0) -> Tuple[str, bool]:
    prompt_q = f"{question} Answer with a single word: yes, no, or maybe."
    try:
        r = requests.post(f"{LLM_SERVICE_URL}/query", json={"query": prompt_q}, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", ""), True
        return f"ERROR HTTP {r.status_code}: {r.text[:150]}", False
    except requests.exceptions.Timeout:
        return "ERROR timeout", False
    except Exception as e:
        return f"ERROR {e}", False


def measure_hops_messages(net: LiveRAGNetwork, questions: List[str], threshold: float = 0.5) -> Tuple[float, float]:
    """Real HTTP calls against the same 3 live data sources, via the
    existing TTL-bounded BFS routing (attack/selective_forward_sim), to get
    a genuine (not synthetic) avg_num_hops under whatever congestion is
    currently active. avg_num_messages approximates the neighbour-notify
    overhead a real P2P gossip layer would add per hop (hops *
    num_query_neighbor) -- this fully-connected 3-peer graph doesn't
    literally send those extra messages today, so this is a documented
    upper-bound estimate, not a separately-measured quantity.
    """
    if not questions:
        return 0.0, 0.0
    hops = []
    for q in questions:
        result = net.topic_aware_query(q, threshold)
        hops.append(result.num_hops)
    avg_hops = sum(hops) / len(hops)
    return avg_hops, avg_hops * net.num_query_neighbor


def run_phase(
    qa_pairs: List[Tuple[str, str]], phase_label: str, hop_network: LiveRAGNetwork,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_question: List[Dict[str, Any]] = []
    successes = 0
    for q, gold in qa_pairs:
        response, ok = query_llm(q)
        if ok:
            successes += 1
        metrics = score_answer(response, [gold])
        per_question.append({"question": q, "gold": gold, "response": response, "ok": ok, **metrics})
        print(f"    [{'OK' if ok else 'FAIL'}] gold={gold:<6} resp={response[:60]!r}")

    numeric_only = [{k: v for k, v in m.items() if isinstance(v, (int, float)) and k not in ("ok",)}
                     for m in per_question]
    agg = average_metrics(numeric_only)

    avg_hops, avg_msgs = measure_hops_messages(hop_network, [q for q, _ in qa_pairs])

    n = len(qa_pairs)
    failed = n - successes
    agg.update({
        "avg_num_hops": avg_hops,
        "avg_num_messages": avg_msgs,
        "avg_query_hit": successes / n if n else 0.0,
        "evaluation_phase": phase_label,
        "successful_queries": successes,
        "failed_queries": failed,
        "query_failure_rate": failed / n if n else 0.0,
    })
    return agg, per_question


def main() -> None:
    p = argparse.ArgumentParser(description="Real DDoS evaluation against the live DRAG deployment")
    p.add_argument("--num_questions", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--severities", nargs="+", default=["low", "mid", "high"], choices=list(SEVERITY_TIERS))
    p.add_argument("--flood_ramp_s", type=float, default=1.0,
                    help="seconds to let flood workers ramp up before the real evaluation queries start")
    p.add_argument("--tier_cooldown_s", type=float, default=60.0,
                    help="seconds to wait after a tier's flood stops before the next tier starts, so the "
                         "60/min rate-limit bucket and any queued load drain instead of carrying over "
                         "into the next tier's baseline/attack measurement")
    args = p.parse_args()

    print("Loading PubMedQA yes/no/maybe questions matched against the loaded corpus...")
    qa_pairs = load_pubmedqa_qa_pairs(args.num_questions, args.seed)
    print(f"  [+] {len(qa_pairs)} questions loaded")

    hop_network = LiveRAGNetwork(use_onchain_scores=False)
    reachable = hop_network.ping_all()
    print(f"  [+] Source reachability: {reachable}")
    if not any(reachable.values()):
        print("\n  [!] No Docker data-source nodes are reachable.")
        print("      Start them with:  docker compose up -d   (repo root)\n")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {"severities": {}}

    # Each tier gets its own freshly-measured baseline (taken immediately
    # before that tier's flood starts) and a cooldown wait before it, rather
    # than reusing one baseline measured before any flooding happened. This
    # keeps each tier's before/after comparison isolated from cumulative
    # exhaustion carried over from the previous tier -- otherwise "high"
    # would be measured on sources already flooded twice in immediate
    # succession beforehand (see problems/ddos_attack_gaps.md #2).
    for i, severity in enumerate(args.severities):
        tier = SEVERITY_TIERS[severity]
        if i > 0:
            print(f"\n  [+] Cooling down {args.tier_cooldown_s:.0f}s before the '{severity}' tier "
                  f"(lets the rate-limit bucket and prior flood's load drain)...")
            time.sleep(args.tier_cooldown_s)

        print(f"\n=== BASELINE ({severity}, no attack) ===")
        baseline_agg, baseline_per_q = run_phase(qa_pairs, "baseline", hop_network)
        print(f"  successful={baseline_agg['successful_queries']} failed={baseline_agg['failed_queries']} "
              f"f1={baseline_agg['f1']:.3f} avail_hit={baseline_agg['avg_query_hit']:.3f}")

        print(f"\n=== POST-ATTACK: {severity} (flood {tier['num_sources']}/3 source(s) "
              f"x {tier['workers_per_source']} workers/source) ===")
        flood = TrafficFlood(DEFAULT_SOURCE_URLS, workers_per_source=tier["workers_per_source"])
        targets = list(DEFAULT_SOURCE_URLS.keys())[: tier["num_sources"]]
        flood.start(targets)
        time.sleep(args.flood_ramp_s)
        try:
            post_agg, post_per_q = run_phase(qa_pairs, "post_attack", hop_network)
        finally:
            flood_stats = flood.stop()

        print(f"  successful={post_agg['successful_queries']} failed={post_agg['failed_queries']} "
              f"f1={post_agg['f1']:.3f} avail_hit={post_agg['avg_query_hit']:.3f}")
        print(f"  flood stats: {flood_stats}")

        results.append({
            "baseline_results": baseline_agg,
            "post_attack_results": post_agg,
            "attack_type": f"pubmedqa_{severity}_ddos_comparison",
        })
        detail["severities"][severity] = {"flood_targets": targets, "flood_stats": flood_stats,
                                           "baseline_per_question": baseline_per_q,
                                           "post_attack_per_question": post_per_q}

    log_dir = os.path.join(_ROOT, "attack_logs", "ddos_sim")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_path = os.path.join(log_dir, f"live_eval_{ts}_ddos_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [+] Comparison JSON written to {summary_path}")

    detail_path = os.path.join(log_dir, f"live_eval_{ts}_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"  [+] Per-question detail written to {detail_path}")


if __name__ == "__main__":
    main()
