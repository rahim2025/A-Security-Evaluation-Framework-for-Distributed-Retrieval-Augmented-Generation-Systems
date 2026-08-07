"""
defense/kb_extraction_defense/calibrate_thresholds.py

Measures a legitimate-user topic-diversity baseline for QueryDiversityThrottle
instead of asserting max_topics_per_window by hand (see
problems/kb_extraction_gaps.md #5: the 82.5% block rate reported in
reports/kb.md was measured with max_topics_per_window=5 against a 10-topic
corpus -- an assumed threshold, never checked against what a real client's
query pattern actually looks like).

No real user telemetry exists for this system, so "legitimate" is modeled
explicitly rather than assumed silently: a session is one client asking
`window_size` questions about a small number of topics it actually cares
about (topic-focused browsing), drawn from the real per-source topic/question
map (attack.kb_extraction.run_attack.load_probe_sets' context_to_title). This
is contrasted against the attacker's own real sampling strategy -- uniform
across the *entire* matched-question pool regardless of topic, which is what
load_probe_sets()/probe_source() actually send (see run_attack.py) -- rather
than against another assumption.

The recommended threshold is the round-up of the legitimate distribution's
99th percentile: high enough that legitimate topic-focused sessions almost
never trip it, while (reported alongside) the attacker's own topic coverage
in the same window shows whether that threshold still has any discriminative
power left on a corpus this small.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.kb_extraction.run_attack import load_probe_sets  # noqa: E402


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def simulate_legitimate_topic_diversity(
    qa_by_question: Dict[str, dict], window_size: int, n_sessions: int, rng: random.Random,
    topics_of_interest_range=(1, 3),
) -> List[int]:
    """One session = one client asking window_size questions drawn only from
    `k` topics it's actually interested in, k ~ Uniform(topics_of_interest_range).
    Returns the distinct-topic count observed in that session's full window
    (matches QueryDiversityThrottle.topics_in_window at window capacity)."""
    by_topic: Dict[str, List[str]] = {}
    for q, info in qa_by_question.items():
        t = info.get("title")
        if t is None:
            continue
        by_topic.setdefault(t, []).append(q)
    all_topics = list(by_topic.keys())
    if not all_topics:
        return []

    results = []
    for _ in range(n_sessions):
        k = rng.randint(*topics_of_interest_range)
        k = min(k, len(all_topics))
        chosen_topics = rng.sample(all_topics, k)
        pool = [q for t in chosen_topics for q in by_topic[t]]
        session_qs = [rng.choice(pool) for _ in range(window_size)]
        touched = {qa_by_question[q]["title"] for q in session_qs}
        results.append(len(touched))
    return results


def simulate_attacker_topic_diversity(
    qa_by_question: Dict[str, dict], window_size: int, n_sessions: int, rng: random.Random,
) -> List[int]:
    """Mirrors the real attack's sampling: uniform across the entire matched
    question pool regardless of topic (attack/kb_extraction/run_attack.py's
    load_probe_sets(), no topic-locality bias)."""
    questions = [q for q, info in qa_by_question.items() if info.get("title") is not None]
    if not questions:
        return []
    results = []
    for _ in range(n_sessions):
        session_qs = rng.sample(questions, min(window_size, len(questions)))
        touched = {qa_by_question[q]["title"] for q in session_qs}
        results.append(len(touched))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate QueryDiversityThrottle's max_topics_per_window "
                                             "against a modeled legitimate-user topic-diversity baseline")
    p.add_argument("--source_index", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--window_size", type=int, default=30)
    p.add_argument("--n_sessions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    probe_sets = load_probe_sets(seed=args.seed)
    info = probe_sets[args.source_index]
    qa_by_question = info["qa_by_question"]
    n_topics = len({v["title"] for v in qa_by_question.values() if v.get("title") is not None})
    print(f"source_{args.source_index} ({info['dataset']}): {n_topics} real topics, "
          f"{len(qa_by_question)} matched questions\n")

    legit = sorted(simulate_legitimate_topic_diversity(qa_by_question, args.window_size, args.n_sessions, rng))
    attacker = sorted(simulate_attacker_topic_diversity(qa_by_question, args.window_size, args.n_sessions, rng))

    print(f"Legitimate sessions (topic-focused, k in [1,3] topics of interest), n={len(legit)}:")
    for pct in (50, 90, 95, 99, 100):
        print(f"    p{pct}: {_percentile(legit, pct):.1f} distinct topics")

    print(f"\nAttacker sessions (uniform across all matched questions, k=all), n={len(attacker)}:")
    for pct in (50, 90, 95, 99, 100):
        print(f"    p{pct}: {_percentile(attacker, pct):.1f} distinct topics")

    recommended = int(_percentile(legit, 99)) + 1
    print(f"\nRecommended max_topics_per_window = ceil(p99 legitimate) + 1 = {recommended}")
    print(f"  (current default in query_diversity_throttle.py: 12; "
          f"reports/kb.md's measured 82.5% block rate used an unmeasured 5)")
    if recommended >= n_topics:
        print(f"  [!] Recommended threshold ({recommended}) >= total topics in this corpus ({n_topics}): "
              "on a corpus this small, a topic-count threshold has little-to-no discriminative power left -- "
              "even the modeled attacker's median session touches most/all real topics. This is a structural "
              "corpus-size limit, not a calibration failure (see problems/kb_extraction_gaps.md #5).")


if __name__ == "__main__":
    main()
