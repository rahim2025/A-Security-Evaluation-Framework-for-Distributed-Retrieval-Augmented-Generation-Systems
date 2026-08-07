"""
attack/ddos_sim/hyperparameter_sensitivity.py

Answers problems/ddos_attack_gaps.md #3 directly instead of only reframing
it in prose: is the mock-mode "100%->10%" collapse (report sec 8.1) an
emergent finding, or close to guaranteed by the default
intensity_min=0.5/intensity_max=1.0 (mean sampled intensity 0.75, comfortably
above the ~0.625 intensity where drop_probability = min(0.95, intensity*0.8)
crosses DOWN_THRESHOLD=0.5)?

Sweeps intensity_min across a grid (intensity_max fixed at 1.0) at a fixed
attack_ratio/strategy/iterations, across multiple seeds, and reports final
availability_percentage after the sweep -- a real dose-response curve
against the model's own mechanism, not a single cherry-picked point.

No Docker required (mock mode only).
"""
from __future__ import annotations

import os
import statistics
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.selective_forward_sim.network_sim import MockRAGNetwork  # noqa: E402
from attack.selective_forward_sim.run_attack import _mock_questions  # noqa: E402
from attack.ddos_sim.run_attack import run_ddos_scenario  # noqa: E402

NUM_PEERS = 20
NUM_ATTACHMENTS = 4
NUM_QUERY_NEIGHBOR = 4
QUERY_TTL = 6
PEER_HIT_PROB = 0.4
THRESHOLD = 0.5
NUM_QUERIES = 100
ITERATIONS = 5
RATIO = 0.3
STRATEGY = "random"
DDOS_DURATION = 60.0
WAVE_INTERVAL_S = 30.0
SEEDS = [0, 42, 123]

# The mechanism's own breakpoint: drop_probability = min(0.95, intensity*0.8)
# crosses DOWN_THRESHOLD=0.5 at intensity = 0.625. Sweep on both sides of it.
INTENSITY_MINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run_one(seed: int, intensity_min: float) -> float:
    network = MockRAGNetwork(
        num_peers=NUM_PEERS, num_attachments=NUM_ATTACHMENTS,
        num_query_neighbor=NUM_QUERY_NEIGHBOR, query_ttl=QUERY_TTL,
        peer_hit_prob=PEER_HIT_PROB, seed=seed,
    )
    questions = _mock_questions(NUM_QUERIES, seed)
    ddos_cfg: Dict[str, Any] = {
        "ddos_duration": DDOS_DURATION, "wave_interval_s": WAVE_INTERVAL_S,
        "intensity_min": intensity_min, "intensity_max": 1.0,
        "cascade_factor": 0.25, "max_cascade_intensity": 0.6,
    }
    rows = run_ddos_scenario(network, questions, THRESHOLD, QUERY_TTL,
                              RATIO, STRATEGY, ITERATIONS, seed, ddos_cfg)
    return rows[-1]["availability_percentage"]


def main() -> None:
    print(f"Fixed: num_peers={NUM_PEERS} ratio={RATIO} strategy={STRATEGY} "
          f"iterations={ITERATIONS} intensity_max=1.0 seeds={SEEDS}")
    print(f"Mechanism breakpoint: intensity=0.625 is where drop_probability "
          f"crosses DOWN_THRESHOLD=0.5\n")
    print(f"{'intensity_min':>14}{'mean_intensity':>16}{'final_avail%':>14}{'stdev':>10}")
    for imin in INTENSITY_MINS:
        vals = [run_one(seed, imin) for seed in SEEDS]
        mean_intensity = (imin + 1.0) / 2.0
        mean_avail = statistics.mean(vals)
        stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{imin:>14.2f}{mean_intensity:>16.3f}{mean_avail:>14.1f}{stdev:>10.2f}")


if __name__ == "__main__":
    main()
