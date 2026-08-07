"""
attack/ddos_sim/validate_flood_ramp.py

Answers problems/ddos_attack_gaps.md #5's flood_ramp_s item empirically:
is 1.0s actually enough time for `workers_per_source` flood threads to
saturate the target before run_live_evaluation.py's timed queries begin?

Starts a real TrafficFlood against one live data source and, every 0.25s,
fires a fresh single probe request measuring its latency -- once probe
latency stops increasing wave-over-wave, the target has reached steady-state
congestion. Reports the elapsed time at which that plateau is first reached,
for each severity tier's worker count (reports/ddos_attack.md's low/mid/high
tiers), so flood_ramp_s can be set from measured evidence instead of an
asserted constant.

Requires `docker compose up -d` (targets source_0 at localhost:8001).
"""
from __future__ import annotations

import os
import sys
import time

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.ddos_sim.live_flood import TrafficFlood  # noqa: E402

TARGET_URL = "http://localhost:8001"
API_KEY = "reliable-derag-secret-2026"
SAMPLE_INTERVAL_S = 0.25
TOTAL_DURATION_S = 5.0
TIERS = {"low": 3, "mid": 6, "high": 10}


def probe_latency() -> float:
    """Fire a fresh single request, return its wall-clock latency (seconds),
    or the timeout value if it errors/times out (treated as fully saturated)."""
    start = time.time()
    try:
        requests.post(f"{TARGET_URL}/query", headers={"X-API-Key": API_KEY},
                       json={"query": "ramp probe", "k": 1}, timeout=10.0)
    except Exception:
        pass
    return time.time() - start


def measure_tier(name: str, workers: int) -> None:
    print(f"\n=== tier={name} (workers_per_source={workers}) ===")
    flood = TrafficFlood({"source_0": TARGET_URL}, workers_per_source=workers)
    flood.start(["source_0"])
    t0 = time.time()
    samples = []
    while time.time() - t0 < TOTAL_DURATION_S:
        elapsed = time.time() - t0
        lat = probe_latency()
        sent = flood.stats["source_0"].as_dict()["requests_sent"]
        samples.append((elapsed, lat, sent))
        print(f"  t={elapsed:5.2f}s  probe_latency={lat:6.3f}s  flood_requests_sent_so_far={sent}")
        time.sleep(max(0.0, SAMPLE_INTERVAL_S - lat))
    flood.stop()

    # Plateau heuristic: first time consecutive-sample latency delta drops
    # below 10% of the running max -- i.e. latency has stopped climbing.
    plateau_t = None
    running_max = 0.0
    for i in range(1, len(samples)):
        running_max = max(running_max, samples[i - 1][1])
        delta = samples[i][1] - samples[i - 1][1]
        if running_max > 0.05 and abs(delta) < 0.1 * running_max:
            plateau_t = samples[i][0]
            break
    print(f"  -> latency plateau first reached at t={plateau_t if plateau_t is not None else 'not reached within '+str(TOTAL_DURATION_S)+'s'}")


def main() -> None:
    r = requests.get(f"{TARGET_URL}/health", timeout=5)
    print(f"source_0 reachable: {r.status_code == 200}")
    for name, workers in TIERS.items():
        measure_tier(name, workers)
        time.sleep(3.0)  # let the source recover between tiers


if __name__ == "__main__":
    main()
