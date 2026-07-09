"""
patch_sfa.py — run from your project root:
    python3 patch_sfa.py

What it does:
  1. Restores selective_forward_attack.py from .bak (gets mock working again)
  2. Appends new classes: SFADetector, SFAMitigation, check_blockchain_status
  3. Writes a new run_attack.py with all modes: mock/stealthy/detection/mitigation/live
"""
import pathlib, shutil, sys

ROOT   = pathlib.Path(__file__).parent
SFA    = ROOT / "attack/selective_forward/selective_forward_attack.py"
RUN    = ROOT / "attack/selective_forward/run_attack.py"
SFA_BK = SFA.with_suffix(".py.bak")
RUN_BK = RUN.with_suffix(".py.bak")

# ── Step 1: restore originals from backup ────────────────────────────────────
if not SFA_BK.exists():
    print(f"ERROR: backup not found: {SFA_BK}")
    print("       Cannot restore. Please re-download the original file.")
    sys.exit(1)

shutil.copy2(SFA_BK, SFA)
print(f"Restored {SFA.name} from backup")

# ── Step 2: verify original has SelectiveForwardingAttack ────────────────────
if "class SelectiveForwardingAttack" not in SFA.read_text():
    print("ERROR: backup does not contain SelectiveForwardingAttack class.")
    print("       The backup may also be corrupted.")
    sys.exit(1)
print("  Verified: SelectiveForwardingAttack found in restored file")

# ── Step 3: append new classes to selective_forward_attack.py ────────────────
APPEND = r'''

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSION: Blockchain status (live RPC check)
# ═══════════════════════════════════════════════════════════════════════════════
BLOCKCHAIN_RPC = "http://localhost:8545"

def check_blockchain_status(timeout=2.0):
    """Real JSON-RPC call to Hardhat node. Falls back gracefully if offline."""
    import requests as _req
    payload = {"jsonrpc": "2.0", "method": "net_peerCount", "params": [], "id": 1}
    try:
        r = _req.post(BLOCKCHAIN_RPC, json=payload, timeout=timeout)
        r.raise_for_status()
        peers = int(r.json().get("result", "0x0"), 16)
        return {
            "online": True, "peer_count": peers, "rpc": BLOCKCHAIN_RPC,
            "status_msg": f"blockchain online — {peers} peers — SFA undetectable on-chain",
        }
    except Exception:
        return {
            "online": False, "peer_count": None, "rpc": BLOCKCHAIN_RPC,
            "status_msg": "blockchain offline — mock mode assumed",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSION: Stealthy (gray-hole) drop helper
# ═══════════════════════════════════════════════════════════════════════════════

def apply_stealthy_drop(network, attack_ratio, strategy, seed, drop_prob):
    """
    Monkey-patch sources with a probabilistic (gray-hole) drop.
    drop_prob=0.1 drops 10% of queries; 0.3 drops 30%, etc.
    Returns (saved_queries_dict, compromised_sids_list) for restore().
    """
    sids = list(network.sources.keys())
    n    = max(1, round(attack_ratio * len(sids)))
    rng  = np.random.default_rng(seed)

    if strategy == "random":
        idxs    = rng.choice(len(sids), size=n, replace=False).tolist()
        targets = [sids[i] for i in idxs]
    else:
        targets = sorted(sids,
                         key=lambda s: network.ssm_scores[s]["reliability"],
                         reverse=True)[:n]

    saved   = {}
    counter = [0]   # mutable counter shared across closures

    for sid in targets:
        src  = network.sources[sid]
        orig = src.query
        saved[sid] = orig
        _rng = np.random.default_rng(seed + abs(hash(sid)) % 9999)
        _dp  = drop_prob
        _cnt = counter

        def _stealthy(question, k=5, _o=orig, _r=_rng, _d=_dp, _c=_cnt):
            if _r.random() < _d:
                _c[0] += 1
                return [], 0.0, False
            return _o(question, k)

        src.query = _stealthy

    return saved, targets, counter


def restore_sources(network, saved):
    """Undo monkey-patching from apply_stealthy_drop or SelectiveForwardingAttack."""
    for sid, orig_fn in saved.items():
        if sid in network.sources:
            network.sources[sid].query = orig_fn


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSION: SFADetector — sliding-window z-score anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════
from collections import defaultdict, deque as _deque


class SFADetector:
    """
    Per-source hit-rate monitor. Flags sources whose hit-rate falls more than
    z_thresh standard deviations below the network mean.
    On flag: decays SSM reliability and usefulness scores by ssm_decay factor.
    """

    def __init__(self, window=50, min_samples=10, z_thresh=1.5,
                 abs_thresh=0.10, ssm_decay=0.5):
        self.window      = window
        self.min_samples = min_samples
        self.z_thresh    = z_thresh
        self.abs_thresh  = abs_thresh
        self.ssm_decay   = ssm_decay
        self._history    = defaultdict(lambda: _deque(maxlen=window))
        self.flagged     = set()
        self.detection_log = []

    def record(self, sid, is_hit):
        self._history[sid].append(int(is_hit))

    def detect(self, ssm_scores=None):
        hitrates = {sid: sum(h) / len(h)
                    for sid, h in self._history.items()
                    if len(h) >= self.min_samples}

        if len(hitrates) < 2:
            return {"flagged_sources": [], "per_source_hitrate": hitrates,
                    "per_source_zscore": {}, "detected": False}

        vals    = list(hitrates.values())
        mean_hr = float(np.mean(vals))
        std_hr  = max(float(np.std(vals)), 1e-9)
        zscores = {sid: (hr - mean_hr) / std_hr for sid, hr in hitrates.items()}

        for sid, z in zscores.items():
            hr = hitrates[sid]
            if z < -self.z_thresh or hr < self.abs_thresh:
                if sid not in self.flagged:
                    self.flagged.add(sid)
                    self.detection_log.append(
                        {"sid": sid, "hit_rate": round(hr, 4), "z_score": round(z, 3)})
                if ssm_scores and sid in ssm_scores:
                    ssm_scores[sid]["reliability"] = max(
                        0, ssm_scores[sid]["reliability"] * self.ssm_decay)
                    ssm_scores[sid]["usefulness"]  = max(
                        0, ssm_scores[sid]["usefulness"]  * self.ssm_decay)

        return {
            "flagged_sources":    list(self.flagged),
            "per_source_hitrate": hitrates,
            "per_source_zscore":  zscores,
            "detected":           len(self.flagged) > 0,
        }

    def summary(self):
        return {"total_flagged": len(self.flagged),
                "flagged_sources": list(self.flagged),
                "detection_log": self.detection_log}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSION: SFAMitigation — blacklist + SSM-ordered rerouting
# ═══════════════════════════════════════════════════════════════════════════════

class SFAMitigation:
    """
    After detection flags suspicious sources:
      1. Blacklist them (skip their .query() calls).
      2. Re-route queries in SSM-reliability-descending order.
    """

    def __init__(self):
        self.blacklist      = set()
        self.mitigation_log = []

    def apply_blacklist(self, detection, network):
        newly = []
        for sid in detection.get("flagged_sources", []):
            if sid not in self.blacklist:
                self.blacklist.add(sid)
                newly.append(sid)
                self.mitigation_log.append({"action": "blacklist", "sid": sid,
                    "ssm": network.ssm_scores.get(sid, {}).get("reliability")})
        return newly

    def query_rerouted(self, network, question, k=5):
        ordered = sorted(network.sources.items(),
                         key=lambda kv: network.ssm_scores[kv[0]]["reliability"],
                         reverse=True)
        hops = 0
        for sid, src in ordered:
            if sid in self.blacklist:
                continue
            docs, score, is_hit = src.query(question, k)
            hops += 1
            if is_hit:
                return True, hops
            if hops >= network.MAX_HOPS:
                break
        return False, hops

    def summary(self):
        return {"blacklisted": list(self.blacklist),
                "log": self.mitigation_log}
'''

with open(SFA, "a") as f:
    f.write(APPEND)
print(f"Appended new classes to {SFA.name}")
print(f"  + check_blockchain_status()")
print(f"  + apply_stealthy_drop()")
print(f"  + restore_sources()")
print(f"  + SFADetector")
print(f"  + SFAMitigation")

# ── Step 4: write new run_attack.py ──────────────────────────────────────────
NEW_RUN = r'''"""
run_attack.py — Selective Forwarding Attack driver for Reliable-dRAG.
Modes: mock | stealthy | detection | mitigation | live

Usage:
  python3 attack/selective_forward/run_attack.py --mode mock
  python3 attack/selective_forward/run_attack.py --mode stealthy
  python3 attack/selective_forward/run_attack.py --mode detection
  python3 attack/selective_forward/run_attack.py --mode mitigation
  python3 attack/selective_forward/run_attack.py --mode live
"""
import argparse, csv, json, pathlib, sys
from datetime import datetime
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from selective_forward_attack import (
    EVAL_DATA, DATA_SOURCES, LLM_SERVICE_URL,
    MockRAGNetwork, SelectiveForwardingAttack,
    apply_stealthy_drop, restore_sources,
    SFADetector, SFAMitigation,
    check_blockchain_status,
)

LOG_DIR = pathlib.Path("attack_logs/selective_forwarding")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def save_json(data, suffix):
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = LOG_DIR / f"attack_{ts}_selective_forward{suffix}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path

def _header(mode, args):
    print("=" * 70)
    print(f"  Selective Forwarding Attack — Reliable-dRAG  [{mode.upper()}]")
    print("=" * 70)
    print(f"  sources={args.sources}  max_hops={args.max_hops}  "
          f"queries={args.num_queries}  hit_prob={args.peer_hit_prob}  seed={args.seed}\n")

def _baseline(network, num_queries):
    questions = [f"q_{i}" for i in range(num_queries)]
    hits = hops_t = ttl = 0
    for q in questions:
        a = network.query(q)
        if a.is_query_hit: hits += 1
        else:              ttl  += 1
        hops_t += a.num_hops
    return {
        "strategy": "baseline", "attack_ratio": 0.0, "num_compromised": 0,
        "compromised_ids": "", "avg_ssm_score_compromised": 0.0,
        "avg_ssm_score_all": 10_000.0,
        "hit_rate":            round(hits / num_queries, 4),
        "avg_hops_per_query":  round(hops_t / num_queries, 2),
        "ttl_exhaustion_rate": round(ttl / num_queries, 4),
        "dropped_queries": 0, "total_queries": num_queries,
        "answered_queries": hits, "exhausted_queries": ttl, "attack": "",
    }

def _run_ratio(network, ratio, strategy, num_queries, seed, drop_prob=None):
    """Run one (ratio, strategy) sweep. drop_prob=None → aggressive 100%."""
    questions = [f"q_{i}" for i in range(num_queries)]

    if drop_prob is not None:
        saved, targets, ctr = apply_stealthy_drop(
            network, ratio, strategy, seed, drop_prob)
    else:
        sfa    = SelectiveForwardingAttack(attack_ratio=ratio, seed=seed)
        n_comp = sfa.apply(network, strategy=strategy)
        targets = list(sfa._compromised_ids) if hasattr(sfa, '_compromised_ids') else []

    hits = hops_t = ttl = 0
    for q in questions:
        a = network.query(q)
        if a.is_query_hit: hits += 1
        else:              ttl  += 1
        hops_t += a.num_hops

    dropped = 0
    if drop_prob is not None:
        dropped = ctr[0]
        restore_sources(network, saved)
    else:
        dropped = getattr(sfa, '_dropped_queries', 0)

    ssm_comp = [network.ssm_scores[s]["reliability"] for s in targets if s in network.ssm_scores]
    ssm_all  = [network.ssm_scores[s]["reliability"] for s in network.sources]
    n_comp   = len(targets)
    mode_tag = f"stealthy_p{drop_prob:.2f}" if drop_prob is not None else "aggressive"

    return {
        "strategy": strategy, "attack_ratio": ratio,
        "num_compromised": n_comp, "compromised_ids": "|".join(targets),
        "avg_ssm_score_compromised": round(float(np.mean(ssm_comp)) if ssm_comp else 0, 2),
        "avg_ssm_score_all":         round(float(np.mean(ssm_all)), 2),
        "hit_rate":            round(hits / num_queries, 4),
        "avg_hops_per_query":  round(hops_t / num_queries, 2),
        "ttl_exhaustion_rate": round(ttl / num_queries, 4),
        "dropped_queries":     dropped,
        "total_queries":       num_queries,
        "answered_queries":    hits,
        "exhausted_queries":   ttl,
        "attack":              mode_tag,
    }

# ─────────────────────────────────────────────── MODE: MOCK ──────────────────

def run_mock(args):
    _header("mock", args)
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = []

    print(f"  {'Strategy':22s}  Ratio  Comp  HitRate  AvgHops  TTLExh  Dropped")
    print("  " + "-" * 66)

    network  = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
    baseline = _baseline(network, args.num_queries)
    results.append(baseline)
    print(f"  {'baseline':22s}  ratio=0.0  comp= 0  "
          f"hit_rate={baseline['hit_rate']:.3f}  "
          f"avg_hops={baseline['avg_hops_per_query']:.2f}  "
          f"ttl_exhaust={baseline['ttl_exhaustion_rate']:.3f}  dropped=0")

    for ratio in args.ratios:
        if ratio == 0.0: continue
        for strategy in args.strategies:
            network = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
            row = _run_ratio(network, ratio, strategy, args.num_queries, args.seed)
            results.append(row)
            print(f"  {strategy:22s}  ratio={ratio:.1f}  comp={row['num_compromised']:2d}  "
                  f"hit_rate={row['hit_rate']:.3f}  "
                  f"avg_hops={row['avg_hops_per_query']:.2f}  "
                  f"ttl_exhaust={row['ttl_exhaustion_rate']:.3f}  "
                  f"dropped={row['dropped_queries']}")

    bhr = baseline["hit_rate"]; bte = baseline["ttl_exhaustion_rate"]
    print(f"\n{'=' * 70}\n  Summary: degradation vs baseline\n{'=' * 70}")
    print(f"  Baseline hit_rate    : {bhr:.3f}")
    print(f"  Baseline ttl_exhaust : {bte:.3f}\n")
    for r in results[1:]:
        print(f"  {r['strategy']:22s}  ratio={r['attack_ratio']:.1f}  "
              f"hit_rate={r['hit_rate']:.3f}  delta={r['hit_rate']-bhr:+.3f}  "
              f"ttl_exhaust={r['ttl_exhaustion_rate']:.3f}  "
              f"delta_ttl={r['ttl_exhaustion_rate']-bte:+.3f}")

    csv_path = LOG_DIR / f"results_seed{args.seed}.csv"
    save_csv(results, csv_path)
    lp = save_json({"timestamp": ts, "attack_type": "selective_forward_mock",
                    "attack_config": vars(args), "results": results},
                   f"_mock_seed{args.seed}")

    chain = check_blockchain_status()
    worst = min(results, key=lambda r: r["hit_rate"])
    print(f"\n  Results saved to: {csv_path.resolve()}")
    print(f"  JSON  -> {lp}")
    print(f"\n  === Thesis Summary ===")
    print(f"  Worst hit_rate     : {worst['hit_rate']:.3f} "
          f"(ratio={worst['attack_ratio']}, strategy={worst['strategy']})")
    print(f"  Max ttl_exhaustion : {max(r['ttl_exhaustion_rate'] for r in results):.3f}")
    print(f"  Blockchain status  : {chain['status_msg']}")

# ──────────────────────────────────────────── MODE: STEALTHY ─────────────────

def run_stealthy(args):
    _header("stealthy — gray-hole probabilistic drop", args)
    DROP_PROBS = [0.10, 0.20, 0.30]
    results = []

    print(f"  {'Strategy':22s}  DropP  Comp  HitRate  Dropped  TTLExh  Delta")
    print("  " + "-" * 70)

    network  = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
    baseline = _baseline(network, args.num_queries)
    results.append(baseline)
    bhr = baseline["hit_rate"]
    print(f"  {'baseline (no attack)':22s}  0.00   0     "
          f"{bhr:.3f}    0        {baseline['ttl_exhaustion_rate']:.3f}    ---")

    for strategy in args.strategies:
        for dp in DROP_PROBS:
            network = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
            row = _run_ratio(network, 1.0, strategy, args.num_queries, args.seed, drop_prob=dp)
            results.append(row)
            print(f"  {strategy:22s}  {dp:.2f}   {row['num_compromised']:2d}    "
                  f"{row['hit_rate']:.3f}    {row['dropped_queries']:5d}    "
                  f"{row['ttl_exhaustion_rate']:.3f}    {row['hit_rate']-bhr:+.3f}")

    csv_path = LOG_DIR / f"results_stealthy_seed{args.seed}.csv"
    save_csv(results, csv_path)
    lp = save_json({"timestamp": datetime.now().isoformat(),
                    "attack_type": "selective_forward_stealthy",
                    "drop_probs": DROP_PROBS, "results": results},
                   f"_stealthy_seed{args.seed}")

    worst = min(results[1:], key=lambda r: r["hit_rate"])
    print(f"\n  Results saved to: {csv_path.resolve()}\n  JSON -> {lp}")
    print(f"\n  === Stealthy Summary ===")
    print(f"  Baseline   hit_rate: {bhr:.3f}")
    print(f"  Worst      hit_rate: {worst['hit_rate']:.3f} ({worst['attack']})")
    print(f"  Insight: gray-hole evades all-or-nothing detection while still degrading service.")

# ─────────────────────────────────────────── MODE: DETECTION ─────────────────

def run_detection(args):
    _header("detection — z-score anomaly detector", args)
    print(f"  Detector: window={args.detect_window}  min_samples={args.detect_min_samples}  "
          f"z_thresh={args.detect_z_thresh}  ssm_decay={args.detect_ssm_decay}\n")
    print(f"  {'Strategy':22s}  Ratio  Detected  Flagged            HitRate  MinSSM")
    print("  " + "-" * 80)

    results = []
    for ratio in [r for r in args.ratios if r > 0]:
        for strategy in args.strategies:
            network  = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
            detector = SFADetector(window=args.detect_window,
                                   min_samples=args.detect_min_samples,
                                   z_thresh=args.detect_z_thresh,
                                   ssm_decay=args.detect_ssm_decay)
            sfa = SelectiveForwardingAttack(attack_ratio=ratio, seed=args.seed)
            n_comp = sfa.apply(network, strategy=strategy)

            hits = ttl = 0
            sids = list(network.sources.keys())
            for i in range(args.num_queries):
                a = network.query(f"q_{i}")
                if a.is_query_hit: hits += 1
                else:              ttl  += 1
                for h in range(min(a.num_hops, len(sids))):
                    detector.record(sids[h],
                                    (h == a.num_hops - 1) and a.is_query_hit)

            det = detector.detect(ssm_scores=network.ssm_scores)
            min_ssm = min(v["reliability"] for v in network.ssm_scores.values())
            flagged_str = ", ".join(det["flagged_sources"]) or "none"

            print(f"  {strategy:22s}  {ratio:.2f}   "
                  f"{'YES':8s}  {flagged_str:18s}  "
                  f"{round(hits/args.num_queries,3):.3f}    {min_ssm:,.0f}"
                  if det["detected"] else
                  f"  {strategy:22s}  {ratio:.2f}   "
                  f"{'NO':8s}  {'none':18s}  "
                  f"{round(hits/args.num_queries,3):.3f}    {min_ssm:,.0f}")

            results.append({"strategy": strategy, "attack_ratio": ratio,
                             "num_compromised": n_comp, "detected": det["detected"],
                             "flagged": flagged_str,
                             "hit_rate": round(hits / args.num_queries, 4),
                             "ttl_exhaust": round(ttl / args.num_queries, 4),
                             "min_ssm": round(min_ssm, 1)})

    csv_path = LOG_DIR / f"results_detection_seed{args.seed}.csv"
    save_csv(results, csv_path)
    lp = save_json({"timestamp": datetime.now().isoformat(),
                    "attack_type": "detection", "results": results},
                   f"_detection_seed{args.seed}")
    n_det = sum(1 for r in results if r["detected"])
    print(f"\n  Results -> {csv_path.resolve()}\n  JSON -> {lp}")
    print(f"\n  === Detection Summary ===")
    print(f"  Detected {n_det}/{len(results)} scenarios ({n_det/len(results)*100:.0f}%)")
    print(f"  SSM decay: {args.detect_ssm_decay}x per flag — misbehaving nodes lose reputation.")

# ──────────────────────────────────────── MODE: MITIGATION ───────────────────

def run_mitigation(args):
    _header("mitigation — blacklist + SSM-ordered rerouting", args)
    print(f"  {'Strategy':22s}  Ratio  Attacked_HR  Mitigated_HR  Recovery  Blacklisted")
    print("  " + "-" * 80)

    results = []
    for ratio in [r for r in args.ratios if r > 0]:
        for strategy in args.strategies:
            # Phase 1: attacked (no mitigation)
            net1 = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
            sfa1 = SelectiveForwardingAttack(attack_ratio=ratio, seed=args.seed)
            sfa1.apply(net1, strategy=strategy)
            hits_atk = sum(1 for i in range(args.num_queries)
                           if net1.query(f"q_{i}").is_query_hit)
            atk_hr   = hits_atk / args.num_queries

            # Phase 2: detection warm-up then mitigation
            net2     = MockRAGNetwork(peer_hit_prob=args.peer_hit_prob, seed=args.seed)
            detector = SFADetector(window=args.detect_window,
                                   min_samples=args.detect_min_samples,
                                   z_thresh=args.detect_z_thresh,
                                   ssm_decay=args.detect_ssm_decay)
            mitigator = SFAMitigation()
            sfa2 = SelectiveForwardingAttack(attack_ratio=ratio, seed=args.seed)
            sfa2.apply(net2, strategy=strategy)

            half = args.num_queries // 2
            sids = list(net2.sources.keys())
            for i in range(half):
                a = net2.query(f"q_{i}")
                for h in range(min(a.num_hops, len(sids))):
                    detector.record(sids[h], (h == a.num_hops-1) and a.is_query_hit)

            det = detector.detect(ssm_scores=net2.ssm_scores)
            mitigator.apply_blacklist(det, net2)

            hits_mit = sum(1 for i in range(half, args.num_queries)
                           if mitigator.query_rerouted(net2, f"q_{i}")[0])
            mit_hr   = hits_mit / (args.num_queries - half) if args.num_queries > half else 0.0
            delta    = mit_hr - atk_hr
            bl_str   = ", ".join(mitigator.blacklist) or "none"

            print(f"  {strategy:22s}  {ratio:.2f}   {atk_hr:.3f}        "
                  f"{mit_hr:.3f}         {delta:+.3f}    {bl_str}")

            results.append({"strategy": strategy, "attack_ratio": ratio,
                             "attacked_hr": round(atk_hr, 4),
                             "mitigated_hr": round(mit_hr, 4),
                             "recovery_delta": round(delta, 4),
                             "blacklisted": bl_str,
                             "detected": det["detected"]})

    csv_path = LOG_DIR / f"results_mitigation_seed{args.seed}.csv"
    save_csv(results, csv_path)
    lp = save_json({"timestamp": datetime.now().isoformat(),
                    "attack_type": "mitigation", "results": results},
                   f"_mitigation_seed{args.seed}")
    avg_rec = float(np.mean([r["recovery_delta"] for r in results]))
    print(f"\n  Results -> {csv_path.resolve()}\n  JSON -> {lp}")
    print(f"\n  === Mitigation Summary ===")
    print(f"  Avg recovery delta : {avg_rec:+.3f}")
    print(f"  Strategy: warm-up {args.num_queries//2} queries → detect → blacklist → reroute")

# ────────────────────────────────────────────── MODE: LIVE ───────────────────

def run_live(args):
    _header("live — real HTTP + blockchain", args)
    import requests

    print("  Checking blockchain ...")
    chain = check_blockchain_status()
    print(f"  {chain['status_msg']}\n")

    if not chain["online"]:
        print("  WARNING: Blockchain offline. Start Docker stack:")
        print("    cd reliable_rag_fixes && docker compose up -d\n")

    sids   = list(DATA_SOURCES.keys())
    n_comp = max(1, round(args.attack_ratio * len(sids)))
    import numpy as _np
    rng    = _np.random.default_rng(args.seed)
    comp   = set(sids[i] for i in rng.choice(len(sids), n_comp, replace=False))

    def http_src(url, q, k=5):
        try:
            hdrs = {"X-API-Key": args.api_key} if args.api_key else {}
            r = requests.post(f"{url}/query", json={"query": q, "k": k},
                              headers=hdrs, timeout=10)
            r.raise_for_status()
            d = r.json()
            docs = d.get("documents", [])
            return docs, float(d.get("score", 0)), bool(d.get("is_hit", len(docs) > 0))
        except Exception:
            return [], 0.0, False

    def http_llm(q):
        try:
            r = requests.post(f"{LLM_SERVICE_URL}/query", json={"query": q}, timeout=30)
            return r.json().get("answer", "")
        except Exception:
            return ""

    def correct(ans, golds):
        al = ans.lower()
        return any(g.lower() in al for g in golds)

    base_ok = atk_ok = 0
    for item in EVAL_DATA:
        q, golds = item["question"], item["answers"]
        for sid, url in DATA_SOURCES.items():
            _, _, hit = http_src(url, q)
            if hit:
                base_ok += int(correct(http_llm(q), golds))
                break

    for item in EVAL_DATA:
        q, golds = item["question"], item["answers"]
        for sid, url in DATA_SOURCES.items():
            if sid in comp: continue
            _, _, hit = http_src(url, q)
            if hit:
                atk_ok += int(correct(http_llm(q), golds))
                break

    base_acc = base_ok / len(EVAL_DATA); atk_acc = atk_ok / len(EVAL_DATA)
    lp = save_json({"timestamp": datetime.now().isoformat(),
                    "mode": "live", "attack_ratio": args.attack_ratio,
                    "compromised": list(comp),
                    "baseline_accuracy": round(base_acc, 4),
                    "attacked_accuracy": round(atk_acc, 4),
                    "accuracy_drop_pp": round((base_acc - atk_acc) * 100, 2),
                    "blockchain": chain}, "_live")

    print(f"  Baseline accuracy  : {base_acc:.4f}")
    print(f"  Attacked accuracy  : {atk_acc:.4f}")
    print(f"  Accuracy drop      : {(base_acc-atk_acc)*100:.1f} pp")
    print(f"  Compromised        : {list(comp)}")
    print(f"  Blockchain         : {chain['status_msg']}")
    print(f"  Log -> {lp}")

# ─────────────────────────────────────────────────── CLI ─────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="SFA — Reliable-dRAG (complete)")
    p.add_argument("--mode", choices=["mock","stealthy","detection","mitigation","live"],
                   default="mock")
    p.add_argument("--sources",          type=int,   default=3)
    p.add_argument("--max_hops",         type=int,   default=3)
    p.add_argument("--peer_hit_prob",    type=float, default=0.4)
    p.add_argument("--ratios",    nargs="+", type=float, default=[0.0,0.34,0.67,1.0])
    p.add_argument("--strategies",nargs="+", default=["random","high_ssm_score"])
    p.add_argument("--num_queries",      type=int,   default=500)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--detect_window",    type=int,   default=50)
    p.add_argument("--detect_min_samples",type=int,  default=10)
    p.add_argument("--detect_z_thresh",  type=float, default=1.5)
    p.add_argument("--detect_ssm_decay", type=float, default=0.5)
    p.add_argument("--attack_ratio",     type=float, default=0.34)
    p.add_argument("--k",                type=int,   default=5)
    p.add_argument("--api_key",          type=str,   default="")
    return p

def main():
    args = build_parser().parse_args()
    {"mock": run_mock, "stealthy": run_stealthy, "detection": run_detection,
     "mitigation": run_mitigation, "live": run_live}[args.mode](args)

if __name__ == "__main__":
    main()
'''

RUN.write_text(NEW_RUN)
print(f"Written new {RUN.name}")

# ── Step 5: verify ────────────────────────────────────────────────────────────
sfa_text = SFA.read_text()
run_text = RUN.read_text()

checks = [
    ("SelectiveForwardingAttack in sfa", "class SelectiveForwardingAttack" in sfa_text),
    ("SFADetector in sfa",               "class SFADetector"               in sfa_text),
    ("SFAMitigation in sfa",             "class SFAMitigation"             in sfa_text),
    ("check_blockchain_status in sfa",   "def check_blockchain_status"     in sfa_text),
    ("run_stealthy in run",              "def run_stealthy"                 in run_text),
    ("run_detection in run",             "def run_detection"                in run_text),
    ("run_mitigation in run",            "def run_mitigation"               in run_text),
    ("from selective_forward_attack in run",
                                         "from selective_forward_attack import" in run_text),
]

print("\n  Verification:")
all_ok = True
for label, ok in checks:
    print(f"  {'✓' if ok else '✗'} {label}")
    if not ok: all_ok = False

print()
if all_ok:
    print("All checks passed. Run:")
    print("  python3 attack/selective_forward/run_attack.py --mode mock")
    print("  python3 attack/selective_forward/run_attack.py --mode stealthy")
    print("  python3 attack/selective_forward/run_attack.py --mode detection")
    print("  python3 attack/selective_forward/run_attack.py --mode mitigation")
else:
    print("Some checks failed — check error output above.")
    sys.exit(1)
