"""
defense/sfa_sim_defense/selective_forwarding_defense.py

Countermeasure for attack.selective_forward_sim.SelectiveForwardingAttack.
Works against either network_sim.MockRAGNetwork or live_network.LiveRAGNetwork
-- both expose `.peers`, `.network` (networkx graph), `.topic_aware_query()`,
and a `._sfa_defense` slot the BFS loop consults every hop.

Three layers
------------
1. Reputation tracking -- every `peer.query()` call is intercepted by a
   wrapper installed in `apply()`. Each result updates an EMA-blended
   per-peer reputation score, starting at 1.0 (trusted by default).
2. Blacklisting -- once a peer has accumulated `min_queries_before_blacklist`
   interactions, if its raw response rate is below `blacklist_threshold`
   it is auto-blacklisted. `suspicion_threshold` gives a softer,
   reputation-based flag (`is_peer_suspicious`) below full blacklisting.
   Capped by `max_blacklist_fraction`: never let auto-blacklisting exclude
   more than that fraction of the *known* peer population. A single fixed
   absolute `blacklist_threshold` cannot tell "this peer is actively
   dropping queries" apart from "the whole population's natural response
   rate is just low" (e.g. a genuinely low-relevance corpus on some real
   sources) -- without a cap, that ambiguity can escalate into
   blacklisting most or all peers, which empties Phase 2's candidate pool
   (see backup_candidates()) and collapses recovery back to zero exactly
   when the defense is needed most. Excluding a *majority* of peers is
   itself a signal the threshold is miscalibrated for this population,
   not evidence to act on blindly -- the same reasoning behind not trusting
   a Byzantine quorum to exceed a bounded fraction of the network.
3. Routing bypass -- `network.topic_aware_query()` checks
   `is_peer_blacklisted()` before calling a peer; if blacklisted it skips
   the call entirely and expands that peer's neighbours at the same hop
   depth instead, recovering reachability without spending extra TTL.

Detection modes (`detection_mode`)
-----------------------------------
"threshold" (default) -- blacklist once raw response rate drops below a
   fixed `blacklist_threshold` (0.05 by default). Simple and effective
   against a black-hole attacker (attack/selective_forward_sim's
   `drop_rate=1.0`, always drops) but structurally blind to a *stealthy*
   attacker that drops only, say, 10-30% of queries -- 0.05 is nowhere
   close to a 70-90% response rate, so a fixed low threshold like this
   can never fire against that attacker regardless of sample size.
"binomial" -- a one-sided binomial significance test against a measured
   `honest_miss_rate` baseline, escalating over `streak_required`
   consecutive significant observations before blacklisting (see
   `_binom_p()`). This is the statistically-principled detector design
   used by attack/selective_forward's `SFADetector`, ported here as an
   explicit opt-in mode rather than this module's default, so a stealthy
   attacker within Uniform(0.10, 0.30) can be caught: the test asks "is
   this peer's miss rate significantly above what an honest peer shows,"
   not "is it blacklist_threshold-bad."

   `honest_miss_rate` defaults to `"auto"` (BUG FIX -- see
   `_effective_honest_miss_rate()`): the baseline is measured LIVE, every
   time the test runs, as the median observed miss rate across the other
   currently-tracked, not-yet-blacklisted peers, rather than a hardcoded
   constant. This module used to ship `honest_miss_rate=0.05` as a static
   default, which silently mismatched its own default mock network
   config (`peer_hit_prob: 0.4`, true honest miss rate ~0.6) -- reproduced
   live, on this project's own default settings, the exact miscalibration
   failure this paragraph already warned about in an earlier revision (an
   inherited/assumed constant, never validated against the actual
   population): 5 of 10 peers blacklisted on every tested ratio, including
   when only 1 was genuinely compromised
   (reports/SFA_Security_Analysis_Report.md sec 12.7). Auto-calibration
   closes that specific failure mode structurally, since there is no
   longer a static constant to go stale -- pass an explicit float instead
   of `"auto"` only if you have a specific pre-measured value you want to
   pin (e.g. reproducing a past run, or a deployment where the honest
   baseline is known in advance and won't drift).

   Verified against a fixed 20%-drop stealthy attacker (2/10 peers
   compromised, honest peer_hit_prob=0.95, 300 queries, 5 seeds):
   "threshold" mode never blacklisted either true attacker (0/2 every
   run -- exactly the blind spot this mode exists to fix). "binomial"
   mode caught 1-2/2 depending on seed, with an honest false positive on
   some runs at streak_required=2; raising streak_required to 3 (this
   module's default) cut false positives roughly 3x across the same
   seeds with similar recall. This is a real, measured trade-off, not a
   guarantee of zero false positives.

   BUG FIX (true sliding window, was growing window): `_binomial_should_
   blacklist()` used to re-test the significance of a peer's CUMULATIVE
   (n, raw_rate) pair every interaction -- n only ever grew, never reset
   or bounded -- unlike the sibling module's `SFADetector`, which re-tests
   over a rolling last-`binom_window` sample (`deque(maxlen=WINDOW)`).
   Two compounding problems with the growing-window version: (1) as n
   grows without bound, even a tiny, noise-level deviation from the
   honest baseline eventually reads as "significant" under a fixed
   `binom_alpha` (classic sequential-testing / "peeking" inflation of the
   false-positive rate as the sample keeps accumulating), and (2) an
   early unlucky patch of misses is diluted rather than aged out by
   later good behavior, so old evidence never leaves the test. Both
   `_binomial_should_blacklist()` (the subject peer's own test) and
   `_effective_honest_miss_rate()` (the other-peers baseline) now draw
   from a fixed-size rolling window (`self._binom_obs`, `binom_window`
   samples per peer, default 40 to match `SFADetector.WINDOW`) instead of
   the cumulative `_query_count`/`_response_count` totals -- those totals
   are unchanged and still drive reputation (Layer 1) and "threshold"
   mode, only the binomial test's own (n, rate) inputs changed.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set


class SelectiveForwardingDefense:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.blacklist_threshold = float(cfg.get("blacklist_threshold", 0.05))
        self.min_queries_before_blacklist = int(cfg.get("min_queries_before_blacklist", 15))
        self.suspicion_threshold = float(cfg.get("suspicion_threshold", 0.10))
        self.reputation_decay = float(cfg.get("reputation_decay", 0.70))
        self.min_peers_for_validation = int(cfg.get("min_peers_for_validation", 2))
        self.reputation_ema_alpha = 1.0 - self.reputation_decay
        # Bounded Phase 2 fallback (see backup_candidates()): once the
        # primary TTL-bounded BFS in network_sim.py/live_network.py runs out
        # of hop budget without a hit, the network gives the defense up to
        # `redundancy_k` extra attempts against peers it trusts and hasn't
        # already tried. This is what lets correct blacklisting translate
        # into a recovered answer when the hop budget alone is too tight to
        # route around a compromised peer -- without it, detection can be
        # 100% correct and still recover nothing (see README). Capped, unlike
        # attack/selective_forward's SFAMitigation.route() Phase 2, which
        # tries up to redundancy_k backups with no bound relative to
        # max_hops at all -- that's a fallback so generous it can mask
        # whether the stated hop budget matters.
        self.redundancy_k = int(cfg.get("redundancy_k", 2))
        # Quorum-preserving cap: auto-blacklisting never excludes more than
        # this fraction of the known peer population, and always leaves at
        # least one peer un-blacklisted regardless of the fraction (see
        # _blacklist_cap()). Set by apply() once the network's peer count is
        # known; defaults to "uncapped" (None) until then so unit-level use
        # of record_peer_interaction() without a network still works.
        self.max_blacklist_fraction = float(cfg.get("max_blacklist_fraction", 0.5))
        self._num_peers: Optional[int] = None

        # "threshold" | "binomial" -- see module docstring "Detection modes".
        self.detection_mode = cfg.get("detection_mode", "threshold")
        # BUG FIX (see _effective_honest_miss_rate() below for the full
        # writeup): honest_miss_rate used to ship as a hardcoded float
        # default (0.05) that silently mismatched this module's own
        # default mock network config -- reproduced live, on this
        # project's own default settings, the exact miscalibration bug
        # this class's docstring already warned against (5/10 peers
        # blacklisted when only 1 was truly compromised;
        # reports/SFA_Security_Analysis_Report.md sec 12.7). Default is
        # now "auto": self.honest_miss_rate stays None and the effective
        # baseline is measured live from the currently-tracked population
        # every time the binomial test runs. Pass an explicit float (as
        # before) to pin a specific pre-measured value instead -- fully
        # backward compatible for any caller that already does so.
        raw_hmr = cfg.get("honest_miss_rate", "auto")
        if isinstance(raw_hmr, str) and raw_hmr.strip().lower() == "auto":
            self.honest_miss_rate: Optional[float] = None
        else:
            self.honest_miss_rate = float(raw_hmr)
        # Auto-calibration needs at least this many OTHER currently-tracked,
        # not-yet-blacklisted peers with enough samples before the median
        # baseline estimate is trusted -- below that, _binomial_should_blacklist
        # abstains (returns False) rather than testing against a guess.
        self.min_peers_for_auto_calibration = int(cfg.get("min_peers_for_auto_calibration", 2))
        self.binom_alpha = float(cfg.get("binom_alpha", 0.05))
        self.streak_required = int(cfg.get("streak_required", 3))
        # True sliding window for the binomial test (see module docstring
        # "BUG FIX"), default 40 to match the sibling module's
        # SFADetector.WINDOW. Independent of min_queries_before_blacklist,
        # which only gates *when* testing starts.
        self.binom_window = int(cfg.get("binom_window", 40))
        self._binom_obs: Dict[int, Deque[int]] = {}
        self._streak: Dict[int, int] = {}
        self.suspicion_level: Dict[int, int] = {}

        self._query_count: Dict[int, int] = {}
        self._response_count: Dict[int, int] = {}
        self._reputation: Dict[int, float] = {}
        self.blacklisted_peers: Set[int] = set()

        self._original_query_fns: Dict[int, Any] = {}
        self._total_bypasses = 0
        self._total_blacklistings = 0
        self._total_validations = 0
        self._blocked_answers = 0
        self._passed_answers = 0
        self._confidence_sum = 0.0
        self._redundant_probes = 0
        self._redundant_hits = 0

    # ── Layer 1: reputation ─────────────────────────────────────────────
    def record_peer_interaction(self, peer_id: int, responded: bool) -> None:
        self._query_count[peer_id] = self._query_count.get(peer_id, 0) + 1
        self._response_count.setdefault(peer_id, 0)
        if responded:
            self._response_count[peer_id] += 1
        self._binom_obs.setdefault(peer_id, deque(maxlen=self.binom_window)).append(0 if responded else 1)

        n = self._query_count[peer_id]
        raw_rate = self._response_count[peer_id] / n
        current = self._reputation.get(peer_id, 1.0)
        alpha = self.reputation_ema_alpha
        self._reputation[peer_id] = (1.0 - alpha) * current + alpha * raw_rate

        if peer_id in self.blacklisted_peers or n < self.min_queries_before_blacklist:
            return

        if self.detection_mode == "binomial":
            should_blacklist = self._binomial_should_blacklist(peer_id)
        else:
            should_blacklist = raw_rate < self.blacklist_threshold

        if should_blacklist and len(self.blacklisted_peers) < self._blacklist_cap():
            self.blacklisted_peers.add(peer_id)
            self._total_blacklistings += 1

    def _effective_honest_miss_rate(self, exclude_peer_id: int) -> Optional[float]:
        """
        Live-measured honest baseline used whenever `honest_miss_rate` is
        left at its default "auto" rather than pinned to a specific
        pre-measured constant (see __init__).

        BUG FIX: this module used to ship a hardcoded `honest_miss_rate`
        default (0.05) that silently mismatched its own default mock
        network config (`peer_hit_prob: 0.4`, i.e. a true honest miss rate
        of ~0.6). Wiring the binomial detector's shipped default into that
        default config blacklisted 5 of 10 peers on every ratio tested,
        including when only 1 was genuinely compromised -- reproduced live,
        on this project's own code, the exact miscalibration bug this
        class's own docstring already warned against (see
        reports/SFA_Security_Analysis_Report.md sec 12.7). A static
        constant is only ever correct for the one population it happened
        to be measured against, and goes stale silently the moment the
        deployment, config, or network size changes.

        This replaces the static constant with the MEDIAN observed miss
        rate across every OTHER currently-tracked, not-yet-blacklisted
        peer with at least `min_queries_before_blacklist` samples --
        median rather than mean specifically because it stays anchored to
        the honest majority as long as compromised peers are a minority
        of the population (the same assumption `max_blacklist_fraction`
        already relies on elsewhere in this class), instead of being
        dragged upward by one or two genuinely compromised peers' high
        miss rates the way a mean would be.

        Returns None if fewer than `min_peers_for_auto_calibration` other
        peers currently have enough data to trust the estimate -- callers
        must treat None as "can't test yet," not as "assume 0."

        Uses each other peer's own rolling `binom_window` sample (see
        module docstring "BUG FIX"), not their all-time cumulative rate,
        so the baseline reflects recent honest behavior rather than
        stale, unboundedly-aged data.
        """
        others = [
            pid for pid in self._query_count
            if pid != exclude_peer_id
            and pid not in self.blacklisted_peers
            and self._query_count[pid] >= self.min_queries_before_blacklist
            and self._binom_obs.get(pid)
        ]
        if len(others) < self.min_peers_for_auto_calibration:
            return None
        rates = sorted(sum(self._binom_obs[pid]) / len(self._binom_obs[pid]) for pid in others)
        mid = len(rates) // 2
        if len(rates) % 2 == 1:
            return rates[mid]
        return (rates[mid - 1] + rates[mid]) / 2.0

    def _binomial_should_blacklist(self, peer_id: int) -> bool:
        """
        One-sided binomial significance test: is this peer's miss rate
        significantly above the honest baseline (either the pinned
        `honest_miss_rate`, or the live "auto"-calibrated estimate -- see
        _effective_honest_miss_rate())? Escalates a running streak of
        significant observations (reset on a non-significant one) and
        only recommends blacklisting once the streak reaches
        `streak_required` -- a single unlucky window on an honest peer
        isn't enough, mirroring attack/selective_forward's SFADetector.

        Tests over this peer's rolling `binom_window` sample
        (`self._binom_obs`), a true fixed-size sliding window matching
        `SFADetector`'s design (see module docstring "BUG FIX") -- not
        the peer's all-time cumulative (n, raw_rate), which is still
        tracked separately in `_query_count`/`_response_count` for
        reputation (Layer 1) and "threshold" mode.
        """
        honest_miss_rate = self.honest_miss_rate
        if honest_miss_rate is None:  # "auto" mode
            honest_miss_rate = self._effective_honest_miss_rate(peer_id)
            if honest_miss_rate is None:
                # Not enough reference peers yet to trust a baseline estimate
                # -- abstain rather than test against a guess (same spirit as
                # min_queries_before_blacklist gating on insufficient
                # per-peer evidence, just applied to population-level
                # evidence instead).
                self._streak[peer_id] = max(0, self._streak.get(peer_id, 0) - 1)
                self.suspicion_level[peer_id] = min(3, self._streak[peer_id])
                return False

        window = self._binom_obs.get(peer_id, deque())
        window_n = len(window)
        miss_rate = (sum(window) / window_n) if window_n else 0.0
        p_value = self._binom_p(miss_rate, window_n, honest_miss_rate)
        if p_value < self.binom_alpha:
            self._streak[peer_id] = self._streak.get(peer_id, 0) + 1
        else:
            self._streak[peer_id] = max(0, self._streak.get(peer_id, 0) - 1)
        self.suspicion_level[peer_id] = min(3, self._streak[peer_id])
        return self._streak[peer_id] >= self.streak_required

    def _binom_p(self, miss_rate: float, n: int, honest_miss_rate: float) -> float:
        """One-sided binomial p-value: is miss_rate > honest_miss_rate?"""
        try:
            from scipy.stats import binomtest  # type: ignore
            k = int(round(miss_rate * n))
            return float(binomtest(k, n, honest_miss_rate, alternative="greater").pvalue)
        except Exception:
            mu = honest_miss_rate * n
            sigma = math.sqrt(n * honest_miss_rate * (1 - honest_miss_rate)) + 1e-9
            z = (miss_rate * n - mu) / sigma
            return max(0.0, 1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    def _blacklist_cap(self) -> int:
        """
        Maximum number of peers auto-blacklisting may exclude at once.
        Uncapped (effectively unlimited) until apply() has told us the real
        peer count; otherwise floor(num_peers * max_blacklist_fraction),
        but never allowed to reach num_peers itself -- at least one peer
        must always remain reachable so Phase 2 (backup_candidates()) is
        never left with zero candidates purely because every peer crossed
        the same fixed threshold.
        """
        if self._num_peers is None:
            return 10**9
        return max(0, min(int(self._num_peers * self.max_blacklist_fraction), self._num_peers - 1))

    def reputation(self, peer_id: int) -> float:
        return self._reputation.get(peer_id, 1.0)

    # ── Phase 2: bounded redundant backup probe ─────────────────────────
    def backup_candidates(self, tried_ids, all_peer_ids: List[int]) -> List[int]:
        """
        Up to `redundancy_k` peers the network hasn't already visited this
        query and that aren't blacklisted, ranked by reputation (an unknown
        peer defaults to reputation 1.0 -- untested, not distrusted).
        Called by network_sim.MockRAGNetwork / live_network.LiveRAGNetwork
        after the primary TTL-bounded BFS exhausts its hop budget.
        """
        candidates = [pid for pid in all_peer_ids if pid not in tried_ids and not self.is_peer_blacklisted(pid)]
        candidates.sort(key=lambda pid: -self.reputation(pid))
        return candidates[: self.redundancy_k]

    def record_redundant_probe(self, hit: bool) -> None:
        self._redundant_probes += 1
        if hit:
            self._redundant_hits += 1

    # ── Layer 2: blacklist / suspicion ──────────────────────────────────
    def is_peer_blacklisted(self, peer_id: int) -> bool:
        return peer_id in self.blacklisted_peers

    def is_peer_suspicious(self, peer_id: int) -> bool:
        return peer_id in self.blacklisted_peers or self.reputation(peer_id) < self.suspicion_threshold

    def filter_trusted_peers(self, peer_ids: List[int]) -> List[int]:
        trusted = [p for p in peer_ids if not self.is_peer_suspicious(p)]
        if trusted:
            return trusted
        non_blacklisted = [p for p in peer_ids if p not in self.blacklisted_peers]
        if non_blacklisted:
            return non_blacklisted  # fallback: allow suspicious-but-not-blacklisted
        return list(peer_ids)  # last resort: avoid total disconnection

    # ── Layer 3: routing-bypass hook, called from network.topic_aware_query() ──
    def record_bypass(self) -> None:
        self._total_bypasses += 1

    # ── apply/remove: wrap peer.query() + install the bypass hook ──────
    def apply(self, network) -> None:
        self._num_peers = sum(1 for p in network.peers if p is not None)
        for pid, peer in enumerate(network.peers):
            if peer is None:
                continue
            original_fn = peer.query
            self._original_query_fns[pid] = original_fn
            peer.query = self._make_wrapper(pid, original_fn)
        network._sfa_defense = self

    def _make_wrapper(self, peer_id: int, orig):
        defense = self

        def _wrapped_query(question, query_confidence_threshold=0.5, *args, **kwargs):
            if defense.is_peer_blacklisted(peer_id):
                defense.record_bypass()
                return None, None, 0.0, False
            result = orig(question, query_confidence_threshold, *args, **kwargs)
            responded = result[0] is not None
            defense.record_peer_interaction(peer_id, responded)
            return result

        return _wrapped_query

    def remove(self, network) -> None:
        for pid, original_fn in self._original_query_fns.items():
            peer = network.peers[pid]
            if peer is not None:
                peer.query = original_fn
        self._original_query_fns.clear()
        if getattr(network, "_sfa_defense", None) is self:
            network._sfa_defense = None

    def reset(self) -> None:
        self._query_count.clear()
        self._response_count.clear()
        self._reputation.clear()
        self.blacklisted_peers.clear()
        self._streak.clear()
        self._binom_obs.clear()
        self.suspicion_level.clear()
        self._total_bypasses = 0
        self._total_blacklistings = 0
        self._total_validations = 0
        self._blocked_answers = 0
        self._passed_answers = 0
        self._confidence_sum = 0.0
        self._redundant_probes = 0
        self._redundant_hits = 0

    # ── validation: cross-peer answer voting, source-blacklist rejection ──
    def validate(self, question: str, candidate_answer, peer_responses: List[Dict[str, Any]]):
        self._total_validations += 1
        for resp in peer_responses:
            pid = resp.get("peer_id", -1)
            if pid >= 0:
                self.record_peer_interaction(pid, responded=bool(resp.get("answer")))

        candidate_source = next(
            (r.get("peer_id") for r in peer_responses if r.get("answer") == candidate_answer), None
        )
        if candidate_source is not None and self.is_peer_blacklisted(candidate_source):
            self._blocked_answers += 1
            return False, 0.05, {"reason": f"Source peer {candidate_source} is blacklisted"}

        trusted_responses = [
            r for r in peer_responses
            if r.get("answer") and not self.is_peer_suspicious(r.get("peer_id", -1))
        ]
        if len(trusted_responses) < self.min_peers_for_validation:
            self._passed_answers += 1
            self._confidence_sum += 0.5
            return True, 0.5, {"validation_skipped": True}

        trusted_answers = [r["answer"] for r in trusted_responses]
        agreement = trusted_answers.count(candidate_answer) / len(trusted_answers)
        confidence = min(1.0, agreement + 0.1 * min(len(trusted_responses), 5) / 5)
        is_valid = agreement >= 0.5
        if is_valid:
            self._passed_answers += 1
        else:
            self._blocked_answers += 1
        self._confidence_sum += confidence
        return is_valid, confidence, {"agreement": agreement}

    # ── stats ────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        suspicious = [
            p for p in self._reputation
            if self.is_peer_suspicious(p) and p not in self.blacklisted_peers
        ]
        return {
            "detection_mode": self.detection_mode,
            "blacklisted_peers": sorted(self.blacklisted_peers),
            "blacklisted_count": len(self.blacklisted_peers),
            "suspicious_count": len(suspicious),
            "total_peer_calls_tracked": sum(self._query_count.values()),
            "total_bypasses": self._total_bypasses,
            "total_blacklistings": self._total_blacklistings,
            "redundant_probes": self._redundant_probes,
            "redundant_probe_hits": self._redundant_hits,
            "peer_reputations": {pid: round(rep, 4) for pid, rep in self._reputation.items()},
            "peer_suspicion_levels": dict(self.suspicion_level) if self.detection_mode == "binomial" else {},
            "total_validations": self._total_validations,
            "blocked_answers": self._blocked_answers,
            "passed_answers": self._passed_answers,
            "block_rate": (self._blocked_answers / self._total_validations) if self._total_validations else 0.0,
            "avg_confidence": (self._confidence_sum / self._total_validations) if self._total_validations else 0.0,
            # Transparency for the binomial-mode calibration fix (see
            # _effective_honest_miss_rate()): "pinned" means honest_miss_rate
            # was explicitly set to a float; "auto" means it's measured live
            # from the current population every call. honest_miss_rate_auto_estimate
            # is a population-wide snapshot for reporting only (no peer
            # excluded) -- the actual per-test value excludes the tested peer.
            "honest_miss_rate_mode": "pinned" if self.honest_miss_rate is not None else "auto",
            "honest_miss_rate_pinned": self.honest_miss_rate,
            "honest_miss_rate_auto_estimate": (
                self._effective_honest_miss_rate(-1)
                if self.honest_miss_rate is None and self.detection_mode == "binomial" else None
            ),
        }
