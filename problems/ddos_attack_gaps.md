# DDoS Attack (`attack/ddos_sim/`) — Known Gaps & Limitations

Tracking doc for issues found during a review of the congestion-based DDoS
simulation (`DDoSAttack`) and the live HTTP flood (`TrafficFlood` +
`run_live_evaluation.py`), reviewed against `reports/ddos_attack.md`. Scope
of the review: modeling assumptions, "God Mode" / privileged-access checks,
and anything that would not survive a thesis committee's scrutiny.

**God Mode check: passed.** No privileged-access cheating was found —
`DDoSDefense` observes attack-induced drops as ordinary failed responses
(the wrapper ordering in `run_defense.py` puts the defense *outside* the
attack's monkey-patch, so it never reads `state.intensity`/`drop_probability`
directly), the live flood is genuine concurrent HTTP load, and error/rate-limit
accounting is reported honestly rather than swallowed. The issues below are
methodological, not fabricated results.

---

## 1. Zero multi-seed evidence (Known, deferred — not yet fixed)

Every attack/defense log actually on disk for this module is `seed42`:

- `attack_logs/ddos_sim/attack_*_ddos_sim_{mock,live}_seed42.*` — no seed0 or seed123 run exists.
- `defense_logs/ddos_sim_defense/defense_*_ddos_sim_{mock,live}_seed42.*` — same.

This directly contradicts CLAUDE.md's rule: *"Do not report single-seed
results as final. Run multiple seeds before writing up any number."*

**Status / why:** the team prioritized getting the remaining five attacks
implemented first and has not yet had time to run the seed sweep for DDoS.
This is not an oversight in the code — `run_attack.py --seed` and
`run_defense.py --seed` already support it — it's just not done yet.

**Action:** re-run the mock sweep at seeds `0, 42, 123` (fast, no Docker
required) and report mean ± variance before the numbers in
`reports/ddos_attack.md` §8.1 are treated as final for the thesis.

---

## 2. Live severity tiers are confounded by run order

In `run_live_evaluation.py`, `low` → `mid` → `high` run sequentially in one
script execution, and `source_0` is flooded in *every* tier (`targets =
list(DEFAULT_SOURCE_URLS.keys())[:tier["num_sources"]]`). There is no
cooldown between tiers, and the Flask-Limiter 60-req/min bucket may not
fully reset between them. The single `baseline` run is also reused across
all three severities rather than re-measured per tier.

**Consequence:** the "high" tier's 85% failure rate is measured on sources
that were already flooded twice in immediate succession beforehand, so the
result can't be cleanly attributed to "high severity" alone vs. cumulative
exhaustion carried over from `low`/`mid`. Combined with §1 (single run, no
repeats), the headline collapse (`semantic_similarity` 0.832→0.137) has no
error bars and one un-isolated confound.

**Action:** either randomize/independently reset tier order (fresh
containers or an explicit cooldown wait ≥ 60s between tiers), or explicitly
document the confound and soften the "clean dose-response curve" framing in
§8.2/§12 of the report.

---

## 3. Mock-mode "catastrophic collapse" is largely a hyperparameter artifact

`intensity_min=0.5, intensity_max=1.0` with `drop_probability =
min(0.95, intensity*0.8)` means a directly-targeted peer's mean sampled
intensity (0.75) sits well above the ~0.625 intensity needed to cross the
`DOWN_THRESHOLD=0.5`. The 100%→10% collapse shown in report §8.1 is close
to guaranteed by these defaults, not an emergent empirical discovery.

**Consequence:** this is fine as a mechanism-validation sanity check, but
if presented as independent proof of "attack effectiveness," a reviewer can
reasonably respond "of course it collapsed, look at your own formula."

**Action:** frame mock-mode results explicitly as mechanism validation, not
primary evidence; lean on the live-flood result as the actual evidence for
the thesis's generalizability claim.

---

## 4. Residual bug in the "fixed" role-token stripping

`drag_llm_service/src/models/open_model.py:59` still runs a blind substring
replace before the new regex fix:

```python
result = result.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").replace("assistant", "")
result = _strip_leaked_role_tokens(result)
```

`.replace("assistant", "")` is not word-boundary-aware — it strips that
substring anywhere in the text, not just a leaked leading role token. The
report frames the leaked-role-token issue as fixed via
`_strip_leaked_role_tokens()`, but the original blind replace still runs
first and is still capable of corrupting legitimate answer text containing
that substring (e.g. "physician assistant").

**Consequence:** low practical risk for this specific PubMedQA yes/no/maybe
task (answers are single words), but it's a real correctness bug, and the
"fixed" framing in the report is only partially true.

**Action:** either replace the blind `.replace("assistant", "")` with a
word-boundary-safe regex, or explicitly scope the limitation in the report
to "safe only because this dataset's gold answers are single tokens."

---

## 5. Minor — worth a caveat, not a fix

- **`flood_ramp_s=1.0`** (delay between starting the flood and firing real
  evaluation traffic in `run_live_evaluation.py`) is asserted, not
  empirically validated, to be sufficient time for `workers_per_source`
  threads to actually saturate the target before the timed evaluation
  queries begin.
- **Shared RNG stream across mock peers**: `MockRAGNetwork` passes a single
  `random.Random` instance to every `MockPeer` (inherited from
  `selective_forward_sim`). Because the DDoS wrapper drops some queries
  before they reach a peer's `.query()`, the number of RNG draws consumed
  differs between attack and baseline runs, shifting downstream peers'
  random outcomes in a path-dependent way. This adds noise, not bias — it's
  already visible in the report's own observation that `hit_rate` doesn't
  decay monotonically with `availability_percentage` — but is worth a
  one-line acknowledgment rather than silence.

---

## 6. Untested interaction with reliability-weighted reranking (low realistic impact, but unverified)

`drag_llm_service/configs/config.yaml` has `rerank_with_reliability: true`
and `reliability_weight: 0.5` **on by default** — every plain `/query` call
fetches on-chain R_i scores and blends them into which retrieved passages
make it into the LLM's prompt (`server.py:581-596`). `reports/ddos_attack.md`
never mentions this toggle or tests with it off.

**Mechanistic assessment (reasoned, not yet empirically verified):** the
wave-based `hit_rate` metric (via `LiveRAGNetwork.topic_aware_query()`)
bypasses `drag_llm_service` entirely — it queries each `drag_data_source`
directly, so that headline metric is **not exposed** to this at all. Only
`run_live_evaluation.py`'s real generation-quality metrics (the
`semantic_similarity 0.832→0.137` collapse) go through `/query` and are
technically subject to it. Mechanistically this is **unlikely to matter
much**: DDoS's damage comes from a flooded source timing out and
contributing *zero* candidates — reranking can only reprioritize among
candidates that were actually returned, it can't compensate for a source
that returned nothing. Toggling `rerank_with_reliability` off would
plausibly leave the reported collapse largely unchanged.

**What would settle this for certain:** whether the three sources'
on-chain R_i values currently differ meaningfully, or sit near a uniform
initialized baseline (in which case the reliability term is close to a
constant added to every candidate and can't reorder anything). Confirmed
that no plain `/query` traffic (which is all four attack modules use)
ever writes a score update — only `/query_analyze` does, and the only
caller of that in the repo is `drag_llm_service/test_service.py`'s
standalone manual smoke test, not wired into any attack. So scores are
plausibly still near baseline, but this was **not checked live** — Docker/
Hardhat was not running at review time and the check was deferred by
request rather than performed.

**Action:** low priority given the mechanistic reasoning above, but if a
committee member asks "did you control for the reliability-reranking
feature," the honest answer right now is "reasoned to be low-impact for
this specific attack, not empirically confirmed." A single read of current
on-chain scores (`get_scores_from_blockchain`/`DragScoresClient`, next time
the stack is up) would close this out cheaply.

---

## Summary table

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | No multi-seed runs (seed42 only) | High | Deferred — known, will run after remaining attacks are implemented |
| 2 | Live severity tiers confounded by run order / shared baseline | Medium-High | Open |
| 3 | Mock-mode collapse is a hyperparameter artifact, not emergent | Medium | Open — needs reframing in report, not a code fix |
| 4 | Blind `.replace("assistant", "")` still live in `open_model.py` | Low-Medium | Open |
| 5 | `flood_ramp_s` unvalidated; shared-RNG path dependence | Low | Open — caveat only |
| 6 | Untested interaction with `rerank_with_reliability` (reasoned low-impact, not empirically confirmed) | Low | Open — cheap to verify, not yet checked (Docker was down at review time) |
