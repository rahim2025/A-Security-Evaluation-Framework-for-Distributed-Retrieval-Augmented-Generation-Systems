# SSM-Score Defense

Everything for the SSM-Score attack — both mechanisms, both defenses —
lives in this one folder, `defense/ssm_defense/`. (It used to be split
across two folders — `ssm_defense/` and `ssm_grounding_defense/` — which
was confusing; the latter has been merged in here and no longer exists.)

Every number below is from a real run against the actual code — the live
Docker stack for grounding-farming, the real deployed Hardhat contract for
key-forgery — not a simulation or an estimate.

> **Scope note.** The project moved beyond attack-only evaluation for SSM:
> the already-existing key-forgery defense is hardened against a known
> gap, and the grounding-farming attack — which had **no defense at all**
> before this work — gets a new one. Both now live in
> `defense/ssm_defense/`.

---

## 1. Two attacks, one name

This repo's SSM-Score attack has two genuinely distinct instantiations,
documented in `.claude/ssm_grounding_farming_plan.md`:

| | Key-forgery (`ssm_score_attack.py`) | Grounding-farming (`grounding_farming_attack.py`) |
|---|---|---|
| Status before this work | Defended (`defense/ssm_defense/`, pre-existing) | **Undefended** |
| Privilege required | The LLM orchestrator's own private key (compromised) | None beyond the source's own — it already has this |
| Mechanism | Forges and submits a score-update transaction directly | Returns broad, generic content via the normal `/poison` instrumentation; the *real* orchestrator computes and submits an honest, correctly-signed transaction based on it |
| What's false | The transaction itself (forged) | The *premise* the transaction is based on (the source didn't actually ground the answer) |
| Thesis framing | Secondary / control-plane finding | **Primary SSM result** — the no-privileged-access, DRAG-parity instantiation |
| Defended by | `drag_contract/contracts/drag_scores.sol` + `defense/ssm_defense/` | `defense/ssm_defense/` |

---

## 2. Key-forgery: closing the patient-throttling gap

The existing defense (`drag_contract/contracts/drag_scores.sol`) already
bounded a single transaction's burst: a caller allow-list, a 2-second
cooldown, a 5,000-point per-transaction delta cap, and absolute score
bounds. The team's own plan doc found the gap directly: **a patient
attacker who throttles each step to just under those caps defeats them
completely** — confirmed with a live run at the time (`AMPLIFY=4000`, 2.5s
between rounds): "3/3 seeds fully succeeded... R/U +20,000."

**Fix:** a cumulative-drift cap. Each source now anchors a rolling epoch
(`EPOCH_DURATION` = 1 hour) with the score values at the anchor; every
update is also checked against **net** movement from that anchor, not just
the single-step delta. `MAX_CUMULATIVE_DELTA_PER_EPOCH` is set to
16,000 — deliberately not a round 20,000, so the attack's own
previously-fully-successful default campaign is now genuinely interrupted
rather than landing exactly at a boundary with nothing visibly blocked.

**Measured result — live, on the actual deployed Hardhat contract** (run
2026-10-05, same `AMPLIFY=4000`/`ROUNDS=5`/`2.5s` config the attack already
uses by default):

```text
Round 1/5: accepted  ->  R=14,000  U=14,000
Round 2/5: accepted  ->  R=18,000  U=18,000
Round 3/5: accepted  ->  R=22,000  U=22,000
Round 4/5: accepted  ->  R=26,000  U=26,000
Round 5/5: REVERTED  ->  CumulativeDriftTooLarge("sources_100", 20000, 16000)
```

**4/5 rounds accepted, final gain capped at +16,000 instead of the
previously-documented +20,000.** Also verified with a dedicated Hardhat
test suite (`drag_contract/test/DragScores.cumulativeDrift.test.js`, 3
tests, all passing): the block happens on exactly the right round, the cap
resets once the epoch rolls over, and an ordinary small legitimate update
is unaffected.

The off-chain monitor (`defense/ssm_defense/ssm_score_defense.py`) was
extended to replay this same rolling-epoch logic against historical
`ScoreRecordUpdated` events, so cumulative-drift violations are detectable
without reading Solidity revert reasons too.

---

## 3. Grounding-farming: a defense built from scratch

### What the attack does

`drag_llm_service/app/server.py` decides whether a source gets positive
on-chain credit using a naive literal-substring check: does the model's
answer text appear somewhere inside this source's retrieved chunk? Not
whether that chunk had anything to do with *producing* the answer.
`grounding_farming_attack.py` exploits this by injecting broad,
topically-aligned filler content (via the same `/poison` endpoint Data
Poisoning uses) engineered to coincidentally contain many plausible short
answers. Every resulting score-update transaction is submitted by the real
orchestrator, correctly authorized, individually within the existing
on-chain bounds — none of §2's defenses can see this, because nothing
about it is a forged transaction. The forgery is semantic, not
cryptographic.

### Attempt 1 (negative result, kept for the record): gate on retrieval relevance

**Hypothesis:** broad, generic filler should score as a weaker, more
marginal retrieval hit than genuinely on-topic content, so require the
matched context to also clear a minimum `rerank_score`/`consensus_score`
(both already computed by `select_top_k_with_defense` / `Reranker.rerank`
— real signals, no extra inference cost) before honoring a substring match
as real grounding.

**Result, measured against the real pipeline and real attack content**
(`offline_validation.py`, 200 real SQuAD questions, realistic per-source
top-`n_contexts` funneling): **disproved.** Filler that survives retrieval
and trips the naive grounding check scores, if anything, *higher* on
`consensus_score` than genuine content (median 1.00 vs 0.67) and
indistinguishably on `rerank_score` (0.61–0.82, squarely inside genuine
content's own 0.47–0.93 range). Gating on these signals would have cost
~19% of genuinely grounded sources their deserved credit while blocking
approximately 0% of the actual attack — worse than no defense at all.

This code (`grounding_verifier.py`, wired into
`drag_llm_service/app/server.py`) is **left in place but disabled by
default** (`retrieval.grounding_defense.enabled: false` in
`drag_llm_service/configs/config.yaml`) — a documented negative result, not
hidden.

### Attempt 2 (effective, enabled by default): ingestion-time content-shape detection — two complementary signals

**Insight:** regardless of how filler scores at retrieval time, *what the
filler text is shaped like* is distinctive. `grounding_farming_attack.py`'s
templates stitch together several short, independent factual claims, each
introduced by a vague, non-committal attribution phrase ("sometimes cited",
"occasionally reference", "in various contexts", "has been associated
with") — because the attacker doesn't know in advance which specific fact a
given query will need, so the content hedges across many unrelated facts
instead of committing to one coherent claim the way genuine prose does.

**Signal A — `content_shape_detector.py`'s `GroundingFarmDetector`** flags a
document once at least `min_hedge_sentences` (default 2) of its sentences
contain this kind of hedged attribution construction.

**Signal A alone is evadable by rewording.** An attacker who swaps the
connective phrases for plain, neutral language (e.g. `"{t} is connected to
{f}."` instead of `"Related information about {t} sometimes cited in this
context includes {f}."`) keeps the exact same exploit — same topic anchor,
same disconnected-fact-per-sentence structure — but trips zero hedge
patterns. Measured: **0/40 (0.0%) detection** against such a reworded
("adaptive") filler set (`content_shape_validation.py`). A real gap, not a
hypothetical one.

**Signal B — `sentence_length_detector.py`'s `SentenceLengthDetector`**
closes that gap by targeting what rewording *cannot* change: the attack's
own stated design goal (`.claude/ssm_grounding_farming_plan.md` §4 — "span
many plausible short-answer entities/phrasings") forces short, simple
`"{topic} {verb} {fact}."` sentences almost by construction, regardless of
which verb or connective is used — there's no room in that structure for
the elaboration and subordinate clauses real encyclopedic prose uses. Flags
a document when its average sentence length (words) is at or below
`max_avg_words` (default 12).

Both are wired into `drag_data_source/app/server.py`'s `/poison` handler —
the same endpoint `PoisonFilter` (the data-poisoning defense) already
gates — as two independent checks; either firing rejects the document.

---

## 4. Measured results

### Offline (`content_shape_validation.py`, real attack-generated filler, real clean corpus documents, 2026-10-05)

| | Signal A (hedge) | Signal B (sentence length) | Combined (OR) |
|---|---|---|---|
| Original filler (as shipped), n=40 | 100.0% | 45.0% | **100.0%** |
| Adaptive filler (reworded), n=40 | 0.0% | 95.0% | **95.0%** |
| Real clean docs (false positives), n=150 | 0.0% | 0.7% | **0.7%** |

### Live — same filler sets, injected through the real `/poison` endpoint against the actual deployed Flask server (not a simulation), 2026-10-05

| Run | Injected | Rejected | Rejected by |
|---|---|---|---|
| Original filler | 0/40 | **40/40 (100%)** | `grounding_farm_shape` (Signal A) |
| Adaptive filler | 2/40 | **38/40 (95%)** | `grounding_farm_short_sentences` (Signal B) — Signal A caught 0, exactly as the offline prediction said it would |

The live numbers match the offline prediction almost exactly, including
the adaptive-attacker case being caught entirely by the second signal after
the first one is fully evaded — direct, real evidence that the two signals
are genuinely complementary rather than redundant. The same methodology
(real deployed system, real attacker code) was used for the key-forgery
result in §2.

---

## 5. Running

```bash
# --- Grounding-farming (defense/ssm_defense/) ---

# Offline (no Docker/GPU needed) -- content-shape numbers above, reproduced
# from a fresh run (original filler, adaptive filler, clean false-positive sweep):
python defense/ssm_defense/evaluation/content_shape_validation.py

# Offline -- the abandoned relevance-gating attempt's numbers:
python defense/ssm_defense/evaluation/offline_validation.py

# Live, requires the Docker stack up (data-source images already copy
# defense/ -- see drag_data_source/Dockerfile):
curl http://localhost:8003/grounding_farm_defense/stats \
  -H "X-API-Key: reliable-derag-secret-2026"

# --- Key-forgery (defense/ssm_defense/) ---

# should now fail on round 5 instead of succeeding 5/5
python attack/ssm_score/run_attack.py

# full before/after + on-chain anomaly scan, logs to defense_logs/
python defense/ssm_defense/evaluation/run_defense.py
```

Toggle off to reproduce the undefended baseline: set
`GROUNDING_FARM_DEFENSE_ENABLED=0` on the data-source containers (same
pattern as `POISON_FILTER_ENABLED`, see
`defense/datapoisoning_defense/README.md`).

---

## 6. Files changed/added

| File | Change |
|---|---|
| `drag_contract/contracts/drag_scores.sol` | **Modified.** New `epochStart`/`epochStartReliabilityScore`/`epochStartUsefulnessScore` fields; `CumulativeDriftTooLarge` check. |
| `drag_contract/test/DragScores.cumulativeDrift.test.js` | **New.** 3 tests: block on the right round, epoch rollover, legitimate update unaffected. |
| `defense/ssm_defense/ssm_score_defense.py` | **Modified.** Off-chain monitor replays the rolling-epoch logic; new `cumulative_drift_violation` anomaly kind. |
| `defense/ssm_defense/README.md` | **Rewritten.** This file — now covers both attacks (previously key-forgery only). |
| `defense/ssm_defense/grounding_verifier.py` | **New.** Attempt 1 (disabled — negative result). |
| `defense/ssm_defense/content_shape_detector.py` | **New.** Signal A. |
| `defense/ssm_defense/sentence_length_detector.py` | **New.** Signal B. |
| `defense/ssm_defense/evaluation/offline_validation.py` + `content_shape_validation.py` | **New.** Real-data validation scripts. |
| `drag_llm_service/app/server.py`, `config.yaml`, `Dockerfile` | **Modified.** Grounding-verification hook (disabled by default); now copies `defense/` into the image. |
| `drag_data_source/app/server.py` | **Modified.** `/poison` now runs both Signal A and B; new `/grounding_farm_defense/stats` endpoint; `/reset` clears both. |
| `docker-compose.yml` | **Modified.** Infrastructure fixes (HF cache mount, xet-transfer disable) needed to get the live run working. |

---

## 7. Honesty notes / honest limitations

- Thresholds (`min_hedge_sentences=2`, `max_avg_words=12`,
  `MAX_CUMULATIVE_DELTA_PER_EPOCH=16,000`, `EPOCH_DURATION=1h`) are
  reasonable defaults calibrated against this corpus's real documents and
  the actual attacks' real parameters (including a simulated adaptive
  rewording for grounding-farming), not exhaustively tuned against a
  held-out split or a real legitimate-traffic baseline.
- **Defeating both grounding-farming signals at once is harder, not
  impossible.** An attacker would need to write longer, more elaborated
  sentences while still cramming a disconnected fact into each one — this
  raises the sophistication/cost bar (and works against the attack's own
  "many documents, cheaply generated" economics) rather than guaranteeing
  prevention against every conceivable rewording.
- **Two negative results are kept on record, not hidden**: the
  relevance-gating attempt (Attempt 1) and an intermediate coherence-based
  signal (average adjacent-sentence word overlap) that looked promising in
  theory but had a 78.7% false-positive rate in practice — real prose also
  has low raw word overlap between sentences (pronouns, synonyms,
  ellipsis), so it wasn't a reliable discriminator.
- The false-positive rate is low and barely rises from Signal B alone
  (0.7%) to the combination with Signal A (also 0.7%, since Signal A alone
  had 0 false positives in this sample) — closing the rewording gap cost
  essentially nothing here. An earlier revision of this doc reported
  0.7%/2.7%/3.3%; that number came from a validation-script bug (sampling
  from a Python `set`, whose iteration order isn't stable across process
  runs, so the "fixed seed" wasn't actually reproducible) — fixed by
  sorting before sampling, and the 0.0%/0.7%/0.7% result above is now
  confirmed identical across repeated runs.
- A very high-volume, consistently excellent honest source could
  theoretically approach the cumulative-drift cap over a busy hour — worth
  measuring against real query volume before treating that threshold as
  production-ready.
