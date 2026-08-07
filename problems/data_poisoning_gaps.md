# Data Poisoning Attack — Gap Analysis

Audit of `attack/datapoisoning/` against the Reliable-dRAG paper (`Reliable-dRAG.md`) and our own
framework goals (`.agents/CLAUDE.md`, `thesis_update.md`). Covers implementation bugs, unrealistic
assumptions ("God Mode" access), and a deeper conceptual gap in how the attack is framed relative to
what the system under test actually claims to defend against.

Every finding below is anchored to a file/line and, where possible, to empirical evidence from
`attack_logs/`. Severity reflects how much it would undermine a thesis defense if left unaddressed.

---

## Part A — Conceptual gap: what are we actually testing?

This is the most important gap and the reason Part B's bugs matter more than they'd otherwise look.

### A1. The "clean baseline" is not clean — it's the paper's own pollution benchmark

The three live data-source containers this attack targets are not a neutral clean deployment. Per
`docker-compose.yml` and `drag_data_source/configs/*.yaml`:

| Source | Port | Backing file | Pollution level (per the paper's own definition) |
|---|---|---|---|
| `sources_0` | 8001 | `data/polluted_token/sources_0.jsonl` | 0% (clean) |
| `sources_20` | 8002 | `data/polluted_token/sources_20.jsonl` | 20% ground-truth tokens replaced |
| `sources_100` | 8003 | `data/polluted_token/sources_100.jsonl` | 100% ground-truth tokens replaced |

These are literally the corpora from the paper's Figure 4 / Table 1 token-level-pollution
environment — the exact benchmark the paper uses to *demonstrate* the reliability mechanism working.

**Consequence:** the "clean_accuracy" reported by `run_attack.py` (e.g. 42.9% in
`theory/DATA_POISONING_DEFENSE.md`) is not a clean-system number. It's the paper's own
already-degraded, unreliable-data-environment number — the exact condition Table 1 shows needs
~500 warmup queries for dRAG to recover from. Any additional degradation we inject on top of this is
confounded with pollution that was already there before our attack script ran.

**Sharper problem:** the `targeted` strategy's example run poisons `sources_100`, which is already
100%-token-polluted by construction. There is very little genuine "additional" damage to attribute to
our attack there — that source was already supposed to be near-worthless to a functioning reliability
mechanism.

**What this requires to fix:** either (a) stand up a genuinely clean 3-source environment (all three
at 0% pollution) as the baseline, poison it ourselves, and measure from there, or (b) keep the current
mixed-pollution deployment but explicitly control for and report the pre-existing pollution level of
each source as a separate variable, so "damage caused by our poisoning" and "damage that was already
baked in" aren't reported as one undifferentiated number.

### A2. The reliability mechanism the paper claims defends against this never actually runs

Traced through `drag_llm_service/app/server.py`:

- Reliability/usefulness scores only update inside `_query_analyze()`
  ([drag_llm_service/app/server.py:900](../drag_llm_service/app/server.py#L900)), gated behind an
  `update_scores=True` parameter that belongs to the **`/query_analyze`** endpoint
  ([server.py:1037](../drag_llm_service/app/server.py#L1037)).
- The attack's evaluation harness (`run_attack.py` → `query_llm()` →
  [run_attack.py:79](../attack/datapoisoning/run_attack.py#L79)) only ever calls **`/query`**, which
  never touches that code path.
- The on-chain contract constructor (`drag_contract/contracts/drag_scores.sol:40`) initializes no
  scores — everything defaults to 0 and stays there.

Result: across the entire attack evaluation, all three of the paper's adaptive mechanisms are
inert:

1. **Reranking's reliability term** (`rerank_with_reliability`, Eq. 4 in the paper, weighted by α) has
   nothing to differentiate on — every source ties at 0.
2. **Retrieval-stage usefulness sampling** (Algorithm 1, step 5: "sample N sources proportionally to
   U_i") isn't exercised at all in this 3-source deployment — `/query` queries all three sources
   unconditionally regardless of U_i.
3. **The scoring feedback loop itself** never fires, so there is no convergence process to speak of —
   not "not yet converged," but structurally incapable of converging given how the harness calls the
   system.

**Why this matters more than a bug report:** it changes what the experiment is actually measuring.
We are not testing "can adversarial poisoning defeat Reliable-dRAG's reliability defense." We are
testing "can poisoning defeat a RAG system with its distinguishing feature switched off." Those are
different claims, and only the first one is thesis-relevant to a system whose entire contribution is
the reliability mechanism.

**Empirical confirmation this is live, not theoretical:** see Part B1 below — the `high_reliability`
strategy's source selection is directly downstream of this dead code path, and the attack_logs prove
it degenerates to a fixed choice.

### A3. What a properly justified version of this attack looks like

Two framings survive contact with the paper and are worth pursuing instead of the current generic
"inject bad docs, measure accuracy drop":

**Framing 1 — Adversarial poisoning vs. the paper's random-noise threat model.**
The paper only ever validates against *uniform random token replacement* (Figure 4). It never tests
content deliberately engineered to win retrieval and reranking. That's a real, citable gap:
*"dRAG's reranker was validated against random noise; we test whether it survives content crafted to
look maximally relevant, injected by a source that has already earned trust."* This is a strictly
stronger and more novel claim than re-running the paper's own pollution methodology — but it requires
Part A2 to be fixed first, because "a source that has already earned trust" is meaningless if R_i
never moves.

**Framing 2 — Cold-start / pre-convergence vulnerability window.**
The paper's own future-work section admits this is open: *"it remains unclear how many queries are
required for dRAG to reach its performance upper bound... we leave this exploration to future
work"* (Reliable-dRAG.md, §7). Poisoning a source immediately after it joins — before enough
legitimate queries have accumulated to build a trust differential — is a gap the authors themselves
flag as uncharacterized. This is arguably the cleanest, most directly extending angle available, and
it turns the "reliability scores never update" bug (A2) into an intentional experimental condition
("we test the pre-convergence window") rather than an oversight, *provided* we also run a
post-convergence condition for contrast.

Either framing needs the same underlying fixes as Part B: a real score-update path, a controllable
clean baseline, and no test-set leakage (below). Without at least one of these framings stated
explicitly, the current results read as "we ran the paper's own stress test without the paper's own
mitigation switched on," which is not a defensible attack claim against this specific system.

---

## Part B — Implementation-level findings

### B1. [RESOLVED — strategy removed] `high_reliability` strategy silently degrades to "always poison sources_0"

**Resolution:** rather than fix this via A2 (which would require standing up a working
blockchain reliability feedback loop — a substantial separate effort, since the LLM service
currently can't even reach the deployed contract), the `high_reliability` strategy was
removed and replaced with `data_rich`: target the source(s) holding the most documents
(via each source's live `/info` doc count), a real, observable signal rather than a broken
one. See `attack/datapoisoning/data_poisoning_attack.py`'s `_select_sources_to_poison()`
and `_get_doc_counts()`. All references to `high_reliability` below are kept as historical
record of the bug, not a description of current behavior.

This is the flagship strategy — the one meant to test "poisoning the most-trusted source causes
maximum damage," which is the data-poisoning analogue of this system's most distinctive feature
(blockchain reliability scoring). It doesn't work as described, and it's directly caused by A2.

- `_select_sources_to_poison()` → `_get_blockchain_scores()` reads `GET /score_events`
  ([attack/datapoisoning/data_poisoning_attack.py:302](../attack/datapoisoning/data_poisoning_attack.py#L302)).
- With no events ever emitted (A2), every source ties at reliability `0.0`.
- Python's `sorted()` is stable, so `sorted_sources[:num_malicious]` just returns the original list
  order — i.e. `sources_0` first, always.

**Empirical proof from `attack_logs/`:**

| Strategy | Sources picked across all logged runs |
|---|---|
| `random` (11 runs) | sources_0, sources_20, sources_100 — well distributed |
| `high_reliability` (5 runs at ratio 0.5) | **sources_0, every single time** |

The two `high_reliability` runs that picked all three sources used `--ratio 1.0` (poison everything),
which is a different, non-differentiating configuration and doesn't contradict the finding.

**Impact:** the 66.7% degradation number attributed to "high_reliability / wrong_answer" in
`theory/DATA_POISONING_DEFENSE.md`'s comparison table is really just "what happens when you poison
sources_0" — indistinguishable from a hardcoded single-source strategy. This is the single most likely
thing a committee member would find if they traced the code, since it directly undercuts the
strategy's name and stated purpose.

**Fix:** before running `high_reliability`, either (a) run a warm-up phase that calls
`/query_analyze` with `update_scores=true` enough times to produce genuine R_i differentiation
(mirroring the paper's own convergence methodology), or (b) seed the contract directly with
distinguishable scores per source. Either way, **log the actual reliability values that drove
selection** in the attack log — right now there is no artifact proving "highest reliability" ever
meant anything other than "first in a Python list."

### B2. [RESOLVED] Query-aware poisoning embeds the literal eval-set questions — test-set leakage

`run_attack.py` hardcoded `target_queries=[item["question"] for item in EVAL_DATA]`
([run_attack.py:172](../attack/datapoisoning/run_attack.py#L172)) and passed it regardless of
`--strategy`. Inside `execute()`, the `if self.target_queries:` block
([data_poisoning_attack.py:161](../attack/datapoisoning/data_poisoning_attack.py#L161)) fired
unconditionally, and `_create_targeted_poison_doc` embeds the **verbatim question text** into every
crafted poison document, guaranteeing near-perfect BM25 + embedding similarity for exactly the
questions later used to measure accuracy.

- Same failure mode CLAUDE.md explicitly warns against for KB Extraction/MIA ("do not use exact
  stored text as probe queries... produced trivially inflated results") — just not caught here.
- It was a confound across the *entire* attack matrix: because this fired for every
  strategy/poison_type combination, comparisons between "random vs. targeted vs. data_rich" or
  "wrong_answer vs. misleading vs. noise" all sat on top of the same dominant, oracle-knowledge
  poisoning signal.

**Resolution:** both `run_attack.py` and `run_all_attacks.py` now accept `--no-query-aware`, which
passes `target_queries=[]` for a genuine black-box run instead. Default behavior (oracle-knowledge,
flag omitted) is unchanged, but every saved log now records `"query_aware": true/false` so old and
new results are distinguishable.

**Controlled comparison** (`theory/DATA_POISONING_DEFENSE.md` §11) — same attack
(`targeted(sources_0,sources_20)/misleading`), same seed (42), same live deployment, only the flag
toggled:

| Mode | Degradation | Result |
|---|---|---|
| Oracle-knowledge (default) | 66.7% | ✅ Success |
| Black-box (`--no-query-aware`) | −50.0% | ❌ Backfired |

This is the clearest single finding across all the data-poisoning work: **most of this attack's
power came from the attacker secretly knowing the exact eval questions, not from the poisoning
mechanism itself.** Every prior result in `theory/DATA_POISONING_DEFENSE.md` (§7-10) should be read
as measuring the oracle-knowledge threat tier specifically (a legitimate scenario — "attacker has
query-log access" — but a strong one), not as a general "poisoning works" claim.

**Full multi-seed black-box sweep now run** (`theory/DATA_POISONING_DEFENSE.md` §12,
`attack_logs/attack_matrix_summary_seeds_2026-07-18_06-26-57.json`) — and it complicates the
single-comparison hypothesis above rather than simply confirming it. It's **not** "black-box
poisoning is mostly ineffective" — it's a near-total inversion:

| Attack | Success rate, oracle-knowledge | Success rate, black-box |
|---|---|---|
| targeted(sources_0,sources_20) / misleading | 3/3 | **0/3** |
| random / noise | 0/3 | **2/3** |
| random / answer_swap | 0/3 | **1/3** |
| data_rich / noise | 0/3 | **2/3** |
| targeted(sources_100) / wrong_answer | 0/3 | 0/3 |
| data_rich / wrong_answer | 0/3 | 0/3 |

The content-crafting strategies (`targeted`/`data_rich` `wrong_answer`/`misleading`) need query
knowledge to be effective at all and collapse to 0/3 without it — intuitive, since their poison text
has no connection to whatever question actually gets asked once it can't embed the question verbatim.
But the plain corpus-noise strategies (`random/noise`, `random/answer_swap`, `data_rich/noise`) that
were completely inert *with* query knowledge become the *only* combos with any success in black-box
mode. No confirmed mechanism for this yet — two untested hypotheses (candidate-slot competition
between the extra oracle-knowledge poison docs and the noise-poisoned ones; eval-set-relative chance
in which random-sampled documents land near the 7 eval questions) are noted in §12, but neither is
verified. Would need per-question retrieval inspection (available in each log's `per_question` field)
to pin down, not done here.

**Bottom line for the thesis:** don't reduce this to a single "black-box is weaker/stronger" claim —
report it as attack-style-dependent, with the two side-by-side tables in §12.

### B3. [MEDIUM] Blockchain reliability state is never reset between runs

`reset_all()` only calls each data source's `/reset`
([data_poisoning_attack.py:189](../attack/datapoisoning/data_poisoning_attack.py#L189)), which rebuilds
the FAISS index — it never touches on-chain scores. Currently masked by A2/B1 (scores never update at
all), but the moment B1 is fixed, this becomes live: `run_all_attacks.py` runs six attack configs
back-to-back with no chain reset, so later matrix entries would inherit reliability state polluted by
earlier attacks' outcomes within the same session. B1 and B3 need to be fixed together, or one invalid
result gets traded for another.

### B4. [RESOLVED] No random seeding anywhere in the attack module

Confirmed via `grep -r "random.seed"` across `attack/` — zero matches (at the time this was written).
`theory/DATA_POISONING_DEFENSE.md` §9 independently hit this in practice, not just in theory: a
defense-on-vs-off comparison on the clean baseline produced a worse aggregate number for the defense,
which turned out to be very likely attributable to single-seed noise rather than a real regression
(each of the 6 combos was a single unseeded run; n=7 questions means one flipped answer is a 14.3-point
swing — see B5).

**Resolution:** `run_attack.py` now accepts `--seed <int>`, calls `random.seed(seed)` before
constructing the attack (deterministically controls all `random.sample`/`random.choice`/`random.choices`
calls in `data_poisoning_attack.py`, since both modules share the same process-wide `random` state),
and logs `"seed"` in `attack_config` in the saved JSON. Verified: identical seed → identical source
selection and decoy content; different seed → different, still-deterministic results.

`run_all_attacks.py` now accepts `--seeds 0 42 123` (or any list), runs every combo once per seed, and
reports mean +/- std for `attacked_accuracy` and `degradation_pct` plus a success rate across seeds,
instead of a single-run YES/NO. Saved to `attack_logs/attack_matrix_summary_seeds_<timestamp>.json`
with both the raw per-seed runs and the aggregated stats. Legacy no-`--seeds` behavior (single unseeded
run per combo) is preserved unchanged for backward compatibility.

**The `{0, 42, 123}` sweep has now been run** against the clean baseline with the defense enabled
(`theory/DATA_POISONING_DEFENSE.md` §10, `attack_logs/attack_matrix_summary_seeds_2026-07-18_03-22-36.json`).
Headline finding: only 1 of 6 combos (`targeted(sources_0,sources_20)/misleading`) has a nonzero success
rate across seeds (3/3, rock-solid at 33.3% ± 0.0%) — every other combo is 0/3, several with negative
mean degradation. This is a dramatically different, far more defensible picture than §7's single-run
"6/6 succeed," and it's now clear that most of that earlier gap was single-seed noise (B4/B5) and the
confounded baseline (A1), not a weak defense. See §10 for the full table and an important infrastructure
caveat: getting one clean, uninterrupted run took several attempts because Docker Desktop kept restarting
mid-sweep and silently reverting the data sources to the polluted corpus via `docker-compose.yml`'s base
config — worth knowing before trusting any single long unattended run in this environment without a
similar safeguard (making the clean config the actual `docker-compose.yml` default, not an override file,
for the run's duration).

### B5. [RESOLVED — caveated] n=7 eval set makes the "success" boolean statistically noisy

Each question is ~14.3 percentage points of accuracy. With clean accuracy already low (42.9% per the
defense doc — itself a consequence of A1, not a truly clean baseline), a single flipped answer swings
the *relative* degradation number (`(clean - attacked) / clean * 100`) by double digits, easily enough
to cross the 10% "success" threshold on its own. Combined with B4 (no seeding), a "successful" vs.
"unsuccessful" verdict for a given combo may just be which way one question happened to flip.

**Resolution:** not fixed by enlarging the eval set (would require re-running every experiment in this
document — out of scope for a documentation-pass fix). Instead, explicitly caveated in
`theory/DATA_POISONING_DEFENSE.md` §13: single-seed numbers (§7-9, §11) are illustrative only; the
multi-seed tables (§10, §12) are what should be quoted, and their nonzero std values (up to 19.2
points) make this exact noise source directly visible rather than hidden. Enlarging the eval set
remains open if a more fundamental fix is wanted later.

### B6. [RESOLVED — caveated] Attack volume is not stealthy

A poisoned source's corpus grows from ~3,200 to ~19,700 docs in the multi-seed sweeps (§10, §12) — a
size increase any basic ingestion-volume monitoring would trivially flag. The paper's own
document-level pollution methodology also tests full-source pollution up to p=100%, so large-scale
corpus alteration isn't unprecedented in this line of work, but it's still an implicit assumption
worth stating outright.

**Resolution:** explicit scope-limitation language added to `theory/DATA_POISONING_DEFENSE.md` §13
and `theory/Data Poisoning Attack — Reliable-dRAG.md` (next to its volume-flooding table): "this
attack assumes no ingestion-volume monitoring by the defender," with the note that a much simpler
defense (cap index growth rate) would catch this specific attack pattern before the dedup/consensus
defense is ever needed — the built defense targets a smarter, volume-capped attacker, not the loud
one actually demonstrated.

### B7. [RESOLVED] Documentation/code default mismatch

`theory/Data Poisoning Attack — Reliable-dRAG.md` stated `--amplify` defaults to 3; the actual argparse
default is 1 ([run_attack.py:165](../attack/datapoisoning/run_attack.py#L165)).

**Resolution:** corrected the documented default to 1, matching the code (not the reverse — every
experiment in this document used the actual code default, so changing the code would have broken
reproducibility of past results for no benefit).

### B8. [RESOLVED — caveated] Substring-match correctness metric

`is_correct()` checks `ans.lower() in response.lower()`. Can false-positive on negated matches (e.g.
"not till September" would count as correct for expected answer "till September") and has no
mechanism to penalize a technically-correct answer buried in otherwise-wrong text.

**Resolution:** the limitation (already partially noted in `theory/Data Poisoning Attack —
Reliable-dRAG.md`) is now spelled out with the specific failure mode (negation false-positives) in
both that doc and `theory/DATA_POISONING_DEFENSE.md` §13, named explicitly as something to state
wherever these accuracy numbers are quoted rather than treating them as exact ground truth. Not
changed in code — reasonable simplification for short-answer QA at this scale; the fix is in how the
numbers are presented, not the metric itself.

### B9. [RESOLVED] `--ratio` silently truncated an explicit `--targets` list

`_select_sources_to_poison()`'s `targeted` branch returned `targets[:num_malicious]`,
where `num_malicious = max(1, int(len(self.data_sources) * self.poisoning_ratio))` is
computed against the **total** source count (3), not the length of the explicitly named
`--targets` list. At the default `--ratio 0.5`, `num_malicious = 1`, so
`--targets sources_0 sources_20` silently poisoned only `sources_0` — the second named
source was never touched, despite every downstream log, table, and doc labeling the
combo `targeted(sources_0,sources_20)`.

**Impact:** this is the flagship, most-cited result in `theory/DATA_POISONING_DEFENSE.md`
(§10's headline "only 1/6 combos reliably succeeds" finding, repeated in §7, §8, §9, §11,
§12) — every one of those rows describes a single-source attack, not the two-source
attack its label claims. A committee member tracing the code would find this immediately;
it was not previously caught because nobody checked what `--targets` actually resolved to
downstream of `--ratio`.

**Fix:** `_select_sources_to_poison()`'s `targeted` branch no longer truncates the
explicit `target_source_names` list — `--ratio` now only pads it with extra random
sources if the named list is *smaller* than the ratio-implied budget, never shrinks it.
`--targets sources_0 sources_20` now always poisons both, regardless of `--ratio`.
**Not retroactive**: every historical result in `theory/DATA_POISONING_DEFENSE.md` and
`theory/Data Poisoning Full Attack Matrix Results (2026-07-21).md` was generated under
the old, truncating behavior and has been annotated with an erratum in both documents
rather than silently re-labeled.

**Verified with a real re-run** (`attack_logs/attack_matrix_summary_seeds_2026-07-30_00-44-31.json`,
`theory/DATA_POISONING_DEFENSE.md` §14): `run_all_attacks.py --seeds 0 42 123 --only
"sources_0,sources_20"` against the live deployment now genuinely poisons both
`sources_0` and `sources_20` every seed (confirmed via each run's `poisoned_source_names`
and `total_injected_docs≈19,700` across two sources vs. the old single-source ~9,850).
The corrected result is **stronger** than the mislabeled one (44.4% ± 19.2% mean
degradation vs. the old 33.3% ± 0.0%), has real seed-to-seed variance instead of the old
run's suspicious exact-zero std, and shows a repeated, multi-question regression pattern
(Q6/Q7 flip correct→incorrect in all 3 seeds) rather than a single flipped answer — a
more defensible number, not just a corrected label.

### B10. [RESOLVED — caveated] `data_rich` degenerates to a fixed choice when doc counts tie

`data_rich` was built (B1) specifically to replace `high_reliability`, whose entire
failure mode was "every source ties, so Python's stable `sorted()` always returns
`sources_0` — indistinguishable from a hardcoded choice." `data_rich` reintroduced the
identical failure mode through a different signal: all three sources currently serve an
identical corpus (same `clean_baseline.jsonl` for all of `sources_0/20/100`, per the A1
fix), so their live `/info` doc counts are tied at every selection point, and the same
stable-sort tie-break silently always resolved to `sources_0` — confirmed directly by the
2026-07-21 full matrix run, where both `data_rich` combos "selected" `sources_0` with no
tie ever disclosed.

**Fix:** `_select_sources_to_poison()`'s `data_rich` branch now (a) logs the actual doc
counts driving selection (`doc_counts_at_selection` in the attack log, per the B1 fix's
own recommendation to log the real values behind "highest X"), (b) shuffles before the
stable sort so a genuine tie resolves randomly instead of always to `sources_0`, and (c)
logs an explicit warning whenever the top count is tied across more than one source, so
"data_rich picked sources_0" can no longer be silently misread as "sources_0 genuinely
had the most documents" when it was actually a coin flip.

**Not fully resolved as a strategy**: this makes the tie-break honest and auditable, but
does not give `data_rich` a real signal to differentiate on while all three sources share
one corpus. If a genuinely document-differentiated `data_rich` result is needed (rather
than a documented tie-break), the deployment needs sources with real doc-count asymmetry
— not attempted here, since it would mean deviating from A1's equal-corpus clean baseline.

---

## What's already well-handled (no action needed)

**The `/poison` / `/reset` / `/info` endpoints.** These are custom additions to
`drag_data_source/app/server.py`, not present in upstream Reliable-dRAG (confirmed — original surface
is just `/health` and `/query`). This is intentionally "God Mode"-shaped (unauthenticated direct write
into the live retriever), but `theory/Data Poisoning Attack — Reliable-dRAG.md` already contains a
solid, ready-to-use justification: frame `/poison` explicitly as a research instrument standing in for
"attacker has gained write access to a decentralized source" (compromised node, malicious contributor,
unvetted upload), not as a vulnerability being claimed against the real system's original API surface.
This framing is standard in RAG/decentralized-system security literature and is genuinely
defensible — just make sure the thesis text states it outright rather than leaving a reader to infer
it.

---

## Priority order for fixes

1. ~~**A2 / B1**~~ — **B1 resolved** by removing `high_reliability` and replacing it with
   `data_rich` (targets the source with the most documents, via `/info` — a real signal,
   no blockchain dependency). **A2 itself is still open**: the reliability feedback loop
   still never engages during a normal `/query`. It's no longer blocking any current attack
   strategy, but it's still relevant to the broader thesis claim about the paper's adaptive
   mechanisms being inert during evaluation (see Part A2), and will need addressing
   separately if the Source Selection Manipulation (SSM) attack is implemented, since SSM's
   entire premise is manipulating this same on-chain score.
2. ~~**A1**~~ — **resolved.** Genuinely clean baseline stood up (`data/polluted_token/clean_baseline.jsonl`,
   verified against `data/polluted_token_squad_backup/sources_0.jsonl`), switchable via
   `docker-compose.clean-baseline.yml`. Measured genuine clean accuracy: **57.1% (4/7)**,
   not the 42.9% (3/7) every prior attack log compared against.
3. ~~**B2**~~ — **fully resolved.** `--no-query-aware` added to both scripts; oracle-knowledge kept as
   the labeled default. Full multi-seed black-box sweep run (`theory/DATA_POISONING_DEFENSE.md` §12):
   result is a near-total inversion, not simple weakening — content-crafting strategies
   (`targeted`/`data_rich` `wrong_answer`/`misleading`) collapse to 0/3 without query knowledge, while
   plain corpus-noise strategies (`random/noise`, `random/answer_swap`, `data_rich/noise`) go from 0/3
   to 1-2/3. Report as attack-style-dependent in the thesis, not a single scalar claim.
4. **B3** — now moot for the current strategy set, since no strategy depends on on-chain
   reliability differentiation anymore (`data_rich` uses `/info` doc counts instead). Revisit
   only if A2 is fixed for SSM's sake and some future data-poisoning strategy is reintroduced
   that reads reliability scores.
5. ~~**B4**~~ — **fully resolved.** `--seed`/`--seeds` implemented, and the `{0, 42, 123}` sweep has
   been run against the clean baseline with the defense on (`theory/DATA_POISONING_DEFENSE.md` §10).
   Result: only 1/6 combos succeeds reliably across seeds. §7-9's single-run numbers are superseded by
   §10 for anything quoted in the thesis — keep them only as illustrations of *why* multi-seeding
   mattered, not as final results.
6. ~~**B5 / B6 / B7 / B8**~~ — **resolved.** B7's default-value mismatch corrected
   (`theory/Data Poisoning Attack — Reliable-dRAG.md`); B5/B6/B8 caveated explicitly in a new
   consolidated "Known limitations" section (`theory/DATA_POISONING_DEFENSE.md` §13), plus the
   `theory/Data Poisoning Attack — Reliable-dRAG.md` walkthrough had its stale pre-fix claims
   corrected in passing (it still described the removed `high_reliability` strategy and the
   already-fixed `_create_text_variants` poison-type bug as current behavior).

---

## Status: all findings resolved or explicitly caveated

Every item in this document is now either fixed with verified evidence (A1, B1, B2, B4, B9, B10) or
explicitly documented as a stated scope limitation (B5, B6, B7, B8) rather than a silent gap. The
one remaining open item, **A2** (blockchain reliability feedback loop never engages), is not blocking
any current data-poisoning strategy — it's relevant only if the separate Source Selection
Manipulation (SSM) attack from the project's CIA-triad framework is implemented later, since SSM's
entire premise is manipulating that same on-chain score.

**Caveat specific to B9/B10 (found during a later re-audit, not part of the original pass above):**
unlike the other resolved findings, fixing these two did not retroactively correct the numbers
already published in `theory/DATA_POISONING_DEFENSE.md` and
`theory/Data Poisoning Full Attack Matrix Results (2026-07-21).md` — both documents were annotated
with an erratum rather than silently edited. Before presenting any `targeted(sources_0,sources_20)`
or `data_rich` result from either document, either (a) cite the erratum alongside it, or (b) re-run
that specific combo under the current code and report the fresh number instead.

**Fully closed for the entire attack matrix, all 6 combos** (`theory/
DATA_POISONING_DEFENSE.md` §14/§15): every combo (`random/noise`, `random/answer_swap`,
`targeted(sources_100)/wrong_answer`, `targeted(sources_0,sources_20)/misleading`,
`data_rich/wrong_answer`, `data_rich/noise`) has now been re-run across seeds {0, 42,
123} under the fixed code, twice — once with `defense.enabled: false` (§15.1, the
genuine attack-only numbers matching the core report's defense-excluded scope) and once
with `defense.enabled: true` (§15.2, kept only as secondary context). The flag was
reverted to `true` after the final run to restore the deployment's committed default.

**Headline change from this full re-run:** with the defense off, 4 of 6 combos succeed
in at least one seed (not just the flagship one) — `random/noise` 1/3, `random/
answer_swap` 2/3, `targeted(sources_0,sources_20)/misleading` 3/3,
`data_rich/noise` 1/3. Only `targeted(sources_100)/wrong_answer` and
`data_rich/wrong_answer` never succeed, consistent with `wrong_answer` being a
weaker poison type than `misleading`/`answer_swap`/`noise` against this eval set.
**Cite `theory/DATA_POISONING_DEFENSE.md` §15.1 for the core report** — it is now the
single most-verified table in this entire audit.
