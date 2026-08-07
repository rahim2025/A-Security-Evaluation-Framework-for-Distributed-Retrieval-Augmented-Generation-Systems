# SSM-Score Redesign: Grounding-Farming Attack — Implementation Plan

**Status:** Implemented and multi-seed validated (§8f, §7b). Remaining: report/thesis-doc updates (§9 step 8).
**Supersedes-as-primary-result:** `attack/ssm_score/ssm_score_attack.py` (key-forgery variant), now relabeled and re-run as the secondary control-plane finding — see [§7](#7-disposition-of-the-existing-key-forgery-attack) and [§7b](#7b-relabeling-defense-interaction-fix-and-multi-seed-re-run-completed).

---

## 1. Why this exists

The current SSM-Score implementation (`attack/ssm_score/ssm_score_attack.py`) submits forged blockchain transactions using **both** the target source's private key and the LLM orchestrator's own private key (`feedbackAndUpdateScoreRecords` is gated `onlyLLMService` in `drag_contract/contracts/drag_scores.sol:148-150`). That contradicts two claims central to the thesis:

- `thesis_update.md`: SSM "requires no privileged access... a regular participant... In a centralized RAG system this attack class does not exist at all."
- The DRAG-parity claim: DRAG's SSM (advertisement manipulation) is executable by a lone peer with no central-authority compromise; the current Reliable-dRAG instantiation is not equivalent in privilege level, breaking "the attack structure and its consequences are identical across both."

Tracing the honest scoring pipeline in `drag_llm_service/app/server.py` found that a source acting alone *does* have one lever over its own on-chain score: **`grounded_by_sources`**, computed at [server.py:856-858](../drag_llm_service/app/server.py#L856-L858):

```python
norm_response = _norm_join(_normalize_for_match(response_text))
selected_contexts_norm = [_norm_join(_normalize_for_match(c["text"])) for c in selected]
grounded_by_sources = [norm_response in ctx for ctx in selected_contexts_norm]
```

This is a **naive literal-substring check**, not a relevance or provenance check. `_normalize_for_match` (`drag_llm_service/src/utils/helper_functions.py:50-51`) just lowercases and strips punctuation to a token list; `_norm_join` joins it back into a string. So the check is: *does the normalized LLM answer appear as a contiguous run of normalized tokens somewhere inside this source's returned chunk?* — regardless of whether that chunk had anything to do with producing the answer.

A source can exploit this without touching the blockchain and without any key beyond its own: return broad, high-coverage filler content designed to contain likely short-answer phrasings, so it gets "grounded" credit — and the resulting positive `reliability_delta`/`usefulness_delta` — on queries it did not meaningfully help answer.

---

## 2. Threat model

| Element | Value |
|---|---|
| Attacker capability | Controls one data source's own retriever/document store (via the existing `/poison` instrumentation endpoint — no different from what Data Poisoning already assumes) |
| Attacker keys required | **None beyond the source's own**, which it already legitimately possesses and uses to sign every `/query` response as normal operation |
| Attacker knowledge | The scoring logic is grounded on literal substring match (`server.py:856-858`) — a property of the deployed system, discoverable by testing/observing score deltas after queries, not requiring source access |
| Goal | Inflate the source's on-chain reliability/usefulness over many queries without deserving it, causing the orchestrator to route future queries toward it at the expense of genuinely higher-quality sources |
| Not required | Stolen keys, forged signatures, direct contract writes, any privileged/insider position beyond "operates one of the participating sources" |

This matches `thesis_update.md`'s SSM definition verbatim and restores parity with DRAG's self-advertisement attack (a lone participant gaming a trust mechanism through the system's own normal interfaces).

---

## 3. Differentiation from Data Poisoning (must survive committee scrutiny)

Both attacks technically use the same delivery primitive (`POST /poison` to inject documents into a source's retriever — `drag_data_source/app/server.py:167-206`). That is expected and fine: DRAG's own Data Poisoning module also varies *strategy* over one primitive. What must be different — and clearly written up as different — is the **objective, the content design, and the measured harm axis**:

| | Data Poisoning | Grounding-Farming (SSM) |
|---|---|---|
| CIA property | Integrity — data plane | Integrity — control plane |
| Content goal | Topically *narrow*, factually *wrong*, targeted at specific queries | Topically *broad*, factually neutral/accurate-sounding, designed for max substring-match coverage across many queries |
| What breaks | The answer the user receives | The trust score's correspondence to actual merit |
| Primary metric | F1 / accuracy degradation vs. baseline | Rate of undeserved "grounded" credit; on-chain score drift vs. a true-merit baseline; resulting query-misdirection rate toward the farming source and away from honest higher-quality sources |
| Attack succeeds even if... | The answer is wrong | The answer is *correct* — the source just fraudulently claims credit for a correct answer it didn't actually ground |

The write-up should lead with that last row: this is a **reputation-fraud / free-riding attack**, not a content-corruption attack. It can succeed while every user-facing answer stays correct, which is precisely why it's a distinct, covert control-plane risk rather than a data-plane one — and it's the reason to measure *misdirection* and *score-vs-merit gap*, not F1.

---

## 3a. Interaction with the existing on-chain SSM defense (must be addressed explicitly)

This codebase already contains a **real, deployed defense** against SSM-Score, added in commit `690aa35` ("Add latest reliability and defense updates") — not scoped as future work, contrary to `.claude/CLAUDE.md`'s "Defenses are scoped as future work for this system." Any write-up of grounding-farming must address this head-on rather than silently working around it, or a committee member who reads `drag_scores.sol` will reasonably ask why the attack ignores a defense that's sitting right there.

**What exists, on-chain (`drag_contract/contracts/drag_scores.sol`), gated inside `feedbackAndUpdateScoreRecords`:**
- `onlyLLMService` caller check — only the registered LLM-service address may call the update function at all (`UnauthorizedCaller`).
- `MIN_UPDATE_INTERVAL` = 2s — a source's score may move at most once per 2 seconds (`UpdateTooFrequent`).
- `MAX_DELTA_PER_UPDATE` = 5,000 — no single transaction may move reliability or usefulness by more than this (`ScoreDeltaTooLarge`).
- `MIN_SCORE`/`MAX_SCORE` = ∓1,000,000 — absolute bounds regardless of history (`ScoreOutOfBounds`).

**Off-chain**, `defense/ssm_defense/ssm_score_defense.py` replays `ScoreRecordUpdated` events and flags any historical transition that would have violated the same four bounds — a monitoring/audit layer mirroring the on-chain enforcement.

The code comments are explicit that all of this was built against the **key-forgery** variant (§7): it stops a forged transaction from directly setting arbitrary scores in one shot, or from being submitted by anyone but the LLM service.

**Why grounding-farming is not blocked by any of these four checks** — this must be demonstrated empirically (run the attack against the live contract with the defense fully active, and confirm zero reverts/flags), not merely asserted:
- It never calls the contract directly and needs no key but its own retriever content — `onlyLLMService` is never implicated.
- `scaled_importance = int(round(score_value * 100))` (`server.py:894`) comes from a bounded similarity/entropy score; realistic per-query deltas are tens, not thousands — far under `MAX_DELTA_PER_UPDATE`.
- Real inference + retrieval per query takes far longer than 2s, so normal query cadence never approaches `MIN_UPDATE_INTERVAL`.
- Farming accumulates small, individually-unremarkable deltas over many queries — nowhere near the absolute bounds at realistic scale.

**The structural reason, for the write-up:** the existing defense hardens the *transaction layer* — who may write, how fast, how much per write. Grounding-farming never adversarially touches that layer; every update it produces is submitted by the legitimate orchestrator, correctly authorized, individually within-bounds, and honestly signed. What's corrupted is the *off-chain input* to that legitimate transaction — the naive substring-match heuristic (`server.py:856-858`) that decides what `scaled_importance` gets computed in the first place. No amount of rate-limiting, delta-capping, or caller authentication catches a forgery that never occurs; only the premise feeding an entirely real transaction is false.

This should be framed as a positive, not a hand-wave: *the first SSM instantiation was fully mitigated by transaction-layer hardening; the second demonstrates that this natural first line of defense is structurally blind to attacks that corrupt the merit signal itself rather than its transport, motivating defense-in-depth beyond blockchain-layer protections.* Recommend adding this as an explicit subsection in the eventual report, with the empirical zero-reverts/zero-flags result as supporting evidence rather than an assumption.

**Also flag to the team, separate from the attack write-up:** the contract modification and `defense/ssm_defense/` module's existence conflict with `.claude/CLAUDE.md`'s stated scope ("Defenses are scoped as future work," "Do not modify the Reliable-dRAG core system"). Worth a decision on whether to update CLAUDE.md to acknowledge this earlier defense pass, or otherwise reconcile the discrepancy before the thesis write-up is finalized.

---

## 4. Content design (must satisfy the actual normalization function)

Because `grounded_by_sources` requires a literal contiguous substring match post-normalization (lowercase, alnum/whitespace tokens only), filler content should be:

1. **Broad-coverage**: span many plausible short-answer entities/phrasings (dates, names, numbers, yes/no/maybe tokens) so that whatever the LLM ultimately generates — for many different queries — has a good chance of appearing verbatim as a token run somewhere in the blob.
2. **Embedding-broad, not just token-broad**: the chunk still has to be *retrieved* in the first place (FAISS similarity + reranker), so it also needs to score reasonably across many query embeddings — e.g. constructed as a concatenation of many short, topically-varied factual sentences rather than one narrow paragraph. This is the "high-connectivity" targeting idea from Data Poisoning, repurposed for breadth instead of depth.
3. **Not adversarially wrong**: unlike Data Poisoning's `wrong_answer`/`misleading` types, content should avoid asserting false facts — the point is to get credit for correct answers, not to cause incorrect ones. Overlap with correct short answers is what's being farmed, not corruption.

Concrete construction strategy to prototype: build each filler document by concatenating N short "answer-shaped" fragments (numbers, named entities, dates, yes/no phrasings) harvested from a general-domain list — or, more cheaply for a first pass, from the actual SQuAD gold-answer distribution itself (attacker doesn't need the *questions*, just the shape of common short answers) — into one retrievable chunk per broad topic cluster.

### 4a. Empirical validation of content strategy (§9 step 2, completed)

Three content strategies were tested live against `sources_100` (injected via `/poison`, tested with `update_scores: false` so no chain writes occurred, reset after each pass), using 15-20 real matched SQuAD questions:

| Strategy | Per-source top-5 retrieval survival | Best rank achieved |
|---|---|---|
| Single blob, 60 unrelated answer fragments, no token budget | 1/15 (7%) | 4th of 5 |
| Answer-type clusters (year / number / person / place-org / misc), token-budgeted to fit under the retriever's 256-token truncation limit | 1/20 (5%) | 1st of 5 |
| **Domain-informed clusters, one filler doc per real article title actually present in `sources_20`/`sources_100`** | **6/20 (30%)** | 1st of 5, plus 3 more in top-3 |

Findings:
- The retriever (`sentence-transformers/all-MiniLM-L6-v2`) truncates at **256 tokens**. The naive blob (~1,950 chars, ~60 disparate sentences) silently lost most of its content to truncation and mean-pooled into a topic-less "average of everything" embedding that scored poorly against any specific query.
- Fixing truncation and giving each cluster coherent, thematically-uniform phrasing (answer-type variant) improved the *rank quality* of hits (4th → 1st) but did **not** move the raw hit rate (~5% either way) — because the mismatch was never about phrasing, it was about topic/subject alignment. Answer-TYPE clustering (years, numbers, names) has no relationship to the actual real-world subjects (Beyoncé, Notre Dame, Chopin, ...) the corpus's queries are about.
- Matching filler content to the corpus's actual **topic distribution** (harvested from the SQuAD dataset's own `title` field, restricted to titles that survive the corpus-match) closed most of the gap: 30% raw survival, ~6x the other two strategies.
- The corpus (`sources_20`/`sources_100`, 500 docs total) is built from only **~10 distinct Wikipedia articles** chunked into many paragraphs. The domain-informed batch (10 filler docs, one per title) therefore already covers the *entire* available topic space for this corpus — further breadth cannot come from more titles, only from more content-density or more sub-topic chunks per title.

**Threat-model implication (must be stated explicitly, not left implicit):** the domain-informed strategy requires the attacker to know the corpus's *topic distribution*, not the exact questions that will be asked. This is the same tier of knowledge Data Poisoning's "high-connectivity targeting" strategy already assumes (§ Attack Mapping table, `.claude/CLAUDE.md`) — domain-awareness, not question-specific targeting — so it's not a new precedent for this framework, but the SSM threat model (§2) should say so directly rather than imply the attacker needs zero domain knowledge at all. A rational operator of a data source realistically observes what topics its own queries and the system's domain concern over time, which is a weaker and more defensible assumption than knowing the corpus contents outright.

**Funnel note for metrics design (§6):** even the winning strategy's raw per-source hits compound down through the pipeline — of 6 per-source hits, only 3 ranked within the top-3 that the orchestrator actually pulls per source (`n_contexts: 3`), and only 1/20 total survived rerank + generation to trip the naive substring grounding check in this small sample. This is expected and consistent with the plan's own framing: the attack's real effect is measured as *cumulative score drift over many query rounds*, not single-query hit rate, so a per-query "would-ground" rate in the 5-15% range, sustained over the dozens/hundreds of rounds the actual eval run performs, is expected to be sufficient to produce a measurable score-vs-merit gap — this should be validated at full scale in §9 step 5-6, not assumed from this small probe alone.

---

## 5. Attack module design

New file: `attack/ssm_score/grounding_farming_attack.py`, mirroring the existing `DataPoisoningAttack` class shape (`attack/datapoisoning/data_poisoning_attack.py`) for codebase consistency:

```python
class GroundingFarmingAttack:
    """
    SSM-Score attack via grounding-heuristic exploitation.

    A source injects broad, high-coverage filler content (via the same
    /poison instrumentation Data Poisoning uses) designed to trip the
    orchestrator's naive substring-match grounding check on many queries,
    farming reliability/usefulness credit it did not earn -- without ever
    touching the blockchain directly or using any key but its own.
    """

    def __init__(
        self,
        target_source: dict,               # {"name": ..., "url": ...}
        coverage_breadth: int = 50,        # number of answer-shaped fragments per filler doc
        num_filler_docs: int = 10,
        amplification_factor: int = 3,     # reuse Data Poisoning's convention
        llm_service_url: str = "http://localhost:9000",
        blockchain_url: str = "http://localhost:8545",
    ): ...

    def build_filler_documents(self) -> list[dict]: ...   # §4 content design
    def inject(self) -> dict: ...                          # POST /poison, reuse DataPoisoningAttack's HTTP pattern
    def reset(self) -> dict: ...                            # POST /reset (already exists on data source)
```

`attack/ssm_score/run_grounding_farming.py` (new, mirrors `run_attack.py`'s shape):

1. Load SQuAD eval set (fix the corpus bug first — see [§7a](#7a-the-corpus-collision-is-a-known-problem-already-fixed-elsewhere-and-still-broken-in-one-place)).
2. Record baseline on-chain scores for all sources.
3. Run N rounds of real queries through `/query` (as a normal user would — **not** attacker-controlled queries; the whole point is these are ordinary queries the farming source happens to get undeserved credit on).
4. After each round, read on-chain scores + capture `grounded_by_sources` outcomes if the orchestrator exposes them (may need read access to `/info`-style diagnostics, or infer from score deltas — see open questions in §8).
5. Compute the metrics in §6.
6. ~~Compare against an honest control source~~ — **considered and dropped, see §8e.** Compare against `sources_20` instead: already untouched, same-domain, comparable size, receiving the identical real query stream, with no new infrastructure required.

---

## 6. Metrics (this is the part that must NOT reuse Data Poisoning's F1-drop framing)

Primary:
- **False-grounding rate**: fraction of queries where the farming source received `grounded=True` credit but was not, by inspection, the source that actually determined the answer (e.g., a genuinely relevant source's content also appears in context and is the true basis for the answer).
- **Score-vs-merit gap**: `(on-chain reliability/usefulness accumulated by farming source) − (on-chain reliability/usefulness accumulated by an untouched comparison source)`, tracked over query rounds — this is the headline curve, analogous to the F1-degradation curve in Data Poisoning but for the control plane. Comparison source is `sources_20` (§8e: a dedicated 4th "honest control" source was considered and dropped as unnecessary and structurally risky).
- **Misdirection rate**: fraction of *later* queries preferentially routed to / weighted toward the farming source as a downstream consequence of its inflated score, per `thesis_update.md`'s stated SSM metric ("the rate at which queries get misdirected to malicious sources"). Report as `farm_source_in_importance_score` (reliability-weighted rerank survival), not raw sampling inclusion — see §8a for why the latter is degenerate in this topology.
- **Crowd-out effect**: reduction in query share / score growth of the legitimate high-quality source(s) over the same period, directly supporting `thesis_update.md`'s second stated harm ("legitimate high-quality sources receive fewer queries than they should"). Empirically confirmed as the actual mechanism behind the accuracy cost — see §8d.

Secondary / sanity:
- Answer-quality (F1) should stay roughly flat or even *improve slightly* relative to baseline while the score-vs-merit gap grows — this was the original expectation; §8d found it does **not** hold once the attack actually escalates (accuracy dropped 60%→52% in the 300-round run, via crowd-out, not content displacement). Report whichever way the multi-seed numbers actually come out, and report the crowd-out mechanism explicitly rather than forcing the "stays flat" framing.

Required per `.claude/CLAUDE.md`: seeds **0, 42, 123**, variance reported, real JSON log artifacts committed to `attack_logs/` — do not repeat the single-run, n=7 evidentiary gap the key-forgery version had.

---

## 7. Disposition of the existing key-forgery attack

Keep it, but:
1. **Relabel** — in both code comments and any report/thesis text — as a **control-plane / orchestrator-key-compromise** attack, not a "malicious source" or "no privileged access" attack. It is a legitimate, separately-interesting finding (a signature scheme that never actually binds to the score value being written — see `message_dict` at `drag_llm_service/app/server.py:944-947`, which only covers `{query, selected_sources}`), just not the flagship "regular participant" SSM story.
2. **Fix the corpus collision** before re-running it — following the codebase's own established pattern, not a new one. See [§7a](#7a-the-corpus-collision-is-a-known-problem-already-fixed-elsewhere-and-still-broken-in-one-place) for the full picture; the short version: switch `attack/ssm_score/run_attack.py:_load_corpus_contexts()` from reading `sources_0.jsonl` unconditionally to **per-source ground-truth loading** — PubMedQA for `sources_0`, SQuAD for `sources_20`/`sources_100` — matching what `attack/kb_extraction/run_attack.py:_load_source_contexts()` and `attack/selective_forward_sim/run_attack.py` already do. Do **not** touch `data/polluted_token/sources_0.jsonl` itself or introduce a separate snapshot file — both are unnecessary once SSM matches its questions against `sources_20`/`sources_100` instead.
3. **Re-run properly**: n=50 (as already configured via `EVAL_SAMPLE_SIZE`), seeds 0/42/123, commit the resulting logs to `attack_logs/` so the report's citations resolve to real, checked-in artifacts (the currently-cited `attack_2026-07-06_18-56-54_ssm_score.json` does not exist anywhere in git history and needs to be replaced with a real one).

### 7b. Relabeling, defense-interaction fix, and multi-seed re-run (completed)

**Relabeling:** `attack/ssm_score/ssm_score_attack.py`'s and `run_attack.py`'s module docstrings and console banners were rewritten to correctly frame this as a **control-plane / orchestrator-key-compromise** attack — the caller must hold `PRIVATE_KEYS["llm_service"]` (`onlyLLMService`-gated), not merely "control a data source." The write-up point carried forward: the signature scheme verified on-chain never binds to the score *values* being written (`message_dict` at `drag_llm_service/app/server.py:944-947` signs only `{query, selected_sources}`), so a compromised orchestrator key can attach any data source's previously-valid signature to fabricated score values. `defense/ssm_defense/run_defense.py` (which imports `SSMScoreAttack` directly to demonstrate the on-chain defense blocking a reckless attack) was checked and is unaffected — it keeps its own local `AMPLIFY=999999`/default `inter_round_delay=0.5` unchanged.

**Defense-interaction bug found and fixed:** the original defaults (`AMPLIFY=999999`, 0.5s between rounds) now revert on round 1 against the real, deployed on-chain defense (`MAX_DELTA_PER_UPDATE=5,000`, `MIN_UPDATE_INTERVAL=2s`, both in `drag_scores.sol`) — confirmed live before touching anything further. Rather than "properly re-running" a script that would now 100% fail, `SSMScoreAttack` gained a configurable `inter_round_delay` parameter (default 0.5, preserving `run_defense.py`'s reckless-attack test scenario byte-for-byte), and `run_attack.py` was switched to defense-aware, cap-respecting defaults: `AMPLIFY=4000` (just under the delta cap), `INTER_ROUND_DELAY=2.5s` (just over the rate limit). A live sanity check (n=5, 3 rounds) confirmed this works cleanly: 3/3 rounds accepted, zero reverts, R/U inflated 10,000→22,000. This is itself a meaningful, reportable contrast with the reckless variant: a defense-aware attacker that throttles to just under both thresholds defeats the transaction-layer defense completely, where a reckless one is fully blocked — the caps stop *bursts*, not *patient* abuse, which is the same "transaction-layer hardening has structural blind spots" theme as §3a, via a different mechanism (throttling vs. semantic corruption).

**Second, unrelated bug found during the first full-campaign attempt:** `drag_data_source/app/server.py` rate-limits each client IP to `RATE_LIMIT_DEFAULT` (default "60 per minute") per source (`flask_limiter`, per-IP via `get_remote_address`), and the orchestrator fans every `/query` out to all three sources from one constant IP (the llm-service container). `run_attack.py`'s `measure_accuracy()` had no pacing, so a 50-question baseline/post pass fired requests fast enough to trip 429s once the campaign's cumulative request rate crossed the cap — confirmed via `docker logs drag-llm-service` showing `WARNING: Failed to query sources_X: 429`. This silently degraded into empty-string predictions that look like (but are not) accuracy effects: the first full-campaign attempt produced nonsensical baseline/post numbers (e.g. seed 42: baseline 0% with 50/50 empty predictions, post 52% with 0/50 empty — a pure artifact of which measurement pass happened to land inside the rate-limited window, not a real effect of the attack). **Checked the already-reported Grounding-Farming campaign logs for the same contamination — clean, zero empty predictions across all 3 seeds — that finding is unaffected.** Fixed by adding `QUERY_DELAY=1.3s` pacing inside `measure_accuracy()` (shared by `run_defense.py` too, since it imports the same function) — same class of issue as `reports/SFA_Security_Analysis_Report.md` sec 12.9, which had already added the (until-now-unused) `RATE_LIMIT_DEFAULT` env-var override to `docker-compose.yml` for exactly this failure mode; pacing the client was chosen over raising the container-level limit to keep the fix entirely attack-layer-side per `.claude/CLAUDE.md`'s "do not modify the Reliable-dRAG core system."

**Clean multi-seed results (seeds 0, 42, 123; n=50; re-run after both fixes, verified zero empty predictions in every pass):**

| Seed | Rounds accepted | R/U delta (`sources_100`) | Accuracy (baseline→post) | Outcome |
|---|---|---|---|---|
| 0 | 5/5 | +20,000 / +20,000 | 66%→56% (**-10pp**) | Full success |
| 42 | 5/5 | +20,000 / +20,000 | 62%→52% (**-10pp**) | Full success |
| 123 | 5/5 | +20,000 / +20,000 | 66%→66% (0pp) | Full success (no measured accuracy cost this seed) |

**The headline contrast for the write-up:** key-forgery is **deterministic** (3/3 seeds, 5/5 rounds every time, zero variance in the score outcome) but requires **orchestrator-key possession** — a far higher privilege bar. Grounding-Farming is **probabilistic** (1/3 seeds caught, §8f) but requires **no privileged access at all** — just control of one's own data source. This is a clean, defensible two-attack SSM story: privilege level trades off against reliability of the outcome, and the framework produced both ends of that trade-off using the same underlying scoring mechanism. Logs: `attack_logs/attack_2026-07-25_00-20-53_ssm_score_key_forgery_seed0.json`, `..._seed42.json` (00-24-17), `..._seed123.json` (00-27-42).

---

### 7a. The corpus collision is a known problem, already fixed elsewhere, and still broken in one place

`data/polluted_token/` is bind-mounted straight into all three data-source containers (`docker-compose.yml:28-29,60-61,90-91`, `./data/polluted_token:/data`), so `sources_0.jsonl` isn't just a file SSM's script reads locally — it's what the live `drag-data-source-0` container actually serves to every query, for every attack. `data/build_pubmedqa_corpus.py` overwriting it for MIA changed what `sources_0` answers with system-wide, not just what SSM's local matching sees.

This already caused real breakage, documented in the affected scripts' own docstrings, and was already fixed three times, inconsistently:

- **`attack/selective_forward_sim/run_attack.py:129-156`** — docstring: *"this script previously crashed on every single run with 'No SQuAD questions matched'... a 100% failure rate."* Fixed via per-source ground truth (PubMedQA for `sources_0`, SQuAD for `sources_20`/`sources_100`).
- **`attack/kb_extraction/run_attack.py:83-155`** — same fix pattern, `_load_source_contexts(source_idx)`. Comment: *"source_0 is PubMedQA now, not SQuAD."*
- **`attack/ddos_sim/run_live_evaluation.py:78-116`** — rewritten to match PubMedQA against `sources_0` directly instead of SQuAD.

SSM should follow this established convention (which is why point 2 above now points at per-source loading, not a new snapshot file) rather than invent a fourth pattern.

**Still broken, independent of anything in this plan:**
- **`attack/selective_forward/run_attack.py:364-379`** — the *live*, canonical SFA module (not `_sim`), the one `reports/Security_Analysis_Report.md` actually cites for its SFA numbers — still has the old single-shared-corpus assumption and an unfixed docstring claiming `sources_0.jsonl` is "the 0%-polluted (clean) copy." It's in the same broken state SSM was in. Fixing SSM does not fix this; flagging so it isn't mistaken for solved.
- **`attack/datapoisoning/run_attack.py:41-53`** sidesteps the crash by using a hardcoded 7-question list instead of dynamic matching, so it won't error — but its framing of `sources_0` as the "0% polluted / clean SQuAD baseline" is now factually wrong, since `sources_0` isn't SQuAD-domain at all anymore. No crash risk, but any result leaning on `sources_0` as the clean reference rests on a stale assumption.

**Footnote for the write-up, not a blocker:** because the orchestrator queries all three sources on every request regardless of which questions are asked, `sources_0` will keep returning off-domain PubMedQA candidates for every SQuAD-domain query SSM sends — harmless (the reranker won't select them) but it means the "3-source" system is effectively "2 SQuAD-relevant sources + 1 noise source" for SSM's purposes going forward. Worth one line in the results write-up.

---

## 8. Open questions / risks to resolve before or during implementation

- **Does the orchestrator expose per-query `grounded_by_sources` outcomes anywhere observable externally** (response metadata, logs, an events endpoint), or must false-grounding rate be inferred indirectly from score deltas? Need to check `server.py`'s `/query` response shape and whether `info_json` (which includes `correctness`/`sampled_sources`/`importance_score`, `server.py:861-866`) is retrievable after the fact via `get_score_events()` (`server.py:485`) or the contract's `ScoreRecordUpdated.info` field — if `info` is logged on-chain per update, that may be enough to reconstruct grounding outcomes without any new instrumentation.
- **Will FAISS retrieval actually surface broad filler content often enough** to matter, or will topically-narrow honest documents consistently outrank it? This needs an early empirical check (inject filler into `sources_100`, run a modest query batch, check `selected`/candidate composition) before committing to full-scale content design — if breadth alone isn't enough to get sampled, the content strategy in §4 needs tightening (e.g. per-topic-cluster filler docs targeted at the query distribution actually in use, rather than one maximally-broad blob).
- **Reuse `/poison` vs. add a distinctly-named endpoint/method**: reusing `/poison` is simpler and arguably strengthens the "same primitive, different strategy" narrative; a distinct name might read more cleanly in the paper. Decide once §3's write-up is drafted — either is technically fine.
- ~~Honest control source~~ — **RESOLVED, see §8e: considered and decided against.** `sources_20` serves as the untouched comparison instead; no 4th container.

---

## 8a. Small-scale validation run (§9 step 3, completed)

`GroundingFarmingAttack` + `run_grounding_farming.py` were built and validated with a small real run (ROUNDS=20, EVAL_SAMPLE_SIZE=20, seed 42, target `sources_100`, log: `attack_logs/attack_2026-07-24_21-16-47_grounding_farming.json`). Every round used real `/query_analyze` calls with `update_scores=True`, so every score-update transaction was submitted by the actual orchestrator, not simulated.

**Confirms §3a empirically:** zero reverts against the existing on-chain SSM defense across all 20 rounds (no `UnauthorizedCaller`/`ScoreDeltaTooLarge`/`UpdateTooFrequent`/`ScoreOutOfBounds`). Score-vs-merit gap after 20 rounds: `sources_100` (farmed) R+56/U+51 vs `sources_20` (untouched) R-19/U+2 vs `sources_0` R+6/U-1 — farming source gained ground an untouched source did not, exactly the effect the thesis needs to show.

**Two honest caveats surfaced, both need to carry into the full campaign and report, not be smoothed over:**

1. **"Misdirection rate" (fraction of rounds where the farm source is in `sampled_sources`) is degenerate at 100% in this 3-source topology** — `n_retrievers: 3` equals the total source count, so usefulness-weighted *sampling* always includes all three sources regardless of score; there is nothing for a low-usefulness source to be excluded from. This metric cannot discriminate here and should not be reported as evidence of attack effect. The non-degenerate signal is **`farm_source_in_importance_score`** (95% in this run) — whether the source survives the *reliability-weighted rerank* into the final top-K — which does respond to the growing score and is the metric to report instead. A real fix for the sampling-level metric would require more sources than `n_retrievers` — this was the main argument for a 4th "honest control" source, and it's the argument §8e concludes isn't strong enough to justify one: `farm_source_in_importance_score` already isn't degenerate and already tells the whole story (confirmed dramatically in §8d's 96.3% result), so the sampling-level metric doesn't need fixing badly enough to warrant new infrastructure.
2. **Accuracy moved 55%→45% (11/20→9/20) rather than staying flat**, contrary to §6's expectation. On n=20 this is within plausible single-seed noise (a 2-question flip), and a plausible mechanism exists (filler occasionally displaces genuinely correct `sources_100` content in the reranked top-K for the subset of eval questions whose real gold context lives in `sources_100`) — but this must be checked at proper scale across seeds 0/42/123 before writing up "F1 stays flat" as a finding. Report whatever the multi-seed numbers actually show, including if that contradicts this expectation.

**Mechanistic note:** `rerank_with_reliability` min-max normalizes reliability across only 3 sources, which stretches even small absolute score differences into a large *relative* reranking influence — this likely explains why a modest ~0.5% absolute score change already produced a 95% final-selection rate and a visible accuracy shift at just 20 rounds. Worth one line in the write-up so the magnitude of on-chain drift needed for real-world impact isn't overstated.

### 8c. First content-design fix attempt failed empirically (0/50, worse than baseline) — root cause identified

Per §8b's recommendation, `GroundingFarmingAttack.build_filler_documents` was reworked to build several sub-topic filler docs per title (`docs_per_title=4`), each anchored on a keyword cluster harvested from the real distribution of questions asked about that title (aggregate word frequency from the full matched question pool, not literal question/answer text). Read-only check (`update_scores=False`, no chain writes) against the identical n=50 sample that exposed the 2% collapse: **result was 0/50 (0%) — worse than the 1/50 (2%) baseline it was meant to improve on.**

Three concrete bugs diagnosed from the actual generated filler text, not guessed at:
1. **Encoding bug defeated keyword exclusion.** The exclude-title-words filter compares `title.split()` (e.g. `"Beyoncé"`) against keywords extracted via an ASCII-only regex (`[A-Za-z]+`, which strips accents to `"beyonc"`/`"beyonce"`). These never match, so mangled fragments of the title itself leaked into its own "distinctive keyword" list (`grounding_farm_beyonc_0: keywords=['beyonce', 'beyonc', 'first', 'album']`), wasting a quarter of that document's keyword budget on noise instead of signal.
2. **Raw keyword frequency surfaces generic words, not title-distinctive ones.** Top-40-by-count picks words like "year", "first", "name", "song", "music", "released" — common across nearly *every* article's question set, not distinctive to this one. This is the textbook failure mode of using raw term frequency instead of a discriminative measure (e.g. TF-IDF: frequency within this title's questions relative to frequency across all titles' questions) — it needs fixing before another attempt.
3. **Template/fragment type mismatch produces incoherent sentences**, e.g. "the year Saint Bernadette Soubirous is mentioned" — `FRAGMENT_TEMPLATES` assumes its `{f}` slot is filled by a year/number, but `short_answer_fragments` mixes all SQuAD answer types indiscriminately. Pre-existing since the original DOMAIN_TEMPLATES design, likely worse now that documents are longer and more numerous.

**Operational note:** the read-only check script crashed mid-run on a Unicode console-encoding error (unrelated to the attack logic — printing a question containing "ł"), which happened *before* its reset call and left `sources_100` poisoned with 40 filler docs until caught and manually reset. No lasting effect (confirmed back to 500/500/0 immediately after), but it's a concrete demonstration of the crash-before-reset risk flagged back when this module was first built. Fixed in the scratchpad checker with a `try/finally` around the reset call; **`run_grounding_farming.py`'s `main()` has the same gap** (its `attacker.reset()` call isn't inside a `finally`) and should get the same fix before the next real on-chain run.

None of this was knowable until it was actually run and the resulting text inspected — confirms the value of testing empirically (read-only first) rather than assuming a design change is an improvement. All three bugs were fixed (Unicode-aware `WORD_RE`, TF-IDF-style discriminative keyword ranking in `run_grounding_farming.py:load_squad_eval`, type-agnostic `FRAGMENT_TEMPLATES` that still repeat `{t}` every sentence) and the same n=50 check re-run twice more (once per fix pass) — **still 0/50 both times.** Keyword quality visibly improved (`grammy, solo, single, woman` for Beyoncé; `sand, piano, liszt, warsaw` for Chopin — genuinely distinctive, not title-fragments-plus-generic-noise anymore), so bugs #1/#2 are confirmed fixed on their own terms. It did not move the retrieval-survival needle at all.

**Conclusion after three iterations, all negative: content-design tuning has hit diminishing-to-zero returns, and the likely explanation is structural, not a phrasing/keyword problem.** Every specific factual question in a 50-question diverse sample is competing against ~500 genuine, high-quality, directly-relevant real documents (the actual SQuAD paragraphs); for most specific questions a real paragraph that actually answers it exists and wins on relevance regardless of how well-crafted the filler is. The n=20 run's 95% engagement rate is now better explained as an artifact of that particular 20-question draw being unusually rich in vague/incomplete-sounding questions (e.g. "Name some adverse effects?", "What group does C support England joining?") that have no single dominant matching real paragraph — not evidence that any content strategy tested here is actually good at competing against real, on-point content.

**Recommendation carried into the next decision point:** stop iterating on content engineering for now. A low (single-digit-percent) but real per-query hit rate is consistent with the plan's own metric philosophy (§6: cumulative score-vs-merit drift over *many* query rounds is the measured effect, not per-query success rate) — the more promising lever is validating that a modest, honestly-measured hit rate still produces a real cumulative score gap when run at proper scale (hundreds of rounds), rather than continuing to chase a higher hit rate through content design.

### 8d. The hit-rate hypothesis was wrong in an important, useful way: this is a feedback loop, not an accumulation

Tested §8c's recommendation directly: a real (`update_scores=True`) 300-round run, same n=50 sample, same sub-topic filler design that scored 0/50 in every read-only check, on a freshly-reset pristine chain (`hardhat-node` restarted and confirmed 10000/10000 beforehand). Log: `attack_logs/attack_2026-07-24_22-14-59_grounding_farming.json`.

**Result: `farm_source_in_importance_score` = 289/300 (96.3%)** — not a modest single-digit rate sustained at scale, a runaway majority. R delta for `sources_100`: **+155**, U delta: **+1183**. Score-vs-merit gap: **+170 reliability / +1272 usefulness** vs `sources_0`, **+186 / +1156** vs `sources_20`. Accuracy: 60.0%→52.0% (not flat this time — see below).

**Why the read-only checks (always 0/50 or 1/50) and this real run (96.3%) disagree so sharply, verified round-by-round, not guessed:** read-only checks use `update_scores=False`, which never lets any source's on-chain score move -- every one of the 50 questions is tested against the *same, frozen, equal* starting reliability for all three sources. The real run's round log shows why that's the wrong test:

```
round 1: farm_selected=True,  R=10000 U=9991   (selected but NOT grounded -- penalized)
round 2: farm_selected=False, R=10000 U=9991   (no change)
round 3: farm_selected=False, R=10000 U=9991   (no change)
round 4: farm_selected=True,  R=10017 U=10008  (GROUNDED+CORRECT -- first real win)
round 5: farm_selected=True,  R=10028 U=10019  (GROUNDED+CORRECT again)
round 6-300: farm_selected=True in nearly every remaining round
```

One early lucky/content-driven grounded hit (round 4) nudged `sources_100`'s reliability score up by a small amount. Because `rerank_with_reliability` min-max-normalizes reliability across only 3 sources (§8a's mechanistic note), that small absolute nudge was immediately stretched into a large *relative* reranking advantage — enough to keep `sources_100` in the selected top-K almost every round from then on, win or lose on any individual query's actual content match. Each additional grounded win reinforces the same advantage. **This is a self-reinforcing feedback loop in the reliability-weighted reranking mechanism, not a stable per-query win rate accumulating linearly.** A frozen-score, one-question-at-a-time read-only check cannot detect this by construction, because the one thing that drives it — the score itself moving between queries — is exactly what `update_scores=False` disables.

This also explains why the earlier flat n=50 real run (§8b, same seed, old single-blurb design) never took off: its round 1 was selected-but-not-grounded (same as this run's round 1 — same seed draws the same question sequence), and it simply never got a *second* qualifying win before the 50 rounds ran out, so it never crossed the threshold into the feedback loop. This run's sub-topic design got one at round 4. **Content design does still matter — for the probability of landing that first seed hit early enough to matter — but the payoff is not a smooth win-rate improvement, it's whether the threshold gets crossed at all.** Expect the eventual multi-seed campaign (§9 step 6) to be closer to bimodal (some seeds catch and escalate dramatically, some stay flat near baseline) than smoothly distributed, which is itself worth reporting rather than averaging away.

**The accuracy drop (60%→52%) is now explained mechanistically, not just observed:** the flip diagnostic shows 5 regressions, **0 of which have gold context in `sources_100`** (`sources_20` in all 5). This rules out the earlier displacement hypothesis (filler directly crowding out `sources_100`'s own correct content) and confirms a different, cleaner mechanism instead: **crowd-out**, exactly as named in §6. `top_k=3` is a fixed, shared budget across all three sources' candidates. Once `sources_100`'s inflated reliability score wins it more of those 3 slots almost every round, `sources_20`'s genuinely relevant candidates get squeezed out of slots they would otherwise have won on merit — even though the filler itself never competes with or resembles that content. This is a stronger, more mechanistically precise result than "F1 stays flat" would have been, and should be reported as the actual finding rather than forced to fit §6's original flat-accuracy expectation.

**Consequences for the rest of this plan:**
- Testing/validating this attack must use real, multi-round, `update_scores=True` runs. Read-only checks are useful only for the narrow question "does this filler survive retrieval at all under a fixed baseline" (§4a's original purpose) -- they cannot evaluate whether a design change helps the actual attack, because the actual attack's success is a threshold/feedback phenomenon that doesn't exist in a frozen-score test.
- The three content-design bug fixes from §8c were not wasted even though they scored 0/50 read-only both before and after -- one of them (or the interaction of the sub-topic design generally) is a plausible reason this run's round 4 landed where the old design's round 1-50 never did. Not proven in isolation (would need repeated real runs / more seeds to attribute cleanly), but consistent with "content still matters for reaching the seed hit."
- Multi-seed reporting (§9 step 6) should report catch/no-catch as an explicit outcome per seed, not just a mean score-vs-merit gap across seeds -- averaging a 96%-engagement run with a 2%-engagement run would produce a misleading middle number that describes neither.
- The chain is no longer pristine after this run (`sources_100`: R10155/U11183, `sources_0`: R9985/U9911, `sources_20`: R9969/U10027) -- meaningful, real evidence this time, not a throwaway validation artifact. `sources_100`'s retriever content is confirmed reset to clean (500/500/0); the on-chain state is intentionally left as-is pending the next decision.

**Blockchain state note:** before this run, the Hardhat chain was pristine (10000/10000, zero events) — confirmed live immediately beforehand. It no longer is: all three sources now hold real accumulated state from this run (`sources_0`: R10006/U9999, `sources_20`: R9981/U10002, `sources_100`: R10056/U10051). Any future run building on this will compound with this state, not start fresh; restarting the `hardhat-node` container is still the reset path if a clean 10000/10000 baseline is wanted before the dedicated multi-seed campaign (§9 step 6).

### 8b. Accuracy-drop investigation (resolved: was a small-sample artifact, but exposed a bigger issue)

Restarted `hardhat-node` for a clean 10000/10000 baseline (confirmed live, and confirmed the LLM service reconnects fine after a Hardhat restart -- no cascading restart of other containers needed), then re-ran with a larger, differently-sampled question set: `EVAL_SAMPLE_SIZE=50, ROUNDS=50`, same seed 42, same target. Added per-question diagnostics to `run_grounding_farming.py` (`gold_context_in` per question -- which of sources_20/100 actually holds its gold context -- plus a baseline-vs-post flip list) to test the run's own displacement hypothesis rather than assume it.

**Result: accuracy did not replicate the drop.** 60.0% baseline -> 58.0% post (30/50 -> 29/50, net 1 question), against 55%->45% (net 2 questions) in the n=20 run. At this sample size the two are statistically indistinguishable -- **the original accuracy drop was very likely a small-sample artifact, not a reproducible effect of the attack.** Of the 3 questions that did flip, only 1 regression's gold context was even in the target source, and one flip was actually an *improvement* -- no concentration pattern supporting the displacement hypothesis either.

**A bigger, unplanned finding surfaced in the process:** the farming mechanism's own engagement rate collapsed between the two runs -- `farm_source_in_importance_score` was 95% (19/20 rounds) in the n=20 run but only **2% (1/50 rounds)** in the n=50 run, using the same 10 domain-informed filler docs and the same target source. This was checked against the obvious confound (a few repeated favorable questions dominating the round count via sampling-with-replacement) and **ruled out**: the n=20 run's rounds covered 13 distinct questions, 12 of which independently showed `farm_selected=True`, so the high rate wasn't a couple of lucky repeats inflating the average -- it was genuinely broad across that draw's specific question set.

The real explanation: `EVAL_SAMPLE_SIZE=20` and `=50` draw two largely different sets of questions from the matched pool (not nested samples), and the 10 domain-informed filler docs are each one *generic, article-level* "background information about X" blurb (§5's `DOMAIN_TEMPLATES`). That generic framing apparently competes well against broad/introductory-sounding questions (which the n=20 draw happened to be unusually rich in) but not against the specific-paragraph-level facts that make up most of a real, diverse question sample -- which is most of what the n=50 draw added. This is consistent with, not contradicting, the original §4a empirical check's ~5% would-trip-grounding rate for the domain-informed strategy -- it's the n=20 real-attack run's 95% that was the outlier, not this one.

**Implications, both need to carry forward:**
- Do not report the n=20 run's numbers (score-vs-merit gap, misdirection rate, or the accuracy delta) as representative of anything -- they reflect one small, apparently favorable draw, exactly the single-seed pitfall `.claude/CLAUDE.md` already warns about, now demonstrated concretely rather than theoretically.
- The content design likely needs another iteration before the real campaign: multiple filler docs per article targeting different sub-facts/paragraph-types (biographical details, specific events, dates, awards -- mirroring real SQuAD paragraph diversity), not one generic per-article blurb, to raise coverage against the full diversity of real questions rather than only the generic-sounding subset. This sharpens the same conclusion §4a already flagged ("further breadth cannot come from more titles, only from more content-density or sub-topic chunks per title") from a hypothesis into something a real run now shows matters a lot.
- For the eventual multi-seed campaign (§9 step 6): use a large, representative `EVAL_SAMPLE_SIZE` (not 20) and enough `ROUNDS` that per-seed variance from question-mix luck averages out, and report variance across seeds 0/42/123 as already required -- this pair of runs shows that variance is not a formality here, it changes the qualitative conclusion.

### 8e. Honest control source: considered, decided against

A dedicated 4th data-source-equivalent instance (§5 step 6, §8 risk #4) was reconsidered after §8a-§8d and dropped, for two reasons:

1. **The original justification for it no longer holds.** The main reason to want a source topology larger than `n_retrievers` was to un-degenerate the sampling-level misdirection metric. But `farm_source_in_importance_score` (the reliability-weighted rerank-survival metric) was already established as the correct, non-degenerate signal to report instead (§8a), and §8d's 96.3% result shows it carries the whole finding on its own -- there's no metric gap left for a 4th source to close.
2. **Building it would risk exactly the kind of change `.claude/CLAUDE.md` warns against.** For a new source to actually affect sampling (and thus be more than decoration), the orchestrator's own configuration would need to change too -- either lowering `n_retrievers` below the new total source count, or adding the source to `data_sources` in `drag_llm_service/configs/config.yaml`. Both are changes to the *evaluated system's* topology/configuration, not to the attack layer sitting on top of it -- in tension with "we are not modifying Reliable-dRAG itself" and "evaluate the system as-is." It would also mean Reliable-dRAG's SSM results no longer run on the same reference 3-source topology the paper itself uses, complicating the DRAG-vs-Reliable-dRAG side-by-side comparison table `.claude/CLAUDE.md` calls for. And practically: one more container is one more thing to debug in an environment that has already produced real, non-obvious failures this session (the API-key auth gap, the Unicode encoding crash) -- added surface area for a benefit that's no longer needed.

**Decision: use `sources_20` as the comparison source instead of building anything new.** It is already untouched by the attack, already same-domain (SQuAD) and comparable in size to `sources_100`, and already receives the identical real query stream every run -- §8a and §8d both already computed the score-vs-merit gap against it and got a real, usable signal (e.g. §8d: `sources_20` R-31/U+27 while `sources_100` gained R+155/U+1183) without standing up anything. No further action needed here; §5 step 6, §6, and §8 risk #4 above are updated to point at this decision.

---

## 8f. Multi-seed campaign results (§9 step 6, completed) — the bimodal hypothesis, confirmed cleanly

Full campaign: seeds **0, 42, 123**, `ROUNDS=300`, `EVAL_SAMPLE_SIZE=100` (up from the 50 used in earlier diagnostic runs — 2,822 matched questions are actually available, so 100 was chosen to stay representative without pushing per-seed runtime past ~20 minutes), target `sources_100`, comparison `sources_20`. Each seed ran against a freshly-restarted, confirmed-pristine chain (10000/10000, verified before every seed) so the three are cleanly comparable and don't compound on each other. Logs: `attack_logs/attack_2026-07-24_23-34-24_grounding_farming_seed0.json`, `..._seed42.json`, `..._seed123.json`.

**One operational bug found and fixed along the way:** seed 0's first attempt crashed on the same Unicode console-encoding issue diagnosed in §8c (a SQuAD answer containing "Ō" broke the default Windows codepage mid-print) — except this time in the production script, not the scratchpad checker, because the `sys.stdout.reconfigure(encoding="utf-8")` fix had only been applied to the checker, not ported back to `run_grounding_farming.py`. The `try/finally` reset still ran correctly (retriever confirmed clean afterward), but the crash happened before the JSON log was written, so seed 0's first attempt produced no artifact. Fixed at the source (`run_grounding_farming.py`, module-level `sys.stdout.reconfigure`) and re-ran seed 0 cleanly on a fresh chain.

**Results:**

| Seed | `farm_selected` rate | R delta (`sources_100`) | U delta (`sources_100`) | Gap vs `sources_20` (R/U) | Accuracy (baseline→post) | Outcome |
|---|---|---|---|---|---|---|
| 0 | **100.0%** (300/300) | **+680** | **+1574** | **+680 / +1574** | 66%→57% (**-9pp**) | **Full escalation** |
| 42 | 0.3% (1/300) | -20 | +20 | -785 / -1122 | 57%→59% (+2pp) | No catch |
| 123 | 11.0% (33/300) | -4 | +216 | -581 / -707 | 59%→60% (+1pp) | No catch (marginal engagement) |

**Catch rate: 1/3 seeds (33%).** This is the cleanest possible confirmation of §8d's bimodal-escalation hypothesis, now with a real number attached instead of a guess from two data points. Averaging across seeds would be actively misleading here: mean R delta = +219 (stdev 400 — larger than the mean), mean gap vs `sources_20` = -229 (stdev 794) — a single averaged number describes none of the three actual outcomes and must not be reported as "the" result. **Report catch rate (1/3) and the per-seed distribution, not a mean.**

**The accuracy cost is conditional on catching, cleanly and consistently across all three seeds — this is the strongest confirmation yet of the crowd-out mechanism from §8d:** the one seed that escalated (0) is also the only one with a real accuracy drop (-9pp, the largest yet observed, larger than §8d's -8pp). The two seeds that didn't catch show accuracy staying flat or *improving slightly* (+2pp, +1pp) — exactly what §6's original expectation predicted, but now understood as conditional on the attack failing to gain traction, not as a general property of the attack. This is a genuinely strong result for the write-up: **F1/accuracy stays flat when the attack doesn't catch, and drops meaningfully when it does** — a precise, mechanistically-grounded, three-seed-confirmed claim, stronger than either the original "stays flat" hypothesis or a blanket "always degrades" claim would have been.

**What this means for the thesis framing:** grounding-farming should be reported as a **real, demonstrated, but probabilistic** attack — capable of a dramatic, self-reinforcing score-vs-merit gap and real accuracy cost via crowd-out when an early grounding win lands, but not guaranteed to trigger on any given deployment/query-mix. This is a more defensible and more interesting claim than either "always works" (contradicted by 2/3 seeds) or "the content design doesn't matter" (contradicted by seed 0's clean 100% escalation) — the honest finding is that the *system's own* reliability-weighted reranking mechanism is what turns a modest initial advantage into a runaway one when the dice land right, which is itself the generalizable, structural point the SSM attack class is supposed to make (per `.claude/CLAUDE.md`'s framing: the vulnerability is a direct consequence of the architectural decision to distribute trust across sources that must self-report, not a fragile artifact of one particular content strategy).

**Open follow-up, not blocking:** with only 3 seeds, "1/3" has wide uncertainty (true catch rate could plausibly be anywhere from ~10% to ~60%). Running a handful more seeds (e.g. 5-10 total) would tighten this estimate for the final write-up, but 3 already satisfies `.claude/CLAUDE.md`'s minimum and produces a reportable, honest number; more seeds is a nice-to-have, not a requirement, given the time cost (~20 min/seed).

---

## 9. Suggested implementation order

1. ✅ Fix SSM's corpus-matching to use per-source ground truth instead of the shared `sources_0.jsonl` (§7.2/§7a).
2. ✅ Empirical check: does broad filler content actually get retrieved/selected at meaningful rates? (§8, risk 2; §4a).
3. ✅ Implement `GroundingFarmingAttack` + filler-content generator (§4-5), validated with real (`update_scores=True`) runs against the live stack (§8a, §8b, §8c, §8d) — including discovering and fixing three content-generation bugs and one crash-safety gap along the way.
4. ~~Stand up the honest control source~~ — **dropped, see §8e.** `sources_20` is the comparison source; no new infrastructure.
5. ✅ Metrics implemented in the run/eval script (§6, as revised by §8a/§8d): score-vs-merit gap, `farm_source_in_importance_score` (not raw sampling), crowd-out via flip diagnostics.
6. ✅ Multi-seed runs (seeds 0, 42, 123) at scale (300 rounds, `EVAL_SAMPLE_SIZE=100`, real `update_scores=True` writes), logs committed to `attack_logs/` with seed embedded in the filename. Result: **1/3 catch rate**, reported per-seed rather than averaged — see §8f.
7. ✅ Re-ran and relabeled the key-forgery attack as the secondary control-plane finding (§7b) — including fixing a defense-interaction bug (old defaults reverted against the real on-chain caps) and an unrelated data-source rate-limit bug that contaminated the first campaign attempt. Result: **3/3 seeds fully succeeded** (deterministic, 5/5 rounds every time, R/U +20,000), accuracy drop 10pp/10pp/0pp across seeds 0/42/123.
8. **NEXT.** Update `reports/Security_Analysis_Report.md` and `.claude/thesis_update.md`'s SSM section to reflect the two-attack structure, the corrected threat model language, and the feedback-loop/crowd-out mechanism (§8d/§8f) as the flagship finding — including the 1/3 catch rate (not a smoothed average) alongside §7b's deterministic-but-higher-privilege key-forgery contrast.

---

## 10. Definition of done

- ✅ `GroundingFarmingAttack` runs end-to-end against the live docker-compose stack and produces committed JSON logs under `attack_logs/`, seeds 0/42/123 (§8f).
- ✅ Reported metrics are score-vs-merit gap, `farm_source_in_importance_score` (misdirection), and crowd-out effect — not F1 degradation as the headline, and not raw sampling-inclusion (degenerate in this topology, §8a).
- ✅ Per-seed catch/no-catch outcome is reported explicitly (§8f: 1/3 caught), not smoothed into a single averaged score-vs-merit gap across seeds (§8f shows the mean/stdev explicitly to demonstrate why averaging would mislead).
- ✅ Accuracy/answer-quality is reported as what the multi-seed numbers actually show — confirmed conditional on catching (-9pp when caught, seed 0; +1 to +2pp when not, seeds 42/123) — this is the finding to report, not the original flat-F1 expectation.
- ✅ The key-forgery attack is re-run post-corpus-fix, multi-seed (seeds 0/42/123, n=50), relabeled as control-plane/orchestrator-key-compromise in code (§7b). 3/3 seeds fully succeeded (deterministic), accuracy drop 10pp/10pp/0pp.
- ⬜ `reports/Security_Analysis_Report.md` §1.3 (Threat Model) and `thesis_update.md`'s SSM section no longer claim "no privileged access" for the key-forgery mechanism, and correctly attribute that claim to the grounding-farming mechanism instead. (Still open — §9 step 8.)
- No dedicated 4th "honest control" data source is built (§8e) — comparison uses `sources_20`.
- ✅ Two operational bugs found and fixed during §7b's re-run, both worth carrying into the report as methodology notes: (1) the pre-existing on-chain defense's `MAX_DELTA_PER_UPDATE`/`MIN_UPDATE_INTERVAL` fully blocks the original reckless key-forgery parameters, requiring defense-aware (cap-respecting) throttling to get a meaningful result; (2) an unthrottled `measure_accuracy()` query loop can trip the data sources' 60/min-per-IP rate limit and silently degrade into empty-prediction noise that looks like an accuracy effect but isn't — checked and ruled out for the already-reported Grounding-Farming logs, fixed going forward via query pacing.
