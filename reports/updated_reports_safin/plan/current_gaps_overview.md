# Framework Gaps Overview — All Implementations (Attacks + Defenses)

**As of:** 2026-09-21 · **Target:** NAACL submission, Aug 12 · **Systems:** DRAG (gossip P2P) + Reliable-dRAG (blockchain reliability)

Consolidated, current-state gap analysis across all six CIA-triad attacks and their defenses. Supersedes the per-attack `problems/*_gaps.md` files as the single planning view; those remain the authoritative detail for each attack. Where a claim below was verified against code/logs this pass it is marked **[verified]**; where it is taken from the team's own reports/gaps files it is marked **[reported]**; inferences are marked **[inferred]**.

Read this top-down: the framework-level gaps (Part 1) are what a NAACL reviewer hits first and matter more than any single attack's residuals. Part 2 is per-attack/defense status. Part 3 is the prioritized run plan for the deadline.

**Status legend:** ✅ done & validated · 🟡 done but small-sample / caveated · 🟠 code written, **not run** · 🔴 open / missing · ⚪ framing-only (no code needed)

---

## Part 1 — Framework-level gaps (highest priority, reviewer-facing)

### F1. Cross-architecture experimental asymmetry 🔴 ⚪
DRAG was run with 3 LLMs × 3 datasets (1 MCQA + 2 open-generation QA) from the original DRAG codebase. Reliable-dRAG's security work uses **one** model (Qwen2.5-1.5B-Instruct) and PubMedQA/SQuAD. The **original Reliable-dRAG paper used two Llama models (3B, 8B) and one dataset (Natural Questions)** — so the current Reliable-dRAG runs match *neither* the DRAG side nor the paper. **[verified against the paper]** This is the single most likely reviewer objection.
- A quantitative head-to-head across systems is **not defensible** as-is (model/dataset/task all differ → any delta is confounded).
- A *framework generalizability* claim is defensible **if** framed as architectural coverage (two trust mechanisms) and disclosed, not as a benchmark comparison.
- Fix: reframe (free) **+** one shared bridge run (same LLM on Natural Questions across both systems) **+** one second LLM on the flagship Reliable-dRAG attacks. See Part 3.

### F2. Three-source topology confounds SFA/DoS (and MIA reliability) 🔴 → in progress
Almost every "odd" Availability result traces to `n_sources = 3`:
- **SFA:** at full redundancy (`max_hops = 3 = n_sources`) the attack has *no* measurable effect; it only bites at hop-limited `max_hops=1`. The whole result is a 3-source artifact. **[reported]**
- **DoS:** "zero redundancy margin" — losing 1 source = 0% failure, 2 = 20%, 3 = 90%. The dose-response is entirely a function of 3. **[reported]**
- **MIA:** reliability min-max normalization over 3 sources; on-chain scores all tie at baseline so the reliability term contributes nothing. **[verified]**
- Fix: **increasing source count (already planned).** This is the highest-value infra change — it retires a large fraction of the honest caveats in the SFA and DoS reports at once. Prioritize SFA/DoS re-runs at the new count.

### F3. Reliable-dRAG live results are largely single-seed / small-sample 🟡
- MIA: 25+25 docs/seed. DoS live flood: single-seed. SFA live: n=30 single-seed (defense), n=200 3-seed (one attack config only). KB defense comparison: single-seed. **[reported/verified]**
- The mock-mode sides are mostly 3-seed (0/42/123); the **live** sides lag. Deadline plan must budget multi-seed live runs on the powerful machine.

### F4. Round-2 fixes are code-only, not executed 🟠
The most recent teammate pass (calibration + rate-limit hardening + defense bug fixes) is written and `py_compile`-checked but **never run against the live stack**, by explicit instruction. Nothing downstream of these is measured:
- MIA: Rev10 calibrated score, `no_retrieval` baseline, `_query_llm_raw` throttle. **[verified]**
- DoS: `LiveClientDefense`, quorum-cap enforcement fix, false-block metric fix, `--seeds`. **[reported]**
- SFA: `honest_miss_rate="auto"`, realistic attackers, `run_realistic_defense_eval`. **[reported]**
- Every current headline number predates these; re-running is required before any of them is citable.

### F5. Docker deployment fragility 🔴
Recurring across MIA, KB, and Data Poisoning: a stale bind-mount silently reverts data sources to the polluted corpus, and Docker Desktop restarts mid-sweep revert to `docker-compose.yml`'s base config. Multiple past results were corrupted this way before being caught. **[reported]**
- Fix before any long unattended run: make the intended corpus the actual compose default for the run's duration (not an override file), and assert corpus identity at run start. This is cheap insurance against losing an overnight run near the deadline.

### F6. Defenses were scoped as "future work" but implemented anyway ⚪
`.claude/CLAUDE.md` scopes defenses out for Reliable-dRAG; they were built on explicit instruction. Not a technical gap, but the thesis text must resolve the framing (the defense work is a contribution, not an accidental scope creep) so a reviewer doesn't read the CLAUDE.md note as an admission. **[reported]**

---

## Part 2 — Per-attack and per-defense status

### 1. Data Poisoning (Integrity) — ✅ attack / ✅ defense (Cross-Peer Validation)
Heavily audited; most implementation bugs resolved with verified re-runs. **[reported]** Open items:
- **A2 (open):** the reliability feedback loop never engages on a normal `/query` (scores stay 0), so the paper's adaptive mechanism is inert during evaluation. Not blocking current DP strategies, but it means the experiment tests "poisoning vs. the reliability feature switched off." Matters for the thesis claim and is a hard dependency for SSM. 🔴
- **Framing:** the best-defended claim is *adversarial content vs. the paper's random-noise-only threat model*, or the *cold-start / pre-convergence window* (which the paper itself flags as open). State one explicitly. ⚪
- **Caveated, not fixed:** n=7 eval set (one flipped answer = 14.3 pts); substring correctness metric (negation false-positives); attack not volume-stealthy (~3.2k→19.7k docs); B9/B10 historical numbers carry errata — re-run or cite the erratum. 🟡

### 2. Source Selection Manipulation (Integrity) — ✅ attack / 🟠🔴 defense (partial)
Two attacks, both multi-seed validated: **Grounding-Farming** (flagship, no privilege, probabilistic 3/6 catch) and **Key-Forgery** (deterministic, orchestrator-key compromise). **[reported]** Open items:
- **No defense for Grounding-Farming** — the on-chain caps defend only Key-Forgery; the flagship's input-layer mechanism (naive substring grounding check) is undefended. This is the report's #1 open item, and **the likely "6th" missing defense** (confirm — see the comment on this doc). 🔴
- On-chain caps stop *reckless* Key-Forgery bursts only; a patient, cap-respecting attacker defeats them 100%. Needs a cumulative-drift monitor. 🔴
- Grounding-Farming catch rate is run-level unstable (seed 123 flipped across identical runs); needs more independent trials to characterize. 🟡
- 429/500 rate-limit errors during SSM runs; round-2 throttle fix is **not run**. 🟠

### 3. Knowledge Base Extraction (Confidentiality) — ✅ attack / ✅ defense (QueryDiversityThrottle)
3-seed, gray-box + blind conditions; corpus-drift and bind-mount bugs fixed and re-verified. Defense calibrated against a simulated legitimate-user baseline. **[reported]** Open items:
- Defense comparison still **single-seed**. 🟡
- Throttle calibrated only for `source_1`; `source_0`/`source_2` use unvalidated thresholds. 🟡
- Phase C (LLM leakage) seed-0 remains error-heavy (6/20 scored) even post-fix. 🟡
- No single-victim targeting mode; threat model rests on the shared hardcoded API key. ⚪

### 4. Membership Inference (Confidentiality) — 🟡 attack / 🟠 defense
**Deeply reviewed this pass. [verified]** AUC ~0.62 (uncalibrated) / lower calibrated — LOW-to-MEDIUM, and the low score is **structural, not a bug** (coarse yes/no/maybe label the base model guesses ~45% right without retrieval). Full detail and fix plan in the **MIA Attack Fix Guide** (living doc). Open items:
- **Root cause is the coarse label.** The only real lever is a RAGLeak-style continuation/perplexity probe (a redesign), ideally on a free-text corpus (Natural Questions / HealthCareMagic). 🔴
- Rev10 calibration, `no_retrieval` baseline, and throttle: **code-only, not run.** 🟠
- Production weights (`decision_match`=1.0) never re-validated against the *calibrated* signal. 🟠
- Phase 2 contrastive collector uses an **unthrottled** `_query_llm_fast()` — running it at scale re-triggers the 429 storm Rev10 fixed. Fix before collecting. 🔴
- Prompt/metric mismatch: the system prompt never asks for yes/no/maybe, so `decision_match` under-fires. 🟡
- Defense (sanitize/obfuscate + Laplace similarity-noise): 13 unit tests pass; **live eval not run** (needs container restart per epsilon). 🟠
- Trained-attacker Phase 1 (0.6658) **is run and real [verified]**, but CI overlaps the linear baseline at n=5 — present as directional, not a clear win, and note it's on the *uncalibrated* signal.

### 5. Denial of Service (Availability) — 🟡 attack / 🟠 defense
Mock: 3-seed + hyperparameter sweep (monotonic dose-response). Live flood: single-seed, real 0%→90% failure curve. **[reported]** Open items:
- Live evaluation **single-seed**; round-2 `--seeds` added but **not run**. 🟠
- `DDoSDefense` (reputation/blacklist/redundant-probe) **never validated against real live congestion** — mock only. 🔴
- `LiveClientDefense` (token bucket / circuit breaker / queue / cache) is **new, uncalibrated params, not run.** 🟠
- Round-2 fixed two real defense bugs (quorum cap declared-but-never-enforced; false-block metric counting real blocks) — **code-only.** 🟠
- Layer asymmetry: both defenses wrap the eval client's retrieval calls, **not** `drag_llm_service`'s own request path — end-to-end answer quality is expected identical across defense conditions. Disclose. ⚪
- Baseline comparisons (centralized / replicated / load-aware) are mock-only; real load-aware routing / production WSGI / admission control absent from the live system. 🔴

### 6. Selective Forwarding (Availability) — 🟡 attack / 🟡 defense
Most mature (866-line report, 8 revisions); live-validated but small-sample. Defense (reputation/blacklist/bypass/redundant-probe) real. **[reported]** Open items:
- Result is a **3-source artifact** (see F2) — source expansion is the main fix. 🔴 → in progress
- Live validation small-sample (n=30 single-seed defense; n=200 3-seed for one attack config only). 🟡
- `honest_miss_rate="auto"` recalibration fix: **code-only, not run.** 🟠
- Binomial detector uses a growing window (not a true sliding window) → measurable false-positive gap vs. the sibling module. 🟡
- Realistic attackers (delay / drift / collusion / adaptive-to-threshold): mock smoke test only, **not run live**; `run_realistic_defense_eval` **not run at all.** 🟠
- Final **LLM answer-correctness** under SFA is unmeasured — the BFS sim is client-side and doesn't affect `drag_llm_service`'s own retrieval. Needs an HTTP interception layer to measure properly. 🔴

---

## Part 3 — Prioritized run plan to Aug 12

Ordered by (reviewer impact ÷ effort). Assumes the powerful-but-shared machine and the planned source-count increase. **Note:** if Aug 12 has already shifted on your calendar, tell me and I'll re-cut the priorities.

**Tier 0 — free, do first (framing, no compute)**
1. Reframe cross-architecture as framework generalizability; state per-system, never cross-system numbers (F1). ⚪
2. Resolve the defense-scope framing in the thesis text (F6). ⚪
3. Land the Docker corpus-identity safeguard before any long run (F5).

**Tier 1 — the source-count expansion (highest structural payoff)**
4. Stand up the larger source deployment; re-run **SFA** and **DoS** at the new count, multi-seed (F2, F3). This retires most Availability caveats.
5. Re-run the Data Poisoning matrix on the new deployment if the corpus setup changes (guard against A1's confounded-baseline class of bug).

**Tier 2 — execute the unrun round-2 code (turns "written" into "measured")**
6. Run the round-2 fixes end-to-end, small scale first: MIA (calibration + throttle), DoS (`--seeds`, `LiveClientDefense`, quorum/false-block fixes), SFA (`honest_miss_rate=auto`, realistic attackers live) (F4). Verify each acceptance check before scaling.

**Tier 3 — credibility breadth**
7. One **second LLM** on the flagship Reliable-dRAG attacks (SSM, SFA, DoS, KB) — kills the "1.5B artifact" critique. Not all six attacks × both models.
8. One **shared bridge run** (same LLM on Natural Questions across DRAG and Reliable-dRAG) for one attack, to license a single controlled comparison (F1).

**Tier 4 — attack-specific**
9. Build the **SSM Grounding-Farming defense** (the missing 6th defense) — provenance verification, not substring presence — plus a cumulative-drift monitor for Key-Forgery.
10. **MIA redesign** (continuation probe on free-text corpus) — the only path to a non-trivial MIA number; also pulls MIA toward the paper's native task. Fix the Phase 2 throttle bug before any contrastive collection.

**Explicitly deferred (out of reach by deadline, state as future work):** full 3×3 matrix on Reliable-dRAG; HTTP interception for real answer-quality-under-SFA; production WSGI / load-aware routing in the live system.

---

*Sources: verified this pass against `attack/Mia_attack/*`, `drag_llm_service/*`, `attack_logs/*`, and the Reliable-dRAG paper (`ReliabledRAG.pdf`); other attacks from the team's own `problems/*_gaps.md`, `reports/updated_reports_safin/*`, and `reports/*_Security_Analysis_Report.md`.*
