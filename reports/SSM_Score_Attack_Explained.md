# SSM-Score Attack, Explained

### Source Selection Manipulation on Reliable-dRAG — in plain terms, and for the team

**What this document is:** a companion to `reports/Security_Analysis_Report.md` (the full technical report) and `.claude/ssm_grounding_farming_plan.md` (the detailed implementation/investigation log). Those documents assume you already know what a smart contract, a reranker, and an AUC score are. This one doesn't. It exists so that anyone — a committee member skimming for five minutes, a teammate who worked on a different attack, or future-you in three months — can understand what this attack actually does and why it matters, without first reading fifty pages of tables.

**What this attack targets:** Reliable-dRAG is a "distributed RAG" system — instead of one central database answering your questions, three independent data sources each hold their own slice of documents, and an orchestrator asks all three, decides which answers to trust, and stitches together a final response. To help make that decision, the system keeps a running *trust score* for each source — like a report card — that goes up when a source's content turns out to be helpful and correct. **SSM-Score (Source Selection Manipulation)** is the name for any attack that games this report card. This project found and validated two different ways to do that, at two very different levels of difficulty.

---

## Part 1 — The Plain-Language Version

Imagine the system as three research assistants — call them A, B, and C — who each keep their own filing cabinet of reference documents. When you ask a question, all three assistants dig through their own cabinet and hand their manager whatever documents they think are relevant. The manager reads what's handed over, writes an answer, and — this is the important part — keeps a report card on each assistant: *did the documents you gave me actually turn out to be useful and correct this time?* Assistants with a better report card get listened to more in the future. It's a completely reasonable system — until you notice how the manager actually checks the report card.

### Cheat #1: gaming a lazy grader (the main attack)

The manager's way of checking "was this assistant's document actually useful" turns out to be lazy: it just checks whether the *final written answer* happens to appear, word-for-word, somewhere inside the documents the assistant handed over. It never checks whether the assistant's documents were the *reason* the answer came out that way.

One assistant figures this out. Instead of handing over carefully chosen, relevant documents, they start handing over big, generic binders stuffed with all kinds of names, dates, numbers, and common phrases loosely related to the topics the office usually deals with — a shotgun approach. Most of the time this doesn't help, because the manager still writes the correct answer using someone else's better, more relevant documents. But every so often, purely by coincidence, the final answer's wording happens to also appear somewhere in the shotgun binder — and the lazy grader gives that assistant full credit for a document that had nothing to do with the real answer.

Here's the twist we found by actually running this for real: **one lucky "credit" like that isn't just a one-time freebie — it snowballs.** The moment that assistant's report card ticks up even slightly, the manager starts trusting them a *little* more on the next question. Trusting them more means their binder gets glanced at more often, which means more chances to coincidentally match another answer, which nudges the report card up again — and because the manager only compares report cards among these three assistants, even a small edge gets blown up into a big one. Once this snowball starts, it tends to keep rolling. If it never gets that first lucky break, though, nothing happens at all — the cheating assistant just looks average forever.

We tested this cheating strategy three separate times (like flipping a coin three times, but for a multi-week experiment instead of a coin). **It fully took off once, and completely fizzled twice.** So this is a real, and when it works, a seriously damaging trick — but it's a gamble, not a guaranteed win. And when it does take off, real damage follows: because the manager only has room to seriously consider a handful of documents per question, the cheating assistant's inflated reputation ends up shoving out a *genuinely* helpful assistant's documents on unrelated questions — meaning honest users get worse answers as a side effect, even though the shotgun binder itself never resembled the correct answer.

### Cheat #2: breaking into the manager's office (the other attack)

The second way to cheat is completely different and much more drastic: instead of gaming what the manager notices, you break into the manager's own office and rewrite an assistant's report card by hand. This requires stealing the *manager's* own keys, not just being a sneaky assistant — a much bigger crime, closer to "compromised the person in charge" than "one employee cut corners."

We also tested this. Unlike the shotgun-binder trick, this one is **not a gamble at all — it works every single time**, because you're not waiting for a lucky coincidence, you're just writing whatever number you want directly onto the report card. The office does have a rule limiting how much any one report-card edit can move the needle at once (to stop someone from writing "PERFECT" in one go) — but an attacker who's willing to make several smaller edits instead of one big one gets around that rule completely, every time we tried it.

### The bottom line, in plain terms

There are two ways to cheat this system, and they trade off against each other exactly the way you'd expect: the low-effort trick (gaming the grader) needs almost no special access but only works about a third of the time; the high-effort trick (breaking into the office) needs a much bigger, harder-to-get level of access, but then it works every time, guaranteed. That trade-off — "easy but unreliable" versus "hard but certain" — is itself the interesting finding: it shows that the *reason* the system is vulnerable isn't a one-off coding mistake, it's baked into the basic idea of "let each assistant partly grade their own performance and trust the aggregate." Any system built that way has to reckon with both ends of this trade-off, not just one.

---

## Part 2 — Semi-Technical Version (for the team)

### The mechanism both attacks share

Reliable-dRAG keeps two on-chain numbers per data source — reliability (`R_i`) and usefulness (`U_i`) — updated via a smart contract (`DragScores.sol`) after every query. The orchestrator's reranker blends semantic relevance with these scores:

$$
\text{score}(c) = (1-w)\cdot\text{sim}(q, c) + w \cdot \hat{R}(\text{source}(c))
$$

`\hat{R}` is **min-max normalized across only the three participating sources.** That detail is load-bearing for everything below: normalizing across so few sources means even a small absolute score gain gets stretched into a large *relative* reranking advantage.

### Attack A — Grounding-Farming (flagship, no privileged access)

**Where it lives:** `attack/ssm_score/grounding_farming_attack.py` + `run_grounding_farming.py`.

The orchestrator decides whether a source "grounded" the final answer with a naive literal-substring check (`drag_llm_service/app/server.py:856-858`):

```python
grounded_by_sources = [norm_response in ctx for ctx in selected_contexts_norm]
```

This checks only whether the model's normalized answer text appears as a contiguous token run *somewhere* inside a source's returned chunk — not whether that chunk caused the answer. `GroundingFarmingAttack` exploits this by injecting broad, domain-informed filler documents (one keyword cluster per real article title in the target corpus, built via TF-IDF-style discriminative scoring so the filler stays on-topic without quoting real answers) via the same `/poison` endpoint Data Poisoning uses — no blockchain interaction, no key beyond the source's own.

**The mechanism we found empirically, not assumed:** a single early, essentially-lucky "grounded" credit nudges the farming source's score up slightly. Because of the min-max normalization above, that nudge is immediately amplified into a near-total reranking advantage, so the source keeps winning the shared top-k slot in nearly every subsequent round — a **self-reinforcing feedback loop**, not a stable per-query win rate. This is why a frozen-score (`update_scores=False`) read-only test consistently measured 0-2% "would-ground" rates for content that, in a real run, drove 96-100% engagement once it crossed the threshold — the read-only test structurally cannot see a dynamic that depends on the score actually moving between queries.

**The accuracy cost, when it escalates, is crowd-out, not content collision:** flip-diagnostic inspection showed zero regressions where the farming source's own filler displaced its own correct content — every regression's gold context lived in the untouched comparison source (`sources_20`). The mechanism is that `top_k` is a small, fixed, shared budget across all three sources' candidates; once the farming source wins more of that budget almost every round, a genuinely relevant source's candidates get squeezed out of slots they'd have won on merit.

**Why the existing on-chain defense (`onlyLLMService` gate, `MAX_DELTA_PER_UPDATE`, `MIN_UPDATE_INTERVAL`, absolute bounds) does not stop this at all:** every transaction this attack produces is submitted by the real, legitimately-authorized orchestrator, individually within every bound. The defense hardens the *transaction layer* (who may write, how fast, how much); this attack corrupts the *input layer* (what the write is even based on). Confirmed empirically: zero reverts, zero flags, across every round of every seed.

### Attack B — Key-Forgery (secondary, orchestrator-key-compromise)

**Where it lives:** `attack/ssm_score/ssm_score_attack.py` + `run_attack.py`.

`feedbackAndUpdateScoreRecords` is gated `onlyLLMService`. `SSMScoreAttack.inflate_scores()` submits a forged update using the `llm_service` private key as transaction sender, signing a fixed probe message with the target source's own key. The contract's signature check (`message_dict`, `drag_llm_service/app/server.py:944-947`) covers only `{query, selected_sources}` — **never the score values themselves** — so a caller who already holds the orchestrator key can attach any source's previously-valid signature to entirely fabricated `reliability`/`usefulness` numbers.

This requires possession of the orchestrator's own key — a materially higher privilege than "controls a data source." It is **not** the "regular participant, no privileged access" SSM story the framework's thesis language originally described; that gap is why Grounding-Farming was built and is now the flagship instantiation, with Key-Forgery relabeled as a secondary, higher-privilege finding.

**The defense-interaction finding that mattered here:** the original attack parameters (`AMPLIFY=999,999`, 0.5s between rounds) now revert on round 1 against the deployed `MAX_DELTA_PER_UPDATE=5,000` / `MIN_UPDATE_INTERVAL=2s` caps — those parameters were never a tuned attack, just the first thing tried. A defense-aware attacker who throttles to just under both caps (`AMPLIFY=4,000`, 2.5s delay) gets **every single round accepted, every seed, deterministically** — the caps stop reckless bursts, not patient, cap-respecting abuse.

---

## Part 3 — Findings and Their Significance

### Multi-seed results (seeds 0, 42, 123, per `.claude/CLAUDE.md`'s requirement)

**Grounding-Farming** (n=100 questions, 300 real query rounds/seed):

| Seed | `farm_source_in_importance_score` | Score-vs-merit gap (R/U vs. untouched source) | Accuracy | Outcome |
|---|---|---|---|---|
| 0 | 100.0% | +680 / +1,574 | 66%→57% (-9pp) | **Full escalation** |
| 42 | 0.3% | -785 / -1,122 | 57%→59% (+2pp) | No catch |
| 123 | 11.0% | -581 / -707 | 59%→60% (+1pp) | No catch |

**Catch rate: 1/3 (33%).** Averaging these numbers would be actively misleading (mean R-gap -229, stdev 794, larger than the mean) — the honest way to report this is catch/no-catch per seed, not a single blended statistic.

**Key-Forgery** (n=50 questions, 5 rounds/seed, defense-aware parameters):

| Seed | Rounds accepted | Score delta (R/U) | Accuracy | Outcome |
|---|---|---|---|---|
| 0 | 5/5 | +20,000 / +20,000 | 66%→56% (-10pp) | Full success |
| 42 | 5/5 | +20,000 / +20,000 | 62%→52% (-10pp) | Full success |
| 123 | 5/5 | +20,000 / +20,000 | 66%→66% (0pp) | Full success |

**Success rate: 3/3 (100%), deterministic.**

### What these numbers mean

1. **The framework's core SSM claim now has a real, working, no-privileged-access instantiation on this system** — something Grounding-Farming provides that the original Key-Forgery implementation did not. That matters directly for the thesis's generalizability argument: DRAG's SSM (a lone peer faking topic advertisements) and Reliable-dRAG's SSM (a lone source faking grounding credit) are now structurally comparable — same privilege level, same "regular participant games a self-reported trust signal" shape — which is exactly the claim the framework needs to hold across architectures.

2. **A probabilistic result is a *stronger*, more defensible finding than a clean 100% would have been**, because we can explain precisely *why* it's probabilistic (a feedback-loop threshold effect) rather than shrugging at inconsistent numbers. It also means real deployments can't dismiss this as "vanishingly rare" — a 1-in-3 chance of a 9-point accuracy hit and a five-figure score distortion is a serious risk in any realistic threat model.

3. **The existing on-chain SSM defense is not "solved, move on"** — it was built for, and is genuinely effective against, one specific attacker profile (reckless, high-magnitude, high-frequency forged updates). Against Grounding-Farming it provides zero protection by construction, and against a merely patient Key-Forgery attacker it provides zero protection empirically. This is the clearest illustration in this whole project of a general security principle: a defense that stops the first attack you throw at it is not the same as a defense that stops the attack *class*.

4. **The two attacks' contrast — probabilistic/low-privilege vs. deterministic/high-privilege — is itself the generalizable, structural point.** It isn't an artifact of this particular corpus or this particular filler-content design; it falls directly out of the architectural decision to distribute trust across sources that must self-report, combined with a small-source-count reranker that amplifies small score changes. Any distributed RAG system built the same way inherits this same shape of vulnerability.

### Where this leaves the project

Grounding-Farming is the flagship SSM result for Reliable-dRAG going forward; Key-Forgery remains a real, secondary, correctly-labeled control-plane finding. No defense currently exists for Grounding-Farming's actual mechanism (the naive grounding check) — this is the single highest-priority open item for this attack surface (see `reports/Security_Analysis_Report.md` §14, recommendation 1). Full technical detail, including every dead end and bug found along the way, is preserved in `.claude/ssm_grounding_farming_plan.md`.
