# Project Security Analysis Report

**Project:** Reliable-dRAG — A Decentralized Retrieval-Augmented Generation System with Blockchain-Secured Source Reliability
**Scope of this report:** Source Selection Manipulation (SSM-Score), Membership Inference Attack (MIA), and Selective Forwarding Attack (SFA) — attack implementations, defenses, and live-verified evaluation results.
**Report type:** Security analysis / thesis-support technical documentation
**Status:** All findings in this report are derived from reading the actual source code and from JSON logs produced by real executions against the live system (Docker containers `drag-hardhat-node`, `drag-llm-service`, `drag-data-source-0/20/100`). Where a claim could not be verified this way, it is explicitly marked as an assumption.

---

## Executive Summary

Reliable-dRAG is a decentralized Retrieval-Augmented Generation (RAG) system. Instead of one organization owning the entire knowledge base, **three independent data sources** (`sources_0`, `sources_20`, `sources_100`) each hold a partial, sometimes-polluted copy of a document corpus, and an **LLM orchestrator service** queries all of them, reranks their answers by an on-chain **reliability/usefulness score**, and generates a final response. The trust scores themselves live on a private Ethereum-compatible blockchain (Hardhat) inside a smart contract called `DragScores`.

This report analyzes three attacks against this architecture, one per leg of the CIA triad's *Integrity*, *Confidentiality*, and *Availability* properties, plus the defenses built and verified against each:

| Attack | CIA Property | Mechanism | Verified Impact (undefended) | Verified Impact (defended) |
|---|---|---|---|---|
| **SSM-Score** (Source Selection Manipulation) | Integrity | Forges blockchain transactions to inflate a malicious source's trust score | Score inflated 10,000 → 5,009,995; accuracy collapsed 42.9% → 0.0% (historical run) | 0/5 attack rounds accepted; score unchanged; 0.0 pp accuracy drop |
| **MIA** (Membership Inference Attack) | Confidentiality | 4-signal **gated** composite (a yes/no/maybe decision-commitment check dominates by construction; embedding similarity, a certainty proxy, and a length ratio only break ties) against a PubMedQA-derived corpus, to infer whether a document is in the corpus | AUC-ROC 0.64–0.70 in 2 of 3 seeds tested (`MEDIUM` privacy risk); 1 seed near-random (`NEGLIGIBLE`, unchanged by gating — see §12) | Two defenses: response sanitization (0% AUC reduction — targets the wrong channel); a new decision-obfuscation defense that **fully neutralizes** the decision-commitment signal (AUC exactly 0.50) while leaving other signals untouched — see §12/§13 for the full 4-revision history |
| **SFA** (Selective Forwarding Attack) | Availability | A compromised-but-online node silently drops 10–30% of queries (gray-hole) | Accuracy unaffected at full redundancy (`max_hops=3`); **0.68→ collapse at hop-limited `max_hops=1`** | Detection: 1/1 compromised node caught; mitigation restores accuracy 0.68 → 1.00 |

**Main finding:** all three attacks are real and implementable against this architecture, but their *measured* severity depends heavily on evaluation methodology — two of the three (MIA and SFA) initially had methodological flaws that made them look artificially weak or artificially strong, and this report documents both the original and corrected methodology so the numbers can be trusted. SSM-Score is the most severe of the three: it requires no probabilistic reasoning, only a single blockchain transaction, and produces total answer corruption once it lands. All three have been defended, and every defense claim in this report is backed by a live before/after comparison, not a theoretical argument.

---

## 1. Project Overview

### 1.1 What the project does

Reliable-dRAG (based on the `yining610/Reliable-dRAG` research system) answers natural-language questions by:

1. Splitting a question corpus (here, the Stanford Question Answering Dataset, SQuAD) across three independent Flask-based **data sources**, each holding roughly the same 500 documents but with a different fraction of them intentionally *token-polluted* (`sources_0` = 0% polluted / clean, `sources_20` ≈ 11% of documents altered, `sources_100` ≈ 59% of documents altered — measured directly by diffing the JSONL files, not by the folder names, which do not match the true pollution percentage). **Note:** this SQuAD-based corpus is what SSM-Score and SFA are evaluated against throughout this report. MIA's corpus was later swapped to a different dataset (PubMedQA) specifically for `sources_0` — see §12.2 for why.
2. Having an **LLM orchestrator** (`drag_llm_service`) query all three sources for every incoming question, rerank the combined candidate passages by a blend of semantic relevance and each source's **on-chain reliability score**, and generate an answer with a language model.
3. Recording each source's performance (was its content used, was the answer correct) as a **feedback transaction** to a smart contract (`DragScores.sol`) deployed on a local Hardhat blockchain, which is the persistent, tamper-evident record of "how much should the system trust this source."

### 1.2 Goal of the attacks analyzed here

Each attack targets a different trust assumption the architecture makes:

- **SSM-Score** assumes the blockchain feedback channel can only be used honestly by the orchestrator to reflect real answer quality. The attack asks: *what happens if an adversary can write directly to that channel?*
- **MIA** assumes that returning a synthesized answer (rather than the raw document) protects the confidentiality of which specific documents are in a given source's corpus. The attack asks: *can an external party, using only the public query API, infer whether a specific piece of content is stored on a specific source?*
- **SFA** assumes a source that responds to health checks and remains reachable is behaving honestly. The attack asks: *what if a node answers most of the time, but secretly drops a fraction of queries — does anyone notice, and does it matter?*

### 1.3 Threat model

| Element | Assumption |
|---|---|
| Attacker capability | Controls one data source's private key (SSM-Score, SFA) *or* is an unauthenticated external client of the public `/query` HTTP API (MIA, and the "external" mode of KB extraction, out of scope here) |
| Attacker goal | Integrity: elevate a low-quality/malicious source's influence. Confidentiality: determine corpus membership of a document. Availability: degrade answer quality/availability while evading detection |
| Attacker knowledge | Full knowledge of the open-source contract ABI and the fact that in this *test* deployment, Hardhat's well-known default private keys are used for every role (`sources_0/20/100`, `llm_service`) — a deliberate stand-in for "attacker has compromised a node's key," not a claim that production keys are public |
| Defender capability | Controls the smart contract source code and the off-chain detector/mitigation logic; cannot change the underlying blockchain's public, permissionless transaction-submission model except via contract-level access control |

### 1.4 Attack category

Using standard ML/security-attack taxonomy:

- **SSM-Score** → **Data/Control-plane Integrity Attack** on a **reputation/trust system** (closely related to Sybil and Byzantine trust-manipulation attacks in reputation systems literature, adapted to a blockchain feedback channel).
- **MIA** → **Membership Inference Attack**, a well-studied privacy attack class against ML systems (Shokri et al., 2017) here adapted to a RAG pipeline instead of a classifier.
- **SFA** → **Selective Forwarding / Gray-Hole Attack**, a classic insider availability attack from wireless sensor network (WSN) and mobile ad-hoc network (MANET) security literature, adapted to a decentralized RAG source topology.

---

## 2. Attack Theory

### 2.A SSM-Score: Trust-Score Falsification

**Definition.** SSM-Score treats each data source's on-chain reliability score `R_i` and usefulness score `U_i` as a *reputation* the system uses to weight that source's influence on the final answer. The attack directly writes an inflated value to that reputation rather than earning it through genuine good answers.

**Why it is used.** In any reputation-weighted system (peer-to-peer networks, recommendation systems, decentralized trust protocols), whoever controls the write path to the reputation store controls, indirectly, the entire system's output — this is a foundational result in reputation-system security research: *reputation is only as trustworthy as its update mechanism*.

**Mathematical intuition.** The reranker computes a blended score for each candidate passage:

$$
\text{score}(c) = (1-w)\cdot\text{sim}(q, c) + w \cdot \hat{R}(\text{source}(c))
$$

where `sim(q, c)` is the query-passage semantic similarity, `w` is `reliability_weight` (configured as `0.5` in this deployment), and `\hat{R}` is the source's reliability score normalized into a comparable range. If `\hat{R}` for the attacker's source is inflated by five orders of magnitude (10,000 → 5,009,995, a 500× increase, observed in a historical run of this exact attack), the second term dominates the sum regardless of `sim(q, c)`, so the reranker selects that source's passages **irrespective of their actual relevance or correctness**.

**Security relevance.** This is an *integrity* attack in the CIA triad: it does not read confidential data (confidentiality) and does not make the system unavailable (availability) — it corrupts the system's *decision process* so that it produces wrong answers while continuing to look operational.

### 2.B MIA: Membership Inference via Embedding Similarity

**Definition.** Membership inference asks a binary question about a specific record `x`: *was `x` used by/available to the target system?* The attacker does not need to extract `x`'s content (that would be extraction/inversion, a different attack) — only to distinguish members from non-members with better-than-random accuracy.

**Why it is used.** In RAG systems, a "member" document that is actually retrieved and placed in the LLM's context tends to produce a *more grounded* answer — closer, in embedding space, to the member document itself — than a "non-member" question, whose answer the model has to guess from parametric knowledge or fabricate. This grounding gap is the fundamental information leak MIA exploits.

**Mathematical intuition — Cosine Similarity.** For two vectors `a, b ∈ ℝᵈ` (here, 384-dimensional sentence embeddings from `all-MiniLM-L6-v2`):

$$
\text{sim}_{\cos}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} \in [-1, 1]
$$

A value near `1` means the response and the candidate context are semantically near-identical (the model grounded its answer in that specific text); a value near `0` or negative means they are unrelated. The MIA attack computes:

$$
s_i = \text{sim}_{\cos}\big(\text{embed}(\text{response}_i),\ \text{embed}(\text{context}_i)\big)
$$

for every probed question `i`, then asks whether the distribution of `s_i` for members is shifted higher than for non-members.

**Beyond cosine similarity alone — the current 4-signal composite.** The implementation now combines four signals into one weighted membership score rather than relying on cosine similarity alone:

$$
\text{score} = 0.20\cdot\text{sim} + 0.10\cdot\text{certainty} + 0.30\cdot\text{length\_ratio} + 0.40\cdot\text{decision\_match}
$$

where `certainty` is a proxy for how confidently the response commits to an answer, `length_ratio` compares response length to the gold answer's length, and `decision_match` (added last, and empirically the strongest of the four — see §12.2) checks whether the response's opening words state the correct yes/no/maybe judgment for the probed question. This mirrors a broader MIA design principle: a linear ensemble of several independently-weak signals can out-perform any single one, *provided* each component is validated to actually vary on the target dataset — an assumption that failed for one component here before being fixed (§12.2).

**Mathematical intuition — AUC-ROC.** Given the similarity scores as a "membership score," a Receiver Operating Characteristic (ROC) curve plots True Positive Rate against False Positive Rate as the decision threshold sweeps across all possible values. The Area Under that Curve is equivalent to the probability that a randomly chosen member scores higher than a randomly chosen non-member:

$$
\text{AUC} = P\big(s_{\text{member}} > s_{\text{non-member}}\big)
$$

`AUC = 0.5` is exactly what a coin flip would achieve — no membership signal. `AUC = 1.0` is perfect separation.

**Security relevance.** This is a *confidentiality* attack: even though the system never returns raw documents, membership itself can be sensitive (e.g., "is this person's medical record in the corpus" leaks private information about who is a patient, independent of the record's contents).

### 2.C SFA: Selective Forwarding (Gray-Hole) and Anomaly Detection Theory

**Definition.** In a multi-hop or multi-source system, a *black-hole* node drops 100% of traffic (trivially detectable by universal failure). A *gray-hole* node drops only a **fraction** `p ∈ (0,1)` of traffic — chosen deliberately low enough to evade naive "is this node ever failing" checks while still degrading the system over many interactions.

**Why it is used.** This is one of the best-studied insider attacks in wireless sensor network and MANET security research, because it exploits the exact gap between *availability* (the node responds to health checks) and *integrity of service* (the node actually does its job every time it's asked).

**Detection theory — the binomial hypothesis test.** If an honest node's true miss probability is `p₀` (the null hypothesis), then in a window of `n` independent queries, the number of misses `k` follows a Binomial distribution `k ~ Binomial(n, p₀)`. The one-sided p-value for "this node's observed miss rate is significantly above what an honest node would produce" is:

$$
p\text{-value} = P(K \ge k \mid K \sim \text{Binomial}(n, p_0)) = \sum_{i=k}^{n} \binom{n}{i} p_0^{\,i} (1-p_0)^{\,n-i}
$$

If this p-value is below a significance threshold `α` (here, `0.05`), the null hypothesis ("this node is honest") is rejected — the observed behavior would be too improbable under honesty. **The entire validity of this test depends on `p₀` being a correct model of real honest behavior** — a theme this report returns to at length in §12, because the original implementation's `p₀` was measured to be wrong by an order of magnitude for this specific deployment.

**Mitigation theory — reputation-weighted routing with redundancy.** Rather than only detecting misbehavior after the fact, a live system can route around suspected nodes: order candidate sources by `(suspicion_level, -reputation_score)` and try them in that order, falling back to a redundant probe of remaining trusted nodes if the first attempt fails. This is a standard fault-tolerance pattern (analogous to circuit breakers in distributed systems) applied to an adversarial setting.

**Security relevance.** This is an *availability* attack — it doesn't corrupt the answer content when it succeeds (the query is just dropped, not answered wrongly), and it doesn't read confidential data; it degrades the *guarantee that a query gets answered at all*.

---

## 3. Attack Architecture

### 3.1 Shared system architecture

```mermaid
graph TB
    subgraph "Blockchain Layer"
        HH[Hardhat Node<br/>local Ethereum-compatible chain]
        DS[DragScores.sol<br/>reliability/usefulness ledger]
        HH --- DS
    end

    subgraph "Data Plane"
        S0[data-source-0<br/>0% polluted]
        S20[data-source-20<br/>~11% polluted]
        S100[data-source-100<br/>~59% polluted]
    end

    subgraph "Orchestration"
        LLM[drag_llm_service<br/>/query endpoint]
        RR[Reliability-weighted<br/>Reranker]
    end

    Client[Client / Attacker] -->|HTTP POST /query| LLM
    LLM -->|query each source| S0
    LLM -->|query each source| S20
    LLM -->|query each source| S100
    S0 --> RR
    S20 --> RR
    S100 --> RR
    LLM -->|read scores| DS
    RR -->|weighted selection| LLM
    LLM -->|generate response| Client
    LLM -->|feedback tx: score update| DS
```

### 3.2 SSM-Score attack architecture

```mermaid
graph LR
    A[Attacker] -->|1 . forge signature using<br/>source's own private key| SIG[Signed feedback message]
    A -->|2 . submit tx as llm_service key| CONTRACT[DragScores.feedbackAndUpdateScoreRecords]
    CONTRACT -->|3 . accepts before fix| SCORE[(sources_100 score:<br/>10,000 → 5,009,995)]
    SCORE -->|4 . reranker reads inflated score| RERANK[Reliability-weighted reranker]
    RERANK -->|5 . always prefers sources_100| CANDIDATE[Polluted context selected]
    CANDIDATE -->|6 . LLM grounds on bad content| WRONG[Wrong / corrupted answer]
```

### 3.3 MIA attack architecture

```mermaid
graph LR
    QA[PubMedQA pqa_labeled rows] -->|split| MEM[Member: context loaded in corpus]
    QA -->|split| NM[Non-member: held-out same-dataset context]
    MEM --> Q1[Send question to /query]
    NM --> Q2[Send question to /query]
    Q1 --> R1[Response text]
    Q2 --> R2[Response text]
    R1 --> EMB1[Sentence embedding + cosine similarity]
    R2 --> EMB2[Sentence embedding + cosine similarity]
    R1 --> LEN1[length_ratio]
    R2 --> LEN2[length_ratio]
    R1 --> CERT1[certainty]
    R2 --> CERT2[certainty]
    R1 --> DEC1[decision_match vs. gold yes/no/maybe]
    R2 --> DEC2[decision_match vs. gold yes/no/maybe]
    EMB1 --> COMP[Weighted composite:<br/>sim*.20+certainty*.10+len*.30+decision*.40]
    EMB2 --> COMP
    LEN1 --> COMP
    LEN2 --> COMP
    CERT1 --> COMP
    CERT2 --> COMP
    DEC1 --> COMP
    DEC2 --> COMP
    COMP --> SCORES[Composite score per question]
    SCORES --> METRICS[AUC-ROC / Accuracy / Precision / Recall / F1]
```

### 3.4 SFA attack architecture

```mermaid
graph LR
    ATK[SelectiveForwardingAttack] -->|monkey-patches| SRC[Compromised source's .query method]
    SRC -->|drop_prob 10-30%| DECISION{Random roll}
    DECISION -->|drop| EMPTY[Return no results]
    DECISION -->|forward| REAL[Return real results]
    EMPTY --> DETECTOR[SFADetector: EWMA + binomial test]
    REAL --> DETECTOR
    DETECTOR -->|suspicion level 0-3| MIT[SFAMitigation: suspicion-aware routing]
    MIT -->|reorders / blacklists| ROUTE[Query routing decision]
    ROUTE --> RESULT[Final answer availability]
```

---

## 4. Implementation Analysis

### 4.A SSM-Score

| File | Purpose | Key elements |
|---|---|---|
| `drag_contract/contracts/drag_scores.sol` | Solidity smart contract holding each source's `ScoreRecord{sourceAddress, timestamp, reliabilityScore, usefulnessScore}`. `feedbackAndUpdateScoreRecords()` is the single write path for scores. | Before the fix: verifies only that *a* valid ECDSA signature from *the source itself* accompanies the update — nothing bounds the magnitude, frequency, or caller of the update. |
| `attack/ssm_score/ssm_score_attack.py` | `SSMScoreAttack` class. `inflate_scores()` reads current scores, adds `amplify` (default `999,999`) to both `reliability` and `usefulness`, signs a fixed probe message with the target source's private key (`_sign_fake_message`), and submits the update via the `llm_service` key, 5 times in a row. | Uses `drag_python_client.DragScoresClient` for all chain I/O; `PRIVATE_KEYS` dict hardcodes the well-known Hardhat test keys for `sources_0/20/100` and `llm_service`. |
| `attack/ssm_score/run_attack.py` | Orchestrates: load SQuAD eval questions whose context is in the loaded corpus → measure baseline accuracy via `/query` → run the attack → measure post-attack accuracy → log everything to `attack_logs/`. | `is_correct()` does substring-or-all-significant-words matching against SQuAD gold answers — a reasonable, non-degenerate correctness check (verified: not trivially always-true). |
| `drag_contract/contracts/drag_scores.sol` (defense) | Adds `onlyLLMService`-style caller check, `MIN_UPDATE_INTERVAL` cooldown, `MAX_DELTA_PER_UPDATE` cap, and `[MIN_SCORE, MAX_SCORE]` bounds directly inside `feedbackAndUpdateScoreRecords`. | Reverts with custom Solidity errors (`UnauthorizedCaller`, `UpdateTooFrequent`, `ScoreDeltaTooLarge`, `ScoreOutOfBounds`) — gas-efficient and unambiguous in logs. |
| `defense/ssm_defense/ssm_score_defense.py` | Off-chain `SSMScoreDefense.scan()` replays `ScoreRecordUpdated` events and flags any historical transition that *would* have violated the same rules — detection/audit layer independent of reading Solidity revert reasons. | |

### 4.B MIA

| File | Purpose | Key elements |
|---|---|---|
| `attack/Mia_attack/mia_attack.py` | `load_membership_documents()` streams PubMedQA's `pqa_labeled` config (1,000 rows), classifies each row's joined context as member (one of the first 500 rows, loaded into `sources_0.jsonl` by `data/build_pubmedqa_corpus.py`) or non-member (the remaining ~500 rows — held out, same dataset/domain, never loaded), and samples both pools with a properly seeded `random.Random(seed)`. `MIAAttack.run()` probes each sampled question via `_query_llm()`, then computes four signals per probe: `_cosine_similarity()` (embedding vs. true context), `_answer_length_ratio()`, `_certainty_score()` (yes/no/maybe commitment vs. hedge detection), and `_decision_match()` (does the response's opening words state the correct gold `final_decision` token). The best-similarity probe's four signals are combined into the weighted composite in §2.B. | Corrected in this project's development history from an earlier version that sampled "non-members" from `sources_20/100.jsonl` — since verified to be ~41–89% *identical* content to `sources_0.jsonl`, not disjoint data. **Dataset itself later swapped from SQuAD to PubMedQA** (§12.2) after SQuAD's Wikipedia-derived content was found to produce an *inverted* AUC-ROC (see §12.2) — a second, independent methodology fix beyond the original non-member-sampling bug. |
| `attack/Mia_attack/run_attack.py` | CLI + logging wrapper; also provides `--dry_run` (synthetic random scores, sanity-checks the metric code produces AUC≈0.5 with no real signal by construction). | |
| `data/build_pubmedqa_corpus.py` | Generator script: writes the first 500 PubMedQA `pqa_labeled` rows into `data/polluted_token/sources_0.jsonl` as `{"htmlid", "html"}` records, replacing the SQuAD-derived corpus specifically for MIA's evaluation. | Does not touch `sources_20/100.jsonl`, which remain SQuAD-derived for the unrelated SSM-Score/poisoning experiments. |
| `defense/mia_defense/mia_defense.py` | `sanitize_response()`: caps response length to 40 words and collapses any contiguous run of >8 words that verbatim-matches the retrieved context to `"..."`, using Python's `difflib.SequenceMatcher.find_longest_match` (a standard longest-common-substring algorithm applied to token sequences). `MIADefenseEvaluator` re-runs the identical probe set through both the raw and sanitized paths for a fair side-by-side comparison, on the same 4-signal composite the attack uses. | Measured to have **no effect on the current strongest signal** (`decision_match`) — the sanitizer only ever touches response length and verbatim overlap, neither of which is what `decision_match` measures (see §12.2, §13). |

### 4.C SFA

| File | Purpose | Key elements |
|---|---|---|
| `attack/selective_forward/selective_forward_attack.py` | `SelectiveForwardingAttack.apply()` monkey-patches `.query()` on a subset of source objects (chosen `random`ly or by `high_ssm_score`) so each call has probability `drop_rate ∈ [0.10, 0.30]` (the `"stealthy"` mode) of returning empty results instead of forwarding. `SFADetector` tracks a sliding window (`deque(maxlen=40)`) of hit/miss observations per node, computes an EWMA-smoothed miss rate, and escalates suspicion via the binomial test in §2.C. `SFAMitigation.route()` orders sources by `(suspicion_level, -ledger_score)`, skips blacklisted nodes, and falls back to a redundant probe. `_SSMChain` is a small in-process SHA-256 hash-chained ledger (a minimal blockchain simulation) tracking each node's reputation. | **Recalibration in this project's development history:** `HONEST_MISS` and `MISS_THRESH` were originally `0.68`/`0.55`, values appropriate for a generic many-node mock topology, not this real 3-source deployment (measured real honest miss rate: `0.000` over 180 live queries) — meaning detection was mathematically incapable of firing regardless of attack strength until recalibrated to `0.05`/`0.08`. |
| `attack/selective_forward/run_attack.py` | `RealSource` wraps one live data-source container as a query-able object. `_evaluate()` aggregates results from all queried sources and checks answer presence (used for baseline/stealthy/detection modes). `naive_route()` (added in this project's development history) provides a fair "undefended routing" comparison for `SFAMitigation.route()`, using the identical first-hit-stops mechanism without suspicion-awareness. | |
| `defense/sfa_defense/run_defense.py` | Runs baseline → stealthy → detection → mitigation (at the deployed `max_hops`) → mitigation (at a dedicated hop-limited `max_hops=1`) back to back and logs one consolidated report. | |

### 4.4 Execution flow across the project

```mermaid
sequenceDiagram
    participant U as Operator (CLI)
    participant A as attack/*/run_attack.py
    participant L as drag_llm_service (:9000)
    participant D as data-source-0/20/100
    participant C as DragScores.sol (Hardhat :8545)

    U->>A: python attack/.../run_attack.py
    A->>L: POST /query {question}
    L->>D: query all 3 sources
    D-->>L: candidate passages
    L->>C: read reliability/usefulness scores
    C-->>L: scores
    L->>L: reliability-weighted rerank + generate
    L-->>A: {"response": "..."}
    A->>A: score similarity / correctness
    A->>A: write JSON log to attack_logs/
```

---

## 5. Attack Workflow

### 5.A SSM-Score — step by step

1. **Baseline measurement.** Sample SQuAD questions whose context is in the loaded corpus; query `/query` for each; record correctness → `baseline accuracy`.
2. **Forge the feedback message.** Build the same JSON message format the honest LLM service would produce (`{"query": "attack_probe", "selected_sources": {...}}`), then sign it with the *target source's own* private key (`sign_message_personal`) — this signature is what the contract checks to authenticate "this source vouches for this feedback."
3. **Submit the inflated update.** Call `feedbackAndUpdateScoreRecords()` using the `llm_service` private key as the transaction sender, passing `new_score = current_score + 999,999` for both reliability and usefulness.
4. **Repeat 5 times** (`ROUNDS = 5`), re-reading the current score each round so the inflation compounds.
5. **Post-attack measurement.** Re-run the same question set; compare accuracy against the baseline.

```mermaid
flowchart TD
    Start([Start]) --> Baseline[Measure baseline accuracy]
    Baseline --> Sign[Sign fake feedback message<br/>with target source's key]
    Sign --> Submit[Submit feedbackAndUpdateScoreRecords<br/>as llm_service]
    Submit --> Check{Contract accepts?}
    Check -->|Yes, undefended| Inflate[Score inflated by +999,999]
    Check -->|No, defended| Reject[Revert: ScoreDeltaTooLarge /<br/>UpdateTooFrequent / UnauthorizedCaller]
    Inflate --> Round{5 rounds done?}
    Round -->|No| Sign
    Round -->|Yes| PostEval[Measure post-attack accuracy]
    Reject --> Round
    PostEval --> Compare[Compare baseline vs. post-attack]
    Compare --> End([End])
```

### 5.B MIA — step by step

1. **Build the member/non-member pools.** Stream PubMedQA's `pqa_labeled` config (1,000 rows); rows 0–499 (written to `sources_0.jsonl` by `data/build_pubmedqa_corpus.py`) are members, rows 500–999 are the held-out, same-dataset non-member pool.
2. **Seeded sampling.** Draw `n_members` and `n_nonmembers` documents using `random.Random(seed).sample()`.
3. **Probe.** For each sampled question, `POST /query {"query": question}`; record the raw response text.
4. **Score four signals.** Encode response and context with `all-MiniLM-L6-v2` for cosine similarity; compute `length_ratio` (response length vs. gold answer length); compute `certainty` (yes/no/maybe commitment vs. hedge detection); compute `decision_match` (does the response's opening words state the correct gold `final_decision` token).
5. **Combine into a weighted composite** (`sim*0.20 + certainty*0.10 + length_ratio*0.30 + decision_match*0.40`) at the best-similarity probe per document.
6. **Aggregate.** Sweep a percentile threshold over all composite scores; compute confusion matrix, accuracy, precision, recall, F1, and AUC-ROC, plus per-signal diagnostic AUCs.

```mermaid
flowchart TD
    Start([Start]) --> Load[Load PubMedQA pqa_labeled, 1000 rows]
    Load --> Split{Row 0-499<br/>= loaded corpus?}
    Split -->|Yes| MemberPool[Member pool]
    Split -->|No| NonMemberPool[Non-member pool: rows 500-999]
    MemberPool --> Sample[Seeded random sample]
    NonMemberPool --> Sample
    Sample --> Probe[POST /query with real question]
    Probe --> Score[Score sim, length_ratio,<br/>certainty, decision_match]
    Score --> Composite[Weighted composite score]
    Composite --> Collect{More probes?}
    Collect -->|Yes| Probe
    Collect -->|No| Threshold[Sweep threshold, compute metrics]
    Threshold --> Report[AUC-ROC, accuracy, precision, recall, F1]
    Report --> End([End])
```

### 5.C SFA — step by step

1. **Select targets.** Choose `attack_ratio` (default 34%) of sources to compromise, either `random`ly or by `high_ssm_score` (targeting the currently highest-reputation node — the PDF's "especially sharp" scenario, since the system is configured to prefer high-scoring sources).
2. **Patch.** Replace each target's `.query()` method with a wrapper that drops the call with probability drawn from `Uniform(0.10, 0.30)` per node.
3. **Run the workload.** Send real SQuAD questions through the (now partially compromised) source set.
4. **Observe (detection mode).** `SFADetector.observe()` records each hit/miss, maintains an EWMA-smoothed miss rate over the last 40 observations, and runs the binomial significance test each time.
5. **Route (mitigation mode).** `SFAMitigation.route()` orders sources by suspicion and reputation, skips blacklisted ones, and falls back to a redundant probe.
6. **Compare.** Naive first-hit routing (no suspicion-awareness) vs. mitigated routing, under the identical attack instance.

```mermaid
flowchart TD
    Start([Start]) --> Select[Select ratio of nodes to compromise]
    Select --> Patch[Monkey-patch .query on target nodes]
    Patch --> Send[Send SQuAD question]
    Send --> Roll{Random roll <<br/>drop probability?}
    Roll -->|Yes| Drop[Return empty result]
    Roll -->|No| Forward[Return real result]
    Drop --> Observe[SFADetector.observe]
    Forward --> Observe
    Observe --> Window{40+ observations<br/>in window?}
    Window -->|No| Send
    Window -->|Yes| Binom[Binomial significance test]
    Binom --> Suspicion{p < 0.05 for<br/>2 consecutive windows?}
    Suspicion -->|Yes| Escalate[Raise suspicion level]
    Suspicion -->|No| Send
    Escalate --> Route[SFAMitigation reorders routing]
    Route --> Compare[Compare naive vs. mitigated accuracy]
    Compare --> End([End])
```

---

## 6. Evaluation Pipeline

All three attacks share the same high-level evaluation shape: **input → prediction → ground truth → scoring → aggregation.**

| Stage | SSM-Score | MIA | SFA |
|---|---|---|---|
| **Input** | SQuAD questions whose context is in the corpus | Sampled member/non-member PubMedQA questions | SQuAD questions, routed through 1–3 sources |
| **Prediction** | LLM's generated answer text | LLM's generated response text | Whether a hit (any document) was returned |
| **Ground truth** | SQuAD gold answer string | Membership label (1 = member, 0 = non-member) | Whether the SQuAD gold answer string appears in retrieved text |
| **Scoring** | Substring/significant-word match → boolean correct/incorrect | 4-signal weighted composite (similarity, certainty, length_ratio, decision_match) → continuous score, thresholded at a percentile | Boolean hit/miss per source per question |
| **Aggregation** | `accuracy = correct / total`; compare baseline vs. post-attack | Confusion matrix, AUC-ROC over all scores | `accuracy = questions_with_a_hit / total`; per-node miss rate for detection |

---

## 7. Metrics Generation

### 7.1 Accuracy (SSM-Score, SFA)

**Purpose:** fraction of questions the system answered correctly (or, for SFA, found any relevant document for).

**Formula:**
$$
\text{Accuracy} = \frac{\text{Correct (or Hit) count}}{\text{Total questions}}
$$

**Range:** `[0, 1]`. **Good:** high and stable under attack (means the attack failed or was mitigated). **Bad:** a large drop from baseline to attacked (means the attack succeeded). **Example:** SSM-Score historical run: `0.429 → 0.0` (100% relative degradation) when the attack succeeded. **Security implication:** this is the *outcome* metric — the one that answers "does the user actually get worse answers."

### 7.2 Accuracy Drop / Effective Drop Rate

**Formula:** `acc_drop = baseline_accuracy − attacked_accuracy` (percentage points). `effective_drop_rate` (SFA) = `total_dropped / total_attempted` — the attack's *actual* realized drop rate, since each compromised node's rate is drawn randomly from `Uniform(0.10, 0.30)`.

**Interpretation:** `acc_drop = 0.0` does **not** always mean "the attack failed" — it can also mean the evaluation topology absorbed the attack via redundancy (see §12). Always read this metric together with the routing/hop configuration.

### 7.3 Precision, Recall, F1 (MIA)

$$
\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad F_1 = \frac{2 \cdot \text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

Here, `TP` = a true member correctly classified as a member at the chosen threshold; `FP` = a non-member incorrectly classified as a member. **Range:** `[0,1]`, higher is "more successful attack." **Limitation specific to this project:** because the classification threshold is fixed at the 50th percentile of the *combined* score distribution (`threshold_pct=50`), precision/recall/F1 are heavily influenced by that percentile choice and are *less* trustworthy than AUC-ROC, which evaluates every possible threshold — this is precisely the "single-threshold accuracy is the wrong metric" issue flagged in this project's own progress tracker and the reason AUC-ROC is the headline MIA metric.

### 7.4 AUC-ROC (MIA) — the headline metric

Already defined mathematically in §2.B. **Range:** `[0,1]`. **Interpretation table:**

| AUC-ROC | Privacy Risk Label (from code) | Meaning |
|---|---|---|
| > 0.90 | 🔴 CRITICAL | Near-perfect membership separability |
| > 0.75 | 🟠 HIGH | Significant, exploitable signal |
| > 0.60 | 🟡 MEDIUM | Moderate, usable-with-effort signal |
| > 0.50 | 🟡 LOW | Slight signal, close to noise |
| ≈ 0.50 | 🟢 NEGLIGIBLE | No better than guessing |

**Measured — three revisions of this attack (see §12.2 for the full history):**

| Revision | Dataset / composite | Seeds tested | AUC-ROC | Tier |
|---|---|---|---|---|
| 1 | SQuAD, `sim*0.85+len*0.15` | 0, 1, 2, 42 | 0.272–0.466 (mean 0.356) | 🟢 NEGLIGIBLE, every run |
| 2 | PubMedQA, `sim*0.70+certainty*0.20+len*0.10` | 0, 1, 42 | 0.517–0.576 (mean 0.546) | 🟡 LOW, every run |
| **3 (current)** | PubMedQA, `sim*.20+certainty*.10+len*.30+decision_match*.40` | 0, 1, 42 | **0.4992–0.7056 (mean 0.616)** | **🟡 MEDIUM in 2/3 seeds; 🟢 NEGLIGIBLE in 1/3** |

**Security implication:** the current (3rd-revision) result is a materially different conclusion from the original: **2 of 3 tested seeds now show a real, moderate membership-inference signal**, driven mainly by a "does the response commit to the correct yes/no/maybe answer" check (`decision_match`) rather than raw text similarity. This is not a fully-solved, guaranteed attack — one seed remains close to random — but it is no longer accurate to describe this system's MIA risk as simply "negligible." See the dedicated `reports/MIA_Security_Analysis_Report.md` for the full per-signal breakdown and the reasoning behind each revision.

### 7.5 Mean Similarity / Similarity Delta, and the newer `decision_match` delta (MIA)

$$
\Delta_{\text{sim}} = \bar{s}_{\text{member}} - \bar{s}_{\text{non-member}}
$$

**Interpretation:** a positive delta means members are, on average, more grounded/exposed than non-members (the expected direction if a leak exists). **Measured (Revision 1, SQuAD):** deltas ranged from `−0.136` to `+0.062` — inconsistent in sign, and *negative* (inverted) in 3 of 4 seeds — a real, systematic inversion, not "no signal" (see §12.2). **Measured (Revision 3, PubMedQA, current):** raw similarity delta is still weak and inconsistent (`−0.017` to `+0.063`), but the newer `decision_match_delta` (same formula, applied to the yes/no/maybe-commitment signal) was **positive in all 3 seeds tested** (`+0.12` to `+0.28`) — the most consistent individual signal measured anywhere in this project, and the main driver of Revision 3's improved AUC-ROC.

### 7.6 Detection Rate (SFA)

$$
\text{Detection Rate} = \frac{|\{\text{compromised nodes correctly flagged as suspected}\}|}{|\{\text{compromised nodes}\}|}
$$

**Measured:** `0/1` before recalibration (mathematically guaranteed to fail — see §12), **`1/1` (1.0)** after recalibration, at `n_questions ≥ 100`. **Security implication:** this is the core "does the defense actually work" number for SFA; a detector that cannot reach `1.0` given clear attack signal and adequate sample size is not a functioning defense, regardless of how sophisticated its statistics look on paper.

### 7.7 Naive vs. Mitigated Routing Accuracy (SFA)

Both computed with the identical first-hit-stops routing mechanism (`naive_route()` vs. `SFAMitigation.route()`), differing only in whether suspicion/reputation informs the routing order. **Measured (hop-limited, `max_hops=1`, `strategy=high_ssm_score`, n=100):** naive `0.69` → mitigated `1.00`. **Security implication:** this isolates *exactly* the value the suspicion-aware routing adds, independent of redundancy — the correct way to claim "the mitigation works," as opposed to comparing against an unrelated aggregate-accuracy baseline (a mistake corrected during this project's development, documented in §12).

### 7.8 On-chain score delta and rounds accepted/blocked (SSM-Score defense)

`rounds_accepted` / `rounds_blocked` out of 5 attack rounds; `scores_before` vs. `scores_after`. **Measured after the fix:** `0/5` accepted, `scores_after == scores_before` exactly. This is a binary, contract-enforced guarantee rather than a statistical estimate — the strongest type of security claim available in this report, because it is provable from the transaction revert reasons, not inferred from behavior.

---

## 8. JSON Metrics Analysis

### 8.1 SSM-Score — `attack_logs/attack_2026-07-06_18-56-54_ssm_score.json` (defended) vs. historical `attack_2026-07-01_..._ssm_score_sources_100.json` (undefended)

| Metric | Undefended (2026-07-01) | Defended (2026-07-06) | Meaning | Interpretation | Impact |
|---|---|---|---|---|---|
| `scores_before` (sources_100) | R=10,000 U=10,000 | R=10,000 U=10,000 | Starting reputation | Identical starting point | — |
| `scores_after` (sources_100) | R=5,009,995 U=5,009,995 | R=10,000 U=10,000 | Reputation after 5 attack rounds | 🔴 500× inflation vs. 🟢 unchanged | Confirms defense holds |
| `rounds_accepted` | 5/5 (implicit — all txs succeeded) | 0/5 | Contract's willingness to accept the forged update | 🔴 fully vulnerable vs. 🟢 fully blocked | Direct security outcome |
| `baseline_accuracy` | 0.429 | 0.58 | Accuracy before attack (different eval configs/dates) | Not directly comparable across runs (different sample) | Context only |
| `post_attack_accuracy` | 0.0 | 0.58 | Accuracy after attack | 🔴 total collapse vs. 🟢 unchanged | Core impact metric |
| `accuracy_drop` | 100% relative (0.429→0.0) | 0.0 pp | Attack effectiveness | 🔴 devastating vs. 🟢 neutralized | Headline result |

**Performance summary:** the undefended contract allowed complete answer corruption; the defended contract allows none. **Strength:** contract-level enforcement is provable and cannot be bypassed by a smarter attacker script (the check is on-chain, not client-side). **Weakness:** the defense's specific numeric thresholds (`MAX_DELTA_PER_UPDATE=5,000`, `MIN_UPDATE_INTERVAL=2s`) were chosen based on this deployment's observed legitimate-feedback magnitudes, not derived from a formal analysis of the widest possible range of legitimate feedback — see §15.

### 8.2 MIA — historical vs. current defense logs

**Historical (`defense_logs/defense_2026-07-06_23-37-31_mia_seed42.json`, Revision 1, SQuAD, 2-term composite):**

| Metric | Undefended | Defended | Meaning | Interpretation | Impact |
|---|---|---|---|---|---|
| `auc_roc` | 0.488 | 0.488 | Membership separability | 🟢 negligible, unchanged | Attack was already weak; defense has nothing to reduce |
| `accuracy` | 0.600 | 0.600 | Threshold-based classification accuracy | 🟡 modestly above chance at 50th-percentile threshold | Consistent with weak/negligible AUC |
| `mean_member_similarity` | 0.1415 | 0.1415 | Avg. grounding for true members | Low absolute grounding | Model rarely echoes source text verbatim |
| `mean_non_member_similarity` | 0.1584 | 0.1584 | Avg. grounding for non-members | **Higher** than member similarity | This inversion was later traced to pretraining overlap with SQuAD — see §12.2 |
| `auc_roc_reduction` | — | 0.0000 | Defense's marginal effect | 🟡 no measurable effect this run | Sanitization had nothing to remove — weak/inverted signal |

**Current (`defense_logs/defense_2026-07-09_17-36-00_mia_seed0.json`, Revision 3, PubMedQA, 4-term composite, n=25/25):**

| Metric | Undefended | Defended | Meaning | Interpretation | Impact |
|---|---|---|---|---|---|
| `auc_roc` | 0.6432 | 0.6432 | Membership separability | 🟡 MEDIUM privacy risk, unchanged by defense | Attack is now real and moderate; defense still finds nothing to remove |
| `accuracy` | 0.600 | 0.600 | Threshold-based classification accuracy | 🟡 well above chance | Consistent with MEDIUM-tier AUC |
| `mean_member_similarity` | −0.0302 | −0.0302 | Avg. raw cosine similarity, members | Weak individually | Similarity alone is not the main driver anymore |
| `mean_non_member_similarity` | −0.0288 | −0.0288 | Avg. raw cosine similarity, non-members | Nearly identical to member similarity | Confirms similarity alone is not what's leaking membership now |
| `auc_roc_answer_match` (= `decision_match`) | 0.6200 | 0.6200 | Does the response commit to the correct yes/no/maybe token | 🟡 the strongest individual signal measured | Member rate 0.72 vs. non-member rate 0.48 — a real, exploitable gap |
| `auc_roc_length_ratio` | 0.6464 | 0.6464 | Response-length-vs-gold-answer ratio | 🟡 second-strongest signal | Contributes meaningfully alongside `decision_match` |
| `auc_roc_certainty` | 0.5200 | 0.5200 | Yes/no/maybe commitment vs. hedge | 🟢 still weak | Redesigned from a completely degenerate hedge-word version; not yet reliable |
| `auc_roc_reduction` | — | **0.0000** | Defense's marginal effect | 🔴 no measurable effect, but for a **new reason** | The sanitizer targets response length/verbatim overlap — neither is what `decision_match` (the real driver) measures. This is a **mechanism mismatch**, not "nothing to remove" — see §12.2, §13 |

### 8.3 SFA — `defense_logs/defense_2026-07-06_23-39-07_sfa_seed42.json`

| Metric | Value | Meaning | Interpretation | Impact |
|---|---|---|---|---|
| `detection.accuracy` | 0.93 | Aggregate accuracy while under attack + detection | 🟢 high (redundancy masks aggregate accuracy) | Aggregate accuracy is not the detection signal — see below |
| `detection.compromised` | `["source_0"]` | Ground truth: which node was attacked | — | — |
| `detection.suspected` | `["source_0"]` | Detector's flagged node(s) | 🟢 exact match | Detection working correctly |
| `detection.detection_rate` | 1.0 | Fraction of compromised nodes correctly caught | 🟢 perfect | Core defense success metric |
| `mitigation.accuracy` (max_hops=3) | 1.0 | Mitigated routing accuracy at full redundancy | 🟢 ceiling | Nothing to improve on at this hop count |
| `mitigation.naive_routing_accuracy` (max_hops=3) | 1.0 | Undefended routing accuracy at full redundancy | 🟢 identical to mitigated | Confirms redundancy alone masks the attack here |
| `mitigation_hop_limited.accuracy` (max_hops=1) | 1.0 | Mitigated routing, worst-case hop budget | 🟢 fully preserved | Real defense value shows up here |
| `mitigation_hop_limited.naive_routing_accuracy` (max_hops=1) | 0.69 | Undefended routing, worst-case hop budget | 🔴 31-point real degradation | This is the number that proves the attack matters |

**Strength analysis:** detection is now statistically sound and empirically perfect at adequate sample size; mitigation provides a measured, real 31-percentage-point recovery in the scenario the threat model actually describes. **Weakness analysis:** at the *currently deployed* configuration (`max_hops = 3 = total sources`), the attack has no measurable effect at all — this is an honest finding about this specific deployment's redundancy level, not a flaw in the attack or defense code.

---

## 9. Performance Dashboard

### SSM-Score (defended contract, live verification)

```
Rounds Blocked
███████████████ 100% (5/5)

Score Drift
█ 0% (0 / 5,000,000 potential inflation)

Accuracy Preserved
███████████████ 100% (0.0 pp drop)
```
🟢 Excellent — attack fully neutralized, provably (on-chain revert), not just observed to fail in one run.

### MIA (3rd revision — PubMedQA corpus, 4-signal composite)

```
AUC-ROC across 3 revisions (mean)
Rev 1 (SQuAD)         ██████████ 36%   🔴 Inverted every seed
Rev 2 (PubMedQA v1)   ██████████████ 55%   🟡 Weak, correct direction
Rev 3 (PubMedQA v2)   ███████████████ 62%   🟡 2/3 seeds MEDIUM tier

Rev 3 AUC-ROC, per seed
Seed 0    ████████████████ 64%   🟢 MEDIUM
Seed 1    █████████████████ 71%   🟢 MEDIUM (crosses "genuine vulnerability")
Seed 42   ████████████ 50%   🔴 NEGLIGIBLE (near-exact random outlier)

Decision-match signal (Rev 3, all 3 seeds — never inverted)
███████████████ 56-64%   🟢 Strongest, most consistent signal found in this project

Defense AUC-ROC Reduction (all 3 revisions, all recorded runs)
· 0%   (Rev 1-2: signal too weak to reduce. Rev 3: real signal, but wrong defense mechanism for it)
```
🟡 Revision 3 is the first time this project's MIA work has crossed into `MEDIUM` privacy risk — in 2 of 3 tested seeds. 🔴 One seed remains a near-exact-random outlier, so this should be read as real, meaningful progress, not a fully solved or fully reliable attack. 🟡 The sanitization defense remains functional but ineffective against the current strongest leak channel (`decision_match`) by construction, not because the signal is weak. See `reports/MIA_Security_Analysis_Report.md` for the complete 3-revision breakdown.

### SFA (recalibrated detector, hop-limited routing test)

```
Detection Rate
███████████████ 100% (1/1 compromised node caught)

Accuracy @ max_hops=3 (current deployment)
███████████████ 94% baseline → 93-94% attacked (minimal drop)

Accuracy @ max_hops=1, naive routing (no defense)
██████████ 69%   🔴

Accuracy @ max_hops=1, mitigated routing
███████████████ 100%   🟢  (+31 points recovered)
```
🟢 Detection: excellent, statistically sound, verified at scale. 🟢 Mitigation: excellent in the scenario it's designed for. 🟡 Current deployment's `max_hops=3` config makes the attack largely moot regardless of defense — a topology observation, not a defense weakness.

---

## 10. Performance Interpretation

| If this metric... | ...it means | Reliability impact | Security impact |
|---|---|---|---|
| SSM-Score `rounds_accepted` **increases** from 0 | The contract-level cap/cooldown/allow-list has a gap | Trust scores could drift again | 🔴 Direct integrity compromise |
| MIA `auc_roc` **increases** toward 1.0 (e.g., after a probe-methodology change) | The system is grounding answers in a way that's more distinguishable per-source | No change to answer quality | 🔴 Growing confidentiality risk |
| MIA `decision_match` delta **stays positive across seeds** (observed: +0.12 to +0.28 in all 3 Revision-3 seeds) | The model reliably commits to the correct yes/no/maybe answer more often for members than non-members | No change to answer quality | 🔴 The most consistent, generalizable leak channel found in this project so far |
| MIA `auc_roc_reduction` **increases** | The sanitization defense is actively suppressing more verbatim leakage | Answers get shorter/less quote-heavy for high-overlap cases only | 🟢 Improving privacy at low utility cost — **but currently measures 0.0000 against the `decision_match` channel specifically, since that defense doesn't touch decision-token commitment at all** |
| SFA `detection_rate` **decreases** below 1.0 | Either sample size dropped below the detector's 40-query warm-up window, or a new stealthier drop rate evades `MISS_THRESH` | No change to answer availability directly | 🔴 Blind spot in insider-threat monitoring |
| SFA hop-limited `naive_routing_accuracy` **decreases** further | The compromised node's drop rate increased, or more nodes were compromised | Users experience more unanswered queries | 🔴 Availability degradation, until mitigation catches up |
| SFA hop-limited `mitigated_routing_accuracy` **decreases** | The suspicion-aware routing is failing to route around a confirmed-bad node (contract bug in `SFAMitigation`) | Users experience unanswered queries despite a "working" defense | 🔴 Defense failure — highest-priority item to investigate |

---

## 11. Experimental Methodology

| Aspect | Detail |
|---|---|
| **Dataset** | SSM-Score and SFA: Stanford Question Answering Dataset (SQuAD v1.1), `rajpurkar/squad`, train split, loaded via HuggingFace `datasets`. **MIA (as of Revision 2 onward):** `qiaojin/PubMedQA` (`pqa_labeled` config, 1,000 rows), loaded into `sources_0.jsonl` only via `data/build_pubmedqa_corpus.py` — switched from SQuAD after SQuAD's Wikipedia-derived content was found to invert MIA's AUC-ROC (see §12.2); `sources_20/100` remain SQuAD-derived for the unrelated SSM-Score/poisoning experiments. |
| **Corpus size** | 500 unique documents loaded per data source (`sources_0/20/100`), each with several SQuAD questions attached. |
| **Pollution levels (measured, not assumed)** | `sources_20`: 57/500 documents (11.4%) differ from `sources_0`. `sources_100`: 297/500 documents (59.4%) differ. Both diverge only in interior tokens — opening sentences are typically preserved. |
| **Evaluation protocol** | Baseline (no attack) → attack applied → post-attack measurement, on the *same* sampled question set per run, for a controlled before/after comparison. |
| **Sample sizes used** | SSM-Score: 50 questions. MIA: 25 members + 25 non-members (50 total). SFA: 60–100 questions (100 recommended; SFADetector requires ≥ 40 observations per node to exit its warm-up window). |
| **Random seeds** | 42 and 0 used across attacks for reproducibility and (for SSM-Score and SFA, always; for MIA, only after a seeding bug fix documented in §12) genuine seed-to-seed variance. MIA's current (Revision 3) evaluation additionally used seed 1, giving 3 seeds total (0, 1, 42) — one of which (42) remains a near-random outlier even after all fixes, an open item (§15). |
| **Environment** | Docker Compose stack: `drag-hardhat-node` (Hardhat local chain, chainId 31337), `drag-llm-service` (Flask + local LLM), `drag-data-source-0/20/100` (Flask + FAISS retrieval, `all-MiniLM-L6-v2` embeddings). |
| **Blockchain** | Hardhat's 20 deterministic default test accounts; `DragScores` contract address is deterministic (`0x5FbDB2315678afecb367f032d93F642f64180aa3`) because it is always the first transaction from account #0 on a fresh chain. |
| **Reproducibility** | All attack/defense scripts accept `--seed`; all runs write a timestamped JSON log to `attack_logs/` or `defense_logs/`, enabling exact replay of the sampled question sets given the same corpus state. |
| **Assumption flagged** | The use of Hardhat's publicly-known default private keys for every role is a deliberate simulation of "attacker has a compromised node's key," standard practice for local blockchain testing — **not** a claim that a production deployment would expose private keys. |

---

## 12. Results Discussion

### 12.1 Why SSM-Score shows the most dramatic result

Unlike MIA and SFA, SSM-Score requires **no probabilistic reasoning or statistical inference** — it is a single deterministic write to a shared, unweighted trust variable. Its severity is essentially unbounded by design (an attacker can write any value, not just a modestly-biased one), which is exactly why the historical undefended run shows a full collapse (42.9% → 0.0%) rather than a partial degradation: once the reranker's reliability term dominates by five orders of magnitude, the semantic-relevance term becomes statistically irrelevant to source selection.

### 12.2 MIA's three revisions: why it initially looked broken, and what changed each time it was fixed

MIA went through **three successive, evidence-driven revisions** in this project, each triggered by a specific defect measured in the one before it. This is summarized here; full per-signal detail lives in the dedicated `reports/MIA_Security_Analysis_Report.md`.

**Pre-Revision-1 bugs (fixed before any of the AUC numbers below were measured):**
1. **Invalid non-member sampling.** The original implementation drew "non-member" content from `sources_20/100.jsonl`, believing them to hold different documents than `sources_0.jsonl`. Directly diffing the files showed they share the same 500 document indices — `sources_20` differs in only 11.4% of documents, `sources_100` in 59.4%. This made "member vs. non-member" partially the *same underlying content*, invalidating the entire premise of the test.
2. **A non-functional random seed.** Passage loading read files sequentially regardless of the `--seed` argument; two different seeds produced byte-identical results.

**Revision 1 (SQuAD corpus, `sim*0.85 + len*0.15` composite):** after fixing both bugs above, the measured AUC-ROC settled at **0.272–0.466 across 4 seeds (mean 0.356) — solidly *below* 0.50 in every single run.** This is not "no signal" — it is a **sign-consistent inversion**. The strongest supporting evidence was that the model answered non-member SQuAD questions correctly from its own pretrained knowledge about as often as member questions: SQuAD is Wikipedia-derived, and the deployed LLM (Qwen2.5-1.5B-Instruct) had almost certainly seen much of that content during pretraining, independent of what the RAG system actually retrieved. `mean_non_member_similarity` (0.1584–0.1592) being slightly *higher* than `mean_member_similarity` (0.1415–0.1761) in more than one run is consistent with exactly this: pretrained, ungrounded answers on generic Wikipedia-style topics can embed *more* centrally/fluently than terse, correctly-grounded member answers.

**Revision 2 (dataset switched to PubMedQA, 3-term composite adding a `certainty` signal):** switching the corpus to `qiaojin/PubMedQA` (biomedical abstracts, far less likely to overlap with the base LLM's pretraining) resolved the inversion — AUC-ROC rose to **0.517–0.576 across 3 sampled seeds (mean 0.546)**, confirming the pretraining-overlap hypothesis. But the new `certainty` signal (a hedge-word-based proxy) was measured **completely degenerate**: PubMedQA elicits short, direct yes/no/maybe answers that essentially never contain a generic hedge word, so `certainty` came out exactly `1.0` for every response, member and non-member alike, in most runs.

**Revision 3 (current — redesigned `certainty`, fixed a blind diagnostic into a 4th signal `decision_match`):** the hedge-word `certainty` was redesigned around explicit yes/no/maybe commitment detection (still weak, no longer exactly degenerate). Separately, the *diagnostic* that had been checking whether a full gold-answer sentence appeared verbatim in the response was found to be **blind** (0.0000 match rate for both groups, every run — PubMedQA's gold field is a full sentence, essentially never quoted verbatim). Replacing it with `decision_match` (does the response's opening words state the correct short yes/no/maybe judgment) turned this from a broken diagnostic into **the single strongest and most consistent signal measured anywhere in this project** — AUC 0.56–0.64, delta always positive across all 3 tested seeds, never once inverted. It was folded into the composite as the heaviest-weighted term (0.40). Result: **AUC-ROC 0.4992–0.7056 (mean 0.616)**, with **2 of 3 seeds now crossing into the `MEDIUM` privacy-risk tier** — the first time this project's MIA work has exceeded `LOW`.

**What remains unresolved:** one seed (42) stayed a near-exact-random outlier (0.4992) even in Revision 3, because `length_ratio` and raw `similarity` are individually still noisy and occasionally invert (unlike `decision_match`, which never did across the seeds tested). This means Revision 3's improvement, while real, is not yet fully reliable across all conditions.

**Revision 4 — a tested hypothesis that turned out wrong, and a defense that worked.** The natural hypothesis for the seed-42 outlier was that a *linear* blend let noisy secondary signals occasionally **outvote** an already-correct `decision_match` reading across the member/non-member boundary. The tested fix: make `decision_match` a **gate** — its weight (0.55) set to strictly exceed the combined maximum of the other three signals (0.45), so cross-boundary outvoting becomes mathematically impossible. **Measured result: the gate produced byte-identical AUC-ROC to the linear blend for 2 of 3 seeds (0.6432 and 0.4992, both to 4 decimal places)** — a genuine negative result. It proves outvoting was never the actual mechanism at play here (decision_match's binary values combined with the other signals' small real magnitudes meant the linear blend already behaved almost exactly like a gate in practice); the real cause of the seed-42 outlier is noise **within** documents that already share the same `decision_match` value — a tie the gate explicitly leaves to the other, weaker signals to break, and therefore cannot fix. Separately, a new defense (`obfuscate_decision()` — prepends a fixed hedging phrase to every response, unconditionally) was built and shown to drive `decision_match`'s own AUC to **exactly 0.5000** in both recorded runs — full, precise neutralization of the attack's strongest signal, the first defense in this project's history to show a real, mechanistically-verified effect rather than "0% reduction, nothing to remove." The overall composite reduction remains modest (+0.008 for seed 0) because `length_ratio` still carries signal this defense doesn't touch, and one seed's raw reduction figure (+0.0528 for seed 42) is flagged as misleading in the dedicated MIA report, since that seed's undefended AUC was already near 0.5 and the "improvement" actually moved further from it in the inverted direction. Full detail: `reports/MIA_Security_Analysis_Report.md`.

### 12.3 Why SFA's detector needed recalibration, and why max_hops changes everything

The original `SFADetector` constants (`HONEST_MISS=0.68`, `MISS_THRESH=0.55`) were carried over from a synthetic many-node mock (`_MockSource`, `hit ⟺ Beta(2,3) ≥ 0.45`) without being re-validated against the real deployment. Measuring the real deployment directly (180 live queries across 3 honest sources) showed a **true honest miss rate of exactly 0.000** — retrieval almost always returns *some* top-k document regardless of true relevance, so "did this source return anything" is nearly always true for an honest node. Against a mis-set `MISS_THRESH=0.55`, a realistic stealthy attacker (10–30% drop rate) could mathematically never cross the threshold — **detection was guaranteed to fail regardless of sample size or attack strength.** After recalibrating to `HONEST_MISS=0.05`, `MISS_THRESH=0.08` (chosen using a `scipy.stats.binomtest` power check against the real measured baseline), detection reached `1.0` at adequate sample size.

Separately, the accuracy impact of the attack is entirely dependent on the routing hop budget, because this deployment's current `n_retrievers` configuration equals the total number of sources (3), so even *naive*, undefended routing eventually tries every source and finds an honest one. Only when the hop budget is deliberately constrained below the source count (`max_hops=1`, matching the PDF's stated threat model of "the system prioritizes high-scoring sources") does the attack show real, measured impact (`0.69` naive vs. `1.00` mitigated). This is not a limitation of the attack or defense implementation — it is an accurate reflection of how much redundancy this specific 3-node deployment currently provides.

---

## 13. Defense Mechanisms

| Defense | Description | How it works | Advantages | Disadvantages | Complexity | Effectiveness (measured) | Residual risk |
|---|---|---|---|---|---|---|---|
| **On-chain caller allow-list** (SSM) | Only the registered `llmService` address may call `feedbackAndUpdateScoreRecords` | `require(msg.sender == llmService)` | Simple, provable, zero runtime cost beyond one comparison | Useless if the `llm_service` key itself is compromised (as simulated here, deliberately, via public test keys) | Low | Partial alone; combined with delta cap, full | Key-compromise of the orchestrator's own signer |
| **On-chain delta cap** (SSM) | Rejects any single update moving a score by more than 5,000 | `require(abs(new-old) <= MAX_DELTA_PER_UPDATE)` | Directly bounds attacker payoff per transaction, regardless of caller | A patient, multi-transaction attacker constrained only by the cooldown could still drift a score over many small legal steps | Low | 🟢 100% blocked in every live test (0/5 rounds) | Slow, many-transaction drift not fully bounded by cap alone |
| **On-chain cooldown** (SSM) | 2-second minimum between updates to the same source | `require(now >= last_update + MIN_UPDATE_INTERVAL)` | Blocks rapid multi-round bursts (exactly what the attack does) | Also throttles legitimate rapid feedback in bursty legitimate traffic | Low | 🟢 Verified: blocks round 2+ of a rapid burst | None significant for this deployment's query rate |
| **On-chain absolute bounds** (SSM) | Score confined to `[-1,000,000, 1,000,000]` | `require(MIN_SCORE <= new <= MAX_SCORE)` | Backstop against slow drift bypassing the delta cap over very many transactions | Bound values are deployment-specific, not derived from a formal legitimate-range analysis | Low | Not yet exercised in a live "slow drift" test (see §15) | Bound choice unverified against true legitimate-feedback distributions |
| **Off-chain audit scan** (SSM) | Replays `ScoreRecordUpdated` events and flags historical violations | Event log replay + same threshold logic as the contract | Works even against a pre-fix contract's historical data; useful for post-incident forensics | Detects, does not prevent — purely observational | Low-Medium | Functional; no anomalies found post-fix (as expected) | Doesn't stop an in-progress attack by itself |
| **Response sanitization** (MIA) | Cap response length + redact long verbatim overlaps with retrieved context | Longest-common-substring redaction via `difflib.SequenceMatcher` | No numeric confidence score needed (the actual API surface here); low latency; no model retraining | Measured 0% AUC reduction in **every** revision tested — but the reason changed: Revisions 1–2 had a weak/absent signal (nothing to remove); **Revision 3's signal is now real (MEDIUM tier in 2/3 seeds), and the defense still shows 0% reduction because its mechanism (length/verbatim-overlap) has nothing to do with the actual driving signal (`decision_match` — does the response commit to the correct yes/no/maybe token). This is a genuine mechanism mismatch, not "nothing to defend."** | Low | 🔴 Verified functional as designed, but structurally unable to address the current strongest leak channel | Doesn't defend against attacks that don't rely on verbatim quoting or response length — `decision_match` is the clearest example found in this project |
| **Decision obfuscation** (MIA, new, Revision 4) | Prepend a fixed hedging phrase to every response, unconditionally, pushing any yes/no/maybe commitment out of the leading-words window `decision_match` inspects | Pure text prepend, no model access | Directly targets the exact mechanism `decision_match` relies on; cheap; the first defense in this project shown to have a real, non-zero, mechanistically-explained effect | **Scoped to a positional check only** — provides no protection against an attacker who scans the whole response instead of just its first few words; overall composite reduction stays modest because `length_ratio` is untouched | Low | 🟢 `decision_match` AUC driven to exactly 0.5000 in both recorded runs (seeds 0, 42) — full neutralization of its target | Adaptive attacker scanning the full response text would likely see `decision_match` return close to its undefended strength; untested against that variant |
| **EWMA + binomial anomaly detection** (SFA) | Per-node sliding-window miss-rate tracking with statistical significance testing | See §2.C formula | Statistically principled; low false-positive rate by construction (`α=0.05`); doesn't need labeled attack data | Needs recalibration whenever the deployment's honest baseline changes (topology-, retrieval-, or corpus-dependent) | Medium | 🟢 1.0 detection rate at `n≥100`, post-recalibration | Needs ≥ 40 observations per node to warm up; blind to attacks below the calibrated threshold |
| **Suspicion-aware routing + blacklist + redundancy** (SFA) | Reorders/skips sources by detected suspicion and reputation | `SFAMitigation.route()`, `_SSMChain` reputation ledger | Measured 31-point accuracy recovery in the hop-limited scenario; degrades gracefully (redundant backup phase) | Provides zero measurable benefit when the deployment already queries every source every time (current `max_hops=3` config) | Medium | 🟢 1.00 vs. 0.69 at `max_hops=1` | Value is entirely conditional on hop-budget configuration — must be paired with a deliberately constrained `n_retrievers` to matter in production |

**Cross-cutting defensive categories present in this project:** Detection ✅ (SFA detector, SSM audit scan), Prevention ✅ (SSM contract caps), Rate limiting ✅ (SSM cooldown), Logging ✅ (all three write structured JSON logs), Output validation/sanitization ✅ (MIA). **Not present in this project (candidates for future work):** adversarial training, prompt filtering/input sanitization for the LLM prompt itself, human-in-the-loop review, sandboxing/isolation of data sources, formal model alignment/guardrails.

---

## 14. Security Recommendations

### High Priority
1. **Deploy the SSM-Score contract fix to any non-test environment before go-live.** This is the only one of the three defenses backed by a cryptographic/consensus-level guarantee rather than a statistical detector — it should be the first thing ported to a production contract.
2. **Decide and document the production `n_retrievers` / hop-budget policy for SFA.** The current deployment's full-redundancy configuration (`max_hops = total sources`) means the SFA mitigation currently provides zero measured benefit; if a future, larger deployment reduces the hop budget for latency/cost reasons, the mitigation becomes load-bearing and should be re-verified at that configuration first.
3. **Re-run the SFA detector calibration whenever the retrieval stack, corpus, or topology changes.** `HONEST_MISS`/`MISS_THRESH` are empirically fit to *this* deployment's measured baseline; changing the number of sources, the embedding model, or `top_k` could shift the true honest miss rate and silently re-break detection the same way the original mock-derived constants did.

### High Priority (MIA-specific, added after Revision 3)
4. **Build and test a decision-obfuscation defense.** The current sanitizer (length cap + verbatim-overlap redaction) is now confirmed to be structurally incapable of addressing the strongest measured leak channel (`decision_match` — whether the response commits to the correct yes/no/maybe token). A defense that hedges or delays the decision token regardless of confidence would directly target this channel; see `reports/MIA_Security_Analysis_Report.md` §13.2 for a concrete design sketch.
5. **Resolve the seed-42 outlier before treating MIA's `MEDIUM`-tier result as reliable.** 2 of 3 tested seeds now show real signal, but one remains near-random — more seeds (5–10) are needed to determine whether this is fixable via further re-weighting or an inherent property of this corpus/model pairing.

### Medium Priority
6. **Extend the MIA evaluation with an adaptive/optimization-based prober** (rather than only single-shot natural questions) — now that a real signal has been found (Revision 3), a more sophisticated adaptive attacker is likely to do at least as well, and probably better.
7. **Add a slow-drift test for the SSM-Score `MAX_DELTA_PER_UPDATE`/bounds design** — verify empirically how many legitimate small updates it would take to reach the `[MIN_SCORE, MAX_SCORE]` ceiling, and whether that number is achievable by a patient attacker within a realistic time window given the cooldown.
8. **Continue redesigning MIA's `certainty` signal.** Now based on yes/no/maybe commitment detection rather than hedge words, but still weak and occasionally inverted — not yet a reliable third signal.

### Low Priority
9. **Consolidate the duplicate/stale `attack/mia` folder** (superseded by `attack/Mia_attack`) to prevent future accidental imports of the unmaintained version.
10. **Add unit tests asserting `naive_route()` and `SFAMitigation.route()` use identical hop-counting semantics**, to prevent the "unfair comparison" class of bug from silently reappearing after future refactors.

---

## 15. Limitations

- **Threat model scope:** all three attacks assume a specific capability (compromised source key, or unauthenticated `/query` access). Attacks requiring a different capability (e.g., compromising the LLM service itself, or a supply-chain compromise of the embedding model) are out of scope.
- **Small-corpus effects:** the 500-document, 3-source corpus is small enough that full redundancy (`max_hops=3`) is cheap and effective; results here may not generalize to a much larger, more sparsely-replicated deployment.
- **SSM-Score bounds are not formally derived.** `MAX_DELTA_PER_UPDATE=5,000` and the `±1,000,000` absolute bounds were chosen to be comfortably above observed legitimate feedback magnitudes and comfortably below the attack's `+999,999` payload — not derived from a worst-case analysis of legitimate feedback distributions across a wider range of possible correctness/grounding signal combinations.
- **MIA no longer has a uniformly negative result, but it is not yet a settled positive one either.** Revision 3 measured `MEDIUM` privacy risk in 2 of 3 tested seeds — real progress over the original negative result — but the third seed remained near-random, and only 3 seeds have been tested with the current composite. Treat the 0.616 mean AUC-ROC as preliminary, not final (see the dedicated MIA report's §15 for the full caveat list).
- **SSM-Score and SFA still use SQuAD**; **MIA (from Revision 2 onward) uses PubMedQA** specifically because SQuAD's Wikipedia-derived content was found to invert MIA's signal via base-LLM pretraining overlap. Results for SSM-Score/SFA may differ on other corpora; MIA's results may differ again on a corpus with different overlap characteristics than PubMedQA.
- **MIA's current strongest signal (`decision_match`) is specific to yes/no/maybe-style QA datasets.** It would not directly transfer to a corpus without an enumerable, checkable answer format (e.g., free-text SQuAD-style answers) — a different signal would need to be found for those.
- **Blockchain simulation uses Hardhat's public test keys** for every role, standard for local development but not representative of a production key-management/HSM setup; the specific "attacker knows the target's private key" capability assumed for SSM-Score and SFA's `high_ssm_score` targeting would require an actual key compromise in production.

---

## 16. Future Improvements

- **MIA:** (1) resolve the seed-42 outlier — more seeds, and/or a non-linear/gated composite that treats `decision_match` as primary rather than one of four linearly-blended terms; (2) build and evaluate the decision-obfuscation defense motivated by §12.2/§13's mechanism-mismatch finding; (3) validate the current weights on a held-out seed set to check for overfitting to seeds 0/1/42; (4) implement an adaptive/optimization-driven prober to establish a stronger upper bound now that a real signal exists to build on.
- **SFA:** extend the detector to a Bayesian sequential-testing framework (updating a posterior over "is this node compromised" after every observation) rather than a fixed-window binomial test, which would reduce the 40-observation warm-up latency.
- **SSM-Score:** explore a commit-reveal or multi-party-attested feedback scheme, so that a single compromised key (source or orchestrator) cannot unilaterally write a score update at all, closing the residual "compromised orchestrator key" risk noted in §13.
- **Cross-cutting:** a unified, larger-scale evaluation harness that runs all three attacks against the same seed/topology/hop-budget matrix in one pass, to produce a single, directly comparable CIA-triad security scorecard for the whole system rather than three independently-scaled reports.
- **Scalability:** re-run all three evaluations against a larger, more realistic corpus (thousands of documents, more than 3 sources) to test whether the small-corpus effects noted in §15 hold at scale.

---

## 17. Final Conclusion

This report analyzed three attacks spanning all three legs of the CIA triad against Reliable-dRAG, and — critically — did not stop at "does the attack code run." For two of the three (MIA, SFA), the first pass of analysis uncovered that the *implementation itself* was silently invalid in ways that would have produced misleading conclusions if taken at face value: MIA's non-member data wasn't actually non-member data, and SFA's detector was calibrated against numbers that made detection mathematically impossible regardless of attack strength. Both were traced to their root cause with direct empirical measurement (diffing the corpus files; measuring the real honest miss rate against the live containers) rather than assumption, fixed, and re-verified live. **MIA went further still**, through two additional revisions after its first "corrected" result: the first corrected version (SQuAD corpus) turned out to be measuring an *inverted* signal caused by base-LLM pretraining overlap with the public evaluation dataset, not a true negative result; switching to a less-overlapping corpus (PubMedQA) and then fixing a blind diagnostic into what became this project's strongest measured signal (`decision_match`) moved MIA from a negative result to a real, if not yet fully reliable, `MEDIUM`-tier finding.

**Key findings:**
- **SSM-Score** is the most severe attack analyzed: a single blockchain transaction can produce total answer corruption (100% relative accuracy collapse in the historical undefended run). The defense built against it is the strongest in this report, because it is enforced by the smart contract itself and provably blocks the exact attack transaction, not merely observed to reduce its success rate.
- **MIA**, after four revisions, now shows a real but partial measured privacy risk: AUC-ROC 0.4992–0.7040 across 3 tested seeds (mean ≈0.613), with **2 of 3 seeds reaching `MEDIUM` privacy risk** — driven mainly by a signal (`decision_match`: does the response commit to the correct yes/no/maybe answer) that never inverted across any seed tested. This supersedes an earlier "negligible risk" conclusion that was itself an artifact of evaluating against a corpus (SQuAD) the deployed LLM had likely already memorized during pretraining. A structural fix tested for the remaining outlier seed (making `decision_match` gate the composite instead of being linearly blended) produced **no measurable change** — a genuine negative result showing the outlier's real cause is noise within, not across, decision-match groupings. A newly-built defense (`obfuscate_decision()`) was shown to **fully neutralize** `decision_match`'s own AUC to exactly 0.50 — the first defense in this project's history with a real, verified effect, though scoped specifically to a positional check.
- **SFA**'s real-world impact is entirely conditional on the routing hop budget: invisible at the currently deployed full-redundancy configuration, and a real, 31-percentage-point-recoverable availability risk the moment the hop budget is constrained — exactly the scenario the original threat model describes.

**Overall security posture:** the integrity attack surface (SSM-Score) is now well-defended with a provable, contract-level guarantee. The confidentiality attack surface (MIA) now shows a real, moderate risk in most tested conditions — a materially different conclusion than this report's earlier "negligible" finding, and one that should be treated as the current best estimate rather than a permanent verdict either way, given the small sample and one unresolved outlier seed. The availability attack surface (SFA) has a working, statistically sound detector and an effective mitigation — but its practical value is currently latent, waiting on a deployment decision (hop-budget/redundancy configuration) rather than a code fix.

**Lessons learned:** the single most consequential activity across this entire analysis was not writing attack or defense code — it was empirically re-deriving the numbers each defense implicitly assumed (the real honest miss rate, the real content overlap between "member" and "non-member" data sources) instead of trusting inherited constants and dataset splits. Every genuine bug found and fixed in this project traced back to exactly that gap between an assumed number and a measured one. MIA's journey adds a further lesson: even a *methodologically correct* evaluation can produce a misleading conclusion if the evaluation corpus itself has an unexamined relationship (here, pretraining overlap) with the system under test — and a diagnostic that measures exactly zero or a constant value is a signal to investigate the check itself, not necessarily evidence that nothing is there. Full detail on MIA's 3-revision history is in `reports/MIA_Security_Analysis_Report.md`.
