# Knowledge-Base (KB) Extraction Attack & Defense — Project Security Analysis Report

**Project:** Reliable-dRAG — Exploring Privacy-Preserving Approaches for Personalized Large Language Models (Distributed Retrieval-Augmented Generation, undergraduate thesis)
**Modules analyzed:** `attack/kb_extraction/run_attack.py` (attack) and `defense/kb_extraction_defense/` (defense) — this report covers **both**, in full, per the request.
**Scope:** This document reverse-engineers the actual KB extraction attack and its countermeasure as implemented in this repository, explains the theory behind every design decision, and interprets real evaluation data produced by running both against the live Docker deployment. It is written for a reader with limited prior security background, while remaining technically precise enough for a thesis committee or security reviewer.

---

## Executive Summary

Reliable-dRAG splits its document collection across three independent `drag_data_source` microservices, each exposing a public `POST /query` retrieval endpoint that a central orchestrator (`drag_llm_service`) calls to gather context before generating an answer. The **KB extraction attack** studied here asks a data-confidentiality question distinct from every other attack in this project: *can an adversary who only ever uses the system's own, legitimate-looking query interface systematically reconstruct the private document collection sitting behind it?*

The answer, measured directly against the live system in this repository across **three seeds (0, 42, 123)**, is **yes, substantially** — a modest sample of realistic questions (54 per source per seed, matched against the real corpus) recovered a stable **37.3%–41.9% mean extraction rate** across sources (per-source std as low as 0.7pp, up to 1.7pp — see §8.1), with **100% accuracy** (everything recovered was genuine, not noise), and **90–100% topic coverage**. A second attack phase probes the LLM itself for indirect leakage of retrieved content through its generated answers; on the two seeds with a clean, error-free sample, it recovers **30% of gold facts in the model's own words** (chunk recovery rate), not the near-zero figure an earlier, error-heavy pass had suggested (§8.2).

**A caveat that belongs up front, not buried in Limitations:** every extraction-rate figure above is a **gray-box, upper-bound** measurement — the probe question set is pre-filtered to guarantee each question's answer document is already confirmed present in the target source, so the attacker never wastes a probe on a topic that source doesn't have. This measures *post-reconnaissance query-targeting efficiency*, not discovery cost from zero prior knowledge. A complementary **blind/cold-start** condition (Phase B', probes drawn without that pre-filter) is also measured and reported side by side (§8.1): it tracks the gray-box number closely on the two SQuAD-domain sources (37.3–39.3% blind vs. gray-box), but is both lower and markedly noisier on the PubMedQA source (32.2% mean, std 6.1pp vs. 1.3pp gray-box) — see §12 for a discussion of why.

During this analysis, a **severe pre-existing configuration bug** was discovered and fixed: the attack script's probe-matching logic assumed all three data sources still served the same SQuAD-derived corpus, an assumption invalidated when `data-source-0`'s content was migrated to PubMedQA for an unrelated fix (`attack/Mia_attack`). This silently made the entire attack **100% non-functional** (it crashed before sending a single probe) until corrected in this analysis — see §4.6 and §12.3.

A purpose-built countermeasure, `defense/kb_extraction_defense`'s `QueryDiversityThrottle`, was also built, implemented, and evaluated in this analysis (no such defense existed in the repository beforehand). Measured live against `source_1` with its **originally-assumed, unmeasured threshold** (`max_topics_per_window=5`): it reduced `extraction_rate` from 0.286 to 0.000 by blocking 82.5% of the attacker's queries. That number is now superseded by a **calibrated re-run**: a purpose-built calibration tool (`calibrate_thresholds.py`) measured a real simulated legitimate-user topic-diversity baseline and derived `max_topics_per_window=4` from its 99th percentile — re-tested at that calibrated threshold (54 probes, matching the attack's own sample size), the defense blocks **74.1%** of queries and reduces `extraction_rate` from **0.390 to 0.134** (a 65.6% relative reduction, not full suppression). This is a more modest but considerably more defensible number than the original, since it is no longer just "aggressive enough to work on a 10-topic corpus with a hand-picked threshold" — see §8.3 and §15.

**Attack category:** Data-Confidentiality Attack / Knowledge-Base (Model) Extraction against a distributed RAG retrieval backend — the RAG-system analogue of a scraping/enumeration attack against a private database, executed entirely through the system's own legitimate API.

**Target system:** The three `drag_data_source` Flask `/query` endpoints (ports 8001–8003) and, for indirect leakage, `drag_llm_service`'s `/query` endpoint (port 9000).

**Security impact:** A significant fraction of each data source's private content can be reconstructed by an attacker holding (or having stolen) the system's single, shared, hardcoded API key — with no rate-based or behavioral anomaly detection standing in the way until the defense built in this analysis is deployed.

---

## 1. Project Overview

### 1.1 What the project does

Reliable-dRAG is a Distributed Retrieval-Augmented Generation (RAG) system. Three independent Flask microservices (`drag_data_source`), each holding its own document shard, are queried by a central orchestrator (`drag_llm_service`), which retrieves relevant passages and asks a language model to answer grounded in them. A blockchain component (`DragScores`, on a local Hardhat Ethereum node) tracks a reliability/usefulness score per source. A family of `attack/*`/`defense/*` modules empirically measures the system's resilience to specific threats — this report covers the pair targeting **content confidentiality**.

### 1.2 Goal of the attack

Unlike the DDoS attack (availability) or the sibling membership-inference attack (does the model know about *one specific* document), KB extraction asks: *how much of the entire private collection can be copied out, wholesale, using only the system's normal query surface?*

### 1.3 Threat model

| Property | Assumption |
|---|---|
| Attacker capability | Sends well-formed HTTP requests to the system's own public `/query` endpoints — no protocol exploitation, no injection, just volume and query selection |
| Attacker knowledge | **External/unauthenticated:** no key, relies on whatever the endpoint exposes without credentials. **Insider/authenticated:** holds the system's API key (a single, fixed, shared secret baked into `docker-compose.yml`) — modelled here as "stolen key," a realistic scenario for a hardcoded, non-rotated credential. Two sub-conditions are measured for the authenticated attacker: **gray-box** (Phase B — probes pre-filtered to guarantee a hit against the target source, an upper-bound on post-reconnaissance efficiency) and **blind/cold-start** (Phase B' — probes sampled without that pre-filter, the realistic floor for an attacker with no prior knowledge of which document subset this deployment hosts) |
| Attacker goal | Reconstruct as much of the network's aggregate document collection as possible, using the fewest queries |
| Ground truth | The evaluation harness has full visibility into every source's actual on-disk content for scoring; the attacker itself never sees this — an evaluation convenience, not part of the attack's real capability |
| Defender capability | Can observe per-client query volume and content (topic) diversity over time; the API-key check and Flask-Limiter's flat rate cap already exist; a topic-diversity-aware throttle (`defense/kb_extraction_defense`) was built and evaluated in this analysis |

### 1.4 Attack category

**Data-Confidentiality / Knowledge-Base Extraction Attack**, black-box and network-native — it uses no vulnerability, only the system's intended query functionality at scale.

---

## 2. Attack Theory

### 2.1 Black-box API enumeration

The attack is a classic **black-box enumeration** strategy: it treats each `/query` endpoint as an oracle that, given an input string, returns some subset of the underlying document collection. No internal system state is ever inspected — every extracted document comes back through the exact same interface a legitimate user would use. This is the same theoretical category as web-scraping or database-enumeration attacks against any search API.

### 2.2 Retrieval / embedding similarity as the extraction mechanism

`drag_data_source`'s retriever (`FastRetriever`) ranks documents by embedding-space similarity to the query. The attack does not need to know this internal mechanism — it only needs enough *diverse* probe queries that, in aggregate, their top-$k$ retrieved documents cover a large fraction of the underlying collection. This is why probe **diversity** (topic-balance, §2.3) matters more than probe **volume** alone: the retrieval oracle returns at most $k=5$ documents per call (server-enforced, see §4.6), so coverage is fundamentally bounded by how many *distinct* regions of embedding space the probe set touches.

### 2.3 Coverage-maximization via topic balance

A naive attacker sending very similar questions repeatedly would retrieve the same handful of documents over and over. The theoretically-motivated countermeasure to that inefficiency (from the attacker's side) is **stratified sampling across topics** — deliberately spreading query selection across the dataset's natural categories (here, SQuAD article titles) so the query budget is spent maximizing *new* coverage rather than depth on a few subjects. This is the same principle behind stratified sampling in statistics: a budget-constrained sample that is deliberately spread across strata (topics) estimates/covers the whole population more efficiently than one drawn uniformly at random from a naturally clustered distribution.

### 2.4 Set-theoretic ground truth for exact extraction

Because `/query` on the raw retrieval endpoint returns **verbatim stored text** (not an LLM paraphrase), correctness is a pure set-membership question:

$$
\text{Extracted}_{\text{correct}} = \text{Extracted} \cap \text{GroundTruth}
$$

This is why (§7, §12) `extraction_accuracy` is measured at exactly 1.0 throughout this analysis — there is no "noisy" or "hallucinated" document a raw retriever can return; a returned passage is either a genuine corpus document or nothing.

### 2.5 Paraphrase-tolerant leakage via embedding similarity and edit distance

The LLM-mediated leakage channel (Phase C, §5) is fundamentally different: the model generates *new* text that may restate a private fact without quoting it verbatim. Pure set-membership scoring would badly under-count this leakage, so the attack borrows the same **cosine similarity** and **Levenshtein edit-distance** machinery used elsewhere in this project (`attack/ddos_sim/nlg_metrics.py`, reused rather than reimplemented) to define a paraphrase-tolerant recovery criterion, the **Chunk Recovery Rate (CRR)**:

$$
\text{recovered} = \big[\, EM = 1 \,\big] \ \lor\ \big[\, SS \geq 0.8 \,\big] \ \lor\ \big[\, \text{EditSim} \geq 0.8 \,\big]
$$

- **Why cosine similarity ($SS$):** it measures the angle between two sentence-embedding vectors, invariant to their length/magnitude — the standard way to ask "do these two texts mean approximately the same thing," independent of exact wording (Embedding Similarity theory).
- **Why edit distance is included alongside $SS$:** cosine similarity can occasionally rate two texts as "close" purely due to shared domain vocabulary even when they differ substantively; word-level edit distance provides an orthogonal, surface-form-based corroborating signal. The disjunction ($\lor$) means a leak is caught if *either* signal fires — a deliberately generous (attacker-favorable, evaluation-conservative) definition, since the goal is to *not* under-count leakage.

### 2.6 Query efficiency as a cost/benefit ratio

$$
\text{query\_efficiency} = \frac{\text{unique correct extractions}}{\text{total queries sent}}
$$

This frames the attack in Decision Theory terms: an attacker operating under a query budget (whether a real rate limit or simple patience) wants to maximize genuine information gained per unit of "cost" (a query, which is also the attacker's primary observable/detectable footprint) — directly motivating the topic-diversity defense in §13, which specifically targets the numerator/denominator relationship this ratio describes.

---

## 3. Attack Architecture

```mermaid
graph TB
    subgraph "Phase A — Unauthenticated"
        A1[probe_source, no API key] --> A2[POST /query x N probes]
        A2 --> A3{HTTP 401?}
        A3 -->|yes, confirmed live| A4[0 docs extracted]
    end

    subgraph "Phase B — Authenticated (insider / stolen key)"
        B1[load_probe_sets<br/>per-source ground truth + probes] --> B2[probe_source, with API key]
        B2 --> B3[Optional: query_gate hook<br/>defense/kb_extraction_defense]
        B3 -->|allowed| B4[POST /query x N probes]
        B3 -->|blocked| B5[Skipped, counted]
        B4 --> B6[Dedup + collect unique docs]
        B6 --> B7[Compare vs ground-truth corpus<br/>exact set membership]
        B7 --> B8[extraction_rate / accuracy /<br/>query_efficiency / topic_coverage]
    end

    subgraph "Phase C — Indirect LLM leakage"
        C1[probe_llm_leakage] --> C2[POST drag_llm_service /query]
        C2 --> C3[Real generated answer]
        C3 --> C4[nlg_metrics: EM, SS, EditSim]
        C4 --> C5[CRR: exact OR SS>=0.8 OR EditSim>=0.8]
    end

    B8 --> D[JSON log: attack_logs/]
    C5 --> D
```

**Inputs:** a probe question set per data source (matched against that source's actual real content — see §4.6), the API key (or its deliberate absence), and — for the defense — a `QueryDiversityThrottle` instance consulted before each request.

**Outputs:** per-source extraction metrics (Phase A/B), per-probe generation-leakage metrics (Phase C), and, when the defense is active, per-query allow/block decisions and throttle statistics — all persisted as timestamped JSON under `attack_logs/`/`defense_logs/kb_extraction_defense/`.

**Components:** `probe_source()` (retrieval-layer extraction + ground-truth scoring), `probe_llm_leakage()` (generation-layer leakage + CRR scoring), `load_probe_sets()` (per-source ground truth and probe construction), `QueryDiversityThrottle` (the defense), `run_attack.py`/`run_defense.py` (CLI orchestration).

---

## 4. Implementation Analysis

### 4.1 `attack/kb_extraction/run_attack.py` — overview

**Purpose:** drive all three attack phases against the live Docker deployment and score the result against real ground truth.

### 4.2 `load_probe_sets(n, seed)` — per-source ground truth construction

**Major function**, and the centerpiece of the corpus-drift fix described in §4.6:

```python
def load_probe_sets(n=PROBE_SAMPLE_SIZE, seed=RANDOM_SEED) -> Dict[int, Dict]:
    ...
    squad_ds = load_dataset("rajpurkar/squad", split="train")
    for idx in (1, 2):
        ground_truth = _load_source_contexts(idx)          # each source's OWN real content
        ...match squad_ds against ground_truth, per source...
    pubmedqa_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ground_truth_0 = _load_source_contexts(0)
    ...match pubmedqa_ds against ground_truth_0...
    return {0: {...pubmedqa...}, 1: {...squad...}, 2: {...squad...}}
```

**Key design decision:** ground truth is loaded **per source**, directly from that source's own on-disk JSONL file (`_load_source_contexts`), not from one shared external reference. This is deliberately simple and always correct by construction: "ground truth" for an extraction attack is precisely *whatever that source actually serves right now* — regardless of whether its content happens to be clean, polluted, or an entirely different dataset from its neighbours.

**Gray-box vs. blind probe construction — the `blind` parameter.** `load_probe_sets(..., blind=False)` (the default, Phase B) drops every candidate question whose answer document isn't already confirmed present in the target source's own corpus (`ctx in ground_truth`) before sampling — every probe sent is a guaranteed hit. `load_probe_sets(..., blind=True)` (Phase B') samples instead from the *entire* matched-question pool for that source's domain, without that pre-filter, modelling an attacker with no prior confirmation of which passages this specific deployment hosts. `main()` runs both conditions back to back and reports them side by side (`phase_b_auth` / `phase_b_auth_blind` in the JSON log) rather than only ever reporting the gray-box ceiling — see §8.1 for the measured gap between the two.

### 4.3 `probe_source(base_url, questions, k, authenticated, ground_truth_contexts, context_to_title, query_gate)` — the extraction engine

```python
for q in questions:
    if query_gate is not None and not query_gate(q):
        blocked_count += 1
        continue
    ...
    resp = requests.post(f"{base_url}/query", headers=headers, json={"query": q, "k": k}, timeout=15)
    if resp.status_code == 401:
        return {"status": "unauthorized", ...}
    ...
    for doc in docs:
        text = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
        if text and text not in collected_docs:
            collected_docs.append(text)
```

Then, the ground-truth-scoring block added in this analysis:

```python
if ground_truth_contexts is not None:
    correct = [d for d in collected_docs if d in ground_truth_contexts]
    result["extraction_rate"] = len(correct) / len(ground_truth_contexts)
    result["extraction_accuracy"] = len(correct) / len(collected_docs)
    result["query_efficiency"] = len(correct) / len(questions)
    if context_to_title:
        result["topic_coverage"] = len(touched_topics) / len(all_topics)
```

**Design decision — the `query_gate` hook.** Rather than duplicating this probing loop inside the defense module, `probe_source()` accepts an optional callable consulted before every request. Returning `False` skips that query with zero HTTP calls made. This lets `defense/kb_extraction_defense/run_defense.py` reuse the exact same, real attack code for both the undefended and defended runs — guaranteeing an apples-to-apples comparison, and following the "reuse, don't duplicate" pattern already established elsewhere in this codebase (`selective_forward_sim`/`sfa_sim_defense`).

### 4.4 `probe_llm_leakage(questions, qa_by_question)` — indirect leakage via generation

```python
gold_answers = (qa_by_question or {}).get(q, {}).get("answers") or []
if answer.strip() and gold_answers:
    em = _nlg_exact_match(answer, gold_answers)
    best_ss = max(_nlg_semantic_similarity(answer, g) for g in gold_answers)
    best_eed = max(normalized_edit_distance_score(answer, g) for g in gold_answers)
    recovered = bool(em >= 1.0 or best_ss >= CRR_SIM_THRESHOLD or best_eed >= CRR_EDIT_THRESHOLD)
```

Only the first 20 sampled probe questions are sent to `drag_llm_service` (a deliberate scope limit — real LLM generation is slow, and this phase's purpose is a leakage *sample*, not exhaustive coverage). Every metric imported here (`exact_match`, `semantic_similarity`, `normalized_edit_distance_score`) is reused from `attack/ddos_sim/nlg_metrics.py`, avoiding a second, divergent scoring implementation.

### 4.5 `defense/kb_extraction_defense/query_diversity_throttle.py` — `QueryDiversityThrottle`

```python
class QueryDiversityThrottle:
    def check_and_record(self, client_key, query_text, topic):
        self._query_counts[client_key] += 1
        if self._cooldown_remaining.get(client_key, 0) > 0:
            self._cooldown_remaining[client_key] -= 1
            return False, "cooldown"
        window = self._window_for(client_key)   # deque(maxlen=window_size)
        window.append(topic)
        if self._query_counts[client_key] >= self.min_queries_before_check:
            if len(self.topics_in_window(client_key)) > self.max_topics_per_window:
                self._cooldown_remaining[client_key] = self.cooldown_queries
                return False, "topic_diversity_exceeded"
        return True, "ok"
```

**Key data structure:** a per-client `collections.deque(maxlen=window_size)` of the most-recent queries' topics — a fixed-size sliding window, so the "diversity" signal is always computed over recent behavior, not a client's entire lifetime history.

**Design decision — why this signal, not a numeric-score noise defense.** An idealized design for this attack family proposes injecting calibrated noise into a per-document similarity/confidence score. That mechanism is **architecturally inapplicable** here: `drag_data_source`'s `/query` returns raw document text (no score exposed for noising in a way that would meaningfully protect content, since the text itself is exactly what's being protected) and `drag_llm_service`'s `/query` returns only `{"response": text}` — no numeric field anywhere in the client-visible response. This same fact is independently established in this repository by `defense/mia_defense/mia_defense.py`'s own docstring for the sibling membership-inference attack; it applies identically here. The throttle instead observes the one signal a real client-facing defense actually has: **how many distinct topics has this client asked about recently**, using the SQuAD article title as a real, already-available proxy for "topic" (the corpus itself carries no separate category field).

### 4.6 The corpus-drift bug — found and fixed in this analysis

**Original code (before this analysis):**
```python
def _load_corpus_contexts():
    path = os.path.join(PROJECT_ROOT, "data", "polluted_token", "sources_0.jsonl")
    ...
```
`sources_0.jsonl` was assumed to hold SQuAD content (the docstring literally states "sources_0.jsonl is the 0%-polluted / clean copy"). Confirmed by direct content inspection during this analysis, `sources_0.jsonl` currently holds **PubMedQA** medical content instead — it was overwritten by `data/build_pubmedqa_corpus.py`, a script written for the unrelated `attack/Mia_attack` fix, and nothing in `kb_extraction` was ever updated to follow that change.

**Consequence:** `load_squad_probes()`'s match-against-`rajpurkar/squad` loop matched **zero** questions against the now-PubMedQA `sources_0.jsonl`, triggering an unconditional `RuntimeError` — a **100% failure rate**, not a partial degradation. This was confirmed directly during this analysis (see transcript: `RuntimeError: No SQuAD questions matched the loaded source documents`) before being fixed.

**Further discovery:** `sources_20.jsonl` and `sources_100.jsonl` (still SQuAD-derived) are no longer identical to *each other* either — direct comparison found only **207 of 500** records byte-identical between them, confirming the three "replica" sources have genuinely diverged over the project's history.

**Fix applied:** ground truth and probe questions are now built **per source** (§4.2), matching each source's own real, current content against the correct upstream dataset (PubMedQA for source 0, SQuAD for sources 1/2, matched independently for each). This is why every JSON result in §8 below reports metrics **per source**, not pooled — a pooled "network-wide" number is no longer a meaningful concept for this deployment.

### 4.7 Server-side cap mismatch — found and fixed

```python
# before: TOP_K = int(os.getenv("TOP_K", "10"))
# after:
TOP_K = int(os.getenv("TOP_K", "5"))   # drag_data_source/app/server.py:260 hard-caps k to 5
```
`drag_data_source/app/server.py` line 260 executes `k = min(int(data.get("k", 5)), 5)` — every prior run requesting `k=10` silently received at most 5 results per query, with the logged `"top_k": 10` field misrepresenting what was actually returned. This is, incidentally, itself a modest existing anti-scraping control (discussed further in §13.2).

---

## 5. Attack Workflow

**Step 1 — Build per-source ground truth and probes.** `load_probe_sets()` loads both the SQuAD and PubMedQA datasets once, matches each against the corresponding source's real on-disk content, and samples `PROBE_SAMPLE_SIZE` (default 54) questions per source.

**Step 2 — Phase A: unauthenticated attempt.** Every source is probed with **no** API key. A real HTTP 401 response is expected and confirms the API-key control is actually enforced (§8, §13.2).

**Step 3 — Phase B: authenticated extraction.** The same probes are re-sent **with** the (shared, hardcoded) API key — modelling an insider or an attacker who has obtained the key. Each response's documents are deduplicated and compared against that source's real ground-truth set.

**Step 4 — Score Phase B.** `extraction_rate`, `extraction_accuracy`, `query_efficiency`, and `topic_coverage` are computed per source.

**Step 5 — Phase C: indirect LLM leakage.** The first 20 probes (SQuAD-domain) are sent through `drag_llm_service`'s `/query` endpoint; each real generated answer is scored against the gold SQuAD answer with EM/SS/EditSim, and CRR is computed over however many probes returned a scoreable, non-empty response.

**Step 6 — Log.** A single JSON report (`attack_logs/attack_<timestamp>_kb_extraction.json`) captures all three phases plus a summary.

```mermaid
flowchart TD
    Start([Start]) --> Load[Load per-source probes<br/>+ ground truth]
    Load --> PhaseA[Phase A: probe without API key]
    PhaseA --> CheckAuth{HTTP 401<br/>on all sources?}
    CheckAuth -->|yes| PhaseB[Phase B: probe WITH API key]
    PhaseB --> Gate{Defense active?}
    Gate -->|yes| Throttle[query_gate: QueryDiversityThrottle]
    Gate -->|no| Send[Send request directly]
    Throttle -->|allowed| Send
    Throttle -->|blocked| Skip[Skip, count as blocked]
    Send --> Collect[Dedup + collect unique docs]
    Skip --> Collect
    Collect --> Score[Compare vs ground truth:<br/>extraction_rate/accuracy/topic_coverage]
    Score --> PhaseC[Phase C: probe drag_llm_service]
    PhaseC --> CRR[Score EM/SS/EditSim -> CRR]
    CRR --> Log[Write JSON log]
    Log --> End([End])
```

---

## 6. Evaluation Pipeline

**Input (Phase A/B):** a per-source probe question list, real; **input (Phase C):** the same questions, sent onward to the LLM.

**Prediction:** raw retrieved document text (Phase A/B) or the LLM's real generated answer text (Phase C).

**Ground truth:** the exact set of documents actually loaded into that source (`_load_source_contexts`) for Phase A/B; the SQuAD gold answer list for Phase C.

**Scoring:** Phase A/B uses exact set membership (`d in ground_truth_contexts`) — appropriate because raw retrieval returns verbatim stored text, so there is no paraphrase channel to account for. Phase C uses the paraphrase-tolerant CRR criterion (§2.5), appropriate because generation genuinely can restate content in different words.

**Aggregation:** per-source sums/ratios for Phase A/B; a single pooled rate (`chunk_recovery_rate`) over however many of the 20 Phase C probes actually returned a scoreable answer for Phase C.

---

## 7. Metrics Generation

#### `extraction_rate`
- **Purpose:** the headline "how much of the private collection did the attacker actually get" number.
- **Formula:** $\dfrac{|\text{Extracted} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$
- **Range:** 0–1.
- **Interpretation:** 🟢 low (<10%) = attack largely contained; 🟡 10–30% = meaningful partial recovery; 🔴 >30% = substantial fraction of the private collection reconstructed.
- **Real example:** source_0 (PubMedQA) `extraction_rate = 0.434` — 217 of 500 real documents recovered from 54 probes.
- **Security implication:** directly quantifies confidentiality loss in the units that matter — fraction of the actual private dataset, not an abstract score.
- **Limitation:** it is bounded above by how many of the source's documents are even *reachable* by any query at all (a document with no semantically-similar probe in the set can never be extracted, regardless of attack sophistication).

#### `extraction_accuracy`
- **Purpose:** "of what was exfiltrated, how much was genuine" — the attack's *precision*.
- **Formula:** $\dfrac{|\text{Extracted} \cap \text{GroundTruth}|}{|\text{Extracted}|}$
- **Range:** 0–1.
- **Real, measured value:** **exactly 1.000 on every one of the three sources.** This is an honest, structural finding, not a coincidence: raw-document retrieval either returns a genuine corpus passage or nothing — there is no hallucination/noise channel at this layer (contrast with Phase C, §12).
- **Security implication:** a defender cannot rely on "the attacker will collect a lot of junk/noise" as an incidental protection at this layer — every single document returned is real.

#### `query_efficiency`
- **Purpose:** the attacker's "return on investment" — genuine content recovered per query sent.
- **Formula:** $\dfrac{|\text{Extracted} \cap \text{GroundTruth}|}{\text{queries sent}}$
- **Real example:** source_0, $217/54 = 4.02$ correct documents per query (each probe returns up to `k=5` candidates, so this is a high, near-maximal efficiency).
- **Security implication:** the primary quantity a rate/volume-based defense tries to drive down — see §13.

#### `topic_coverage`
- **Purpose:** breadth — did the attack touch most subject areas or concentrate on a few?
- **Formula:** $\dfrac{|\{\text{topics of correctly-extracted docs}\}|}{|\{\text{all topics in ground truth}\}|}$ (SQuAD article title used as "topic"; **not computed for PubMedQA**, which carries no comparable field — reported as `null`/`n/a` rather than guessed).
- **Real example:** source_1 (SQuAD, sources_20) `topic_coverage = 1.0` — all 10 real article topics touched with only 54 probes.
- **Security implication:** high coverage with a small query budget is itself informative — it shows the topic-balanced sampling strategy (§2.3) is working as theoretically predicted.

#### Chunk Recovery Rate (`chunk_recovery_rate`, CRR)
- **Purpose:** the headline Phase C metric — does the LLM leak private content in its own words, not just verbatim.
- **Formula:** fraction of scored probes where $EM \geq 1$ OR $SS \geq 0.8$ OR $\text{EditSim} \geq 0.8$ (§2.5).
- **Range:** 0–1.
- **Real, measured value:** **0.0** over 5 scored probes.
- **Interpretation:** 🟢 in this specific measurement — but see §12 for why this number should not be over-interpreted given the small scored sample.

#### Average Semantic Similarity / Edit Similarity (`avg_semantic_similarity`, `avg_edit_similarity`)
- **Formula:** mean, over scored probes, of the best-matching-gold cosine similarity / normalized edit similarity (identical formulas to §7.2 of the DDoS report — reused, not reimplemented).
- **Real, measured value:** `avg_semantic_similarity = 0.517`, `avg_edit_similarity = 0.303`.
- **Interpretation:** moderate semantic closeness without crossing the 0.8 CRR threshold — the LLM's answers are topically related but not close enough to count as a genuine leak under this (deliberately strict) criterion.

#### Raw volume counters (`docs_extracted`, `chars_extracted`, `kb_size_mb`, `total_leaked_chars`)
- **Purpose:** the simplest, most literal measure of "how much data left the system" — useful for a plain-language executive summary even without any ground-truth interpretation.
- **Real example:** 595 total documents, 590,587 characters (≈0.56 MB) extracted across all three sources from just 162 total probes (54 × 3).

---

## 8. JSON Metrics Analysis

### 8.1 Attack — Phase B (gray-box, authenticated) and Phase B' (blind/cold-start), per source, 3 seeds

Sources: `attack_logs/kb_extraction/attack_2026-07-19_01-{08-57,09-50,11-58}_kb_extraction.json` (seeds 0, 42, 123 respectively), each a genuine 54-probe-per-source run against the live deployment, gray-box and blind conditions both measured in the same run.

**Phase B — gray-box (probes guaranteed to hit the target source):**

| Source | Dataset | seed=0 | seed=42 | seed=123 | mean ± std | extraction_accuracy | topic_coverage |
|---|---|---|---|---|---|---|---|
| source_0 | pubmedqa | 0.402 | 0.434 | 0.422 | **0.419 ± 0.013** | 1.000 (all seeds) | n/a (no topic field) |
| source_1 | squad | 0.402 | 0.390 | 0.386 | **0.393 ± 0.007** | 1.000 (all seeds) | 1.00 (10/10, all seeds) |
| source_2 | squad | 0.356 | 0.366 | 0.396 | **0.373 ± 0.017** | 1.000 (all seeds) | 0.90–1.00 |

**Phase B' — blind/cold-start (probes sampled without the ground-truth pre-filter):**

| Source | Dataset | seed=0 | seed=42 | seed=123 | mean ± std | vs. gray-box mean |
|---|---|---|---|---|---|---|
| source_0 | pubmedqa | 0.408 | 0.284 | 0.274 | **0.322 ± 0.061** | −0.097 (23% relative drop, and 4.7× the gray-box variance) |
| source_1 | squad | 0.372 | 0.390 | 0.384 | **0.382 ± 0.008** | −0.011 (~3% relative drop, comparable variance) |
| source_2 | squad | 0.376 | 0.380 | 0.364 | **0.373 ± 0.007** | ~0.000 (no measurable drop) |

| Metric | Value | Meaning | Interpretation | Impact |
|---|---|---|---|---|
| `extraction_rate`, gray-box (source_0) | 0.419 ± 0.013 | ~42% of the entire real PubMedQA collection reconstructed, stable across 3 seeds | 🔴 Substantial confidentiality loss | The single largest real-data finding in this report |
| `extraction_rate`, blind (source_0) | 0.322 ± 0.061 | Still recovers ~32% with zero reconnaissance, but far noisier seed-to-seed | 🟡 Real floor, lower and less predictable than the gray-box ceiling | Confirms gray-box numbers are an upper bound, not the only realistic threat level — see §12 |
| `extraction_accuracy` (all sources, both conditions) | 1.000 | Every extracted document was genuine | 🔴 No incidental noise protection exists at this layer | Confirms the attack's haul is 100% usable, not diluted by junk, regardless of probe-selection condition |
| `topic_coverage` (source_1) | 1.00 | All 10 real topics touched, every seed | 🔴 Full breadth achieved with a modest query budget | Validates the topic-balanced-sampling theory (§2.3) empirically |
| Phase A status (all sources) | `unauthorized` | Real HTTP 401 on every unauthenticated attempt | 🟢 The one control that is working correctly | Confirms the API-key gate is live and enforced |

**Performance Summary:** the gray-box (post-reconnaissance) attack achieves a stable 37–42% extraction rate with perfect accuracy and near-total topic coverage, consistent across all 3 tested seeds — the multi-seed spread (0.7–1.7 percentage points of std) is tight enough that a single-seed run would not have been misleading, but it's now backed by evidence rather than assumption. The blind/cold-start floor is meaningfully lower and much noisier on the PubMedQA source specifically, while barely distinguishable from the ceiling on both SQuAD sources.

**Strength Analysis:** the topic-balanced probe strategy is empirically validated — `topic_coverage = 1.0` on source_1 shows the attack achieves full breadth cheaply, exactly as the theory in §2.3 predicts. The gray-box numbers are also now known to be a genuine, tight, reproducible ceiling rather than a single lucky sample.

**Weakness Analysis:** the attack is entirely dependent on holding a valid API key — Phase A's clean, universal 401 rejection shows the *unauthenticated* path is fully closed; all measured damage comes from the authenticated/insider scenario, which is a narrower (though realistic, given the key is a fixed, non-rotated shared secret) threat model. Separately, every gray-box number in this report is an **upper bound on post-reconnaissance efficiency**, not a claim about zero-knowledge attacker capability — the blind condition shows that ceiling holds up well for the SQuAD sources but overstates real-world efficiency by roughly a quarter on the PubMedQA source.

### 8.2 Attack — Phase C (LLM indirect leakage), 3 seeds

**This section supersedes the original single-run Phase C result.** The run originally cited here (`error_count=15/20`) predated a Docker bind-mount fix (§12.3) that had `sources_20`/`sources_100` silently reading a stale, different checkout's data — a confound that also happened to inflate Phase C's error rate. Re-run post-fix at 3 seeds:

| Seed | `probes_sent` | `error_count` | `chunks_scored` | `chunk_recovery_rate` (CRR) | `avg_semantic_similarity` | `avg_edit_similarity` | `total_leaked_chars` |
|---|---|---|---|---|---|---|---|
| 0 | 20 | 14 | 6 | 1.000 | 1.000 | 1.000 | 54 |
| 42 | 20 | **0** | **20** | **0.300** | 0.615 | 0.325 | 179 |
| 123 | 20 | **0** | **20** | **0.300** | 0.546 | 0.300 | 267 |

**Seed 0 is the outlier, not the other two.** Seeds 42/123 score all 20/20 probes cleanly (the bind-mount fix genuinely closed the original error-rate problem for those runs); seed 0 still shows 14/20 errors and its `CRR=1.000` is drawn from only 6 scored probes — too small a sample to trust, and *not* representative of the two clean runs. **The headline number for this attack should be CRR ≈ 0.30 (seeds 42/123), not the 0.0 originally reported and not seed 0's 1.000** — the original "zero leakage" finding was an artifact of an error-heavy n=5 sample from a corrupted data-mount, not a real property of the system. A genuine, non-trivial fraction (30%) of Phase C's scored probes leak a gold fact into the LLM's own generated wording, closely matching or exceeding the strictness threshold (§2.5) even without verbatim quoting.

### 8.3 Defense comparison — original (unmeasured threshold) vs. calibrated threshold

**Run 1 — original, hand-picked threshold.** Source: `defense_logs/kb_extraction_defense/defense_2026-07-10_20-08-34_kb_extraction_throttle.json`, source_1 (SQuAD), 40 probes, `max_topics_per_window=5`, `min_queries_before_check=8` — chosen without reference to any measured legitimate-user behavior.

| Metric | Attack Only | Attack + Defense | Reduction |
|---|---|---|---|
| `extraction_rate` | 0.286 | 0.000 | −0.286 (100%) |
| `topic_coverage` | 1.00 | 0.00 | −1.00 (100%) |
| `docs_extracted` | 143 | 0 | −143 |
| queries blocked | 0/40 | 33/40 (82.5%) | — |

**Run 2 — calibrated threshold (supersedes Run 1 as the headline result).** `defense/kb_extraction_defense/calibrate_thresholds.py` simulates a legitimate user as one client asking `window_size` questions about 1–3 topics of genuine interest (2,000 simulated sessions), contrasted against the attacker's own real, uniform-across-all-topics sampling strategy (also 2,000 simulated sessions). Measured on source_1 (10 real topics): legitimate sessions touch p50=2, p99=3 distinct topics; the attacker touches p50=9, p90–p100=10 — a clean separation, giving a *derived*, not guessed, `max_topics_per_window = ceil(p99_legitimate) + 1 = 4`. Re-tested at this calibrated threshold, `min_queries_before_check=15`, 54 probes (matching the attack's own default sample size): source `defense_logs/kb_extraction_defense/defense_2026-07-19_00-07-11_kb_extraction_throttle.json`.

| Metric | Attack Only | Attack + Defense | Reduction |
|---|---|---|---|
| `extraction_rate` | 0.390 | **0.134** | **−0.256 (65.6% relative)** |
| `topic_coverage` | 1.00 | **0.90** | −0.10 (10%) |
| `docs_extracted` | 195 | **67** | −128 |
| queries blocked | 0/54 | **40/54 (74.1%)** | — |

| Metric | Value | Meaning | Interpretation | Impact |
|---|---|---|---|---|
| `total_flagged` (Run 1) | 2 | The throttle's diversity threshold was crossed twice (once on the way into the first cooldown, once immediately after it expired) | 🟢 The defense fired promptly | See §12.4 for why this pattern (flag → cooldown → immediate re-flag) occurs |
| `queries_blocked_fraction` (Run 1) | 0.825 | Over four-fifths of the attacker's probe budget was wasted on blocked, silently-rejected requests | 🟡 Strong suppression, but the threshold was never validated against real legitimate-user behavior | Superseded below — kept for comparison, not the number to quote as final |
| `queries_blocked_fraction` (Run 2, calibrated) | 0.741 | Nearly three-quarters of the attacker's budget still wasted, using a threshold derived from a measured legitimate-user model instead of a guess | 🟢 Strong suppression that survives being held to a real standard | The number to actually cite: `extraction_rate` cut by two-thirds (0.390→0.134), not driven all the way to zero |

**Performance Summary:** the original, hand-picked threshold (`max_topics_per_window=5` against a 10-topic corpus) drove `extraction_rate` to exactly zero — a suspiciously clean result that turned out to be an artifact of the threshold being set relative to a small topic space with no reference to real usage. Once the threshold is derived from an explicit, measured legitimate-user model instead, the defense is still strong (74.1% of queries blocked, extraction cut by two-thirds) but no longer perfect — a materially more defensible, if less dramatic, result. `min_queries_before_check` was also raised (8→15) as part of the same calibration pass, giving legitimate short sessions more room before being evaluated at all.

---

## 9. Performance Dashboard

```
EXTRACTION RATE per source, gray-box (Phase B, mean of 3 seeds)
  source_0 (pubmedqa) ████████░░░░░░░░░░░░  41.9% ± 1.3pp  🔴
  source_1 (squad)    ███████░░░░░░░░░░░░░  39.3% ± 0.7pp  🔴
  source_2 (squad)    ███████░░░░░░░░░░░░░  37.3% ± 1.7pp  🔴

EXTRACTION RATE per source, blind/cold-start (Phase B', mean of 3 seeds)
  source_0 (pubmedqa) ██████░░░░░░░░░░░░░░  32.2% ± 6.1pp  🟡 (well below gray-box, noisy)
  source_1 (squad)    ███████░░░░░░░░░░░░░  38.2% ± 0.8pp  🔴 (tracks gray-box closely)
  source_2 (squad)    ███████░░░░░░░░░░░░░  37.3% ± 0.7pp  🔴 (tracks gray-box closely)

EXTRACTION ACCURACY (all sources, both conditions, all seeds)
  ████████████████████ 100.0%  🔴 (no noise channel at this layer)

TOPIC COVERAGE
  source_1 ████████████████████ 100%  🔴
  source_2 ██████████████████░░  90-100%  🔴

DEFENSE EFFECT, calibrated threshold (source_1, 54 probes — see §8.3)
  Extraction rate, undefended  ███████░░░░░░░░░░░░  39.0%
  Extraction rate, defended    ██░░░░░░░░░░░░░░░░░░  13.4%  🟢
  Queries blocked              ███████████████░░░░░  74.1%  🟡

PHASE C — Indirect LLM leakage (seeds 42/123, clean 20/20 samples)
  Chunk Recovery Rate  ██████░░░░░░░░░░░░░░  30.0%  🟡 (real signal — supersedes the earlier 0.0% artifact)
  Avg Semantic Sim.    ████████████░░░░░░░░  58.1%  🟡
```

| Indicator | Meaning |
|---|---|
| 🟢 Excellent (low attack success / strong defense) | |
| 🟡 Moderate | |
| 🔴 Poor (attack succeeding significantly) | |

---

## 10. Performance Interpretation

| Metric | If it **increases** | If it **decreases** |
|---|---|---|
| `extraction_rate` | More of the private collection reconstructed — worse for the defender | Attack contained, or defense working |
| `extraction_accuracy` | Attacker's haul is cleaner/more usable | Attacker is wasting effort on noise (not observed at this layer, see §7) |
| `topic_coverage` | Attack achieving broader reconnaissance of the whole collection | Attack confined to a narrow subject area (lower overall confidentiality risk even at the same raw `extraction_rate`) |
| `query_efficiency` | Attacker extracting more per query — a more dangerous, harder-to-rate-limit attacker | Attacker burning queries with little to show — exactly what the throttle in §13 is designed to force |
| `chunk_recovery_rate` (CRR) | LLM increasingly leaking private facts in its own words | Generation layer not restating retrieved content closely enough to count as a leak |
| `queries_blocked_fraction` (defense) | Defense actively suppressing attacker traffic | Defense inactive or attacker below its detection threshold |

**Reliability/Security:** `extraction_accuracy = 1.0` combined with rising `extraction_rate` is the clearest possible confidentiality-loss signal — there is no ambiguity or noise to discount.
**Detection:** `topic_coverage` climbing rapidly relative to query count is exactly the anomaly the throttle in §13 is built to detect — a legitimate user's natural topic distribution should be far narrower over a comparable number of queries.

---

## 11. Experimental Methodology

| Aspect | Detail |
|---|---|
| **Datasets** | `qiaojin/PubMedQA` (`pqa_labeled`, train split) for source_0; `rajpurkar/squad` (train split) for sources 1/2 — matched independently per source against that source's real on-disk content |
| **Ground truth** | Direct, per-source read of `data/polluted_token/sources_{0,20,100}.jsonl` |
| **Probe sample size** | 54 per source (attack, both Phase B and B'), 40 (defense Run 1, original threshold), 54 (defense Run 2, calibrated threshold) |
| **`top_k`** | 5 (corrected from a stale default of 10 — see §4.7) |
| **Random seeds** | **0, 42, 123** for the attack (Phase A/B/B'/C, §8.1–8.2); defense comparison (§8.3) is still single-seed — see §15 |
| **Defense thresholds (evaluated)** | Run 1 (original, unmeasured): `window_size=30`, `max_topics_per_window=5`, `min_queries_before_check=8`, `cooldown_queries=20`. Run 2 (calibrated via `calibrate_thresholds.py`): `max_topics_per_window=4` (derived from a simulated legitimate-user 99th percentile), `min_queries_before_check=15`, same `window_size`/`cooldown_queries` |
| **Environment** | Docker Compose (`hardhat-node`, `data-source-0/20/100`, `llm-service`), evaluated live, both from a native Windows environment and the user's WSL2 `.venv` |
| **API key** | The single, fixed, shared secret configured in `docker-compose.yml` (`reliable-derag-secret-2026`) — used as-is for the authenticated/"insider" phase, deliberately withheld for the unauthenticated phase |
| **Reproducibility** | `RANDOM_SEED={0,42,123} python attack/kb_extraction/run_attack.py`; `python defense/kb_extraction_defense/run_defense.py --probe_sample_size 54 --max_topics_per_window 4 --min_queries_before_check 15` (calibrated); `python defense/kb_extraction_defense/calibrate_thresholds.py` to reproduce the calibration itself |

---

## 12. Results Discussion

### 12.1 Why `extraction_accuracy` is exactly 1.0 on every source

This is a structural property of the retrieval layer, not a measurement coincidence: `FastRetriever.search()` can only ever return passages that genuinely exist in its own index. There is no generative step, so there is no mechanism by which a "wrong" or fabricated document could be returned. This is a meaningful contrast with Phase C, where generation *can* diverge from ground truth — which is exactly why extraction_accuracy-style scoring is inappropriate for Phase C and CRR was built instead.

### 12.2 Why the original Phase C run had 15/20 errors, and why that finding didn't survive

`probe_llm_leakage()` sends real generation requests to `drag_llm_service`, which internally fans out to all three data sources and waits up to 10 seconds per source (`drag_llm_service/app/server.py`'s `query_data_sources()`). The original 75% error rate was *not*, as first suspected, ordinary transient load — it was traced (§12.3) to `sources_20`/`sources_100` being served from a stale Docker bind-mount pointing at a different, older checkout of this repository. Once that mount was corrected, Phase C's error rate dropped to 0/20 on two of three re-tested seeds (§8.2). Seed 0 still shows a high (14/20) error rate in the corrected environment, so some genuine transient-load sensitivity remains — but it is no longer the dominant explanation, and it no longer justifies treating `chunk_recovery_rate` as unmeasurable: two of three seeds now provide a full, clean 20-probe sample.

### 12.3 Why the corpus-drift bug (§4.6) matters beyond "a crash" — and a second, deeper instance of the same failure mode

This is not merely a software bug — it is a **security-relevant configuration drift** with real implications: a defender who ran this attack script *before* this analysis's fix would have concluded, incorrectly, that the system was completely safe from this attack ("it crashes, so extraction must not be possible"), when the true state was "the attack script itself is broken, the system's actual exposure was never measured." This is a cautionary example of why an attack/evaluation harness's own correctness must be verified independently of its pass/fail output — a script that always fails to run is indistinguishable, from a log-reading standpoint, from a script confirming the system is secure.

A second, more subtle instance of the identical failure mode was found later in this same analysis: `sources_20`/`sources_100` extraction rates were, at one point, reading exactly `0.000` under a live run. Root cause was a **Docker bind-mount pointing at a separate, older clone of this repository** on the same machine (a stale `docker compose up` target, not touched by a container restart, only by recreating the containers against the correct mount) — the running containers were serving pre-migration content while this repo's ground-truth loader read the current, migrated files. `source_0` coincidentally matched (same record count both sides), which is why only two of three sources showed the failure. This also fully explains, and fixes, §12.2's Phase C error rate. The lesson repeats: an extraction number of exactly `0.000`, or an error rate that looks like "the system is broken," is a signal to verify the *harness's* environment before concluding anything about the system under test.

### 12.4 Why the original defense threshold blocked so aggressively (82.5%) on this corpus — and why the calibrated one (74.1%) is the number to trust

With only 10 real topics available and the original, unmeasured `max_topics_per_window=5`, any topic-balanced attacker crosses the diversity threshold almost immediately (well before `window_size=30` queries accumulate). Once flagged, the client is blocked for `cooldown_queries=20` queries; because the sliding window is **not cleared** on cooldown, the moment the cooldown expires the window still contains the same diverse topic history, so the throttle **re-flags immediately** — explaining the observed pattern (`total_flagged: 2`, `total_blocked: 33`) of one long near-continuous block rather than several short ones. This is an intentional, defensible design choice (a legitimate long-lived diverse user would also need re-evaluating, not silently re-trusted) but means the effective, real-world block duration is longer than `cooldown_queries` alone would suggest on a small-topic corpus.

This is exactly why the threshold was recalibrated (§8.3, Run 2): a threshold chosen without reference to real legitimate behavior can't be distinguished from a threshold that happens to work only because it was set aggressively relative to a small, fully-known topic space. The calibrated run's 74.1% block rate and 65.6% relative extraction-rate reduction are lower than the original 82.5%/100% — that is the *expected and correct* effect of replacing a guess with a measurement, not a regression in the defense's quality.

### 12.5 The gray-box/blind gap is real but domain-dependent, not uniform

§8.1's blind-mode (Phase B') numbers track the gray-box ceiling closely on both SQuAD sources (within 1-3 percentage points, comparable variance) but sit noticeably lower and far noisier on the PubMedQA source (0.322 ± 0.061 vs. 0.419 ± 0.013 gray-box — nearly 5× the standard deviation). A plausible mechanism: SQuAD's per-article structure means a randomly-sampled question is still fairly likely to land near this source's actual 500-document subset regardless of pre-filtering, while PubMedQA's medical-question space is broader and more topically dispersed relative to the 500-document subset actually loaded, so which specific unfiltered sample a seed happens to draw matters more. This was not independently verified against the corpus's topic structure — flagged as the explanation that best fits the observed data, not a confirmed mechanism.

---

## 13. Defense Mechanisms

### 13.1 `QueryDiversityThrottle` (built and evaluated in this analysis)

| Aspect | Detail |
|---|---|
| **Description** | Per-client sliding-window detector: flags and temporarily blocks a client once the number of *distinct topics* touched in its recent query history exceeds a threshold |
| **How it works** | `deque(maxlen=window_size)` of recent topics per client; on each query, if `len(distinct topics in window) > max_topics_per_window` (after `min_queries_before_check` queries), the client enters a `cooldown_queries`-length block |
| **Advantages** | Targets the specific gap Flask-Limiter's flat volume cap leaves open (a topic-balanced attacker can stay under 60/min while still touching an anomalous number of subjects); reuses real, already-available "topic" ground truth (SQuAD article title) rather than inventing a synthetic signal; bounded cooldown, not a permanent ban |
| **Disadvantages** | No numeric-score-noise option exists for this system (§4.5) — the defense can only *block*, not subtly degrade, a suspected attacker, which is a coarser, more detectable intervention; on a small-topic corpus (this deployment: 10 topics), thresholds tuned for a larger, more diverse real corpus could over-trigger |
| **Implementation complexity** | Low — a single sliding-window class, no changes to the retrieval or generation pipeline required; designed to be wired into a Flask `before_request` hook (same pattern already used by `check_api_key()`) |
| **Effectiveness (measured)** | Original, unmeasured threshold: `extraction_rate` 0.286→0.000, 82.5% blocked. **Calibrated threshold (headline number, §8.3):** `extraction_rate` 0.390→0.134 (65.6% relative reduction), `topic_coverage` 1.00→0.90, 74.1% blocked — strong but no longer total suppression |
| **Residual risk** | An attacker aware of the threshold could deliberately narrow its topic diversity (sacrificing coverage for stealth) to stay under `max_topics_per_window` — the defense would not detect this slower, narrower-scope variant; not evaluated in this analysis |

### 13.2 Existing real infrastructure controls

| Mechanism | Status | Effectiveness against this attack |
|---|---|---|
| API-key gate (`check_api_key`) | ✅ Deployed, confirmed working (Phase A: universal 401) | Fully blocks the unauthenticated path; provides no protection once the (single, shared, hardcoded) key is known |
| Flask-Limiter (60/min per IP) | ✅ Deployed | Caps raw request volume but does not address topic-balanced extraction specifically (the same gap `QueryDiversityThrottle` was built to close) |
| Server-side `k<=5` cap (`server.py:260`) | ✅ Already present (confirmed and corrected for in this analysis, §4.7) | A modest, incidental anti-scraping control — halves the per-query yield versus an uncapped retriever, though not designed as a deliberate defense |

---

## 14. Security Recommendations

**High Priority**
1. Rotate the shared, hardcoded API key (`reliable-derag-secret-2026`) and issue distinct, revocable per-client credentials — the entire Phase B/insider threat model in this report exists because the key is fixed and universal.
2. Deploy `QueryDiversityThrottle` server-side (a `before_request` hook analogous to the existing `check_api_key()`), not just as an offline evaluation script.

**Medium Priority — done**
3. ~~Recalibrate `max_topics_per_window`/`min_queries_before_check` against this deployment's actual topic count and a measured legitimate-user query-diversity baseline before relying on the module's generic defaults in production.~~ **Done:** `calibrate_thresholds.py` derives `max_topics_per_window=4` from a simulated legitimate-user 99th percentile; re-tested result in §8.3, Run 2.
4. ~~Extend Phase C's leakage evaluation to a larger, more reliable sample.~~ **Done:** re-run at 3 seeds post-infra-fix; 2 of 3 now score a clean 20/20 (§8.2). Seed 0 still shows a high error rate — not fully closed, see §15.

**Low Priority**
5. Add a per-client cooldown-window reset/decay so a flagged-then-cleared legitimate client isn't immediately re-flagged purely because its stale window hasn't cycled out yet (§12.4).
6. Consider a single-victim-targeting evaluation mode (currently absent — extraction is only measured network-wide per source) to answer "how easily can one specific document be extracted," a different and arguably more realistic confidentiality question for a specific sensitive record.
7. Extend the defense's calibration (Recommendation 3) to source_0 and source_2 — it was only run against source_1; the other two sources still use unvalidated thresholds if the defense were deployed against them.

---

## 15. Limitations

- **No single-victim targeting mode** — ground truth and metrics are computed per-source (network-wide), not for one specific target document; the attack's realistic capability against a *particular* sensitive record is not directly measured.
- **Corpus heterogeneity limits cross-source comparison** — since the three sources now serve genuinely different content (§4.6), comparing their `extraction_rate` numbers side by side (§8.1) is informative but not a controlled, apples-to-apples comparison.
- **Gray-box (Phase B) numbers are an upper bound, not a claim about a zero-knowledge attacker** — see the Executive Summary caveat and §12.5. The blind/cold-start floor (Phase B') is now measured and reported alongside, closing most of this gap, but it still assumes the attacker correctly guessed the corpus's *domain* (PubMedQA/SQuAD), just not which specific ~500-document subset is loaded.
- **Phase C's seed-0 sample is still small and error-heavy (6/20 scored)** — unlike seeds 42/123 (20/20 clean), seed 0's high error rate persisted even after the Docker bind-mount fix (§12.2, §12.3); its `CRR=1.000` should not be quoted, only the seeds-42/123 mean (~0.30).
- **CRR thresholds (0.8/0.8) are conventional, not empirically calibrated** for this specific deployment's model and corpus.
- **Defense comparison (§8.3) is still single-seed** — unlike the attack (§8.1/§8.2, now 3 seeds), neither the original nor the calibrated defense run has been repeated across seeds, so no variance estimate exists for `extraction_rate_reduction`/`queries_blocked_fraction`.
- **Defense calibration was only performed against source_1** — source_0 (no topic field, PubMedQA) and source_2 were not separately recalibrated; deploying the defense against them with source_1's threshold is unvalidated.
- **Attacker identity is still a valid peer/keyholder**, not a fully unauthenticated outsider with zero system access — consistent with, and inherited from, the same scoping limitation noted in the design documentation this attack family follows.

---

## 16. Future Improvements

- Wire `QueryDiversityThrottle` into `drag_data_source/app/server.py`'s real request path (a drop-in `before_request` patch, following the exact pattern `defense/mia_defense/README.md` already documents for its own text-level defenses).
- Replace the fixed-size query-count window with a genuine time-based sliding window, reducing sensitivity to an attacker's request pacing.
- Extend the legitimate-user calibration (`calibrate_thresholds.py`) to source_0 and source_2, and re-run the defense comparison across seeds to put error bars on `extraction_rate_reduction`.
- Build a single-victim-targeting attack/evaluation mode to answer the narrower, arguably more realistic "can one specific sensitive document be extracted" question.
- Investigate why Phase C's seed-0 run still shows a high error rate post-infra-fix while seeds 42/123 don't, before treating the 20/20 clean sample as the reliably reproducible norm.

---

## 17. Final Conclusion

This report reverse-engineered, fixed, and empirically evaluated both the KB extraction attack and a newly-built countermeasure against the live Reliable-dRAG deployment, across **three seeds** for the attack. In the process, it uncovered and corrected two independent, previously-undetected configuration bugs that had silently invalidated earlier measurements — a stale corpus assumption that made the attack's evaluation 100% non-functional (§4.6), and a Docker bind-mount pointing at a separate, older checkout that had zeroed out two sources' extraction rates and starved Phase C's sample (§12.3) — findings as significant to this project's security posture as the attack's own measured results, since a broken or environmentally-contaminated evaluation is indistinguishable from a genuinely secure system without independent verification.

**Key findings:** with only a modest, realistic probe budget (54 questions) and the system's single shared API key, an attacker reconstructs a stable **37.3–41.9% mean extraction rate** (std 0.7–1.7pp across 3 seeds) of each data source's real private content with **perfect accuracy** and up to **100% topic coverage** — this is a **gray-box, upper-bound** measurement (probes guaranteed to hit the target corpus); a blind/cold-start condition, also now measured, tracks it closely on the SQuAD sources and sits meaningfully lower and noisier (32.2% ± 6.1pp) on the PubMedQA source. Phase C's LLM-leakage channel, initially reported as a 0.0 chunk-recovery-rate finding, turned out to be an artifact of the same environment bug — corrected, it shows a real **~30% chunk recovery rate** on the two seeds with a clean sample, not "no leakage." A purpose-built topic-diversity throttle, evaluated live against the real deployment, was first measured with an unvalidated, hand-picked threshold (100% suppression, 82.5% blocked) and then re-measured with a threshold *derived from a simulated legitimate-user baseline* — the calibrated, more defensible result is a 65.6% relative extraction-rate reduction and 74.1% blocked, strong but no longer total (§8.3, §12.4).

**Overall effectiveness:** the attack is highly effective given a valid API key and no defense, and this conclusion is now backed by multi-seed evidence rather than a single run. The defense's *mechanism* is validated and effective against the specific topic-balanced strategy this attack uses; its calibrated-threshold result is a real, if more modest, block rate rather than the artificially total suppression the original hand-picked threshold produced.

**Security impact:** confirmed and substantial — this deployment's content-confidentiality guarantee currently rests entirely on the secrecy of one fixed, non-rotated shared key, with no behavioral defense in production to fall back on until the module built in this analysis is actually deployed.

**Lessons learned:** exact-match ground-truth scoring at the retrieval layer is unambiguous (accuracy is always 1.0, because there is no noise channel), but this same clarity vanishes the moment generation enters the pipeline (Phase C) — a reminder that RAG-system security analysis must treat the retrieval and generation layers as distinct threat surfaces with genuinely different evaluation methodologies, not a single pass/fail measurement. A second lesson, learned twice over in this report (§4.6 and §12.3): an anomalous result — a crash, an exact-zero rate, an inflated error count — is as likely to be a broken harness or contaminated environment as it is to be a genuine security finding, and only checking the underlying environment, not just the output number, tells the two apart. A third, this time methodological rather than infrastructural: a "clean" 100%-suppression or zero-leakage result is itself worth suspecting until the measurement it rests on (an unvalidated threshold, an error-starved sample) has been checked — both of this report's most dramatic original numbers (82.5%→0.000 suppression, 0.0 CRR) turned out to be artifacts of exactly that kind, and both real, corrected findings underneath them are still genuine security concerns, just less extreme ones.
