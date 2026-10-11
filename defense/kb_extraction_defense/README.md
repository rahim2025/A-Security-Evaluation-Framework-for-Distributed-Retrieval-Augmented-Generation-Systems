# KB Extraction Defense

This folder is the single reference for defending the Reliable-dRAG data sources against knowledge-base (KB) extraction. It covers an audit of the defense that existed before, the replacement design, how it was calibrated and evaluated, the results (simulated and live on the Docker stack), and the limits of the defense.

The attack under defense is `attack/kb_extraction/`. Defenses are an extension beyond the original attack-only scope, added by team decision.

**Contents:** 1. Summary · 2. Audit of the previous defense · 3. Threat model · 4. Design · 5. Deployment · 6. Evaluation method · 7. Results · 8. Limitations · 9. Reproducing · 10. Files

---

## 1. Summary

| | Before | After |
|---|---|---|
| What the defense knows | needs a topic label per query (none exists on a real server) | server-side only: caller id, returned doc ids, its own corpus embeddings |
| Default configuration | `max_topics_per_window=12` (no effect, see 2.2) | calibrated from legitimate traffic only |
| Guarantee | none | at most `B` distinct documents per identity per epoch (`B/N` of the corpus) |
| Evaluated against | one fixed list of 54 questions | static probes plus adaptive crawlers (random, frontier, local, dwell, decoy-diluted, Sybil), 3 seeds |
| Live validation | client-side gate in the attack script | real HTTP against the Docker data source |

Headline result (SQuAD-backed source 1, N=500 docs, seeds 0/42/123, 600 query attempts):

- Undefended crawlers recover **39–94%** of the KB. The deployed Flask-Limiter (60/min) changes nothing, only the speed.
- With the guard they recover **9–23%**, and the first flag fires after 10–126 queries.
- Focused legitimate users (1–3 topics) see a **3.3–4.6%** chance of any degraded query and a **0.1–0.2%** drop in retrieval hit-rate.
- Live over HTTP the numbers are the same as in simulation (e.g. frontier crawler 0.913 → 0.160).

What does not work, stated up front (details in section 8):

- Broad legitimate readers (browsing, 5-topic analysts) are indistinguishable from extraction by content on a 500-doc / 10-topic corpus and are throttled by the breadth signal.
- On a topically homogeneous corpus (source 0, PubMedQA) the breadth signal gives little; the guarantee falls back to the budget bound (about 52% of the corpus there).
- Extraction through the LLM answer channel is outside this guard.

---

## 2. Audit of the previous defense

Files: `query_diversity_throttle.py`, `run_defense.py`, `calibrate_thresholds.py` (kept as the "v1" baseline).

### 2.1 What it did

A per-client sliding window counts the distinct *topics* among the client's last 30 queries and blocks the client for 20 queries if the count exceeds `max_topics_per_window`.

### 2.2 Why it was not a sufficient defense

1. **It needs labels a server does not have.** The topic of a query was taken from the attack script's own `context_to_title` dictionary, so the gate ran inside the attacker's process. Server-side, nothing labels a query with a topic. In our simulator v1 is given the topic of the top-1 retrieved document, which is an *oracle upper bound* on what it could do deployed.
2. **The default threshold does nothing.** With the default `max_topics_per_window=12` and the corpus having about 10 topics, the threshold cannot be exceeded: the extraction rate is identical to no defense (`oracle_v1_default12` equals `none` in every row of 7.1).
3. **Evaluated against a weak attacker.** The only attack was 54 fixed dataset questions, which recover about 39% of the KB even with no defense. A defense validated only against it says little about an adaptive crawler.
4. **No bound on total leakage.** It limits a burst of topics, not how much is eventually recovered. A patient attacker paces below the threshold.
5. **No caller identity.** Keyed on a client string, so identity splitting defeats it.
6. **Inconsistent logs and no variance.** Single runs, no multi-seed reporting.

### 2.3 What was kept

The idea that topical breadth distinguishes a sweeping crawler from a focused user is sound. The new design keeps it but derives the "topic" from the server's own corpus instead of requiring labels, and adds a budget so the guarantee does not depend on detection.

---

## 3. Threat model

- **Asset:** the documents stored in a data source. `POST /query` returns the top-5 full documents.
- **Adversary:** holds the shared `X-API-Key` (stolen key, insider, any non-orchestrator client) and can call `/query` directly. Not the orchestrator.
- **Adversary capability:** adaptive, black-box on content. It uses only returned documents to choose the next query (keyword snowball). It may know the public embedding model (white-box worst case for the frontier/dwell attackers). It may spread queries over several identities.
- **CLAUDE.md rule honoured:** no attacker ever sends stored document text as a probe; every query is a 3-keyword bag or a generic public seed question.
- **Attackers evaluated** (`attack/kb_extraction/adaptive_attack.py`):

| attacker | strategy |
|---|---|
| `static54` | the original fixed list of 54 dataset questions |
| `snow_random` | keyword snowball, random next query |
| `snow_frontier` | picks the candidate least similar to anything already retrieved (maximises new docs per query) |
| `snow_local` | stays near its previous query to look like a focused user |
| `*_dil3` | interleaves 3 repeated queries after each real one to dilute discovery velocity |
| `snow_dwell` | breadth-aware: exhausts one region of the corpus before moving to the next, to keep the number of regions per window small |

- **Out of scope:** extraction through the LLM answer (section 8.4), a compromised orchestrator, and spoofing of the client identity.

---

## 4. Design

Implemented in `extraction_guard.py` (`ExtractionGuard`) and wired into `drag_data_source/app/server.py`.

### 4.1 Principle

An extractor's objective is to maximise *new documents per query*. The server can observe which document ids each caller has already received. A legitimate caller keeps returning to a small working set; an extractor cannot be efficient and look repetitive at the same time. A cumulative budget bounds the damage of any strategy that evades detection.

### 4.2 Signals (label-free)

- **Breadth.** At start-up the data source clusters its own document embeddings (spherical k-means, K=10, numpy only). A query's *region* is the cluster of its top-1 result. Breadth = number of distinct regions among the caller's last 30 queries. Flag when breadth ≥ θ (θ=6) after at least 10 queries. Using the union of all 5 results instead of top-1 was tried and gave about 90% false positives because neighbours spill across clusters.
- **Discovery velocity** and **query dispersion** are implemented as optional signals. They were measured to be weak (6.2) and are disabled in the shipped configuration. When several signals are enabled they are AND-ed.

### 4.3 Graded response

```
ALLOW   -> serve normally
DEGRADE -> serve k=1 result for 20 queries           (first and second flag)
FREEZE  -> serve only documents already served to this caller   (third flag, or a budget is hit)
```

A false positive therefore costs a few one-result answers; re-reading already-seen documents is never blocked.

### 4.4 Budgets (the guarantee)

- `client_budget` B: distinct documents served per caller per epoch. Set to the 99.5th percentile of the working set of *legitimate focused users*, measured on calibration data (section 6.2).
- `global_budget` (optional): distinct documents across all untrusted callers; the Sybil backstop (7.4).
- Leakage per identity per epoch is at most B/N of the corpus regardless of the attack. Epoch = 24 h by default.

### 4.5 Trusted caller

The LLM service reads documents in bulk legitimately. It authenticates with a second credential, `X-Orchestrator-Token`, and is exempt. Everyone else holding only the API key is guarded. `drag_llm_service/app/server.py` sends the token on both of its data-source calls.

---

## 5. Deployment

Changed outside this folder (instrumentation only, retrieval logic untouched):

| file | change |
|---|---|
| `drag_data_source/app/server.py` | builds the guard (clusters from `retriever.emb`), guard path in `/query`, `GET /extraction_guard/stats`, `POST /extraction_guard/reset`, `X-Guard-Action` response header |
| `drag_llm_service/app/server.py` | `X-Orchestrator-Token` header on data-source requests |
| `docker-compose.yml` | `ORCHESTRATOR_TOKEN` on the services; `EXTRACTION_GUARD_ENABLED` (default 1), `EXTRACTION_GUARD_DEBUG`, `EXTRACTION_GUARD_CONFIG` (JSON overrides) on the data sources |

Shipped defaults (`DEFAULT_GUARD_FIELDS`): window 30, min_queries 10, breadth_thresh 6, n_clusters 10, degrade_k 1, cooldown 20, escalate_after_flags 3, client_budget 202, epoch 86400 s. The budget 202 was calibrated on source 1; set it per source (source 0 calibrates to 262) with `EXTRACTION_GUARD_CONFIG='{"client_budget": 262, "breadth_thresh": 10}'`.

Change the default `ORCHESTRATOR_TOKEN` (`orch-secret-2026`) outside a lab.

Disable the guard with `EXTRACTION_GUARD_ENABLED=0`.

The guard keys on `request.remote_addr`. Behind a proxy this must be the real client address.

---

## 6. Evaluation method

### 6.1 Simulator

`sim_server.py` is an in-process stand-in for `/query`: the real corpora (500 docs each), real MiniLM embeddings and the real top-k retrieval rule; only the HTTP hop is removed. Logical time is used for the rate limit and epochs. This allows 3 seeds × many attackers in minutes. Section 7.5 checks it against the live stack.

Defenses compared: `none`; `ratelimit60` (the deployed Flask-Limiter); `oracle_v1_default12` and `oracle_v1_tuned5` (v1 with oracle topic labels); `guard`.

### 6.2 Calibration without leakage

Thresholds and B are chosen on calibration seed 1001 using **legitimate traffic only** for the false-positive constraint, and attackers `snow_random`/`snow_frontier` for the tie-break. Test seeds are 0/42/123, disjoint from the calibration seed. Selection rule: FP ≤ 5% on protected classes, then minimum attacker ER. The grid is K ∈ {6, 10, 16}, θ ∈ {3,…,10}, with and without velocity, plus a budget-only fallback.

### 6.3 Legitimate users

`legit_workloads.py` generates sessions from the corpus QA pairs: focused on 1/2/3/5 topics (20–200 queries), a browsing Markov model (stay probability 0.7/0.9) and aggregate/analyst behaviour. **Protected classes** are focused 1–3 topics: the defense must not affect these. Broad classes are reported but not protected (see 8.1). Utility is measured as retrieval hit-rate (is the gold document in the top-5).

### 6.4 Metrics

Extraction rate ER = fraction of the corpus documents recovered by the attacker after a fixed number of query attempts (blocked queries count as attempts). Also first flagged query, ER at first flag, protected-class FP (share of sessions with at least one degraded query) and hit-rate drop. Mean ± std over seeds 0/42/123.

Reproducibility note: attacker sampling never iterates over Python sets (hash randomisation); results were verified identical across two processes.

---

## 7. Results

All numbers are generated by `report_tables.py` from `defense_logs/kb_extraction_defense/offline/*.json`.

### 7.1 Extraction rate after 600 attempts, source 1 (SQuAD, N=500)

| attacker | none | ratelimit60 | v1 default12 (oracle) | v1 tuned5 (oracle) | **guard (label-free)** |
|---|---|---|---|---|---|
| static54 | 0.393 ± 0.007 | 0.393 ± 0.007 | 0.393 ± 0.007 | 0.110 ± 0.032 | **0.183 ± 0.022** |
| snow_random | 0.938 ± 0.011 | 0.938 ± 0.011 | 0.938 ± 0.011 | 0.111 ± 0.040 | **0.158 ± 0.040** |
| snow_frontier | 0.913 ± 0.017 | 0.913 ± 0.017 | 0.913 ± 0.017 | 0.077 ± 0.009 | **0.160 ± 0.006** |
| snow_local | 0.887 ± 0.010 | 0.887 ± 0.010 | 0.887 ± 0.010 | 0.170 ± 0.067 | **0.227 ± 0.045** |
| snow_frontier_dil3 | 0.641 ± 0.016 | 0.641 ± 0.016 | 0.641 ± 0.016 | 0.062 ± 0.004 | **0.088 ± 0.002** |
| snow_local_dil3 | 0.485 ± 0.024 | 0.485 ± 0.024 | 0.485 ± 0.024 | 0.117 ± 0.001 | **0.131 ± 0.005** |

Reading it:

- The rate limit changes nothing at this query count (it bounds speed, not total recovered).
- v1 at its default is a no-op.
- v1 with an oracle and tuned to 5 topics is slightly better than the guard, but needs labels a server does not have. The label-free guard is within about 1–8 points of that upper bound.

Breadth-aware `snow_dwell` was evaluated separately (`dwell_source1.json`, same seeds): undefended 0.901, guard **0.146**, budget only 0.405; with decoys (`snow_dwell_dil3`) 0.552 → **0.149**.

First flagged query and ER at first flag (guard): static54 13.7 / 0.123; snow_random 19.0 / 0.123; snow_frontier 10.0 / 0.099; snow_local 40.3 / 0.177; snow_frontier_dil3 25.0 / 0.068; snow_local_dil3 126.0 / 0.123.

### 7.2 Legitimate-user impact, source 1 (share of sessions with at least one degraded/frozen query)

| defense | browsing 0.7 | browsing 0.9 | focused1 | focused2 | focused3 | focused5 | **protected FP** | **hit-rate drop** |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v1 default12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| v1 tuned5 (oracle) | 0.858 | 0.458 | 0 | 0.042 | 0.050 | 0.550 | 0.031 | 0.013 |
| **guard** | 0.842 | 0.533 | 0 | 0.042 | 0.092 | 0.633 | **0.044** | **0.002** |

Protected FP is the mean over focused1–3. Broad readers are throttled, with or without labels (8.1).

### 7.3 Ablation, source 1 (ER_final, mean over seeds)

| variant | static54 | random | frontier | local | frontier_dil3 | local_dil3 | protected FP |
|---|---|---|---|---|---|---|---|
| budget only | 0.393 | 0.405 | 0.404 | 0.405 | 0.407 | 0.405 | 0.000 |
| velocity + dispersion + budget | 0.222 | 0.233 | 0.247 | 0.404 | 0.407 | 0.405 | 0.025 |
| breadth + budget, no escalation | 0.186 | 0.404 | 0.404 | 0.405 | 0.265 | 0.221 | 0.033 |
| breadth + velocity + budget | 0.198 | 0.278 | 0.224 | 0.367 | 0.407 | 0.405 | 0.025 |
| **breadth + budget + escalation (shipped)** | 0.183 | 0.158 | 0.160 | 0.227 | 0.088 | 0.131 | 0.033 |

- Velocity and dispersion fail against `local` and decoy-diluted crawlers (per-attacker detection score of velocity at window 30 is 0.002 for `snow_local_dil3` and 0.05 for `snow_frontier_dil3`, `signals_source1.json`), which is why they are not shipped.
- Without escalation, a flagged crawler is degraded to k=1 and keeps crawling until the budget; escalation to FREEZE is what contains it early.
- Budget only is a flat 0.40 = B/N, independent of the attacker. That is the guarantee; the signals reduce it further.

The ablation's protected FP (0.033) uses the seed-0 legitimate suite only, while the compare table (0.044) averages seeds 0/42/123; both are below the 5% target.

### 7.4 Sybil (identity splitting), source 1, `snow_frontier`

| identities | per-client budget only | + global budget |
|---|---|---|
| 1 | 0.160 ± 0.006 | 0.160 ± 0.006 |
| 4 | 0.439 ± 0.011 | 0.404 ± 0.000 |
| 16 | 0.739 ± 0.011 | 0.404 ± 0.000 |

A per-client budget alone is defeated by splitting across identities. The global budget caps the union at B/N, at the cost that heavy legitimate load from many untrusted clients could exhaust it. It is optional (`global_budget`) and off by default.

### 7.5 Budget bound versus corpus size

Federated union corpora (real sources concatenated), same B per identity, 2500 attempts:

| corpus | N | B | bound B/N | frontier undefended → budget only | local+dil3 undefended → budget only |
|---|---|---|---|---|---|
| source 1 (N=500) | 500 | 195 | 0.390 | 0.996 → 0.391 | 0.887 → 0.392 |
| union | 1000 | 233 | 0.233 | 0.972 → 0.236 | 0.688 → 0.234 |
| union | 1500 | 285 | 0.190 | 0.909 → 0.191 | 0.567 → 0.191 |

Observed leakage equals B/N to the third decimal, and B grows much more slowly than N (the working set of a focused user does not scale with the corpus), so the bound tightens on larger KBs.

### 7.6 Live validation (Docker, real HTTP), source 1

Before = the orchestrator-token path (exempt, equals the old behaviour); after = plain API key (guarded). Same deployment, same retrieval code, `RATE_LIMIT_DEFAULT` raised so the 60/min limiter does not contaminate the result (the script aborts on any HTTP 429). `live_eval.py`, seeds 0/42/123, 600 attempts.

| attacker | undefended | guarded | first flag |
|---|---|---|---|
| static54 | 0.393 ± 0.007 | 0.183 ± 0.022 | 13.7 |
| snow_random | 0.938 ± 0.011 | 0.158 ± 0.040 | 19.0 |
| snow_frontier | 0.913 ± 0.017 | 0.160 ± 0.006 | 10.0 |
| snow_local_dil3 | 0.485 ± 0.024 | 0.131 ± 0.005 | 126.0 |

Legitimate focused sessions (24 sessions, 1–3 topics, 20–200 queries): **0 sessions affected, hit-rate 0.882 with and without the guard.**

The live numbers match the simulator (compare 7.1), which validates the simulator as a stand-in. An end-to-end question through the orchestrator (`POST :9000/query`) still returns an answer with the guard on and the data-source guard counters unchanged, confirming that the token wiring exempts the LLM service.

### 7.7 Second corpus, source 0 (PubMedQA, N=500)

PubMedQA has no native topic field. Its "topics" in `eval_common.py` are sklearn k-means pseudo-labels, which differ from the guard's own clustering, so focused-user behaviour there is only loosely tied to the regions the guard sees. Calibration therefore picks a very conservative breadth threshold (10 of 10 clusters) and B=262.

| attacker | none | v1 tuned5 (oracle) | guard |
|---|---|---|---|
| static54 | 0.419 ± 0.003 | 0.091 ± 0.018 | 0.303 ± 0.019 |
| snow_random | 0.941 ± 0.002 | 0.083 ± 0.025 | 0.457 ± 0.026 |
| snow_frontier | 0.899 ± 0.011 | 0.093 ± 0.016 | 0.449 ± 0.105 |
| snow_local | 0.893 ± 0.009 | 0.091 ± 0.026 | 0.463 ± 0.087 |
| snow_frontier_dil3 | 0.597 ± 0.022 | 0.098 ± 0.012 | 0.315 ± 0.029 |
| snow_local_dil3 | 0.471 ± 0.015 | 0.078 ± 0.016 | 0.483 ± 0.029 |
| snow_dwell | 0.905 ± 0.006 | 0.123 ± 0.026 | 0.525 ± 0.001 |
| snow_dwell_dil3 | 0.537 ± 0.040 | 0.141 ± 0.008 | 0.509 ± 0.019 |

Protected-class FP 0.6% and hit-rate drop 0.000. The guard roughly halves what undefended crawlers get on this corpus, but the strongest adaptive crawlers reach the budget bound (0.525 ≈ B/N = 0.524). The oracle v1 does much better here because it is *given* the pseudo-topic labels the legitimate workload was generated from; this is an upper bound, not a deployable result. Sybil and scale on this source follow the same pattern as source 1 (`sybil_source0.json`: 16 identities, 0.882 per-client vs 0.526 with the global budget; `scale_source0.json`: leakage tracks B/N).

---

## 8. Limitations

1. **Broad legitimate readers.** On a 500-doc, 10-topic corpus a user who reads across 5 topics, or browses with a high topic-switch rate, behaves like a crawler by content. The guard throttles 53–84% of browsing sessions and 63% of 5-topic sessions (7.2). Oracle v1 does the same, so this is a property of the problem at this corpus size, not of the implementation. The remedy is the graded response (a flagged reader still gets one result per query and can re-read) and per-user identity/quotas, not detection.
2. **Corpus structure dependence.** The breadth signal needs a corpus with real topical clusters. On a homogeneous corpus (7.7) the shipped guarantee is the budget bound only.
3. **Bound is a fraction, not a wall.** B/N is 39% here and shrinks with N (7.5), but on a small KB a patient attacker still gets that share per identity and per epoch; the budget refills each epoch.
4. **Identity.** Keyed on the caller IP. Rotating IPs defeats the per-client budget; only the optional global budget bounds that (7.4), and it can be exhausted by legitimate load.
5. **LLM answer channel.** Extraction through `drag_llm_service` `POST /query` (answers, not documents) is a different, much slower channel: earlier measurement was about 10 characters of KB content per query versus roughly 2.8–5.2 k characters per query through raw `/query` (approximate, from a previous session, not re-measured here). The orchestrator's `/query` and `/query_analyze` are unauthenticated and unthrottled. Authenticating and rate-limiting them is recommended but is **not enabled** by default, to avoid breaking the sibling attack/defense scripts.
6. **Orchestrator compromise and token theft** give an attacker the exempt path. Out of scope.
7. **Simulator versus deployment.** The simulator validated against the live stack for source 1 only; sources 0 and 2 live runs were not done. Source 2 offline was not run.
8. **Calibration scope.** Thresholds are per corpus. The shipped defaults come from source 1; other sources should be re-calibrated (`offline_eval.py calibrate --source N`).

---

## 9. Reproducing

```bash
# offline (simulator), about 1 h per source on CPU
python defense/kb_extraction_defense/evaluation/offline_eval.py all --source 1
python defense/kb_extraction_defense/evaluation/offline_eval.py all --source 0
python defense/kb_extraction_defense/evaluation/report_tables.py --source 1      # prints the tables above

# live (Docker): rebuild data sources + llm-service, raise the HTTP rate limit for the measurement
RATE_LIMIT_DEFAULT="100000 per minute" EXTRACTION_GUARD_DEBUG=1 \
  docker compose up -d --build data-source-0 data-source-20 data-source-100 llm-service
python defense/kb_extraction_defense/evaluation/live_eval.py --source 1
```

Results are written to `defense_logs/kb_extraction_defense/{offline,live}/`.

Individual experiments: `offline_eval.py {threat,signals,calibrate,compare,ablate,sybil,scale} --source N`.

---

## 10. Files

```
kb_extraction_defense/
├── README.md                  this document
├── extraction_guard.py        the defense (imported by drag_data_source/app/server.py)
├── evaluation/                not deployed
│   ├── eval_common.py         corpus + embedding loading
│   ├── legit_workloads.py     legitimate-user session models
│   ├── sim_server.py          in-process /query stand-in with pluggable defenses
│   ├── offline_eval.py        multi-seed experiments (threat/signals/calibrate/compare/ablate/sybil/scale)
│   ├── live_eval.py           the same attackers over HTTP against Docker
│   └── report_tables.py       markdown tables from the result JSONs
└── legacy_v1/                 superseded; kept as the comparison baseline
    ├── query_diversity_throttle.py
    ├── run_defense.py
    └── calibrate_thresholds.py
```

Adaptive attackers live with the other attacks: `attack/kb_extraction/adaptive_attack.py`.
