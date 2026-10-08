# Similarity-only Laplace Noise Defense (MIA) — Implementation Notes

**Status: code written and offline-unit-tested (13/13 pass, no live service
needed). Live evaluation NOT run** — needs Docker services up and, per the
defense's own operational requirement, a config edit + container restart
per epsilon setting, which the user asked to run themselves later.

## Flow traced before writing anything

`POST /query` (`drag_llm_service/app/server.py`) → `query_data_sources()` →
candidates with a retriever `"score"` each → `defense_cfg =
retrieval_config.get('defense', {})`; if `defense_cfg['enabled']`, both
`/query` and `/query_analyze` call **`select_top_k_with_defense()`**
(`drag_llm_service/src/retriever/defense.py`) as their single shared
selection path. Inside it: `reranker.embed_query_and_candidates()` →
`dense = cand_embs @ q_emb` (raw cosine similarity, embeddings L2-normalized
by `Reranker`, so this is exactly cosine ∈ [-1, 1]) → `_minmax(dense)` →
optional `orig_score_weight` blend with the retriever's own prior score →
optional `reliability_weight` blend with on-chain `R_i` → `dedup_candidates()`
→ optional `consensus_weight` blend → `argsort` → top-k. If `defense.enabled`
is false, `/query`/`/query_analyze` instead call `reranker.rerank_with_reliability()`
or `reranker.rerank()` directly (same fusion math, no dedup/consensus).

This project's deployed `config.yaml` already has `defense.enabled: true`, so
`select_top_k_with_defense()` is already the live code path — confirmed
before deciding where to insert the noise step.

## Where the noise goes, and why there specifically

Inserted at exactly one point: `dense` (raw cosine similarity, right after
it's computed, right before `_minmax(dense)`) inside `select_top_k_with_defense()`.
Every downstream line — `_minmax`, hybrid/BM25 blending, `orig_score_weight`,
`reliability_weight`, `dedup_candidates`, `consensus_scores`, the final
`argsort` — is **unmodified code now operating on a perturbed input**, not new
fusion math. This is what "preserve the current ranking/fusion architecture"
meant in practice: one line changes what `dense` *is*, nothing else changes
what happens to it.

Why this point and not, say, the already-`_minmax`'d `fused` score: cosine
similarity has a fixed, data-independent range ([-1, 1] by construction,
since `Reranker._embed_texts()` L2-normalizes), which is what a Laplace
mechanism's `sensitivity` parameter needs to be a real constant, not an
artifact of whichever candidates happen to be in this particular query's
pool. `_minmax`'d `fused` is rescaled per-query against that query's own
min/max, so "sensitivity" for it isn't a fixed number — adding noise there
would need a different, weaker justification.

## Why BM25 and `orig_score` are untouched

The task specified "similarity-only." In this codebase, "similarity" is
specifically the dense embedding score `mia_attack.py`'s own MIA attacker
also uses (same model, `all-MiniLM-L6-v2`) — the actual signal the attack
exploits. BM25 is a distinct lexical-overlap score computed by
`Reranker._bm25_scores()`; `orig_score` is the upstream retriever's own
score. Neither is what MIA's cosine-similarity signal measures, so neither
was touched.

## Config

`drag_llm_service/configs/config.yaml`, under the existing
`retrieval.defense:` block (flat-key style, matching `dedup_enabled` /
`consensus_weight`'s existing convention — not a new nested sub-dict):

```yaml
similarity_noise_enabled: false      # true reverts to noiseless == disabled by default
similarity_noise_epsilon: 1.0        # Laplace privacy parameter
similarity_noise_sensitivity: 2.0    # cosine similarity range [-1,1] -> 2.0
similarity_noise_seed: 42            # reuses this file's model.seed convention; independent stream
```

## Seed / reproducibility mechanism

The repo's existing convention (`model.seed: 42` in this same config file,
threaded into `open_model.py`'s generation call) is a single integer config
value, not a global RNG singleton. Reused the same shape here
(`similarity_noise_seed`), but derived a **per-query** sub-seed
(`_derive_seed()`, `hashlib.sha256(f"{seed}:{query}")`) rather than reusing
one fixed seed for every query for the life of the process — same base
seed + same query text always reproduces the identical noise draw (needed
for the reproducibility test and for offline comparability), while
different queries aren't all perturbed by one frozen vector. Used
`hashlib` (already a transitive stdlib dependency, deterministic across
processes) instead of Python's built-in `hash()`, which is salted per
process (`PYTHONHASHSEED`) and would silently break reproducibility across
separate runs. `numpy.random.default_rng(seed)` draws the actual noise —
`numpy` is already this file's only import beyond stdlib, so no new random
framework was introduced.

## Honest caveats

- **Fixed per-query noise, not fresh-per-call.** The same query, sent
  twice, gets the identical perturbation both times (by design — see
  above). This is the right behavior for reproducible *evaluation*, but it
  means this is not a formal per-query differential-privacy guarantee with
  a composing privacy budget across repeated queries — it's a calibrated
  obfuscation layer, sized using DP's noise-calibration formula
  (`b = sensitivity/epsilon`), not a certified DP mechanism. This
  distinction matters if the eventual writeup wants to claim a formal ε-DP
  guarantee rather than "Laplace-calibrated similarity obfuscation."
- **Only reachable via `select_top_k_with_defense()`.** If a deployment
  sets the outer `defense.enabled: false` (which also disables
  dedup/consensus), `/query` falls back to `reranker.rerank_with_reliability()`
  / `reranker.rerank()`, and the new noise step — which lives only inside
  `select_top_k_with_defense()` — will not run regardless of
  `similarity_noise_enabled`. This was a deliberate scope boundary (no
  `server.py` routing change), not an oversight: this repo's deployed
  config already has `defense.enabled: true`, so the new defense is
  reachable as shipped. Documented rather than silently left ambiguous.
- **Evaluation requires a container restart per epsilon**, since config is
  loaded once at server startup — there is no per-request override. See
  `run_similarity_noise_defense.py`'s "Operational requirement" docstring
  section for the exact sequence. This mirrors how this project's own MIA
  report already treats live-config changes (Revision 7's infrastructure
  note).
- **Task-quality metric measures member-document questions only** (real
  questions about content that IS in the corpus) — the natural
  "legitimate user" proxy, scored with this repo's existing F1/EM
  implementation (`attack/ddos_sim/nlg_metrics.py`, reused unmodified).
