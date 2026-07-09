# MIA (Membership Inference Attack) Defense

Defends against `attack/Mia_attack`, which sends a real PubMedQA question to
`/query` and, on the best-similarity probe per document, computes a **gated**
composite membership score (4th revision):

    membership_score = decision_match*0.55 + similarity*0.20 + certainty*0.10 + length_ratio*0.15

with `decision_match`'s weight (0.55) exceeding the combined maximum of the
other three (0.20+0.10+0.15=0.45) **by construction** — a document with a
correct decision-match always scores strictly higher than one without,
regardless of the other three signals, which only re-rank *within* a gate
tier. AUC-ROC over this composite score across many probes is the leakage
metric (this is the fix already applied for the PDF's "MIA — '50%
neutralised' claim is weak" issue: AUC-ROC instead of single-threshold
accuracy). **This defense module must score undefended vs. defended on the
same composite metric the attack actually uses** — scoring plain cosine
similarity instead would compare against a metric the attacker no longer
relies on.

**Weight history (4 revisions):**

| Revision | Formula | Why it changed |
|---|---|---|
| 1 (SQuAD) | `sim*0.85 + len*0.15` | Mirrored original DRAG formula, hop term folded into similarity |
| 2 (PubMedQA) | `sim*0.70 + certainty*0.20 + len*0.10` | Hop term replaced by hedge-word `certainty` instead of folded away |
| 3 (PubMedQA, linear) | `sim*0.20 + certainty*0.10 + len*0.30 + decision_match*0.40` | Re-calibrated from live measurement: hedge-word `certainty` measured degenerate; `length_ratio` measured strongest yet had the smallest weight; `decision_match` (explicit yes/no/maybe-token detection, replacing a previously blind full-sentence answer-match check), measured the **strongest and most consistent signal of any tested** and was folded in — but as a *linearly blended* term, so a bad seed's noise in the other three could still occasionally outvote it |
| **4 (current, GATED)** | `decision_match*0.55 + sim*0.20 + certainty*0.10 + len*0.15` | `decision_match` promoted from a blended term to a **gating** term: its weight now strictly exceeds the other three combined, so it can never be outvoted across a decision_match boundary — see "What the gate actually fixed" below for the honest, measured result of this change |

**Dataset switch (this session): SQuAD → PubMedQA.** The attack originally
used SQuAD (Wikipedia-derived) for both corpus and probes. Across 4 seeds
(0, 1, 2, 42; n=25/25), that measured composite AUC-ROC ≈ 0.356 on
average — *below* 0.50 in every single run, a systematic inversion, not
"no signal." The most likely cause: the deployed LLM (Qwen2.5-1.5B-Instruct)
was almost certainly pretrained on Wikipedia, so it answers *non-member*
SQuAD questions from memory nearly as well as *member* ones. Switched to
`qiaojin/PubMedQA` (`pqa_labeled` config) — biomedical research abstracts the
base LLM is far less likely to have memorized — via
`data/build_pubmedqa_corpus.py`, which writes the first 500 of PubMedQA's
1,000 `pqa_labeled` rows into `data/polluted_token/sources_0.jsonl` as the
loaded corpus; the remaining ~500 rows serve as a genuinely disjoint,
same-distribution held-out non-member pool (**same domain**, not a different
corpus like legal contracts — a cross-domain split would let a detector
succeed by recognizing topic rather than genuine membership, making any AUC
lift ambiguous).

**Earlier attack methodology fixes (prior session, still in effect):** the
attack used to draw "non-member" passages from `sources_20/100.jsonl`,
believing them to be different content from `sources_0.jsonl` — they
weren't, so that split was invalid. It also had a `random.Random(seed)`
sampling bug that made every `--seed` produce identical results, and probed
with an artificial passage-prefix "seed phrase" instead of a real question.
All fixed; see `attack/Mia_attack/mia_attack.py`'s docstring for details.

## Why this defense, not "calibrated noise on confidence scores"

The original DRAG system's MIA defense added noise to a numeric confidence
score. Reliable-dRAG's `/query` endpoint doesn't return one --
`drag_llm_service/app/server.py` returns only `{"response": "<text>"}`
(see line ~615). The only channel available to the attacker is the response
*text itself*, so the defense caps what that text can leak:

1. **Response length cap** (`MAX_RESPONSE_WORDS = 40`) — a long response
   gives an external embedding model more surface to match against source
   text.
2. **Longest-common-run redaction** (`MAX_OVERLAP_WORDS = 8`) — if the
   response contains a contiguous run of words copied from the retrieved
   context longer than this, that run is collapsed to `...`. This directly
   targets the mechanism the attack's own hypothesis relies on ("member
   responses closely paraphrase, often quote, the real KB text").

Non-member responses aren't quoting anything relevant to begin with, so
they're essentially untouched — only genuine verbatim leakage gets capped,
which is what pulls the member/non-member similarity distributions together
and drives AUC-ROC back toward 0.5 without degrading answer quality for
normal queries.

## Files

- `mia_defense.py` — `sanitize_response()` (length cap + verbatim-overlap
  redaction, existing) and `obfuscate_decision()` (new: prepends a fixed
  hedge phrase to defeat positional `decision_match` detection), plus
  `MIADefenseEvaluator` (runs the attack's exact probe set against the live
  LLM service, scores raw, sanitized-only, and sanitized+obfuscated
  responses on the identical gated composite score the attack uses, and
  reports all three side by side, along with the same diagnostics the
  attack itself reports)
- `run_defense.py` — CLI runner, logs to `defense_logs/`

## Maintaining the same dataset and scoring as the attack

`MIADefenseEvaluator` imports `load_membership_documents()`, `SIM_WEIGHT`,
`CERTAINTY_WEIGHT`, `LEN_WEIGHT`, `DECISION_WEIGHT`, `_answer_length_ratio()`,
`_certainty_score()`, and `_decision_match()` straight from
`attack.Mia_attack.mia_attack` — the exact same sampling *and* the exact same
composite scoring the attack uses: real PubMedQA questions, members whose
context is loaded into `data/polluted_token/sources_0.jsonl`, non-members
whose context is genuinely absent from it (a disjoint, same-distribution
held-out PubMedQA split, not a different corpus), scored with the same
weighted similarity+certainty+length+decision_match formula. No separate
dataset, scoring, or sampling logic is introduced for the defense
evaluation. Each world's "best-of-several-probes" selection is made
independently (undefended vs. defended attacker only ever see their own
world's responses), mirroring `_probe_documents()` in the attack module.

## A real, honest result — four revisions, each fixing what the last one measured

**Revision 1 (SQuAD, 2-term):** mean AUC-ROC 0.356 across 4 seeds — *inverted*
in every run. Diagnosed as pretrained-knowledge camouflage (Qwen2.5 already
knew Wikipedia-derived SQuAD facts regardless of retrieval).

**Revision 2 (PubMedQA, 3-term):** switching corpus removed the inversion —
AUC ≥ 0.50 in all 3 sampled runs — but the lift was modest, and the new
`certainty` signal (hedge-word based) measured **completely degenerate**:
PubMedQA's terse yes/no/maybe answers essentially never contain a hedge
word. Separately, the answer-match diagnostic was **blind** (`0.0000` match
rate for both groups in every run) because it checked for the full
`long_answer` sentence verbatim, which a generated response almost never
quotes.

**Revision 3 (PubMedQA, linear 4-term):** redesigned `certainty` around
yes/no/maybe commitment detection; fixed the blind diagnostic into
`decision_match` (does the response's opening words state the correct
`final_decision` token) — which turned out to be the **strongest and most
consistent signal measured in this project** (AUC 0.62/0.64/0.56 across 3
seeds, delta always positive, never inverted) and was folded in as a 4th
linearly-blended term. Result: AUC-ROC **0.6432, 0.7056, 0.4992** (mean
0.616) — 2 of 3 seeds `MEDIUM` tier, but seed 42 stayed a near-random
outlier despite `decision_match` itself being positive (+0.12) that same run.

**Revision 4 (current) — gating, tested, and an honest surprise:**

The hypothesis behind Revision 3's seed-42 outlier was that a *linear* blend
let `length_ratio`'s occasional inversion **outvote** an already-correct
`decision_match` reading across the member/non-member boundary. The fix
tested here: make `decision_match` a **gate** (weight 0.55, strictly
exceeding the other three combined at 0.45) so that outvoting is
mathematically impossible — any document with a correct decision-match
scores above any document without one, no matter what the other three
signals do.

**Measured result: the gate produced byte-identical AUC-ROC to the linear
blend for 2 of 3 seeds (0.6432 for seed 0, 0.4992 for seed 42 — both to 4
decimal places), and a negligible change for the third (0.7056 → 0.7040).**
This is not a bug — it is a real, informative negative result: it shows the
seed-42 problem was **never actually caused by cross-boundary outvoting** in
the first place. Because `decision_match` is binary (0 or 1) and the other
three signals' real measured magnitudes are small (raw similarity ~±0.02–0.08,
length_ratio ~0.04–0.09), the linear blend was *already* behaving almost
exactly like a gate in practice for this data, even without the weight
inequality being mathematically guaranteed. The actual cause of the seed-42
outlier is **intra-tier** noise: among documents that share the same
`decision_match` value (a "tie" the gate explicitly leaves to the other three
signals to break), `length_ratio` and `similarity` are themselves inverted
for seed 42 specifically — and no gating scheme fixes bias *within* a tier,
only override *across* tiers. This is a more precise diagnosis than the
original hypothesis, obtained by actually testing the fix rather than
assuming it would work.

**Defense result — the first non-zero `auc_roc_reduction` in this project's
history, with an important caveat:** a new `obfuscate_decision()` defense
(prepends a fixed hedging phrase to every response, pushing any yes/no/maybe
commitment out of the leading-words window `decision_match` inspects) was
built and evaluated. Combined with the existing `sanitize_response()`:

| Seed | `decision_match` AUC, undefended → defended | Composite AUC, undefended → defended | `auc_roc_reduction` |
|---|---|---|---|
| 0 | 0.62 → **0.50** (fully neutralized) | 0.6432 → 0.6352 | **+0.0080** |
| 42 | 0.56 → **0.50** (fully neutralized) | 0.4992 → 0.4464 | +0.0528 (see caveat below) |

`obfuscate_decision()` **fully neutralizes `decision_match` in both runs** —
`auc_roc_answer_match` (the field now holding `decision_match`) lands at
exactly `0.5000`, with a `0.0000` match rate for *both* groups, confirming
the defense does exactly what it was designed to do: defeat the specific
positional check. But the overall composite `auc_roc_reduction` is modest
for seed 0 (+0.0080) because `length_ratio` (AUC 0.6384 even after
obfuscation) still carries real signal the defense doesn't touch at all —
neutralizing one channel doesn't neutralize all of them.

**Caveat on seed 42's `+0.0528` figure:** `auc_roc_reduction` is defined as
`undefended_auc − defended_auc`, which only cleanly means "moved toward
random" when both values are on the same side of 0.5. Seed 42's undefended
AUC (0.4992) was already almost exactly 0.5; the defended value (0.4464)
moved *further away* from 0.5 in the *opposite* direction (more inverted),
not closer to it. Measured by **distance from random** (`|auc − 0.5|`)
instead, seed 42 actually got slightly *worse* (0.0008 → 0.0536), not
better — the raw reduction number is technically positive but should not be
read as "the defense improved privacy here." This is disclosed rather than
quoted uncritically, since the naive number alone is genuinely misleading
for a seed whose undefended AUC straddles 0.5.

**Honest scope of `obfuscate_decision()`:** it defeats a *positional* check
(does the answer commit in its first few words) — it does not remove the
correct decision from the response text at all, and provides zero protection
against an attacker who scans the *whole* response instead of only its
opening words. This scope limitation is documented directly in
`obfuscate_decision()`'s own docstring in `mia_defense.py`, not only here.

This should be read as **real, substantive, multi-round progress with one
precise negative result (the gate) and one precise positive result (the
obfuscation defense) — not a solved attack or a solved defense.** See
`reports/MIA_Security_Analysis_Report.md` for the full before/after evidence
across all four revisions and remaining limitations.

## Regenerating the corpus (required once before running against PubMedQA)

```bash
python data/build_pubmedqa_corpus.py
```

This overwrites `data/polluted_token/sources_0.jsonl` with 500 PubMedQA
abstracts. **Important:** on this machine, the live `drag-data-source-0`
container's volume mount was found to point at a *different* checkout
(`.../reliable de-rag rahim/Reliable-dRAG-main/data/polluted_token`), not
this repo's `data/` folder — check `docker inspect drag-data-source-0
--format '{{ range .Mounts }}{{ .Source }} -> {{ .Destination }}{{println}}{{end}}'`
before assuming an edit here takes effect live. Copy the regenerated
`sources_0.jsonl` into whatever directory that mount actually resolves to,
then call `POST http://localhost:8001/reset` (or restart the container) to
force it to re-embed from the new file — no image rebuild is needed, since
`FastRetriever` builds its FAISS index in memory at startup/reset time from
the JSONL on disk.

## Running the evaluation

Requires `sentence-transformers`, `scikit-learn`, `numpy` (already needed by
`attack/Mia_attack`; install with `pip install sentence-transformers
scikit-learn numpy` if not already present):

```bash
python defense/mia_defense/run_defense.py
python defense/mia_defense/run_defense.py --seed 0   # repeat with --seed 1, 2 for thesis variance
```

This evaluates the defense against the **live, unmodified** LLM service —
it queries exactly as the attack does, then applies `sanitize_response()`
and, separately, `sanitize_response()` + `obfuscate_decision()` combined, to
the raw response before scoring, so undefended vs. sanitized-only vs.
sanitized+obfuscated AUC-ROC are all directly comparable on identical
probes. No redeploy is required to see the comparison.

## Deploying it for real (optional)

To have `drag_llm_service` actually apply this to every response (not just
in the evaluation harness), patch `drag_llm_service/app/server.py` right
after generation, using the `selected` reranked contexts that are already in
scope at that point:

```python
# drag_llm_service/app/server.py, in the /query handler, after:
response_text = model.generate(prompt)

# MIA defense: cap verbatim leakage of retrieved context into the response
from defense.mia_defense.mia_defense import sanitize_response
context_texts = [str(c["text"]).strip() for c in selected]
response_text = sanitize_response(response_text, context_texts)

return jsonify({"response": response_text}), 200
```

This requires rebuilding/restarting the `llm-service` container (a larger,
GPU-backed image — much slower than the `hardhat-node` rebuild used for the
SSM-Score defense), so it wasn't hot-patched into your live container in
this session. Apply it when convenient:

```bash
docker compose build llm-service
docker compose up -d llm-service
```
