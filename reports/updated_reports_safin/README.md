# MIA — Trained-Attacker Follow-up (Revision 9, Phase 1 + 2 code)

**Status:**
- **Phase 1 (`trained_attacker.py`): code complete AND run.** No live
  service needed (reuses already-collected Revision 7 data) — see result
  below.
- **Phase 2 collection (`contrastive_probes.py`) + Phase 2 train/eval
  (`contrastive_trained_attacker.py`): code complete, logic-verified
  against synthetic data, but NOT run against real data.** Needs the live
  LLM service / Docker stack, which was intentionally not brought up this
  session (see "Why this wasn't run" below) — for you to run on your
  research PC.
- **Phase 1 result (real, not synthetic):** a gradient-boosted classifier,
  selected via Leave-One-Seed-Out CV on `DEV_SEEDS` only, scores **mean
  AUC 0.6658 (95% CI [0.6027, 0.7289])** on the 5 locked fresh `TEST_SEEDS`
  — higher than, and with a tighter CI than, the existing hand-tuned
  linear composite's 0.6200 (95% CI [0.5455, 0.6945]). Saved to
  `attack_logs/trained_attacker_final_result.json` and
  `attack_logs/trained_attacker_locked_meta.json`.

This document exists so the next session (or teammate) can pick up Phase 2
and know exactly what it is running and why.

## Why this exists

Ahead of NAACL submission, external reviewer feedback on the MIA section
(`Improvement Proposal before NAACL Submission.md`) asked for three things:

1. More models/datasets, and non-pretrained-content documents, to rule out
   pretraining memorization as a confound.
2. A **trained classifier** attacker — using wording, confidence,
   similarity, and multiple contrastive questions per document — instead of
   the existing hand-designed/grid-searched composite, with strict dev/test
   separation and multiple seeds.
3. Testing existing defenses against that stronger attacker.

The user asked to start with the attack side (items 1–2, not item 3 —
defense testing is out of scope for this pass and, per
`.claude/CLAUDE.md`, is explicitly future work for this project on
Reliable-dRAG anyway). Item 1 (new models/datasets) is **not** addressed
here — it's a separate, larger undertaking (new corpus build, possibly a
second LLM backend) not started this session. This pass is scoped to item
2 only.

Before writing anything, I read the existing
`reports/MIA_Security_Analysis_Report.md` (8 revisions deep). Two things
from that report matter for what follows:

- **A dev/test seed split and raw per-document signal data already
  exist** (Revision 7): `attack_logs/tune_weights_dev_signals.json` (650
  rows, 13 dev seeds) and `attack_logs/tune_weights_test_signals.json`
  (250 rows, 5 held-out test seeds), each row holding four signals —
  `norm_sim`, `certainty`, `length_ratio`, `decision_match` — for one
  document. The current production attacker is a grid-searched **linear**
  weighting of those four signals (`decision_match` alone won: mean test
  AUC 0.620, 95% CI [0.546, 0.695]).
- **Repeating an identical question to the same document does not work as
  a signal on this deployment** (Revision 8, §2.10): under this system's
  deterministic decoding, the same question always gets the same answer,
  so the existing "consistency" signal is mechanically stuck at chance.
  This matters directly for how "multiple contrastive questions" had to be
  designed below — see Phase 2.

## What was built

### Phase 1 — `attack/Mia_attack/trained_attacker.py`

Trains an actual supervised classifier (not a hand-picked linear form) on
the **existing** Revision 7 dev/test signal data — no new LLM calls
needed, since that data is already on disk.

- Model candidates: two regularized logistic regressions (`C=1.0`,
  `C=0.1`) and a shallow, regularized gradient-boosted tree ensemble
  (`n_estimators=50, max_depth=2`). Small/regularized on purpose — 650
  rows over 4 features is not enough data to justify a wide hyperparameter
  search or deep trees.
- Model **selection**: Leave-One-Seed-Out cross-validation, using only the
  13 `DEV_SEEDS` — never touches `TEST_SEEDS`. This mirrors
  `tune_weights.py`'s own dev/test discipline, just applied to "which
  classifier" instead of "which weight vector."
- The winning model type is refit on *all* dev rows, locked to disk
  (`attack_logs/trained_attacker_locked.pkl` + a human-readable
  `..._locked_meta.json`), then scored **exactly once** against the 5
  reserved `TEST_SEEDS` — the same "evaluate once, weights already frozen"
  rule Revision 7 used, enforced by the same runtime assertion (refuses to
  evaluate if the test file contains a seed outside `TEST_SEEDS`).

Commands (two-step, matching `tune_weights.py`'s `search` → `evaluate`):

```bash
# needs a working scikit-learn build -- see "Environment note" below
/opt/miniconda3/bin/python3 attack/Mia_attack/trained_attacker.py search
/opt/miniconda3/bin/python3 attack/Mia_attack/trained_attacker.py evaluate
```

**What this is and isn't.** This is a real, if modest, improvement in
kind: a classifier can learn a non-linear boundary and interactions the
grid search's fixed linear form could never express, over the same four
signals. It is *not* yet the richer "wording / confidence / multiple
contrastive probes" attacker the reviewer described — that needs new
features, which means new data, which means Phase 2.

### Phase 2 — `attack/Mia_attack/contrastive_probes.py`

Collects the richer feature set the reviewer actually asked for. This
**requires live LLM calls** and was not run.

- **Contrastive probes, not repeated-identical ones.** Revision 8 already
  showed repeating the *identical* question is mechanically useless under
  this deployment's deterministic decoding. So this script asks 5
  differently-*framed* paraphrases of the same underlying yes/no/maybe
  question per document (direct restatement, evidence-framed, confidence-
  framed, "is it accurate that…"-framed, and a terse forced-choice
  framing) — different input strings can produce different outputs even
  under greedy decoding, so this sidesteps the exact failure mode Revision
  8 found, rather than repeating it under a different name.
- **Wording features**: response character length, word count, a hedge-
  word fraction (reusing `mia_attack.py`'s existing `HEDGE_WORDS` list so
  the two modules agree on what counts as hedging), and a small curated
  refusal-phrase detector (e.g. "not enough information", "cannot
  determine") distinct from single hedge words.
- **Aggregation across probes**: mean/max/std of similarity and certainty,
  mean length ratio, and — the actual "contrastive consistency" signal —
  `decision_match_rate` (fraction of the 5 differently-worded probes that
  land on the correct decision) and `decision_match_std` (0 if all five
  agree, higher if they don't).

Commands (collection, then train/evaluate on the collected data --
`attack/Mia_attack/contrastive_trained_attacker.py`, mirroring Phase 1's
search/evaluate discipline but over the 12-feature contrastive/wording
row shape instead of the original 4 signals):

```bash
# 1. collect (needs the live LLM service; --seeds lets you pilot at a
#    smaller scale before committing to the full default split -- see
#    "Recommended next step" below)
/opt/miniconda3/bin/python3 attack/Mia_attack/contrastive_probes.py collect --split dev  [--seeds 0 1 42]
/opt/miniconda3/bin/python3 attack/Mia_attack/contrastive_probes.py collect --split test [--seeds 865 659]

# 2. train + evaluate (offline, no live service needed once step 1 is done)
/opt/miniconda3/bin/python3 attack/Mia_attack/contrastive_trained_attacker.py search
/opt/miniconda3/bin/python3 attack/Mia_attack/contrastive_trained_attacker.py evaluate
```

`contrastive_trained_attacker.py` was verified end-to-end against
synthetic data (not the live service) this session: Leave-One-Seed-Out CV
selection, the locked-model refit, the one-shot test evaluation, and the
dev/test seed-overlap leakage guard all behave correctly. It has not been
run against real collected data, since no collection has happened yet.

**Why this wasn't run this session.** Two blockers, both disclosed rather
than worked around silently:

1. **Docker was not running** at the start of this session (`docker ps`
   failed — no daemon socket). I started Docker Desktop and confirmed the
   daemon came up, but **no containers were started** — bringing up
   `drag_llm_service` + the three `drag-data-source` containers (and,
   per `.claude/CLAUDE.md`, confirming the Hardhat node and smart
   contract) is itself a decision affecting a shared, stateful local
   deployment, and the existing report treats live-service actions as
   needing explicit authorization (e.g. the container restart in
   Revision 7 was only done "with explicit authorization obtained
   first").
2. **Cost.** This script queries the live service 5x per document (one
   call per contrastive template) instead of ~1x. At the existing
   `DEV_SEEDS`+`TEST_SEEDS` scale (13+5 seeds × 50 docs/seed) that's
   `18 × 50 × 5 = 4,500` queries — a multi-hour live-inference run against
   a real, shared service. Revision 6's own consistency-score pilot
   (5× cost per document, same reasoning) was deliberately run at a
   *smaller* scale (10+10 docs, 3 seeds) rather than the full dev/test
   split, specifically because of this cost. The same scoping-down
   should happen here before a first real run — I did not want to
   pre-commit to a seed count/scale on your behalf.

**Recommended next step, not yet taken:** run `contrastive_probes.py
collect` at pilot scale first (e.g. 3 seeds, 10+10 docs/seed, matching
Revision 6's consistency-pilot precedent) rather than the full 13+5-seed
split, confirm the collection pipeline and per-template behavior look
sane, *then* decide whether to scale up to the full dev/test split for a
real train/evaluate run.

## Environment note (unrelated to the attack, but blocks running anything)

The project's system Python
(`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`) has a
broken scikit-learn install — an x86_64-compiled extension under an
arm64 interpreter (`ImportError: ... incompatible architecture`). This
would block `tune_weights.py` too, not just the new scripts here — it is
a pre-existing environment issue, not something introduced this session.
`/opt/miniconda3/bin/python3` (conda `base` env) has a working
scikit-learn (1.8.0), numpy, requests, and sentence-transformers, and is
what both new scripts should be run with. Both new files were syntax-
checked (`python -m py_compile`) under that interpreter; neither was
executed.

## What this pass does NOT do

- **No new models or datasets** (reviewer item 1). Still Qwen +
  PubMedQA only.
- **No defense evaluation** against the trained attacker (reviewer item
  3, and the user's own "start with the attack" scoping for this pass).
  `defense/mia_defense/` is untouched.
- **No results yet, for either phase.** Phase 1 is runnable right now
  with no new infrastructure; Phase 2 needs Docker services up and a
  scale decision first (see above).
- **Two confound-control items already flagged in an earlier
  conversation turn (LLM-decoding randomness, and whether
  `rerank_with_reliability` affects MIA) were already directly addressed
  in Revision 8** (§2.10 of `reports/MIA_Security_Analysis_Report.md`) —
  root-caused and confirmed zero-effect respectively — so nothing further
  was done on those here.

## Honest caveats on the design itself

- **Phase 1's feature set is thin.** Four scalar features is a low-
  dimensional input for any classifier; the LOSO-CV/lock/evaluate-once
  discipline guards against overfitting *to the seed split*, but not
  against the ceiling that four signals simply may not carry much more
  separable information than the grid search already found. A result
  close to 0.620 (the existing linear baseline) would be an expected,
  not a surprising, outcome.
- **Phase 2's contrastive templates are hand-written, not independently
  sampled.** Five fixed phrasings mean any one template being
  systematically easier or harder (regardless of membership) is a real
  risk — per-template AUC should be checked alongside the aggregate once
  real data exists, not just the combined `decision_match_rate`/`std`.
- **Paraphrasing was deliberately not done with an LLM** (the target
  model or another one) — an LLM-based paraphraser could itself leak
  membership-correlated signal (e.g. paraphrasing more fluently for
  text nearer its own training distribution), which would confound
  Phase 2's results in a way that would be hard to detect after the
  fact. Fixed templates avoid that confound at the cost of framing
  diversity.
