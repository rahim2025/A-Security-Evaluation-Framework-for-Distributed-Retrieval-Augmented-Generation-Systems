# Data Poisoning Defense — Ingestion-Time Content Filter (`datapoisoning_defense`)

Defends against `attack/datapoisoning`, which injects malicious documents
through each data source's `/poison` endpoint to degrade retrieval quality
and, consequently, LLM answer accuracy.

**Scope note:** the project's `CLAUDE.md` originally scoped this evaluation
as attack-only, with defenses as future work. This module is an intentional
scope expansion requested mid-project — flag it to the team/supervisor as a
deliberate addition, not an oversight in the original plan.

## What changed

`drag_data_source/app/server.py`'s `/poison` handler previously accepted any
`{"id", "text"}` document and indexed it unconditionally — no check on
content, volume, or novelty. It now runs every candidate document through
`PoisonFilter` (`poison_filter.py`) before indexing it, and reports what it
rejected in the response body (`rejected_count`, `rejections`). A new
`GET /poison_filter/stats` endpoint exposes cumulative detection counts for
the process.

This mirrors the same principle `defense/ssm_defense` used on-chain for the
score-manipulation attack: **enforce at the boundary the attack actually
breaches**, not only in an offline report.

## Mechanism — four independent signals, any one rejects

| Signal | What it catches | Why |
|---|---|---|
| **Lexical red flags** | `wrong_answer`, `misleading`, `targeted` poison types | Generic self-referential override/correction language — "disregard this context," "has been deliberately altered," "the verified and official answer is," `[CORRUPTED]` markers. Phrased around the *behavior* (claiming to override/correct context, self-labeling as corrupted/authoritative), not lifted verbatim from this repo's attack script, so it generalizes beyond this exact wording. |
| **Gibberish tokens** | `noise` poison type | Fraction of long (≥4 char) alphabetic tokens with no vowel, OR a run of several such tokens in a row — catches a short nonsense block even when diluted inside a long legitimate-looking document. |
| **Near-duplicate / template reuse** | `amplification_factor`, `question_variants`, repeated `targeted` templates | Character-shingle Jaccard similarity against other documents in the same `/poison` request and a bounded recent-history window (2000 docs). An attacker who doesn't amplify has negligible retrieval impact anyway, so this signal covers the intensities that actually matter. |
| **Burst-volume cap** | `heavy`/`medium` intensity floods | A single `/poison` request may add at most `min(200, 15% of current corpus size)` documents; the rest of an oversized batch is dropped before the other three checks even run. |

Configurable via `PoisonFilter.__init__` — see docstrings in `poison_filter.py`
for every parameter. Toggle the whole thing off with env var
`POISON_FILTER_ENABLED=0` (see **Running** below).

## What this does NOT catch (by design)

`answer_swap` poisoning reuses a real, well-formed document from elsewhere in
the corpus attached to the wrong record. Its text is indistinguishable from
legitimate content by any content-only signal — no amount of lexical,
gibberish, or duplicate-detection tuning fixes this. Closing it would need
provenance/identity binding between a document and the source authorized to
serve it (e.g. signed documents, or cross-referencing claimed vs. actual
`record_id`), which is out of scope for this module. Measured detection rate
on `answer_swap` is ~2–11% (opportunistic catches via near-duplicate only,
not a real defense against this poison type) — see **Honesty note**.

## Offline validation (no Docker required)

`offline_validation.py` generates the exact document shapes
`DataPoisoningAttack` would send to `/poison` (all four poison types, with
`question_variants` and query-aware `targeted` docs included) and classifies
them with a fresh filter, plus a false-positive sweep on real clean corpus
documents. Run it any time with `python defense/datapoisoning_defense/offline_validation.py`.

Measured results (seed 42, `SQuAD`/polluted_token corpus, 30 base samples ×
2 variants + 7 query-aware targeted docs × 2 variants per scenario):

| Poison type | Oracle-knowledge detection | Black-box detection | Dominant signal |
|---|---|---|---|
| `wrong_answer` | 100% (74/74) | 100% (60/60) | lexical |
| `misleading` | 90.5% (67/74) | 100% (60/60) | lexical, + near-dup |
| `noise` | 52.7% (39/74) | 46.7% (28/60) | near-dup, + gibberish |
| `answer_swap` | 10.8% (8/74) | 1.7% (1/60) | near-dup only (opportunistic) |

False-positive rate: **2.0% (3/150)** real clean documents wrongly rejected —
one contained the literal word "poisoned" in legitimate context, two were
genuine near-duplicate articles already present in the corpus (the corpus
itself contains some duplication).

### Adaptive-attacker note

`misleading`'s oracle-knowledge run missed 7/74 docs on the lexical signal
specifically because of the attack script's own `_perturb_text` variant
generator: one of its four perturbations splits the document text exactly in
half and inserts `[CORRUPTED]` at the midpoint, which can land inside the
embedded phrase ("deliberately alt" | "ered") and break the regex match. The
near-duplicate signal still caught all 7, because the surrounding text
structure is unchanged — but this is a genuine illustration that a
signature-based lexical filter alone is evadable by trivial text
perturbation, and multi-signal defense-in-depth matters. Worth noting
explicitly in any writeup of this module's limitations.

### Honesty note

Thresholds (`gibberish_token_ratio=0.12`, `near_dup_jaccard=0.55`,
`max_injection_fraction=0.15`, etc.) are reasonable starting defaults chosen
by inspecting this corpus's real clean-document statistics, not
exhaustively tuned against a held-out validation set. The `noise` detection
rate in particular is capped by how close this specific random-letter
generator's statistics (~28% chance a random 6-char token has no vowel) sit
to real text's own incidental no-vowel-token rate in this corpus (a web-scraped
dataset with lists/acronyms) — pushing the gibberish threshold lower to catch
more noise risks rejecting legitimate acronym-dense passages. Re-validate
with `offline_validation.py` before relying on these numbers for anything
beyond this evaluation.

## Running

### Rebuild the data-source images (required for the live fix to take effect)

```bash
# from the repo root
docker compose build data-source-0 data-source-20 data-source-100
docker compose up -d data-source-0 data-source-20 data-source-100
```

### Live before/after comparison

```bash
# BEFORE (undefended baseline) — set POISON_FILTER_ENABLED=0 on the data-source
# containers (docker-compose.yml or an override file), rebuild/restart, then:
python attack/datapoisoning/run_attack.py --evaluate --seed 42 --intensity heavy

# AFTER (defended) — restore POISON_FILTER_ENABLED=1 (the default), rebuild/restart, then:
python defense/datapoisoning_defense/run_defense.py --seed 42 --intensity heavy
```

Use matching `--strategy`/`--poison-type`/`--ratio`/`--amplify`/`--seed`/
`--intensity` flags between the two runs for a fair before/after comparison —
`run_defense.py` accepts the same options `run_attack.py` does. Logs go to
`attack_logs/` (before) and `defense_logs/datapoisoning_defense/` (after).

### Offline-only validation (no Docker/GPU needed)

```bash
python defense/datapoisoning_defense/offline_validation.py
```
