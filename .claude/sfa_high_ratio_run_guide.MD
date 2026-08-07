# Reliable-dRAG selective forwarding — high-ratio live run

**Purpose:** establish whether Phase-2 redundant-probe recovery survives when only one
honest source remains. This is the only remaining question the three-source deployment can
answer, since every ratio below the two-source threshold compromises exactly one peer and
therefore repeats the existing experiment.

**Outcome either way is reportable.** Recovery holding at one honest source strengthens the
finding; recovery breaking gives you the point where it breaks and turns a single
measurement into a two-point curve. There is no null result available here.

---

## Step 1 — Determine the ratio before running anything

Open `attack/selective_forward_sim/selective_forwarding_attack.py`, lines 92–101
(`select_targets()`), and read how `num_compromise` is derived from `ratio` and peer count.

| If the code uses | Use ratio | Gives |
|---|---|---|
| `int()` or `floor()` | `0.67` | `int(2.01) = 2` |
| `ceil()` | `0.40` | `ceil(1.2) = 2` |
| `round()` | `0.60` | `round(1.8) = 2` |

Do not guess. With three peers, `0.67` under `ceil()` yields 3, which compromises every
source and produces a total blackout rather than the test you want.

**Confirm from the output, not the input.** The attack JSON records `num_compromise`. If it
reads anything other than 2, stop and adjust the ratio before running the remaining seeds.

---

## Step 2 — Preconditions

1. `docker compose ps` shows all three data sources and the orchestrator healthy.
2. **Verify the question loader is on the post-§12.8 code.** Earlier live runs silently fell
   back to a six-question generic set (`"What is the capital of France?"`) when corpus
   matching returned nothing, with no warning printed. Confirm the run is drawing real
   PubMedQA questions before trusting any output. This is the failure mode most likely to
   waste the run.
3. Keep `--min_request_interval_s 1.1`, which holds the client under the server's
   60-per-minute per-source limit. Do not raise the server limit and lower the throttle
   together for this run; the existing setting is known good.

---

## Step 3 — Command

Identical to §12.12's protocol apart from `--ratio`, so the two configurations remain
directly comparable.

```bash
python attack/selective_forward_sim/run_attack.py --mode live --single \
    --ratio <from step 1> --strategy high_connectivity --n_questions 200 \
    --min_request_interval_s 1.1 --seed {0,42,123}
```

Run all three seeds. Unlike the ratio-0.3 case, **the seed is expected to matter here.** At
one compromised source, two honest sources remain and every question is answerable, which
is why §12.12 saw zero variance. At two compromised sources, whether a query succeeds
depends on whether the single remaining source holds content for that question, and the
seed determines which 200 questions are drawn. Variance across seeds is the informative
signal, not noise.

Estimated wall time: roughly 4 minutes minimum per seed from throttling alone, plus
generation. Budget an hour for all three.

---

## Step 4 — Acceptance checks

Per run, before using any number:

- [ ] `num_compromise == 2`
- [ ] `rate_limited_total == 0`
- [ ] `error_total == 0`
- [ ] Question set is corpus-matched PubMedQA, not the six-question fallback
- [ ] `dropped/total` consistent with two compromised sources rather than one

If `rate_limited_total` or `error_total` is non-zero, the run is uninterpretable. That was
the §12.9 confounder, and it produced plausible-looking but meaningless numbers the first
time.

---

## Step 5 — Reading the result

| Observed | Interpretation | What to write |
|---|---|---|
| Hit rate stays 1.000 | Recovery survives on a single honest source. The mock's TTL-bounded model overstates damage at every severity this deployment can express | Live evaluation validates the mechanism across the full expressible range; mock sweep carries severity |
| Hit rate falls, TTL exhaustion rises | Recovery degrades once redundancy is nearly exhausted. You now have two points and a direction | Report both configurations as a two-point comparison and state where recovery begins to fail |
| Hit rate collapses toward 0 | Recovery depends on at least two honest sources | Strong finding: names the redundancy floor the mechanism requires |

Variance across the three seeds is itself a result. Zero variance would mean the single
honest source covers every question drawn; non-zero variance quantifies how often it does
not.

---

## Step 6 — Reporting into Chapter 5

Table 5.19 becomes a two-configuration table rather than one row. **Add a
`Sources Compromised` column and place it before `Ratio`.** That column is the real severity
variable on a three-source deployment, and showing it pre-empts the obvious question about
why a ratio sweep was not run: ratios from 0.1 to roughly 0.5 all resolve to one compromised
source and are the same experiment.

Report each configuration as mean ± std across the three seeds.

Then update:

- **5.2.5 prose** — replace the open-question framing with the two-point comparison
- **5.6 Limitations** — the concession narrows from "single configuration" to the accurate
  statement that a three-source deployment admits only two distinct severities, so no
  dose-response curve is obtainable live and the mock sweep supplies it
- **4.4 provenance** — the clause reading "though at a single configuration" can be dropped
- **Figure spec** — if hit rate varies between the two configurations, the Reliable-dRAG
  selective forwarding figure gains a live overlay on the mock curve, which makes the
  mock-versus-live divergence visible rather than described

---

## What not to run

- Additional seeds at ratio 0.3. The determinism is documented and understood; more seeds
  produce identical rows.
- A strategy comparison. All three sources are tied at 10000.0 reliability, so the stable
  sort makes `high_connectivity` deterministic, and with an equalised corpus which source is
  compromised carries no information.
- Intermediate ratios. Every value between the one-source and two-source thresholds repeats
  an experiment you already have.
