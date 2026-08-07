# Results Re-run & Validation Tracker
**Thesis:** Security Evaluation Framework for Distributed RAG Systems
**Target:** Final defence, 10 days out
**Scope:** Chapter 5 (Result Analysis) — what must be re-run, re-validated, varied, or cut

---

## Priority legend

| Tier | Meaning |
|---|---|
| **P0** | Correctness problem visible in the current draft. An examiner reading the table will find it. Fix even if nothing else moves. |
| **P1** | Weakens a load-bearing claim. Do if time allows. |
| **P2** | Improves the report. Cut without regret if the clock runs out. |
| **CUT** | Remove from the report entirely. |
| **WRITE** | No re-run needed — fix in prose/table formatting only. |

**Type column:** `RERUN` = new experiment run · `VERIFY` = check against implementation, may need no run · `WRITE` = writing-only fix · `CUT` = delete

---

## DRAG (primary demonstration — parameter sweeps)

### P0 — must fix

- [ ] **Data poisoning — Table 4.1** · `RERUN`
  - **Issue:** n=20 eval questions. Every F1 is a multiple of 5.0. Non-monotonic cells (HD+Repl 0.70 > HD+Repl+Val 0.55 at PR 30%; Rand no-def improves from 20%→30% poisoning) are 2–3 question swings, i.e. noise.
  - **Action:** Re-run with **n ≥ 100** eval questions. Keep the existing sweep shape (PR × strategy × defence) — no new rows.
  - **Also:** Drop the F1 column for MMLU (single-letter answers make F1 ≡ EM; it carries no information). Keep F1 for News/Medical where it diverges.
  - **Accept when:** poisoning ratio 20%→30% degrades monotonically for every strategy; defence ordering (None < Repl < Repl+Val) holds.

- [ ] **DDoS — Table 4.9** · `RERUN` / `VERIFY`
  - **Issue:** 0% query success rate reported alongside F1 of 0.05–0.55. Contradictory — if no query succeeds, F1 should be ~0.
  - **Action:** Determine whether QSR and F1 are measured over different denominators, then either fix the metric or re-run. Add a **ratio axis** (currently pinned at 0.3 only).
  - **Accept when:** QSR and F1 are mutually consistent and the ratio sweep is monotonic.

- [ ] **Node removal — Table 4.7** · `CUT`
  - **Issue:** Non-monotonic across ratios (0.2 worse than 0.3), identical results across all three targeting strategies at every ratio, Avg Hops of 0.235 unexplained.
  - **Action:** **Delete the table and section.** Node removal is not one of the six framework attacks and is a resilience test, not a security attack. Removing it eliminates a liability and buys space.

### P1 — should fix

- [ ] **KB extraction — Table 4.2** · `RERUN`
  - **Issue:** SS = 100%, ER ≡ CRR, EED = 0.00 across every config. Signature of probing with exact stored questions → trivially inflated baseline.
  - **Action:** Add the **blind / topic-level probe condition** via the existing `use_dataset_questions=False` pathway. Report gray-box and blind side by side (mirrors the Reliable-dRAG KB report's Phase B / Phase B' treatment).
  - **Report cost:** +1 column pair, no new rows.

- [ ] **Routing manipulation / SSM — Tables 4.11, 4.12** · `RERUN`
  - **Issue:** Attack parameters pinned at a single point (6 attackers, topic claim ratio 0.6). This is the flagship novelty claim and has the *least* variation of any DRAG attack.
  - **Action:** Sweep **claim ratio** or **attacker count** (one axis, 3–4 points). Highest-value addition after the n fix.
  - **Also:** Avg hops recorded as 0.0 in both conditions — contradicts the hop-overhead cost claimed for cross-peer validation. Verify against logs before citing a hop-cost trade-off.

- [ ] **KB extraction defence — Table 4.3** · `RERUN`
  - **Issue:** Defended ER = 0.00 in most rows. The 20-query cap is uncalibrated, and block rate rises with KB size mechanically (larger KB needs more queries under a fixed budget), so the suppression is partly an artifact.
  - **Action:** Calibrate the threshold against a simulated legitimate-user baseline — reuse the `calibrate_thresholds.py` method already built for Reliable-dRAG. Report the calibrated number as the headline.
  - **Note:** A more modest, defensible number beats a suspiciously perfect one.

- [ ] **MIA — Table 4.5** · `RERUN`
  - **Issue:** Small n (TPR granularity suggests ~10 members). Model sweep axis (Llama 3B / Gemma 2B / Qwen 0.5B) is good and should be kept.
  - **Action:** Re-run at the same n used for poisoning. Sweep shape unchanged.

### P2 / write-only

- [ ] **MIA defence — Table 4.6** · `WRITE`
  - **Issue:** News undefended row has TPR = FPR = 1.0 — degenerate threshold, accuracy is just tracking the 70/30 base rate.
  - **Action:** Report **AUC-only** for that row, with the existing footnote promoted into the prose. No re-run needed.

- [ ] **KB defence diagnostics — Table 4.4** · `VERIFY`
  - **Issue:** Anomaly-flagged queries = 11 and peers flagged = 1 in *every one* of six conditions. The draft's own footnote flags this as suspicious.
  - **Action:** Confirm against the implementation that the detector keys on query-pattern structure (topic diversity in a fixed window) rather than content. If confirmed, state it as expected behaviour. Likely no re-run.

- [ ] **SFA defence — Table 4.8** · `WRITE`
  - **Status:** Clean and monotonic. No re-run.
  - **Action:** Convert the 5-point ratio sweep to a **figure**; retain only endpoints (0.1 and 0.5) in the table. Saves 6 rows.

- [ ] **DDoS defence — Table 4.10** · `WRITE`
  - **Status:** Good structure (intensity × strategy × defence). No re-run.
  - **Action:** Consider figure treatment to reduce 13 rows.

---

## Reliable-dRAG (portability demonstration — multi-seed)

> **Scope reminder:** defences on this system are **out of report scope**. Attack results only. Each attack needs one line in a consolidated table, not a section. Exception: SSM-Score gets its own subsection (novelty claim).

- [ ] **DDoS live mode** · `RERUN` · **P1**
  - **Issue:** The headline result — 90% query failure, semantic similarity 0.83 → 0.04 — is **single-seed** by the report's own admission (§15). Mock mode already has 3 seeds.
  - **Action:** Re-run live evaluation at **3 seeds (0, 42, 123)**. Report mean ± std.

- [ ] **SFA** · `RERUN` · **P1**
  - **Issue:** All §8 figures are `trials=1`, single-seed. Live sweeps are n=30, one seed. Report explicitly says "not yet statistically robust."
  - **Action:** 3 seeds minimum on the configuration that goes in the report.

- [ ] **SSM-Score** · `WRITE` · **P0**
  - **Issue:** Draft table shows 3 seeds; the underlying report ran each seed **twice** (6 independent trials, 3 escalated).
  - **Action:** Report as **3/6 trials**, not 1/3 seeds. The probabilistic outcome is the finding — present it as such rather than as inconsistency.

- [ ] **KB extraction** · `NONE`
  - Already 3 seeds with per-source std. Compress to one row in the consolidated table.

- [ ] **MIA** · `NONE`
  - Over-provisioned (13 dev + 5 fresh held-out seeds, AUC 0.620, 95% CI [0.546, 0.695]). Compress to **one row**: headline AUC + CI + n.

- [ ] **Data poisoning** · `RERUN` · **P2 (likely infeasible in 10 days)**
  - **Issue:** No analysis report exists. Leaves Integrity resting on SSM alone for system two.
  - **Action if time:** Run and document. **If not:** state explicitly in 5.6 Limitations as a known coverage gap rather than leaving it silently absent.

---

## Cut list (do this first — it's free)

- [ ] Node removal table + section (Table 4.7)
- [ ] Per-seed row explosion → collapse all multi-seed results to `mean ± std (n=k)`
- [ ] Redundant F1 column on MMLU tables
- [ ] Reliable-dRAG defence results (already out of scope — confirm none leaked into draft)
- [ ] Dataset dimension inside individual tables → one consolidated dataset-sensitivity table

---

## Blocked / awaiting material

- [ ] **KB defence implementation files** — on other machine. If not retrieved in time, state as implemented-but-not-measured in 5.3 rather than omitting silently.
- [ ] **`SKILL.md` + `references_*` files** — needed before generating report LaTeX.

---

## If the clock runs out

Do these in order and stop wherever you stop:

1. n ≥ 100 on data poisoning (Table 4.1)
2. DDoS QSR/F1 contradiction (Table 4.9)
3. Cut node removal (Table 4.7)
4. KB blind-probe condition (Table 4.2)
5. SSM parameter sweep (Table 4.11)
6. Reliable-dRAG DDoS + SFA seeds

Items 1–3 are visible defects in the current draft. Everything below 3 is strengthening, not repair.

---

## Status summary

| System | Re-run | Verify | Write-only | Cut |
|---|---|---|---|---|
| DRAG | 5 | 1 | 3 | 1 |
| Reliable-dRAG | 2–3 | 0 | 1 | 0 |
