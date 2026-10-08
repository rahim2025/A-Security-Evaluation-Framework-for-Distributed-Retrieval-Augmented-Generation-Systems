# Progress Report — MIA, Selective Forwarding and DoS on Reliable-dRAG

**Project:** Security evaluation of Reliable-dRAG (second system in the generalizability study, after DRAG)
**Prepared for:** Dr. Swakkhar Shatabda
**Scope of this report:** the three attacks where work has been done since the last update: Membership Inference (MIA), Selective Forwarding (SFA) and Denial of Service (DoS). It covers what each attack measures, what was wrong or weak, what we changed, and what is still unverified.

---

## 0. Executive summary

| Attack | CIA property | Main change | Where it stands |
|---|---|---|---|
| MIA | Confidentiality | Fixed a measurement flaw: the score could not tell "the database helped" from "the model already knew". Added a no-retrieval control, a free-text corpus, larger pools, multi-seed statistics and failure handling. | Logic fixed. Fresh multi-seed runs on the live system are pending. |
| SFA | Availability (covert) | Added harder, more realistic attackers. Fixed the defense's detector (wrong baseline, no time window). | Logic verified offline in simulation. Not yet run against the live system. |
| DoS | Availability (overt) | Added a baseline comparison and a real client-side defense. Fixed two bugs in that defense and one in the metric. | Logic verified offline with unit tests. Live evaluation not yet run. |

**Important caveats:**
- **Nothing in this report is a final thesis number.** The recent changes were checked by reading the code, syntax checks, unit tests and mock-mode runs. They were not run end to end against the live Docker/Hardhat stack. The final multi-seed runs (seeds 0, 42, 123) still have to be done on the machine that can run the language model.
- **Defenses are outside the agreed scope.** The agreed deliverable is attack-only. The defense work below was done on explicit instruction, and its purpose is to confirm that the attacks are real and measurable. It is not a claim that defenses are part of the contribution.

---

## 1. Membership Inference Attack (MIA)

### 1.1 What the attack asks

Confidentiality has two attacks: KB Extraction asks what is in the database, and MIA asks only whether a particular record exists in it. An attacker who cannot read the database sends carefully chosen questions about a candidate document and looks at how the system answers. If the answer is grounded in that document, the document was probably retrieved, so it is probably a member.

Following the RAGLeak method (Feng et al., ACISP 2025), the signal is how closely the answer matches the target document. We report AUC-ROC across thresholds, not one accuracy figure. A value of 0.50 means no leakage. Values above about 0.70 mean a real vulnerability.

### 1.2 The problem we found

The attack measured "membership" mainly by checking whether the answer to a yes/no/maybe question was correct. That has a structural weakness. A correct answer can come from two places:

1. **Retrieval:** the system found the document in the database. This is the signal we want.
2. **The model's own training:** the language model (Qwen2.5-1.5B) already knows the fact, or just guesses "yes". PubMedQA is about 55% "yes", so guessing already gets a fair number right.

Because of the second path, non-member documents also get answered correctly. That shrinks the gap between members and non-members, and the AUC drifts toward 0.50. Earlier SQuAD-based runs were not just near chance but consistently below it, which means the signal was inverted. The likely cause is that SQuAD comes from Wikipedia and the model had already seen it.

The more serious issue was that nothing in the output told us this was happening. A run could produce a meaningless AUC with no indication of why.

### 1.3 What we changed

**A. Control measurement (the core fix).** Every probe is now answered twice: once through the normal retrieval pipeline and once with retrieval switched off, using the same model and prompt. A document counts as evidence of membership only when the retrieval answer is right and the no-retrieval answer is wrong. In other words, we measure what the database added, so knowledge the model already had cancels out. The uncorrected signal is kept as a diagnostic so earlier results stay comparable.

**B. A harder, fairer dataset.** PubMedQA labels are only yes/no/maybe, which limits how much signal the attack can ever see. We added a second corpus, HealthCareMagic: long, free-text answers from doctors to real patients. The domain is the same, so the confound is comparable, but the answer is long-form text and not a three-way label. For this corpus the score is the retrieval lift: how much closer the answer is to the target document with retrieval than without.

**C. Larger, cleaner member and non-member pools.**
- 5,000 members and 5,000 non-members (previously 25 vs 25 in the default configuration).
- Boilerplate answers that would match anything are dropped.
- Non-members are de-duplicated against members, so a near-copy cannot leak membership.
- The default probe count is now 100 per class.

**D. Honest statistics.** The attack runs with seeds 0, 42 and 123 and reports mean ± standard deviation. It also reports a bootstrap confidence interval on the AUC. This follows the rule that single-seed results are not acceptable.

**E. Failures are not data.** Previously, if the language model service timed out or errored, the empty response was scored as similarity 0. That created fake "member" signals. For example, a lift of +0.42 appeared when only the no-retrieval call had failed, and runs ended up as all zeros once the service stopped answering. Now the attack retries once, skips the document if it still fails, aborts after 5 consecutive failures, and records the number of failed documents in the log.

**F. Rate-limit handling.** The data sources accept 60 requests per minute, and each question fans out to all three of them. Measured live, 64% of requests were being rejected (HTTP 429), which silently degraded retrieval and therefore the score. We added pacing between calls and retry-with-backoff, and extended the same protection to the contrastive-probe module, which sends about five times as many calls.

**G. Provenance and stale-data guard.** Every log now records the corpus hash and the code version. The attack also checks that the corpus the system loaded is the one we think we are testing. This protects against a stale mounted file producing results for the wrong dataset.

### 1.4 Probe-design rule

We do not use exact stored text as a probe. Probes are paraphrased or keyword-level. The earlier DRAG work used exact text and produced inflated results, so this rule is kept here from the start.

### 1.5 Status and open points
- Fixed in code; the new HealthCareMagic multi-seed run is pending on the live stack.
- The result is conditional on the control measurement being valid, so the first live run should confirm that member and non-member distributions separate after retrieval and overlap before it.
- Remaining known limitation: the language model is not fully deterministic from run to run. This is why we report variance across seeds and do not rely on one run.

### 1.6 Defense (scoped as future work, implemented for validation only)
A similarity-noise defense adds calibrated Laplace noise to the retrieval similarity scores, so the score the attacker can exploit becomes less reliable. It passes its offline unit tests (13 of 13). It is not a formal differential-privacy guarantee: the noise is fixed per query for reproducibility, so it is "calibrated obfuscation". Live evaluation has not been run.

---

## 2. Selective Forwarding Attack (SFA)

### 2.1 What the attack is

SFA is the covert availability attack. An insider peer earns trust and then silently drops some of the queries it should forward. Reliable-dRAG prefers high-reliability sources, so a peer that has built a high score and then starts dropping does maximum damage at minimum visibility. This is why the attack matters especially for this system.

### 2.2 Weakness we addressed: the attacker model was too easy

The original attacker dropped queries at a fixed rate. A real adversary would behave less predictably. We added more realistic attackers as a subclass, so the validated baseline stays untouched:

- **Delay attacker:** answers slowly instead of staying silent. This is a different failure mode from dropping.
- **Drifting attacker:** the drop rate rises and falls over time instead of staying fixed, so there is no stable signature to detect.
- **Colluding attackers:** several compromised peers take turns. Only one is on duty for any query, so each looks individually mild while the total damage is the same.
- **Threshold-adaptive attacker:** a worst case in which the attacker knows the defense's parameters and drops just enough to stay under the detection line.

In the mock run the adaptive attacker was never detected (0 true positives across two seeds) while still causing measurable damage. This is a result in its own right: it shows where a fixed-threshold detector fails.

### 2.3 Weaknesses fixed in the detector

1. **Wrong honest baseline.** The binomial detector tests each peer against an assumed "honest" miss rate. That was hard-coded to 5%, but in the default network the true honest rate is about 60%. Every honest peer therefore looked guilty. With the shipped defaults it blacklisted 5 of 10 peers even when only one was compromised. The baseline is now measured automatically as the median miss rate of the other tracked, non-blacklisted peers. The median is used so that a minority of compromised peers cannot drag the estimate up.
2. **No true time window.** The detector now uses a rolling window of recent behaviour. This matters for the central scenario: a peer that behaves well to build a high reliability score and then starts dropping would otherwise be hidden by its good history. Resetting the defense now also clears the window.

### 2.4 How the defense is evaluated

For each attacker type we run a defended and an undefended sweep with the same seed, and measure:
- **False positives:** honest peers wrongly blacklisted.
- **Detection delay:** how many queries pass before a compromised peer is caught.
- **Extra latency and network cost:** overhead of the defense (bypass routes and redundant probes).
- **Low-relevance honest peers:** a no-attack scenario where honest peers naturally respond rarely. Any blacklisting there is a false positive by definition. This tests the main ambiguity of the approach: a low response rate can mean either "dropping" or "this peer simply has little relevant content".

### 2.5 Status and limits
- Verified offline in mock mode. Not yet run against the live containers.
- **Not done: answer-quality impact.** The SFA simulation runs on the client side, so a simulated drop does not change what the real language-model service retrieves. Measuring the true effect on answer quality would need interception in front of the real data sources.
- **Not done: larger testbed.** Only three real peers exist. More would need new containers, corpus splits and on-chain identity registration.

---

## 3. Denial of Service (DoS)

### 3.1 What the attack is

DoS is the overt availability attack. An outside attacker floods the system so legitimate queries lose access to sources, and we measure how hit rate and answer quality degrade as pressure increases. It is the visible counterpart to SFA.

### 3.2 What we added

**Repeated trials.** The live flood evaluation previously ran with a single seed. It now accepts several seeds and reports mean and standard deviation of the availability drop at each severity.

**Baseline comparison.** To show whether any defense helps, we compare against reference designs:
- a centralized network (a single node);
- replicated retrieval (query everyone, succeed if anyone answers, at higher cost);
- ordinary uninformed rate limiting, versus load-aware routing.

Two baselines were deliberately not reproduced and are marked as such, not faked: turning reliability-aware routing on and off (this is a property of the real service, not of the mock) and a true centralized RAG product (a different system, not a configuration of this one).

**A real client-side defense.** The earlier defense existed only in the simulation. We built one that can be tested against live traffic, with four mechanisms:
- a rate limiter per peer;
- a circuit breaker that opens when a peer keeps failing, probes it later, and closes again when it recovers;
- a per-peer queue limit;
- a short-lived cache of recent answers.

The evaluation compares no defense, the simulation defense and this live defense under a real flood. It reports availability, latency, answer quality, defense cost, and false blocks (blocking after the attack has stopped).

### 3.3 Bugs found and fixed
1. **Unenforced safety cap.** The defense had a "maximum fraction of peers it may mark unavailable" setting, but nothing enforced it. Under a heavy flood every peer could be blocked at once, which is worse than no defense. The cap is now enforced.
2. **The cap's first fix defeated itself.** We found this by running the unit tests, which we had written earlier but never executed. The first fix checked each peer in isolation, so when several peers were congested each one saw the others already counted and let itself through. In a three-peer test with two congested peers, none were reported blocked. The cap is now decided once per call: the first allowed peers (in a fixed order) are blocked, and in a total outage the true state is reported without the cap. The re-run gives the intended result: one of the two congested peers is blocked and the other is let through by the cap.
3. **Division by zero** when a peer's rate-limit capacity is set to zero. It is now scored as the least healthy.
4. **Misleading false-block metric.** The metric counted every block ever made, including the correct ones during the flood. It now counts only blocks that occur after the attack has stopped.

### 3.4 Result so far (mock mode)
At the scale of the earlier validated report (20 peers, 100 questions, 3 seeds), the defense recovered hit rate directionally, from 0.503 to 0.533. That is a modest gain. A small initial test showed the defense performing worse than none. We re-ran it at full scale, and the small test turned out to be sampling noise, not a wiring error.

### 3.5 Status and limits
- Unit tests for the live defense pass (11 of 11). The live flood evaluation has not been run.
- **Asymmetry to state clearly:** the defense protects the client's retrieval layer, but the language-model service makes its own separate calls to the data sources, and the defense does not wrap those. End-to-end answer quality is therefore not improved by it. This is documented in the evaluator and should be reported as a limitation, not hidden.
- **Not done:** larger deployments, different resource limits, other datasets and models. These are infrastructure work.

---

## 4. How this supports the thesis claim

The claim is that the CIA-triad framework from DRAG generalizes to a structurally different distributed RAG system. The work above supports that in three ways:

- **MIA:** the same attack category carries over, but Reliable-dRAG needed a control for model pretraining knowledge. That is a concrete adaptation point to report in the DRAG vs Reliable-dRAG comparison.
- **SFA:** the reliability-scoring mechanism makes the attack more damaging here. The adaptive attacker shows that a defense tuned to a fixed threshold can be evaded.
- **DoS:** the same overt attack applies, but the system's source-health routing and its two-layer structure (client retrieval vs the service's own calls) change what a defense can protect.

---

## 5. What remains before these results can go in the thesis

1. Bring up the Docker/Hardhat stack on the machine that can run the language model. The development laptop is Apple Silicon and the service depends on vLLM/CUDA.
2. Run the baseline (no attack) first, to record the reference F1.
3. Run MIA on PubMedQA and HealthCareMagic with seeds 0, 42 and 123, and confirm that the control measurement separates the members.
4. Run SFA and DoS in live mode with the same three seeds.
5. Fill in the DRAG vs Reliable-dRAG comparison table using these numbers.
6. Report variance for every figure, and state the limitations in sections 2.5 and 3.5 explicitly.

---

*Sources for this report: `reports/updated_reports_safin/` (MIA_SCORE_MECHANISM_FIX, MIA_SFA_DDOS_FIXES_ROUND2 and ROUND3, SELECTIVE_FORWARDING, DDOS, SIMILARITY_NOISE_DEFENSE, VERIFICATION_STATUS) and the commit history on branch `tanzim_safin`.*
