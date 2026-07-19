# CLAUDE.md — Reliable-dRAG Security Evaluation

## Who We Are

We are a five-person undergraduate thesis team from a Bangladeshi university, supervised by Dr. Swakkhar Shatabda. Our thesis is titled **"Exploring Privacy-Preserving Approaches for Large Language Models"**.

---

## What We Are Doing in This Codebase

We are **not modifying Reliable-dRAG itself**. We are building a security evaluation layer *on top of it* — attack modules that probe the system through its existing interfaces, exactly as an adversary would.

This is the second system in our generalizability evaluation. Our primary system is **DRAG** (Xu et al., 2025), a peer-to-peer gossip-based distributed RAG. We built and validated a full CIA-triad security framework on DRAG. We are now porting that framework here to demonstrate it generalizes across distributed RAG architectures.

The thesis contribution is the **framework**, not the attacks in isolation. Every implementation decision should serve the argument: "the same structured approach to distributed RAG security applies here as it did on DRAG."

---

## The Framework — Six Attacks, CIA Triad

Each CIA property has two attacks targeting structurally distinct mechanisms. This is the taxonomy we defend in the thesis.

| Property | Attack | Mechanism |
|---|---|---|
| Integrity | Data Poisoning | Data-plane — what a source *stores* |
| Integrity | Source Selection Manipulation (SSM) | Control-plane — which source gets *trusted* |
| Confidentiality | KB Extraction | Content — what is *in* the store |
| Confidentiality | Membership Inference (MIA) | Presence — whether a record *exists* |
| Availability | Denial of Service (DoS) | Overt, external resource exhaustion |
| Availability | Selective Forwarding | Covert, insider withholding |

**Defenses are scoped as future work** for this system. Attack-only evaluation is sufficient to demonstrate generalizability.

---

## Why Reliable-dRAG Specifically

Reliable-dRAG shares the same distributed-retrieval paradigm as DRAG: independent data sources, no central knowledge base, a routing/selection mechanism that decides which source serves each query. This makes the attack surface structurally comparable.

Critically, Reliable-dRAG introduces a **blockchain-based reliability scoring mechanism** (R_i / U_i scores, updated via smart contract on each query). This is a concrete, novel instantiation of our **Source Selection Manipulation** attack — something that exists in this system's architecture but has no equivalent in DRAG. It is the most distinctive contribution of this generalization effort.

---

## Attack Mapping to This System

| Attack | Instantiation Here | Priority |
|---|---|---|
| **Data Poisoning** | Inject polluted documents at the data-source level before retrieval. The system's own evaluation already uses token/document-level pollution — our adversarial version is the malicious equivalent. | High — start here |
| **SSM (Score Manipulation)** | Manipulate R_i / U_i feedback to inflate a malicious source's reliability score on-chain. Targets the smart-contract update path (Algorithm 2 in the paper). This is the novel attack. | High — most important |
| **KB Extraction** | Systematically query the LLM service interface to reconstruct private source content. Use keyword/topic-level probes, *not* exact stored text. Report CRR, SS, and EED metrics. | Medium |
| **Membership Inference** | Probe via confidence or grounding signals. Use paraphrased probes from the start. Report AUC-ROC across thresholds, not single-point accuracy. | Medium |
| **DoS** | Take data sources offline; measure hit-rate and answer-quality degradation as a function of sources removed. | Medium |
| **Selective Forwarding** | A source earns a high R_i score, then silently withholds responses. Especially impactful here because the system *prioritises* high-scoring sources, so a forwarding-dropper causes maximum damage at minimum visibility. | Medium |

---

## What We Already Have (from DRAG)

All six attacks and three defenses are implemented and working on DRAG. The base classes, config-driven architecture, and evaluation metrics (F1, EM, Precision, Recall, CRR, SS, EED) are already built. The goal here is **porting and adapting**, not building from scratch.

Key things that will need adapting:
- The network interface (DRAG uses a Python gossip simulation; Reliable-dRAG uses HTTP endpoints + Docker + a local Hardhat blockchain node).
- The attack entry points (DRAG peers are directly accessible Python objects; here we go through the LLM service and data-source APIs).
- SSM has no DRAG equivalent — this is a net-new implementation.

---

## Environment Setup Notes

Reliable-dRAG runs via **Docker Compose** with a local **Hardhat Ethereum node** for the blockchain component. Both must be running before any experiments. The smart contract is pre-deployed on the local testnet.

**Before doing anything else:**
1. Confirm Docker is running and `docker-compose up` succeeds.
2. Confirm the Hardhat node is live and the smart contract is deployed.
3. Run the clean baseline (no attacks) and record F1 / accuracy to establish ground truth before touching anything.

---

## What Good Results Look Like

We need, at minimum:
- A **clean baseline** on Natural Questions (the dataset used in the paper).
- **Data poisoning** results at light / medium / heavy intensity, with F1 degradation curves.
- **SSM** results showing reliability-score manipulation inflates a malicious source's usage share.
- At least two or three remaining attacks with clear before/after metrics.
- **Multi-seed runs** (at least seeds 0, 42, 123) with variance reported. Single-seed results are not acceptable for the thesis.
- A **DRAG vs. Reliable-dRAG comparison table** — same attack, both systems, side by side. This table is the generalizability proof.

---

## What to Avoid

- **Do not use exact stored text as probe queries** in KB Extraction or MIA. This was a mistake in earlier DRAG work that produced trivially inflated results. Use keyword or paraphrased probes from the start.
- **Do not claim defenses are in scope** for this system. Attack-only is the agreed deliverable.
- **Do not modify the Reliable-dRAG core system** unless strictly necessary for instrumentation. We are evaluating the system as-is.
- **Do not report single-seed results** as final. Run multiple seeds before writing up any number.

---

## Relationship to the Thesis Argument

Every result here feeds one claim: *the CIA-triad security framework developed on DRAG generalizes to a structurally different distributed RAG architecture.* Keep that sentence in mind when deciding what to implement, what to prioritize, and how to frame results. If a result doesn't serve that claim, it is secondary.
