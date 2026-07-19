# Introduction and Current Goals

Based on the CIA triad, we sought to employ 6 attacks in our evaluation framework to assess the security of the distributed RAG system it's being applied to. Alongside the original gossip base P2P [DRAG](https://github.com/xuchenhao001/DRAG) codebase, we are planning to apply the the framework to [Reliable-dRAG](https://github.com/yining610/Reliable-dRAG), as well as considering federated RAG approaches as well with [FedRAG](https://github.com/VectorInstitute/fed-rag). The current goal now is to apply the evaluation framework to both DRAG and another codebase to showcase generalizability of the framework in context of distributed RAG systems, alongside showcasing how the evaluation can be used improve a system by applying defenses after evaluation on the DRAG codebase.

---

# Security Evaluation Framework: Attack Overview

## Data Poisoning Attack (Integrity)

Distributed RAG systems store knowledge across multiple independent sources, each of which contributes to the pool of information the system draws from when answering queries. This distribution of knowledge ownership creates a corresponding distribution of trust — the system must assume that each source's knowledge is accurate, because it has no inherent mechanism to verify the content of what any source contributes.

The Data Poisoning Attack exploits this assumption. A compromised source injects incorrect, misleading, or deliberately confusing knowledge into its local knowledge store. When queries subsequently arrive that match the poisoned content, the system retrieves and presents the corrupted information as legitimate context, causing the LLM to generate degraded or wrong answers. The attack is amplified by injecting multiple copies of each poisoned entry and by constructing semantically similar question variants that map to the same wrong answer, maximising the probability that incoming queries match the poisoned data rather than any clean version.

What makes the source selection strategy meaningful is that not all sources contribute equally to the system's output. Targeting high-connectivity or data-rich sources causes disproportionately greater degradation than random selection, because those sources serve more queries and hold more of the knowledge the system relies on. Replica-aware targeting goes further — identifying topics whose content is replicated across multiple sources and poisoning all replicas simultaneously, eliminating the redundancy that would otherwise dilute the attack's effect.

What makes this attack distinct from external interference is the threat model it assumes: the attacker has write access to at least one source's knowledge store. This is an insider or compromised-participant model, not an external one. In a centralized RAG system this threat reduces to a single point of compromise — one database, one target. In a distributed system, the attack surface scales with the number of sources, and the damage scales with how strategically the attacker chooses which sources to compromise.

**What is measured:** F1 score and answer quality degradation compared to a clean baseline, as a function of the poisoning ratio, the source selection strategy, and the amplification factor applied.

## Source Selection Manipulation Attack (Integrity)

Distributed RAG systems need a way to decide which source to send each query to. Every such system implements some mechanism for this — DRAG uses TARW, where peers advertise topic expertise and the routing algorithm uses those advertisements to direct queries toward relevant peers; Reliable-dRAG uses a blockchain-based reliability scoring system, where sources accumulate trust scores over time and the LLM orchestrator preferentially routes queries toward higher-scoring sources. These mechanisms work efficiently, but they share a common trust assumption: that participants are honest about their own capability or quality.

The Source Selection Manipulation Attack breaks that assumption. A malicious participant joins the system and manipulates whatever signal the source selection mechanism uses to evaluate trustworthiness or relevance — falsely advertising topic expertise it doesn't hold, or artificially inflating its own reliability score. The selection mechanism, seeing this participant as highly relevant or highly trusted, preferentially directs queries toward it. The malicious source then receives queries it has no legitimate knowledge for, and returns garbage answers or fabricated responses.

The damage is twofold. First, answer quality degrades because queries land on the wrong source. Second, legitimate high-quality sources receive fewer queries than they should — the malicious participant has effectively crowded them out of the selection priority queue.

What makes this attack meaningful for the framework is that it requires no privileged access. The attacker is just a regular participant that misrepresents its own capabilities or trustworthiness through the normal interfaces the system already provides. In a centralized RAG system this attack class does not exist at all — there is only one knowledge base and nothing to select between. The vulnerability is a direct consequence of the architectural decision to distribute knowledge across independently operating sources, each of which must be trusted to represent itself honestly. The specific manipulation technique is mechanism-dependent — advertisement-based in DRAG, score-based in Reliable-dRAG — but the attack structure and its consequences are identical across both.

**What is measured:** the rate at which queries get misdirected to malicious sources, and the resulting F1 and answer quality degradation compared to baseline, across both the advertisement-manipulation instantiation in DRAG and the score-manipulation instantiation in Reliable-dRAG.

## Knowledge Base Extraction Attack (Confidentiality)

The knowledge stored in a distributed RAG system represents the collective intellectual property of its participants — curated datasets, proprietary documents, or domain expertise that individual source owners have contributed. This knowledge is not intended to be publicly accessible in raw form; users are expected to receive answers to specific queries, not bulk access to underlying content. The system's query interface is designed for retrieval, not disclosure.

The Knowledge Base Extraction Attack treats the query interface itself as an exfiltration channel. An external attacker, operating with no access beyond what any legitimate user possesses, submits a systematic series of topic-balanced queries drawn from the same domain as the system's knowledge. Each query elicits a response that incorporates retrieved knowledge chunks as context. By collecting these responses at scale — carefully selecting queries to achieve broad topic coverage rather than concentrating on a single subject — the attacker progressively reconstructs a substantial portion of the underlying knowledge base without ever accessing it directly.

The effectiveness of this attack rests on a structural property of RAG systems: they are designed to be informative. The more accurately and completely a system answers queries, the more of its underlying knowledge it necessarily exposes through those answers. There is no clean separation between being useful and being extractable. This tension is not resolvable through query rate limiting alone, because the attack requires no unusual query volume — it requires only breadth of coverage.

This attack assumes no privileged access and no knowledge of the system's internal architecture beyond the fact that it is a RAG system operating over some domain. It is equally applicable to any distributed RAG system that exposes a query interface, making it the most directly portable attack in the framework.

**What is measured:** Extraction Rate (the fraction of the total knowledge base recovered), Chunk Recovery Rate (the fraction of individual knowledge chunks recovered), Semantic Similarity between extracted and ground-truth content, and Effective Edit Distance, all as a function of query budget and topic coverage breadth.

## Membership Inference Attack (Confidentiality)

When a RAG system retrieves knowledge in response to a query, its behaviour is not uniform across all queries. Queries that closely match knowledge the system holds produce responses with different characteristics — higher confidence, greater specificity, more detailed answers, faster or more direct routing — than queries about topics the system has no relevant data for. This behavioural asymmetry is a functional requirement of the system: it is how useful retrieval differs from an uninformed guess. But it is also a signal that an external observer can exploit.

The Membership Inference Attack uses this signal to determine whether specific data points are present in the system's knowledge base without ever accessing that knowledge base directly. An adversary operating purely through the query interface submits probing queries and observes characteristics of the responses — semantic similarity between the response and the query, answer length and specificity, and indicators of routing behaviour such as how directly and confidently the system responds. These features are used as discriminative signals to classify whether the queried content is a member of the training or knowledge corpus or not.

The attack operates as a black-box adversary: no access to model weights, no access to source internals, no knowledge of the system's topology. It requires only the ability to submit queries and observe responses. This makes it applicable to any distributed RAG system that exposes a query interface, regardless of its internal architecture. The severity of the privacy leak scales with model capability — larger and more capable models tend to produce more detailed, contextually rich responses for known content, inadvertently making the membership signal stronger rather than weaker.

In a distributed RAG system, the membership inference risk has an additional dimension absent from centralized systems: because knowledge is partitioned across sources, a successful inference attack can reveal not only whether a data point exists in the system, but potentially which source or which topic partition holds it, exposing the distribution of knowledge ownership across participants.

**What is measured:** Membership inference accuracy, precision, recall, and a Privacy Risk Score across query types and model configurations, measuring how reliably response characteristics discriminate between member and non-member queries.

## Denial of Service Attack (Availability)

Distributed RAG systems process queries through a pipeline of interdependent components — routing logic, knowledge retrieval, and response generation — each of which consumes finite computational resources. Under normal operating conditions these resources are sufficient to handle expected query volumes within acceptable latency bounds. The Denial of Service Attack targets this resource budget directly: by submitting query volume significantly in excess of what the system is designed to handle, an external attacker attempts to exhaust the processing capacity of one or more components, degrading response quality, increasing latency, or causing queries to fail entirely.

Unlike attacks that require insider access or network participation, the DoS attack is purely external. The attacker needs only the ability to submit queries at volume — the same interface available to any legitimate user. There is no privileged position required, no network presence to maintain, and no knowledge of internal system architecture needed beyond the existence of a query endpoint.

An important methodological note applies to how this attack is characterized in simulation environments. Sequential simulation architectures cannot model genuine resource contention — queries are processed one at a time and the system never experiences true concurrent load. In such environments, the DoS scenario is more accurately described as a **load sensitivity analysis**: a controlled measurement of how system performance — answer quality, query success rate, and response coherence — degrades as query throughput increases beyond baseline levels. This characterization is still informative, establishing the system's performance envelope under stress even if it does not replicate the conditions of a real volumetric attack. In service-based architectures with real HTTP endpoints, concurrent request generation produces a closer approximation of genuine resource exhaustion, enabling a more direct characterization of availability risk.

The adversarial dimension that distinguishes this from random load fluctuation is target selection: an attacker who concentrates traffic on the system's highest-cost component — typically the LLM inference endpoint — causes greater degradation per query than distributing load evenly, because inference is computationally bottlenecked in a way that routing and retrieval are not.

**What is measured:** Query success rate, answer quality (F1), and response latency degradation as a function of query load, with explicit acknowledgment of simulation constraints where applicable and comparison between sequential and concurrent execution models where the target architecture permits.

## Selective Forwarding Attack (Availability)

Distributed RAG systems rely on their constituent sources to respond to queries honestly and completely. No source is compelled to contribute useful responses — participation is assumed cooperative, and the system's retrieval logic has no inherent way to distinguish a source that genuinely lacks relevant knowledge from one that is deliberately withholding it.

The Selective Forwarding Attack exploits this assumption. A compromised source remains visible and reachable within the system — it does not go offline, and the system continues to direct queries toward it — but when queries arrive, it silently drops them, returning empty or null responses instead of retrieving relevant knowledge. From the system's perspective, this source looks healthy. It is present, it accepts queries, and it occupies whatever routing priority or reliability score it has legitimately earned. But every query it receives produces nothing useful, consuming a retrieval attempt without contributing to an answer.

The key distinction from simple source failure is the adversarial intent behind which sources are compromised. A source going offline randomly is a reliability problem — it happens by chance and system designers plan for it through replication and redundancy. Selective forwarding is a deliberate attack where the adversary chooses which sources to compromise for maximum disruption: targeting high-priority or high-reliability sources causes far greater degradation than targeting peripheral or low-scored ones, because the system preferentially routes queries toward them.

What makes this attack meaningful for the framework is that it is invisible to naive health checks. The compromised source passes any liveness test — it is reachable and responsive — while systematically undermining retrieval quality. In a centralized RAG system this attack reduces to total system failure, which is immediately detectable. In a distributed system, the damage is partial and silent, making it significantly harder to diagnose.

**What is measured:** hit rate degradation, answer quality decline, and the difference in impact between randomly selected compromised sources and strategically selected high-priority ones, as a function of what fraction of sources are compromised.

---

# Application on Federated RAG system

In the federated setup, attacks target clients and their data, and we measure how the global system performance and privacy are affected.

## 1. Client Data Poisoning

We pick some clients as malicious and inject wrong or misleading examples into their local data. When the system aggregates data across clients, those poisoned items reduce the overall retrieval quality. The drop in scores shows how much the poisoned clients hurt the system.

### Step-by-Step

- **Setup:** The dataset is split across clients (IID). A subset of clients is marked malicious based on `poisoning_ratio` and `attack_strategy`.
- **What the attacker does:** On each malicious client, some local examples are **poisoned** (e.g., wrong/swapped answers). The attack can also **amplify** poisoned items and create **question variants**, so bad data appears multiple times.
- **How impact is measured:** A global store is rebuilt from all clients (clean + poisoned). Retrieval quality is evaluated (EM/F1/BLEU). The difference between baseline and post-attack is the **quality drop** caused by malicious client data.
- **Defense in the run:** A quarantine-style inspection runs; if a client is flagged, its data is excluded. If no one is quarantined, defended scores stay the same as post-attack.

## 2. Client Membership Inference

We test whether an attacker can guess if a query belongs to a specific client's private data. The attack tries to infer membership using retrieval confidence. If accuracy is high, it means privacy leakage.

### Step-by-Step

- **Goal:** Decide if a query likely belongs to a specific client's private data.
- **How it works:**
  - "Member queries" are taken from the target client.
  - "Non-member queries" are taken from other clients.
  - The attack checks retrieval confidence and predicts membership using a threshold.
- **Metric meaning:**
  - **Membership Acc = 1.0** means perfect leakage.
  - **Membership Acc = 0.5** means random-guess level.
- **Defense in the run:** Score masking is applied (reduces confidence signals).

## 3. Client Knowledge Extraction

We send many queries to the system to recover as much stored knowledge as possible. The recovery percentage shows how much of a client's data can be extracted. Defenses like rate limiting and anomaly detection try to reduce that recovery rate.

### Step-by-Step

- **Goal:** Recover as much stored knowledge as possible via repeated querying.
- **How it works:** The attacker issues many queries (either dataset questions or generic topic templates). Any retrieved nodes are counted as "recovered."
- **Metric meaning:**
  - **KB Recovery % = 100%** means the attacker recovered everything.
  - Lower % after defenses means the defense is working.
- **Defense in the run:** Query rate limiting and anomaly detection block suspicious query patterns, reducing recovery.

---

> **Important note:** The federated runner measures **retrieval correctness**, not LLM generation quality. So the drop you see is "retrieval quality degradation" caused by poisoned clients, not full LLM output quality.

---

# Defense Overview

In order to showcase how a system would benefit from our framework, we also applied several defense mechanisms to the DRAG system:

## Defense Against Data Poisoning: Cross-Peer Validation

The data poisoning attack succeeds because the system, by default, trusts whatever any individual source returns. A poisoned source can return a fabricated answer and the system has no way to distinguish it from a legitimate one. Cross-peer validation addresses this by making agreement a prerequisite for trust: instead of accepting the first answer the routing algorithm retrieves, the system collects responses from multiple peers and only accepts an answer if enough of them concur.

**How it works.** When a query is processed, the system extends its routing to gather responses from a minimum of three peers rather than stopping at the first match. These responses are then compared against each other. Because real-world answers from different peers rarely use identical wording even when they are correct, the comparison uses fuzzy string matching — two answers are treated as equivalent if they are at least 85% similar by sequence matching. The matching answers are grouped together and the group with the most members is treated as the majority position. A candidate answer is accepted only if it belongs to the majority group and that group represents at least 60% of all peer responses collected. If no clear majority exists — for instance, when all peers return genuinely different answers — the system defaults to accepting the candidate rather than rejecting it, preserving availability at the cost of reduced protection in ambiguous cases.

**Why it works against poisoning.** A poisoned source is, by definition, a minority. In a network where most sources hold clean knowledge, the fabricated answer from one or a few compromised peers will be outvoted by the correct answer from the honest majority. The attacker would need to compromise more than 60% of the peers consulted for any given query to push a poisoned answer through validation — a significantly harder bar than compromising any single peer. The fuzzy matching prevents the attacker from evading detection by making minor lexical variations to their poisoned answers, since semantically similar legitimate answers will still cluster together and form a majority.

**The cost.** The defense increases message overhead. A query that would normally complete in 3–4 routing hops now requires up to 6, because the system needs to gather enough peer responses to perform meaningful validation. This is the primary trade-off: improved integrity at the cost of increased network load. The configuration allows up to 2 additional routing hops beyond the baseline to collect the minimum required peer responses.

## Defense Against Knowledge Extraction: Layered Extraction Prevention

The knowledge extraction attack succeeds through systematic, broad querying — sending enough carefully distributed queries to reconstruct a significant portion of the knowledge base over time. A single defensive measure is unlikely to stop a determined attacker, because any one barrier can be worked around given enough time or creativity. The defense therefore applies three complementary layers, each targeting a different dimension of extraction behavior, so that bypassing one layer does not make the others irrelevant.

---

### Layer 1 — Query Rate Limiting

The most direct property of an extraction attack is volume: an attacker needs to send many queries to extract many data points. The rate limiter exploits this by capping how many queries a single peer or user can submit within a rolling time window. The window tracks query timestamps continuously, discarding entries that fall outside the recent window and counting only those that remain. If that count exceeds the configured limit of 20 queries per window, the requesting peer is temporarily blocked for a fixed cooldown period before they can query again.

This alone reduces extraction effectiveness roughly proportionally to how much it cuts query volume. An attacker who could previously send 44 queries and recover 26 data points is now capped at 20 queries, limiting their recoverable data points to approximately half that. The defense is computationally cheap and takes effect before any extraction succeeds, since the block is applied the moment the query count exceeds the threshold rather than after damage is done. Its primary weakness is that a sufficiently patient or distributed attacker — one operating from multiple accounts or IP addresses — can work around it by spreading queries across identities.

---

### Layer 2 — Response Perturbation

Rate limiting targets extraction by volume; response perturbation targets extraction by quality. Even if an attacker successfully retrieves a response, what they receive has been deliberately degraded at the character level before it reaches them. The system intercepts the raw knowledge chunk before it is returned and injects 15% character-level noise — substituting, inserting, deleting, or swapping individual characters — so that the structured content the attacker captures is subtly corrupted.

The key design decision is that the perturbation targets only the raw structured data returned in the retrieval pipeline, not the LLM-generated final answer presented to the user. A human reading the response sees a clean, readable answer because the LLM generates its response using the original clean context. But an attacker parsing the underlying structured output — as automated extraction tools do — receives a corrupted version. This directly attacks the metrics that extraction success depends on: exact match fails the moment even a single character differs, semantic similarity drops if enough characters are corrupted, and effective edit distance increases proportionally to the noise injected. The defense is most effective against automated extraction pipelines that rely on precise string recovery, and less effective against attackers who use the human-readable LLM output rather than the raw retrieved chunks.

---

### Layer 3 — Extraction Anomaly Detection

The first two layers address extraction by volume and by output quality. The third layer addresses extraction by behavioral pattern. It is motivated by a consistent characteristic of extraction attacks that distinguishes them from legitimate usage: normal users ask questions about a small number of related topics, while an extraction attacker deliberately spreads queries across as many topics as possible to maximize coverage of the knowledge base.

The detector tracks three signals per peer over the course of their query session: total query volume, the number of unique topics queried, and the ratio between the two — topic diversity per query. A legitimate user asking 44 questions about a single subject produces a diversity ratio close to zero. An extraction attacker asking 44 questions distributed across 29 different topics produces a ratio of approximately 0.66. The detector flags a peer when any of three threshold conditions are met: total queries exceeding 20, unique topics exceeding 6, or a diversity ratio exceeding 0.6. Once flagged, the peer's subsequent queries can be silently dropped, throttled, or returned with deliberately misleading responses.

The anomaly detector catches what the rate limiter misses: a low-volume but highly systematic attacker who stays under the query cap but probes many topics in each session. Conversely, the rate limiter catches what the anomaly detector might miss: a high-volume attacker who focuses narrowly on one topic. Together they cover both dimensions of extraction strategy.

---

### Combined Effect

The three layers form a defense-in-depth stack that operates at different points in the query lifecycle. Rate limiting applies before a query is processed and caps total extractable volume. Anomaly detection applies during the session and identifies behavioral patterns that rate limiting alone would not catch. Response perturbation applies immediately before the response is returned and degrades the quality of whatever data the attacker does manage to collect. An attacker who circumvents one layer — for example, by spreading queries slowly across multiple sessions to evade the rate limiter — still faces behavioral flagging from the anomaly detector and output corruption from the perturbation layer. The combination makes large-scale, high-fidelity knowledge extraction significantly harder to execute even against a patient and adaptive adversary.