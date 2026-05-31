# Plan: Generalised Security-Evaluation Framework for FedRAG

## Context
- DRAG (distributed-rag-poison-defense) has: DataPoisoning, KBExtraction, MembershipInference, **NodeAvailability** attacks + CrossPeerValidation, QueryRateLimiter, ResponsePerturbation, ExtractionAnomalyDetector defenses.
- FedRAG has: DataPoisoning, KnowledgeExtraction, MembershipInference attacks + ClientDataPoisoningDefense, ClientQueryRateLimiter, ExtractionAnomalyDetector, ScoreMaskingKnowledgeStore defenses.
- **Gap in FedRAG**: NodeAvailability attacks, CrossPeerValidation defense, ResponsePerturbation defense.
- Goal: Port the missing DRAG attack/defense pieces into FedRAG so both architectures can be evaluated with a single, modular framework.

## Implementation Steps

### 1. Add NodeAvailabilityAttack to fed_rag/attacks/
- Create `src/fed_rag/attacks/node_availability.py`
- Attack types: node_removal, byzantine, partition, ddos, sybil
- Works on a list of "nodes" (peers or clients), architecture-agnostic
- Export in `src/fed_rag/attacks/__init__.py`

### 2. Add missing defenses to fed_rag/defenses/
- Create `src/fed_rag/defenses/network_defenses.py`:
  - `CrossPeerValidation`: Majority voting across peer/client responses
  - `ResponsePerturbation`: Perturb retrieved knowledge before returning
- Export in `src/fed_rag/defenses/__init__.py`

### 3. Extend federated security evaluation runner
- Update `src/fed_rag/security/federated_evaluation.py` to include:
  - Node availability attack simulation (client dropout, byzantine, sybil)
  - Cross-peer validation defense
  - Response perturbation defense
  - Metrics: availability_percentage, failed_queries, byzantine_responses

### 4. Create unified runner at repo root
- Create `run_unified_security_evaluation.py`:
  - Single entry point running all attacks (poisoning, membership, extraction, node availability)
  - Produces evaluation matrix with all metrics
  - Works with both DRAG configs and FedRAG synthetic data

### 5. Update evaluation matrix writer
- Extend `MATRIX_COLUMNS` in `src/fed_rag/utils/evaluation_matrix.py` to include node-availability fields:
  - "Availability %", "Defended Availability %", "Byzantine Nodes", "Sybil Nodes"

### 6. Verify
- Run `python -m py_compile` on new/modified files
- Check for import errors / circular deps
- Ensure existing tests still pass
