# FedRAG Security Evaluation Framework — Session Context

## Project Goal
Build a **modular, generalised security-evaluation framework** that works for both:
- **DRAG** (Distributed RAG — peer-to-peer network)
- **FedRAG** (Federated RAG — federated fine-tuning of retriever + generator)

The framework should evaluate attacks (poisoning, membership inference, knowledge extraction, node availability) and defenses, producing a unified evaluation matrix.

## What We Did in This Session

### 1. Added missing DRAG attacks/defenses to FedRAG
- **New attack**: `src/fed_rag/attacks/node_availability.py`
  - Types: `node_removal`, `byzantine`, `partition`, `ddos`, `sybil`
  - Works on any sequence of nodes (peers or clients), architecture-agnostic
- **New defenses**: `src/fed_rag/defenses/network_defenses.py`
  - `CrossPeerValidation` — majority voting across peer/client responses
  - `ResponsePerturbation` — noise / mask / truncate on retrieved metadata

### 2. Cleaned up FedRAG architecture to match DRAG's cleanliness
**Before** (cluttered):
- 6 top-level scripts: `demo_attack.py`, `evaluate_real_attack.py`, `poison_one_client.py`, `real_attack_experiment.py`, `run_security_evaluation.py`, `run_federated_security_evaluation.py`
- Massive 1000-line `src/fed_rag/security/federated_evaluation.py`

**After** (clean):
- **One entry point**: `run_security_evaluation.py` — dispatches by `--mode {centralized,federated}`
- **Demos moved**: `example_scripts/` (out of repo root)
- **Evaluators package**: `src/fed_rag/evaluators/`
  - `config.py` — SecurityConfig, DefenseConfig, loaders
  - `metrics.py` — shared metrics (EM, F1, BLEU, jaccard, HashingRetriever)
  - `centralized.py` — real-model evaluator (single store)
  - `federated.py` — synthetic-client evaluator (multi-client simulation)
- **Config-driven** like DRAG:
  - `config/security.yaml` — attack toggles (`enable_attack`, `enable_membership_inference`, `enable_extraction`, `enable_node_availability`)
  - `config/defense.yaml` — defense toggles (master `enabled` + per-defense flags)

### 3. Extended evaluation matrix
Added columns for node availability:
- `Availability %`
- `Post-Attack Availability %`
- `Defended Availability %`
- `Byzantine Nodes`
- `Sybil Nodes`

## Current Architecture

```
fed-rag/
├── run_security_evaluation.py          ← ONE entry point
├── example_scripts/                    ← demos (moved from root)
│   ├── demo_attack.py
│   ├── evaluate_real_attack.py
│   ├── poison_one_client.py
│   └── real_attack_experiment.py
├── config/
│   ├── security.yaml                   ← attack true/false switches
│   └── defense.yaml                    ← defense true/false switches
├── src/fed_rag/
│   ├── evaluators/                     ← NEW: clean evaluator package
│   │   ├── __init__.py
│   │   ├── config.py                   ← SecurityConfig, DefenseConfig, loaders
│   │   ├── metrics.py                  ← shared metrics + HashingRetriever
│   │   ├── centralized.py              ← centralized mode (real models)
│   │   └── federated.py                ← federated mode (synthetic clients)
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── node_availability.py        ← NEW (ported from DRAG)
│   │   ├── data_poisoning.py
│   │   ├── kb_extraction.py
│   │   └── membership_inference.py
│   ├── defenses/
│   │   ├── __init__.py
│   │   ├── network_defenses.py         ← NEW (ported from DRAG)
│   │   └── client_side.py
│   ├── security/
│   │   └── __init__.py                 ← thin re-export of evaluators
│   └── utils/
│       └── evaluation_matrix.py        ← updated with availability columns
```

## Key Design Decision: Client-Level vs System-Level Evaluation

### What exists now
**Centralized mode** → evaluates the **whole system** (single global knowledge store). This is inherently system-level — exactly like DRAG's network view.

**Federated mode** → evaluates **individual clients** one at a time:
- Data poisoning: poisons client 3, tests client 3
- Membership inference: probes client 5's store
- Knowledge extraction: extracts from client 2
- Node availability: drops client 7, tests client 7

### The architectural gap
DRAG evaluates the **whole system** — when a peer is attacked, the entire network degrades. Our federated mode does **not** show what happens to the **global system** after aggregation.

In real federated learning:
1. Clients train locally on their data
2. They send model updates (not raw data) to a server
3. Server aggregates updates into a global model
4. **The global model is what users actually interact with**

So poisoning client data → poisons local updates → poisons the global model → degrades the whole system.

### What needs to be added
**System-level evaluation within the federated context**:

After client-level attacks, build a **global knowledge store** from ALL client data combined (respecting dropped clients, excluding quarantined ones if defense is on). This represents "the world after aggregation." Run the same 4 attacks against this global store.

This gives **two rows per attack** in the matrix:

| Attack | Scope | Measures |
|--------|-------|----------|
| Client Data Poisoning | One client's local store | Baseline vs poisoned client's BLEU/EM/F1 |
| System Data Poisoning | Global store built from ALL clients (including poisoned) | Baseline vs global poisoned BLEU/EM/F1 |
| Client Membership Inference | One client's store | Can attacker guess if query is in *that client*? |
| System Membership Inference | Global store from all clients | Can attacker guess if query is in *any client*? |
| Client Knowledge Extraction | One client's store | How many nodes recoverable from *that client*? |
| System Knowledge Extraction | Global store from all clients | How many nodes recoverable from *whole system*? |
| Client Node Availability | One dropped client's store | What happens to one client when dropped? |
| System Node Availability | Global store from *active* clients only | What happens to whole system when some clients drop? |

### How to implement
In `src/fed_rag/evaluators/federated.py`, after the client-level evaluation blocks, add a **system-level evaluation block**:

1. Build `global_clean_store` from all clean client data (already exists as `clean_store`)
2. If poisoning enabled: build `global_poisoned_store` from all poisoned client data combined
3. Run attacks against the global store(s)
4. Add matrix rows with `"System ..."` prefix

## How to Run

```bash
# Enable all attacks in config
# config/security.yaml:
#   enable_attack: true
#   enable_membership_inference: true
#   enable_extraction: true
#   enable_node_availability: true
#   node_attack_type: 'byzantine'

# Enable defenses in config
# config/defense.yaml:
#   enabled: true
#   client_data_poisoning_defense:
#     enabled: true
#   score_masking:
#     enabled: true
#   cross_peer_validation:
#     enabled: false
#   query_rate_limiter:
#     enabled: true
#   extraction_anomaly_detector:
#     enabled: true

# Run federated mode
python run_security_evaluation.py --mode federated \
    --num-clients 10 --num-examples 120 --seed 7 \
    --enable-node-availability --node-attack-type byzantine

# Run centralized mode (real models)
python run_security_evaluation.py --mode centralized \
    --dataset mmlu --llm llama32_3b --seed 0
```

## Open Questions / Next Steps

1. **System-level evaluation in federated mode** — the main missing piece. Needs implementation in `federated.py` after client-level blocks.

2. **Model-level aggregation simulation** — currently we aggregate knowledge stores (documents). In real FL, clients aggregate retriever **weights**, not documents. For a more realistic FedRAG evaluation, we should simulate weight poisoning / Byzantine gradients. This is a deeper architectural addition.

3. **DRAG integration** — Can we run the same framework against DRAG's actual codebase? The `NodeAvailabilityAttack` and `CrossPeerValidation` are designed to work on any node sequence, so they should be reusable.

4. **Unified matrix across both systems** — The centralized mode uses real data (MMLU, medical, news). The federated mode uses synthetic data. For a truly unified comparison, we should allow federated mode to use the same real datasets split across clients.

5. **Defense effectiveness measurement** — Currently defenses are boolean on/off. We should measure *how much* the defense mitigates the attack (delta metrics).

## Key Files Modified in This Session
- `src/fed_rag/attacks/node_availability.py` ← NEW
- `src/fed_rag/attacks/__init__.py` ← updated exports
- `src/fed_rag/defenses/network_defenses.py` ← NEW
- `src/fed_rag/defenses/__init__.py` ← updated exports
- `src/fed_rag/evaluators/__init__.py` ← NEW
- `src/fed_rag/evaluators/config.py` ← NEW
- `src/fed_rag/evaluators/metrics.py` ← NEW
- `src/fed_rag/evaluators/centralized.py` ← NEW (extracted from old runner)
- `src/fed_rag/evaluators/federated.py` ← NEW (extracted from old runner)
- `src/fed_rag/security/__init__.py` ← slimmed down
- `src/fed_rag/security/federated_evaluation.py` ← DELETED
- `src/fed_rag/utils/evaluation_matrix.py` ← added availability columns
- `config/security.yaml` ← added node availability flags
- `config/defense.yaml` ← NEW
- `run_security_evaluation.py` ← rewritten as unified dispatcher
- `run_federated_security_evaluation.py` ← DELETED
- `demo_attack.py` → `example_scripts/`
- `evaluate_real_attack.py` → `example_scripts/`
- `poison_one_client.py` → `example_scripts/`
- `real_attack_experiment.py` → `example_scripts/`
