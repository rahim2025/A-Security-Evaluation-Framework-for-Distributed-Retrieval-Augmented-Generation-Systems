# FedRAG Security Evaluation Framework

A complete adversarial attack and defense library built on top of the
[FedRAG](https://github.com/VectorInstitute/fed-rag) federated retrieval-augmented
generation toolkit.

---

## What is included

### Attacks (`src/fed_rag/attacks/`)

| Attack | Class | What it does |
|---|---|---|
| Data Poisoning | `DataPoisoningAttack` | Injects malicious (query, response) pairs into client training data |
| Membership Inference | `MembershipInferenceAttack` | Infers whether a query appears in a client's knowledge store via retrieval confidence |
| Knowledge Extraction | `KnowledgeExtractionAttack` | Recovers stored knowledge nodes by probing with templated queries |
| Node Availability | `NodeAvailabilityAttack` | Simulates node removal, Byzantine clients, DDoS, network partition, and Sybil injection |

### Defenses (`src/fed_rag/defenses/`)

| Defense | Class | What it counters |
|---|---|---|
| Client Data Poisoning Defense | `ClientDataPoisoningDefense` | Data Poisoning — quarantines clients with high poison-marker density |
| Score Masking | `ScoreMaskingKnowledgeStore` | Membership Inference — replaces real similarity scores with a fixed public value |
| Cross-Peer Validation | `CrossPeerValidation` | Node Availability — majority-votes answers across surviving peers |
| Query Rate Limiter | `ClientQueryRateLimiter` | Knowledge Extraction — caps total queries per client per window |
| Extraction Anomaly Detector | `ExtractionAnomalyDetector` | Knowledge Extraction — flags high-volume, broad-topic query floods |
| Response Perturbation | `ResponsePerturbation` | Extraction / MIA — adds noise to retrieved answers before returning |

### Security Evaluation Framework (`security_framework/`)

A new high-level orchestration layer added on top of the existing FedRAG
attack and defense code.  It requires **no GPU, no Hugging Face model
download, and no Ollama server** — it uses a deterministic hash-based
embedding retriever so every experiment is fully reproducible on any machine.

| Module | Purpose |
|---|---|
| `security_framework/simulator.py` | `FedRAGSimulator` — lightweight federated RAG deployment mock |
| `security_framework/pipeline.py` | `SecurityPipeline` — orchestrates all attacks + defenses |
| `security_framework/reporter.py` | `SecurityReport` — formats results as JSON / Markdown / CSV |

---

## Quick start

### Option 1 — Full evaluation (no ML dependencies)

```bash
python run_full_evaluation.py
```

This runs all four attacks with all defenses enabled and writes results to
`results/`.

### Option 2 — Custom evaluation

```bash
python run_full_evaluation.py \
    --num-clients 20 \
    --num-examples 500 \
    --seed 42 \
    --poison-type answer_swap \
    --node-attack-type byzantine \
    --output-dir my_results/
```

### Option 3 — Single attack

```bash
python run_full_evaluation.py --attack data_poisoning
python run_full_evaluation.py --attack membership_inference
python run_full_evaluation.py --attack knowledge_extraction
python run_full_evaluation.py --attack node_availability
```

### Option 4 — Disable defenses (measure raw attack impact)

```bash
python run_full_evaluation.py --no-defense
```

### Option 5 — Your own dataset

```bash
python run_full_evaluation.py --dataset-jsonl path/to/data.jsonl
```

Dataset JSONL format (one JSON object per line):

```json
{"query": "What is federated learning?", "response": "A distributed ML paradigm where training happens on-device.", "topic": "machine_learning"}
```

### Option 6 — Python API

```python
from security_framework import FedRAGSimulator, SecurityPipeline

sim    = FedRAGSimulator(num_clients=10, num_examples=200, seed=0)
pipe   = SecurityPipeline(sim, defense_enabled=True)
report = pipe.run_all()

report.print_summary()
report.save("results/")
```

---

## Full-stack evaluation (requires ML dependencies)

To run against a real sentence-transformer retriever and optionally an
Ollama LLM, use the original entry points:

```bash
# Install extras
pip install fed-rag[huggingface]
pip install ollama pyyaml datasets

# Centralized evaluation (single knowledge store)
python run_security_evaluation.py --mode centralized

# Federated evaluation (multiple clients)
python run_security_evaluation.py --mode federated

# With Ollama LLM for answer generation
python run_security_evaluation.py --mode centralized --use-ollama-generation
```

---

## Output files

After running `run_full_evaluation.py` the `results/` directory contains:

| File | Description |
|---|---|
| `security_report.json` | Complete raw results for every attack |
| `SECURITY_REPORT.md` | Human-readable Markdown summary |
| `attack_matrix.csv` | Flat CSV — one row per attack, suitable for Excel / Pandas |

The original evaluators additionally write:

| File | Description |
|---|---|
| `EVALUATION_MATRIX.md` | Centralized attack evaluation matrix |
| `FEDERATED_EVALUATION_MATRIX.md` | Federated attack evaluation matrix |
| `benchmark_attack_results.json` | Raw centralized results |
| `federated_attack_results.json` | Raw federated results |
| `evaluation_matrix.csv` | Centralized CSV |
| `federated_evaluation_matrix.csv` | Federated CSV |

---

## Pre-computed results

`attack_matrix.json` contains pre-computed results for all attacks on a
10-client, 200-example synthetic dataset.  Load them without running any
experiment:

```python
import json
from pathlib import Path

matrix = json.loads(Path("attack_matrix.json").read_text())
for row in matrix["results"]:
    print(row["attack"], "→ delta_f1 =", row.get("delta_f1"))
```

---

## Architecture

```
fed-rag/
├── src/fed_rag/
│   ├── attacks/                   # Attack implementations
│   │   ├── data_poisoning.py      #   DataPoisoningAttack
│   │   ├── membership_inference.py#   MembershipInferenceAttack
│   │   ├── kb_extraction.py       #   KnowledgeExtractionAttack
│   │   └── node_availability.py   #   NodeAvailabilityAttack
│   ├── defenses/                  # Defense implementations
│   │   ├── client_side.py         #   ClientDataPoisoningDefense, ScoreMasking,
│   │   │                          #   CrossPeerValidation, QueryRateLimiter,
│   │   │                          #   ExtractionAnomalyDetector
│   │   └── network_defenses.py    #   ResponsePerturbation
│   └── evaluators/                # Evaluation runners
│       ├── centralized.py         #   run() — single knowledge store
│       ├── federated.py           #   run() — multi-client simulation
│       ├── config.py              #   SecurityConfig, DefenseConfig
│       └── metrics.py             #   BLEU, EM, F1, HashingRetriever
│
├── security_framework/            # NEW: high-level orchestration layer
│   ├── __init__.py
│   ├── simulator.py               #   FedRAGSimulator (no ML deps)
│   ├── pipeline.py                #   SecurityPipeline
│   └── reporter.py                #   SecurityReport
│
├── run_full_evaluation.py         # NEW: single-command entry point
├── run_security_evaluation.py     # Original entry point (ML deps required)
├── attack_matrix.json             # NEW: pre-computed results
├── config/
│   ├── security.yaml              # Attack configuration
│   └── defense.yaml               # Defense configuration
└── example_scripts/
    ├── demo_attack.py             # Original demo
    ├── full_pipeline_demo.py      # NEW: API usage examples
    └── ...
```

---

## Configuration

`config/security.yaml` — toggle individual attacks and set parameters:

```yaml
security:
  enable_attack: true
  poisoning_ratio: 0.2
  poison_type: answer_swap        # wrong_answer | misleading | noise | answer_swap
  amplification_factor: 2
  enable_membership_inference: true
  membership_threshold: 0.8
  enable_extraction: true
  enable_node_availability: true
  node_attack_type: node_removal  # node_removal | byzantine | partition | ddos | sybil
  node_attack_ratio: 0.3
```

`config/defense.yaml` — enable/configure each defense:

```yaml
defense:
  enabled: true
  client_data_poisoning_defense:
    enabled: true
    quarantine_threshold: 0.25
  score_masking:
    enabled: true
    public_score: 0.5
  cross_peer_validation:
    enabled: true
    min_agreement_ratio: 0.6
    voting_method: majority
  query_rate_limiter:
    enabled: true
    max_queries_per_client: 18
  extraction_anomaly_detector:
    enabled: true
    max_queries_threshold: 20
    topic_diversity_threshold: 0.6
  response_perturbation:
    enabled: false
    perturbation_level: 0.15
    mode: noise
```

---

## Extending the framework

### Add a new attack

1. Create `src/fed_rag/attacks/my_attack.py` with a class that has an
   `execute()` method returning a result dataclass.
2. Export it from `src/fed_rag/attacks/__init__.py`.
3. Add a runner to `security_framework/pipeline.py` and call it from
   `SecurityPipeline.run_all()`.

### Add a new defense

1. Create or extend `src/fed_rag/defenses/` with your defense class.
2. Export it from `src/fed_rag/defenses/__init__.py`.
3. Wrap the corresponding attack runner in `pipeline.py`.

---

## License

Same as the upstream FedRAG project. See `LICENSE` in the repository root.
