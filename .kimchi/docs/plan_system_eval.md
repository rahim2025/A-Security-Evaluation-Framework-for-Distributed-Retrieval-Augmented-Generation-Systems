# Plan: System-Level Security Evaluation for FedRAG

## Goal
Create a system-level evaluation framework for FedRAG that mirrors DRAG's
`simulator.py` approach, enabling both systems to be evaluated on the **same
metrics** and producing a **unified comparison matrix** for the paper.

## Current Gap
- DRAG evaluates the **full system**: baseline → attack → post-attack, measuring
  exact_match, precision, recall, f1, bleu, rouge, semantic_similarity,
  num_hops, num_messages, query_hit, query_failure_rate, availability.
- FedRAG evaluates **individual attacks** (centralized/federated) but lacks a
  unified baseline-vs-post-attack system-level runner with degradation metrics.
- FedRAG metrics are limited to em, f1, bleu, jaccard (missing DRAG metrics).

## Implementation Plan

### 1. Extend `src/fed_rag/evaluators/metrics.py`
Add DRAG-compatible metrics:
- `precision`, `recall` (token-based, like DRAG)
- `rouge1`, `rouge2`, `rougeL` (via `rouge-score`)
- `semantic_similarity` (via `sentence-transformers`)
- `edit_distance`, `normalized_edit_distance` (via `python-Levenshtein`)
- `bigram_overlap`, `trigram_overlap`
- `query_failure_rate`, `successful_queries`, `failed_queries`
- `avg_num_hops`, `avg_num_messages`, `avg_query_hit`

Gracefully handle missing dependencies with try/except.

### 2. Create `src/fed_rag/evaluators/system.py`
System-level evaluator that mirrors DRAG's `simulator.py` flow:

```
Load dataset → Build knowledge store → BASELINE EVAL
→ Apply attack(s) → POST-ATTACK EVAL
→ Compute degradation → Write matrix + JSON + terminal summary
```

Supports:
- **Data Poisoning Attack**
- **Node Availability Attack** (node_removal, byzantine, partition, ddos, sybil)
- **Knowledge Base Extraction Attack**
- **Membership Inference Attack**

Collects per-query metadata: `num_hops`, `num_messages`, `is_query_hit`,
`relevant_score`, `relevant_knowledge`.

### 3. Update `src/fed_rag/utils/evaluation_matrix.py`
Add new matrix columns for system-level metrics:
- `Baseline Precision`, `Post-Attack Precision`, `Defended Precision`
- `Baseline Recall`, `Post-Attack Recall`, `Defended Recall`
- `Baseline Rouge1/2/L`, `Post-Attack Rouge1/2/L`
- `Baseline Semantic Sim`, `Post-Attack Semantic Sim`
- `Baseline Query Hit`, `Post-Attack Query Hit`
- `Query Failure Rate Baseline`, `Query Failure Rate Post-Attack`
- `Avg Num Hops`, `Avg Num Messages`
- `Availability %`, `Post-Attack Availability %`

### 4. Create `run_system_evaluation.py`
New top-level runner:
```sh
python run_system_evaluation.py \
  --mode centralized \
  --dataset mmlu \
  --llm llama32_3b \
  --attack poisoning \
  --output-dir logs/system_evaluation
```

Modes:
- `centralized`: single knowledge store, real sentence-transformer retriever
- `federated`: multi-client simulation

### 5. Export from `src/fed_rag/evaluators/__init__.py`
Add `run_system` entry point.

### 6. Add dependencies to `pyproject.toml`
- `nltk>=3.9`
- `rouge-score>=0.1.2`
- `python-Levenshtein>=0.26.0` (or `Levenshtein`)

## Files to Modify
| File | Change |
|------|--------|
| `src/fed_rag/evaluators/metrics.py` | Add DRAG metrics |
| `src/fed_rag/evaluators/system.py` | New system-level evaluator |
| `src/fed_rag/evaluators/__init__.py` | Export `run_system` |
| `src/fed_rag/utils/evaluation_matrix.py` | Extend matrix columns |
| `pyproject.toml` | Add `nltk`, `rouge-score`, `python-Levenshtein` |
| `run_system_evaluation.py` | New runner script |

## Expected Output
- `logs/system_evaluation/SYSTEM_EVALUATION_MATRIX.md`
- `logs/system_evaluation/system_evaluation.csv`
- `logs/system_evaluation/system_evaluation.json`
- Terminal summary with baseline vs post-attack degradation (DRAG-style)
