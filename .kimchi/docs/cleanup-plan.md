# Cleanup Plan: Make FedRAG as clean as DRAG

## Problem
FedRAG has **6 top-level scripts** doing similar security evaluation, and one **massive 1000-line** `src/fed_rag/security/federated_evaluation.py`. DRAG has **1 entry point** (`simulator.py`) and clean `modules/`.

## Goal
- One top-level entry point that reads configs and dispatches evaluations.
- Evaluation logic split into clean, focused modules under `src/fed_rag/evaluators/`.
- Demo / experiment scripts moved out of the repo root.

## Steps

### 1. Create `src/fed_rag/evaluators/` package
- `config.py`    — SecurityConfig, DefenseConfig, `load_security_config()`, `load_defense_config()`
- `metrics.py`   — `tokenize`, `token_f1`, `simple_bleu`, `jaccard`, helpers
- `centralized.py` — centralized evaluation logic (extracted from current `run_security_evaluation.py`)
- `federated.py`   — federated evaluation logic (extracted from `security/federated_evaluation.py`)
- `__init__.py`  — exports `run_centralized`, `run_federated`, `SecurityConfig`, `DefenseConfig`

### 2. Rewrite `run_security_evaluation.py`
- Single entry point.
- `--mode {centralized,federated}` (default: centralized).
- Reads `config/security.yaml` + `config/defense.yaml`.
- Delegates to `evaluators.centralized` or `evaluators.federated`.

### 3. Remove `run_federated_security_evaluation.py`
- Now redundant (covered by unified runner).

### 4. Move demos out of repo root
- `demo_attack.py` → `example_scripts/demo_attack.py`
- `evaluate_real_attack.py` → `example_scripts/evaluate_real_attack.py`
- `poison_one_client.py` → `example_scripts/poison_one_client.py`
- `real_attack_experiment.py` → `example_scripts/real_attack_experiment.py`

### 5. Clean up `src/fed_rag/security/`
- Remove `src/fed_rag/security/federated_evaluation.py` after extraction.
- Keep `src/fed_rag/security/__init__.py` minimal.
