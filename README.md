# FedRAG Security Evaluation Pipeline

Federated Retrieval-Augmented Generation (FedRAG) security evaluation with 4 adversarial attacks,
20 federated clients, and end-to-end LLM generation using llama3.2:3b via Ollama.

---

## What This Does

Evaluates how adversarial attacks degrade a federated RAG system across two layers:

  Retrieval layer  — How much each attack corrupts knowledge store retrieval quality (EM, F1, SemSim)
  LLM layer        — How poisoned context changes llama3.2:3b news summaries (F1, Semantic Similarity)

Dataset : heegyu/news-category-dataset — 20 news headlines, one per federated client
Attacks : Data Poisoning | Node Availability | Knowledge Extraction | Membership Inference (MIA)

---

## Prerequisites

  Python      3.10+   (inside WSL Ubuntu 22.04)
  Ollama      latest  (running on Windows host, reachable at localhost:11434)
  llama3.2:3b         (pulled via Ollama)
  GPU         optional (all-MiniLM-L6-v2 retriever; falls back to CPU)

---

## One-Time Setup

### 1. Enter project directory
  cd /mnt/d/safin/distributed-rag-poison-defense-fed-rag-safin/distributed-rag-poison-defense-fed-rag-safin

### 2. Create and activate virtual environment
  python3 -m venv venv
  source venv/bin/activate

### 3. Install Python dependencies
  pip install -e .
  pip install datasets requests sentence-transformers nltk

### 4. Pull the LLM (run in Windows PowerShell, NOT WSL)
  ollama pull llama3.2:3b
  ollama serve

### 5. Verify Ollama is reachable from WSL
  curl http://localhost:11434/api/tags
  # Should return JSON listing llama3.2:3b

---

## Run the Full Pipeline

  source venv/bin/activate

  python run_system_with_llm_news_20clients.py \
    --llm-model llama3.2:3b \
    --num-samples 20 \
    --num-clients 20 \
    --attacks poisoning node_availability extraction membership_inference \
    --output-dir logs/news_llama_20clients_all_attacks

Runtime: ~10-20 minutes depending on GPU/CPU speed.

---

## Generate HTML Report

  python generate_news_report.py
  explorer.exe FedRAG_News_Report.html

---

## Optional Flags

  --llm-model       llama3.2:3b   Any Ollama model tag (mistral, gemma2:2b, etc.)
  --num-samples     20            Number of news articles to evaluate
  --num-clients     20            Number of simulated federated clients
  --seed            0             Random seed for reproducibility
  --output-dir      logs/...      Where JSON results are saved
  --attacks         (all 4)       Space-separated list of attacks to run

---

## Expected Results

Phase 1 - Retrieval Layer (no LLM):

  Attack                  Baseline EM   Post-Attack EM   Impact
  Data Poisoning          1.00          0.70             -30%  <-- vulnerable
  Node Availability       1.00          1.00              0%   <-- resilient by design
  Knowledge Extraction    1.00          1.00              0%   <-- measures leakage not quality
  Membership Inference    1.00          1.00              0%   <-- measures privacy not quality

Phase 2 - LLM Generation Layer (llama3.2:3b):

  Scenario                F1      Semantic Similarity
  Baseline (clean)        ~0.23   ~0.48
  After poisoning         ~0.20   ~0.44   <-- LLM reads corrupted context
  Node/Extraction/MIA     ~0.23   ~0.48   <-- no change (expected)

NOTE: Exact Match (EM) = 0 for all LLM results is CORRECT and EXPECTED.
Free-text summarisation never exactly matches ground truth.
F1 and Semantic Similarity are the real metrics for news generation.

---

## Output Files

  logs/news_llama_20clients_all_attacks/
  |-- system_evaluation.json        Phase 1 retrieval-layer results (all 4 attacks)
  |-- llm_generation_results.json   Phase 2 LLM-layer results (baseline + poisoning)

  FedRAG_News_Report.html           Full interactive HTML report

---

## What Was Fixed and Implemented

  Fix 1 - LLM output "2" bug
    Problem : llama3.2:3b treated news headlines as MMLU multiple-choice, answered "2"
    Solution: Rewrote prompt to demand ONE sentence; uses only top-1 retrieved doc

  Fix 2 - Response validator
    Problem : Model still occasionally output bare numbers on first attempt
    Solution: _clean_response() detects MCQ-pattern answers, retries up to 3x

  Fix 3 - Poisoning LLM evaluation showed n/a
    Problem : fed_rag.attacks.get_attack does not exist in this version
    Solution: Manual store poisoning - directly corrupts metadata["answer"] of 30% of nodes

  Fix 4 - Attacks appeared harmless at LLM layer
    Problem : LLM only ran on clean store so no attack damage was visible
    Solution: _run_llm_on_store() runs LLM on both clean and poisoned stores

  IMPORTANT: No changes made to the core fed_rag/ library.
  All fixes are in run_system_with_llm_news_20clients.py and generate_news_report.py only.

---

## Pipeline Architecture

  Query (news headline)
       |
       v
  Retriever (all-MiniLM-L6-v2 · GPU)
       |
       v
  Knowledge Store <--- attacks injected here (Phase 1)
  (20 news nodes, one per federated client)
       |
       v
  Top-1 Retrieved Doc
       |
       v
  LLM Generator (llama3.2:3b via Ollama) <--- Phase 2 reads poisoned context
       |
       v
  Generated Answer (1-sentence news summary)
       |
       v
  Metrics vs Ground Truth (EM | F1 | BLEU | Semantic Similarity)

---

## Federated Client Distribution (20 Clients, IID Split)

  Client 00: THE WORLDPOST      Client 10: STYLE & BEAUTY
  Client 01: STYLE & BEAUTY     Client 11: HEALTHY LIVING
  Client 02: BLACK VOICES       Client 12: WORLD NEWS
  Client 03: WOMEN              Client 13: TASTE
  Client 04: POLITICS           Client 14: HOME & LIVING
  Client 05: WELLNESS           Client 15: PARENTING
  Client 06: IMPACT             Client 16: CRIME
  Client 07: WELLNESS           Client 17: WELLNESS
  Client 08: RELIGION           Client 18: POLITICS
  Client 09: GOOD NEWS          Client 19: ENTERTAINMENT

All client knowledge is pooled into one in-memory store of 20 nodes before evaluation.

---

## Troubleshooting

  Connection refused on Ollama
    Run "ollama serve" in Windows (not WSL). Check port 11434 is not blocked by firewall.

  ModuleNotFoundError: fed_rag
    Run: pip install -e .   (inside activated venv)

  ModuleNotFoundError: datasets
    Run: pip install datasets

  LLM outputs "2" again
    Confirm run_system_with_llm_news_20clients.py contains the _clean_response function.

  Old HTML report still shows "2"
    Re-run: python generate_news_report.py   to overwrite the stale file.

  HF Hub unauthenticated warning
    Set: export HF_TOKEN=your_token   OR ignore it — dataset still downloads fine.
Step 3 — Run the pipeline

python run_system_with_llm_news_20clients.py \
  --llm-model llama3.2:3b \
  --num-samples 20 \
  --num-clients 20 \
  --attacks poisoning node_availability extraction membership_inference \
  --output-dir logs/news_llama_20clients_all_attacks

Step 4 — Generate and open the report

python generate_news_report.py && explorer.exe FedRAG_News_Report.html