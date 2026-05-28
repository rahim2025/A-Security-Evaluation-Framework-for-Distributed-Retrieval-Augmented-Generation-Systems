"""
DRAG Scalability Testing Suite
================================
Tests the DRAG distributed RAG system across 4 scaling dimensions:
  1. Dataset Volume     — vary data.num_samples
  2. Network Size       — vary rag.num_peers
  3. Attack Intensity   — vary security.poisoning_ratio
  4. Dataset Diversity  — swap full datasets (MMLU / medical / news)
"""
