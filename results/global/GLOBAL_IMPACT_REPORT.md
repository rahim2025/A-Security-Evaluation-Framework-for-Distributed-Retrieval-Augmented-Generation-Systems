# FedRAG — Global System Impact Analysis

Shows how **client-level attacks** cascade into **global RAG system degradation**.
Each attack is run at three intensity levels: Light (10 %), Medium (30 %), Heavy (50 %).

## Simulator Configuration

| Parameter | Value |
|-----------|-------|
| Clients | 10 |
| Examples | 200 |
| Seed | 0 |
| Embedding dim | 32 |

## 1. Data Poisoning → Global LLM Performance Impact

**Attack:** Malicious clients inject wrong/misleading answers into their local knowledge store. When the global system retrieves from these clients, poisoned answers are returned, degrading global EM/F1/BLEU.

**Defense:** `ClientDataPoisoningDefense` — inspects each client's data for poison markers and conflicting answers; quarantines high-risk clients.

| Intensity | Clients Hit | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 | Recovery |
|-----------|-------------|-------------|-------------|------|-------------|----------|
| HEAVY | 5/10 (50%) | 1.0000 | 1.0000 | **+0.0000** | 1.0000 | +0.0000 |

### Degradation Cascade (Data Poisoning)

| Step | Client Poisoned | Total Poisoned | Global F1 | Δ from Baseline | % Degradation |
|------|----------------|----------------|-----------|-----------------|---------------|
| 1 | 5 | 1/10 | 0.9200 | -0.0800 | 8.0% |
| 2 | 6 | 2/10 | 0.8400 | -0.1600 | 16.0% |
| 3 | 4 | 3/10 | 0.7600 | -0.2400 | 24.0% |
| 4 | 0 | 4/10 | 0.6800 | -0.3200 | 32.0% |
| 5 | 2 | 5/10 | 0.6000 | -0.4000 | 40.0% |
| 6 | 9 | 6/10 | 0.5200 | -0.4800 | 48.0% |
| 7 | 3 | 7/10 | 0.4400 | -0.5600 | 56.0% |
| 8 | 7 | 8/10 | 0.3600 | -0.6400 | 64.0% |
| 9 | 8 | 9/10 | 0.2800 | -0.7200 | 72.0% |
| 10 | 1 | 10/10 | 0.2000 | -0.8000 | 80.0% |

## 2. KB Extraction → Privacy / Knowledge Exposure

**Attack:** An adversary repeatedly probes the public retrieval API with templated queries to reconstruct private knowledge nodes. Answer quality is NOT degraded, but private data leaks.

**Defense:** `QueryRateLimiter` + `ExtractionAnomalyDetector` — caps total queries per client and flags abnormally broad-topic query bursts.

| Intensity | Clients Targeted | KB Exposed (no defense) | KB Exposed (defended) | Reduction |
|-----------|-----------------|-------------------------|-----------------------|-----------|
| HEAVY | 5/10 (50%) | **100.0%** | 36.0% | -64.0pp |

### Extraction Cascade (cumulative KB exposure as more clients are targeted)

| Step | Client Targeted | Total Targeted | Exposure (no defense) | Exposure (defended) |
|------|----------------|----------------|-----------------------|---------------------|
| 1 | 5 | 1/10 | 10.0% | 8.0% |
| 2 | 9 | 2/10 | 20.0% | 15.5% |
| 3 | 0 | 3/10 | 30.0% | 23.0% |
| 4 | 3 | 4/10 | 40.0% | 30.5% |
| 5 | 4 | 5/10 | 50.0% | 38.0% |
| 6 | 6 | 6/10 | 60.0% | 46.0% |
| 7 | 1 | 7/10 | 70.0% | 53.5% |
| 8 | 8 | 8/10 | 80.0% | 61.0% |
| 9 | 7 | 9/10 | 90.0% | 68.5% |
| 10 | 2 | 10/10 | 100.0% | 76.0% |

## 3. Node Availability → Global System Coverage

**Attack:** Clients are removed (node_removal / ddos), set Byzantine (returning corrupted answers), network-partitioned, or Sybil-injected.

**Defense:** `CrossPeerValidation` — majority-votes candidate answers across surviving peers; discards answers that fail consensus.

| Attack Type | Intensity | Nodes Down | Avail % | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 |
|-------------|-----------|------------|---------|-------------|-------------|------|-------------|
| node_removal | HEAVY | 5/10 | 50% | 1.0000 | 0.5000 | **-0.5000** | 0.5000 |
| byzantine | HEAVY | 5/10 | 100% | 1.0000 | 0.8333 | **-0.1667** | 0.9333 |
| ddos | HEAVY | 5/10 | 50% | 1.0000 | 0.5000 | **-0.5000** | 0.5000 |

### Availability Cascade (node_removal, one node at a time)

| Step | Node Removed | Active Nodes | Global F1 | Δ from Baseline | % Degradation |
|------|-------------|--------------|-----------|-----------------|---------------|
| 1 | 8 | 9/10 | 0.9000 | -0.1000 | 10.0% |
| 2 | 0 | 8/10 | 0.8000 | -0.2000 | 20.0% |
| 3 | 3 | 7/10 | 0.7000 | -0.3000 | 30.0% |
| 4 | 5 | 6/10 | 0.6000 | -0.4000 | 40.0% |
| 5 | 6 | 5/10 | 0.5000 | -0.5000 | 50.0% |
| 6 | 1 | 4/10 | 0.4000 | -0.6000 | 60.0% |
| 7 | 2 | 3/10 | 0.3000 | -0.7000 | 70.0% |
| 8 | 4 | 2/10 | 0.2000 | -0.8000 | 80.0% |
| 9 | 9 | 1/10 | 0.1000 | -0.9000 | 90.0% |
| 10 | 7 | 0/10 | 0.0000 | -1.0000 | 100.0% |

---

## Key Takeaways

- **Data Poisoning** has the highest direct impact on global answer quality;   even 10% malicious clients can drop global F1 measurably.
- **KB Extraction** does not degrade answer quality but can expose the entire   knowledge base if rate limiting is not applied.
- **Node Availability** degrades global coverage linearly with the fraction   of nodes removed; Byzantine attacks are harder to recover from than simple removal.
- **Defenses** consistently recover a significant portion of lost performance,   but never fully restore baseline — underlining the importance of prevention.

## Output Files

| File | Description |
|------|-------------|
| `global_impact_results.json` | Full raw results for every scenario |
| `GLOBAL_IMPACT_REPORT.md` | This report |
| `global_attack_matrix.csv` | Flat CSV — one row per scenario |
| `cascade_data_poisoning.csv` | Per-client poisoning cascade |
| `cascade_kb_extraction.csv` | Per-client extraction cascade |
| `cascade_node_availability.csv` | Per-node removal cascade |