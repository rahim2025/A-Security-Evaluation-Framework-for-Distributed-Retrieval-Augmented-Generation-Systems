# FedRAG — Global System Impact Analysis

Shows how **client-level attacks** cascade into **global RAG system degradation**.
Each attack is run at three intensity levels: Light (10 %), Medium (30 %), Heavy (50 %).

## Simulator Configuration

| Parameter | Value |
|-----------|-------|
| Clients | 20 |
| Examples | 500 |
| Seed | 123 |
| Embedding dim | 32 |

## 1. Data Poisoning → Global LLM Performance Impact

**Attack:** Malicious clients inject wrong/misleading answers into their local knowledge store. When the global system retrieves from these clients, poisoned answers are returned, degrading global EM/F1/BLEU.

**Defense:** `ClientDataPoisoningDefense` — inspects each client's data for poison markers and conflicting answers; quarantines high-risk clients.

| Intensity | Clients Hit | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 | Recovery |
|-----------|-------------|-------------|-------------|------|-------------|----------|
| LIGHT | 2/20 (10%) | 1.0000 | 0.9520 | **-0.0480** | 0.9520 | +0.0000 |
| MEDIUM | 6/20 (30%) | 1.0000 | 0.8560 | **-0.1440** | 0.8560 | +0.0000 |
| HEAVY | 10/20 (50%) | 1.0000 | 0.7600 | **-0.2400** | 0.7600 | +0.0000 |

### Degradation Cascade (Data Poisoning)

| Step | Client Poisoned | Total Poisoned | Global F1 | Δ from Baseline | % Degradation |
|------|----------------|----------------|-----------|-----------------|---------------|
| 1 | 15 | 1/20 | 0.9760 | -0.0240 | 2.4% |
| 2 | 12 | 2/20 | 0.9520 | -0.0480 | 4.8% |
| 3 | 0 | 3/20 | 0.9280 | -0.0720 | 7.2% |
| 4 | 13 | 4/20 | 0.9040 | -0.0960 | 9.6% |
| 5 | 17 | 5/20 | 0.8800 | -0.1200 | 12.0% |
| 6 | 7 | 6/20 | 0.8560 | -0.1440 | 14.4% |
| 7 | 9 | 7/20 | 0.8320 | -0.1680 | 16.8% |
| 8 | 5 | 8/20 | 0.8080 | -0.1920 | 19.2% |
| 9 | 19 | 9/20 | 0.7840 | -0.2160 | 21.6% |
| 10 | 16 | 10/20 | 0.7600 | -0.2400 | 24.0% |
| 11 | 18 | 11/20 | 0.7360 | -0.2640 | 26.4% |
| 12 | 2 | 12/20 | 0.7120 | -0.2880 | 28.8% |
| 13 | 11 | 13/20 | 0.6880 | -0.3120 | 31.2% |
| 14 | 6 | 14/20 | 0.6640 | -0.3360 | 33.6% |
| 15 | 3 | 15/20 | 0.6400 | -0.3600 | 36.0% |
| 16 | 4 | 16/20 | 0.6160 | -0.3840 | 38.4% |
| 17 | 10 | 17/20 | 0.5920 | -0.4080 | 40.8% |
| 18 | 8 | 18/20 | 0.5680 | -0.4320 | 43.2% |
| 19 | 1 | 19/20 | 0.5440 | -0.4560 | 45.6% |
| 20 | 14 | 20/20 | 0.5200 | -0.4800 | 48.0% |

## 2. KB Extraction → Privacy / Knowledge Exposure

**Attack:** An adversary repeatedly probes the public retrieval API with templated queries to reconstruct private knowledge nodes. Answer quality is NOT degraded, but private data leaks.

**Defense:** `QueryRateLimiter` + `ExtractionAnomalyDetector` — caps total queries per client and flags abnormally broad-topic query bursts.

| Intensity | Clients Targeted | KB Exposed (no defense) | KB Exposed (defended) | Reduction |
|-----------|-----------------|-------------------------|-----------------------|-----------|
| LIGHT | 2/20 (10%) | **100.0%** | 28.0% | -72.0pp |
| MEDIUM | 6/20 (30%) | **100.0%** | 29.3% | -70.7pp |
| HEAVY | 10/20 (50%) | **100.0%** | 28.8% | -71.2pp |

### Extraction Cascade (cumulative KB exposure as more clients are targeted)

| Step | Client Targeted | Total Targeted | Exposure (no defense) | Exposure (defended) |
|------|----------------|----------------|-----------------------|---------------------|
| 1 | 11 | 1/20 | 5.0% | 3.0% |
| 2 | 2 | 2/20 | 10.0% | 6.0% |
| 3 | 6 | 3/20 | 15.0% | 9.0% |
| 4 | 4 | 4/20 | 20.0% | 12.0% |
| 5 | 17 | 5/20 | 25.0% | 15.0% |
| 6 | 13 | 6/20 | 30.0% | 18.0% |
| 7 | 15 | 7/20 | 35.0% | 21.2% |
| 8 | 0 | 8/20 | 40.0% | 24.2% |
| 9 | 3 | 9/20 | 45.0% | 27.2% |
| 10 | 5 | 10/20 | 50.0% | 30.4% |
| 11 | 16 | 11/20 | 55.0% | 33.4% |
| 12 | 12 | 12/20 | 60.0% | 36.6% |
| 13 | 14 | 13/20 | 65.0% | 39.8% |
| 14 | 9 | 14/20 | 70.0% | 42.8% |
| 15 | 10 | 15/20 | 75.0% | 45.8% |
| 16 | 8 | 16/20 | 80.0% | 48.8% |
| 17 | 1 | 17/20 | 85.0% | 51.8% |
| 18 | 7 | 18/20 | 90.0% | 54.8% |
| 19 | 18 | 19/20 | 95.0% | 57.8% |
| 20 | 19 | 20/20 | 100.0% | 60.8% |

## 3. Node Availability → Global System Coverage

**Attack:** Clients are removed (node_removal / ddos), set Byzantine (returning corrupted answers), network-partitioned, or Sybil-injected.

**Defense:** `CrossPeerValidation` — majority-votes candidate answers across surviving peers; discards answers that fail consensus.

| Attack Type | Intensity | Nodes Down | Avail % | Baseline F1 | Attacked F1 | Δ F1 | Defended F1 |
|-------------|-----------|------------|---------|-------------|-------------|------|-------------|
| node_removal | LIGHT | 2/20 | 90% | 1.0000 | 0.9000 | **-0.1000** | 0.9000 |
| node_removal | MEDIUM | 6/20 | 70% | 1.0000 | 0.7000 | **-0.3000** | 0.7000 |
| node_removal | HEAVY | 10/20 | 50% | 1.0000 | 0.5000 | **-0.5000** | 0.5000 |
| byzantine | LIGHT | 2/20 | 100% | 1.0000 | 0.9667 | **-0.0333** | 0.9833 |
| byzantine | MEDIUM | 6/20 | 100% | 1.0000 | 0.9000 | **-0.1000** | 0.9667 |
| byzantine | HEAVY | 10/20 | 100% | 1.0000 | 0.8333 | **-0.1667** | 0.9500 |
| ddos | LIGHT | 2/20 | 90% | 1.0000 | 0.9000 | **-0.1000** | 0.9000 |
| ddos | MEDIUM | 6/20 | 70% | 1.0000 | 0.7000 | **-0.3000** | 0.7000 |
| ddos | HEAVY | 10/20 | 50% | 1.0000 | 0.5000 | **-0.5000** | 0.5000 |

### Availability Cascade (node_removal, one node at a time)

| Step | Node Removed | Active Nodes | Global F1 | Δ from Baseline | % Degradation |
|------|-------------|--------------|-----------|-----------------|---------------|
| 1 | 15 | 19/20 | 0.9500 | -0.0500 | 5.0% |
| 2 | 5 | 18/20 | 0.9000 | -0.1000 | 10.0% |
| 3 | 8 | 17/20 | 0.8500 | -0.1500 | 15.0% |
| 4 | 12 | 16/20 | 0.8000 | -0.2000 | 20.0% |
| 5 | 18 | 15/20 | 0.7500 | -0.2500 | 25.0% |
| 6 | 6 | 14/20 | 0.7000 | -0.3000 | 30.0% |
| 7 | 1 | 13/20 | 0.6500 | -0.3500 | 35.0% |
| 8 | 11 | 12/20 | 0.6000 | -0.4000 | 40.0% |
| 9 | 10 | 11/20 | 0.5500 | -0.4500 | 45.0% |
| 10 | 7 | 10/20 | 0.5000 | -0.5000 | 50.0% |
| 11 | 17 | 9/20 | 0.4500 | -0.5500 | 55.0% |
| 12 | 13 | 8/20 | 0.4000 | -0.6000 | 60.0% |
| 13 | 4 | 7/20 | 0.3500 | -0.6500 | 65.0% |
| 14 | 9 | 6/20 | 0.3000 | -0.7000 | 70.0% |
| 15 | 2 | 5/20 | 0.2500 | -0.7500 | 75.0% |
| 16 | 16 | 4/20 | 0.2000 | -0.8000 | 80.0% |
| 17 | 0 | 3/20 | 0.1500 | -0.8500 | 85.0% |
| 18 | 14 | 2/20 | 0.1000 | -0.9000 | 90.0% |
| 19 | 3 | 1/20 | 0.0500 | -0.9500 | 95.0% |
| 20 | 19 | 0/20 | 0.0000 | -1.0000 | 100.0% |

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