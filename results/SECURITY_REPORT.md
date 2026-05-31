# FedRAG Security Evaluation Report
**Generated:** 2026-05-30 23:36:52

## Architecture
True per-client local InMemoryKnowledgeStore — DRAG-style attack flow.
Each attack targets individual client local stores; server aggregates across all clients.

## Simulator
| Parameter | Value |
| --- | --- |
| Clients | 150 |
| Examples | 1500 |
| Seed | 0 |

## Results

### Data Poisoning
- Malicious clients: 49 / 150
- Poison type: wrong_answer
- Baseline F1: 1.0000
- Post-attack F1: 0.8600  Δ=-0.1400
- Defended F1: 0.6993

### Membership Inference
- Attack accuracy: 1.000  TPR=1.000  FPR=0.000
- Defended accuracy: 0.500

### Knowledge Extraction
- Recovered: 6 / 6 nodes (100.0%)
- Defended: 6 / 6 nodes (100.0%)

### Node Availability
- Attack type: node_removal
- Affected: 45 / 150 clients
- Baseline F1: 1.0000
- Post-attack F1: 0.6407  Δ=-0.3593
- Defended F1: 0.8707
