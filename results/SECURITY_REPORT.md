# FedRAG Security Evaluation Report

**Generated:** 2026-05-27T14:17:27.288739+00:00

## Simulator Configuration

| Parameter | Value |
| --- | --- |
| Clients | 10 |
| Examples | 200 |
| Seed | 0 |
| Embedding dim | 32 |

## Attack Results

### Data Poisoning

| Metric | Value |
| --- | --- |
| Baseline EM | 1.0000 |
| Post-Attack EM | 0.8400 |
| Defended EM | 0.8000 |
| Delta EM | -0.1600 |
| Baseline F1 | 1.0000 |
| Post-Attack F1 | 0.8400 |
| Defended F1 | 0.8000 |
| Delta F1 | -0.1600 |
| Baseline BLEU | 1.0000 |
| Post-Attack BLEU | 0.8400 |
| Defended BLEU | 0.8000 |
| Delta BLEU | -0.1600 |
| Membership Acc | None |
| KB Recovery % | None |
| Runtime (s) | 0.2489 |

### Membership Inference

| Metric | Value |
| --- | --- |
| Baseline EM | None |
| Post-Attack EM | None |
| Defended EM | None |
| Delta EM | None |
| Baseline F1 | None |
| Post-Attack F1 | None |
| Defended F1 | None |
| Delta F1 | None |
| Baseline BLEU | None |
| Post-Attack BLEU | None |
| Defended BLEU | None |
| Delta BLEU | None |
| Membership Acc | 1.0000 |
| Defended Membership Acc | 0.5000 |
| KB Recovery % | None |
| Runtime (s) | 0.0025 |

### Knowledge Extraction

| Metric | Value |
| --- | --- |
| Baseline EM | None |
| Post-Attack EM | None |
| Defended EM | None |
| Delta EM | None |
| Baseline F1 | None |
| Post-Attack F1 | None |
| Defended F1 | None |
| Delta F1 | None |
| Baseline BLEU | None |
| Post-Attack BLEU | None |
| Defended BLEU | None |
| Delta BLEU | None |
| Membership Acc | None |
| KB Recovery % | 100.0000 |
| Defended KB Recovery % | 45.0000 |
| Runtime (s) | 0.0045 |

### Node Availability (node_removal)

| Metric | Value |
| --- | --- |
| Baseline EM | 1.0000 |
| Post-Attack EM | 0.7000 |
| Defended EM | 0.7500 |
| Delta EM | -0.3000 |
| Baseline F1 | 1.0000 |
| Post-Attack F1 | 0.7000 |
| Defended F1 | 0.7500 |
| Delta F1 | -0.3000 |
| Baseline BLEU | 1.0000 |
| Post-Attack BLEU | 0.7000 |
| Defended BLEU | 0.7300 |
| Delta BLEU | None |
| Availability Before % | 100.0000 |
| Availability After % | 70.0000 |
| Membership Acc | None |
| KB Recovery % | None |
| Runtime (s) | 0.0000 |

## Defense Summary

| Defense | Target Attack | Status |
| --- | --- | --- |
| ClientDataPoisoningDefense | Data Poisoning | Active |
| ScoreMasking | Membership Inference | Active |
| CrossPeerValidation | Node Availability | Active |
| QueryRateLimiter | Knowledge Extraction | Active |
| ExtractionAnomalyDetector | Knowledge Extraction | Active |
