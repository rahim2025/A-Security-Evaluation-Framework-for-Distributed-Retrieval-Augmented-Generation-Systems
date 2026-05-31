# Federated Training Evaluation Matrix

- **Attack**: `gradient_flip`
- **Malicious clients**: [1]
- **Rounds**: 2

| Round | EM | F1 | BLEU | Semantic |
|------|-----|-----|------|----------|
|     1 | 0.950 | 0.950 | 0.950 | 0.996 |
|     2 | 0.050 | 0.050 | 0.050 | 0.520 |

## Degradation

- **exact_match**: 0.9500 → 0.0500 (-0.9000)
- **f1**: 0.9500 → 0.0500 (-0.9000)
- **bleu**: 0.9500 → 0.0500 (-0.9000)
- **semantic_similarity**: 0.9962 → 0.5202 (-0.4760)
