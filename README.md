<!-- markdownlint-disable-file MD033 MD041 -->

<img src="https://d3ddy8balm3goa.cloudfront.net/vector-fed-rag/logo-dark.svg" width="175" align="right" alt="logo"/>

# FedRAG

---------------------------------------------------------------------------------------

[![Linting](https://github.com/VectorInstitute/fed-rag/actions/workflows/lint.yml/badge.svg)](https://github.com/VectorInstitute/fed-rag/actions/workflows/lint.yml)
[![Unit Testing and Upload Coverage](https://github.com/VectorInstitute/fed-rag/actions/workflows/unit_test.yml/badge.svg)](https://github.com/VectorInstitute/fed-rag/actions/workflows/unit_test.yml)
[![codecov](https://codecov.io/github/VectorInstitute/fed-rag/graph/badge.svg?token=JjJBPckP8v)](https://codecov.io/github/VectorInstitute/fed-rag)
[![GitHub License](https://img.shields.io/github/license/VectorInstitute/fed-rag)](https://github.com/VectorInstitute/fed-rag/blob/main/LICENSE)
![GitHub Release](https://img.shields.io/github/v/release/VectorInstitute/fed-rag)
[![DOI](https://zenodo.org/badge/918377874.svg)](https://doi.org/10.5281/zenodo.15092361)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/VectorInstitute/fed-rag)

FedRAG is an open-source framework for fine-tuning Retrieval-Augmented
Generation (RAG) systems across both centralized and federated architectures.

## Simplified RAG fine-tuning across centralized or federated architectures

### Advanced RAG fine-tuning

Comprehensive support for state-of-the-art RAG fine-tuning methods that can be
federated with ease.

### Work with your tools

Seamlessly integrates with popular frameworks including HuggingFace,
LlamaIndex, and LangChain — use the tools you already know.

### Lightweight abstractions

Clean, intuitive abstractions that simplify RAG fine-tuning while
maintaining full flexibility and control.

## Installation

### From package managers

```sh
# pypi
pip install fed-rag

# conda-forge
conda install -c conda-forge fed-rag
```

> [!NOTE]
> Extras for `fed-rag` are also available, such as the HuggingFace extra, which
> can be installed via `pip install fed-rag[huggingface]`

### From source

```sh
git clone https://github.com/VectorInstitute/fed-rag.git
cd fed-rag

# install using pip
pip install -e .

# or, install using uv, our package manager tool of choice
uv sync --all-extras --group dev --group docs
```

## Documentation

For more detailed documentation, visit our [official documentation site](https://vectorinstitute.github.io/fed-rag/).

- [Getting Started](https://vectorinstitute.github.io/fed-rag/getting_started/essentials/)
- [Quick Starts](https://vectorinstitute.github.io/fed-rag/getting_started/quick_starts/)
- [Examples](https://vectorinstitute.github.io/fed-rag/examples/)
- [API Reference](https://vectorinstitute.github.io/fed-rag/api_reference/)
- [Glossary](https://vectorinstitute.github.io/fed-rag/glossary/)

> [!TIP]
> This README provides a high-level overview, but our official documentation is
> updated more frequently with the latest features, tutorials, and API changes.
> For the most current information, please refer to the documentation site.

## Examples

Check out our [examples directory](examples/) for more detailed usage examples:

- Basic RAG fine-tuning with federated learning
- Implementing RA-DIT with FedRAG
- Custom federated aggregation strategies
- Integration with popular LLM frameworks

## Security Evaluation (Knowledge-Store Level)

FedRAG includes experimental attack utilities for evaluating **knowledge-store
security** against data corruption and availability attacks.  These evaluators
measure how much retrieval quality degrades when the knowledge base is
compromised — **they do not model federated training** (no local client
 training, no FedAvg aggregation, no Flower rounds).

- `DataPoisoningAttack` corrupts stored question–answer pairs.
- `MembershipInferenceAttack` infers whether query-like text is present in the
  knowledge store using retrieval confidence.
- `KnowledgeExtractionAttack` probes the retrieval interface to recover stored
  nodes.
- `NodeAvailabilityAttack` models node-removal, Byzantine, partition, DDoS,
  and Sybil attacks.

> **Scope note:** The evaluators all query a **single pooled knowledge store**
> built from the (possibly corrupted) dataset.  The "federated" label only
> means data is IID-split across clients for attack attribution (e.g. "poison
> client 3's data", "drop client 5"); no actual federated learning occurs.

### Evaluation modes

Three modes are supported.  All produce the **same evaluation matrix**
(columns: EM, precision, recall, F1, BLEU, ROUGE, semantic similarity,
edit distance, n-gram overlap, query-failure rate, availability) so that
DRAG and FedRAG results are directly comparable on architecture-agnostic
metrics.

#### 1. Centralized (single knowledge store)

```sh
python run_security_evaluation.py --mode centralized \
  --dataset mmlu --llm llama32_3b --seed 0
```

Uses DRAG's MMLU dataset config and `sentence-transformers/all-MiniLM-L6-v2`
retriever. Writes `logs/security_evaluation/EVALUATION_MATRIX.md`.

#### 2. Federated (multi-client simulation)

```sh
python run_security_evaluation.py --mode federated \
  --num-clients 10 --num-examples 240 --seed 7
```

Simulates IID data splits across clients, runs poisoning,
membership-inference, extraction and node-availability attacks, and applies
client-side defences (poisoning quarantine, score masking, rate limiting,
anomaly detection).  **The underlying query target is still a single pooled
knowledge store.**

#### 3. System-level (baseline → attack → post-attack)

Mirrors DRAG's `simulator.py` flow exactly:

```sh
python run_security_evaluation.py --mode system \
  --num-clients 10 --attacks poisoning node_availability extraction \
  --dataset mmlu --seed 0
```

This mode evaluates the **whole system** by:

1. Building a baseline knowledge store from clean data.
2. Running all queries and recording answer-quality metrics.
3. Applying the selected attack(s) (corrupts the store or drops clients).
4. Re-running the same queries against the attacked store.
5. Computing degradation: `baseline − post-attack`.
6. Writing a unified matrix with architecture-agnostic fields:
   `exact_match`, `precision`, `recall`, `f1`, `bleu`, `rouge1/2/L`,
   `semantic_similarity`, `edit_distance`, `bigram/trigram_overlap`,
   `query_failure_rate`, `availability_percentage`.

All three modes write Markdown matrix, CSV matrix, and raw JSON results.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## Citation

If you use FedRAG in your research, please cite our library:

```bibtex
@software{Fajardo_fed-rag_2025,
author = {Fajardo, Andrei and Emerson, David},
doi = {10.5281/zenodo.15092361},
license = {Apache-2.0},
month = mar,
title = {{fed-rag}},
url = {https://github.com/VectorInstitute/fed-rag},
version = {0.0.27},
year = {2025}
}
```

> [!NOTE]
> The above citation may not reflect the most recent version of the library. We
> recommend using the Github citation widget (i.e. "Cite this respository") to
> obtain a citation entry reflecting the latest released version.

## License

FedRAG is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

FedRAG is developed and maintained by the [Vector Institute](https://vectorinstitute.ai/).
