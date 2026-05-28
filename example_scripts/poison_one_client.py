#!/usr/bin/env python3
"""
Simulate a federated learning scenario with IID data distribution across multiple clients,
apply a data‑poisoning attack to a single client (default client 0), and report the
impact on the MMLU benchmark using a simple Exact‑Match metric.

The script does **not** run a full Flower FL server – it only demonstrates how the
poisoned data would affect the aggregated knowledge store that a FedRAG system would
use.  This mirrors the existing ``run_security_evaluation.py`` flow but adds a
client‑level split.
"""

import argparse
import random
from typing import List, Dict

from datasets import load_dataset
from fed_rag.attacks import DataPoisoningAttack
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.huggingface import HFSentenceTransformerRetriever
from fed_rag.data_structures import KnowledgeNode, NodeType

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_mmlu(split: str = "test") -> List[Dict[str, str]]:
    """Load the MMLU dataset and flatten it into the ``{query, response, topic}``
    schema expected by :class:`DataPoisoningAttack`.

    The HuggingFace ``cais/mmlu`` dataset stores the correct answer as an integer
    index into ``choices``.  For simplicity we convert the index to the corresponding
    letter (A‑D) and treat that letter as the *response*.
    """
    # ``load_dataset`` returns a ``Dataset`` object; we request the ``test`` split
    # (the original repo uses only the test split for evaluation).
    raw = load_dataset("cais/mmlu", "all", split=split)

    examples: List[Dict[str, str]] = []
    for row in raw:
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        # ``answer`` can be an int or a list containing a single int (as in the
        # test fixture). Normalise it to an int.
        if isinstance(answer_idx, list):
            answer_idx = answer_idx[0]
        # Map 0‑3 -> A‑D; fallback to the raw integer string.
        answer_letter = "ABCD"[answer_idx] if 0 <= answer_idx < 4 else str(answer_idx)
        examples.append({"query": question, "response": answer_letter, "topic": "MMLU"})
    return examples


def iid_split(data: List[Dict[str, str]], num_clients: int, seed: int) -> List[List[Dict[str, str]]]:
    """Shuffle ``data`` with ``seed`` and split it into ``num_clients`` chunks.
    The split is as even as possible – any remainder is distributed to the first
    ``remainder`` clients.
    """
    rng = random.Random(seed)
    rng.shuffle(data)
    total = len(data)
    base, rem = divmod(total, num_clients)
    splits: List[List[Dict[str, str]]] = []
    start = 0
    for i in range(num_clients):
        size = base + (1 if i < rem else 0)
        splits.append(data[start : start + size])
        start += size
    return splits


def build_store(examples: List[Dict[str, str]], retriever: HFSentenceTransformerRetriever) -> InMemoryKnowledgeStore:
    """Create an ``InMemoryKnowledgeStore`` from ``examples``.
    Each example becomes a ``KnowledgeNode`` whose embedding is obtained from the
    ``retriever`` (sentence‑transformer model).
    """
    nodes: List[KnowledgeNode] = []
    for ex in examples:
        emb = retriever.encode_context(ex["query"]).squeeze(0).tolist()
        nodes.append(
            KnowledgeNode(
                node_type=NodeType.TEXT,
                text_content=ex["query"],
                embedding=[float(v) for v in emb],
                metadata={"topic": ex["topic"], "answer": ex["response"]},
            )
        )
    return InMemoryKnowledgeStore.from_nodes(nodes)


def evaluate_exact_match(dataset: List[Dict[str, str]], store: InMemoryKnowledgeStore, retriever: HFSentenceTransformerRetriever) -> float:
    """Return the Exact‑Match (EM) score on ``dataset`` using ``store``.
    For each query we retrieve the top‑1 node and compare its stored ``answer``
    metadata with the ground‑truth ``response``.
    """
    if not dataset:
        return 0.0
    correct = 0
    for ex in dataset:
        query_emb = retriever.encode_query(ex["query"]).squeeze(0).tolist()
        hits = store.retrieve(query_emb=query_emb, top_k=1)
        if hits:
            _, node = hits[0]
            pred = node.metadata.get("answer", "")
            if pred.strip().lower() == ex["response"].strip().lower():
                correct += 1
    return correct / len(dataset)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated data‑poisoning demo (MMLU)")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of IID clients")
    parser.add_argument(
        "--poison-client",
        type=int,
        default=0,
        help="Zero‑based index of the client to poison (default: 0)",
    )
    parser.add_argument(
        "--poisoning-ratio",
        type=float,
        default=0.1,
        help="Fraction of the chosen client’s records to poison",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Load and prepare the MMLU examples
    # -------------------------------------------------------------------
    all_examples = load_mmlu()
    print(f"Loaded {len(all_examples)} MMLU examples.")

    # Split IID across clients
    client_data = iid_split(all_examples.copy(), args.num_clients, args.seed)
    for i, chunk in enumerate(client_data):
        print(f"Client {i}: {len(chunk)} examples")

    # -------------------------------------------------------------------
    # Apply poisoning to the selected client
    # -------------------------------------------------------------------
    if not (0 <= args.poison_client < args.num_clients):
        raise ValueError("poison_client index out of range")
    attacker = DataPoisoningAttack(
        poisoning_ratio=args.poisoning_ratio,
        poison_type="wrong_answer",
        mode="replace",
        seed=args.seed,
    )
    poisoned_result = attacker.execute(client_data[args.poison_client])
    client_data[args.poison_client] = poisoned_result.poisoned_examples
    print(
        f"Client {args.poison_client} poisoned: {len(poisoned_result.poisoned_examples)} examples ("
        f"{len(poisoned_result.poisoned_indices)} indices)"
    )

    # -------------------------------------------------------------------
    # Build knowledge stores – clean (no poison) vs. aggregated poisoned data
    # -------------------------------------------------------------------
    retriever = HFSentenceTransformerRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    clean_store = build_store(all_examples, retriever)
    # Re‑assemble the federated data after poisoning
    aggregated_examples = [ex for client in client_data for ex in client]
    poisoned_store = build_store(aggregated_examples, retriever)

    # -------------------------------------------------------------------
    # Evaluate Exact‑Match on the original test set
    # -------------------------------------------------------------------
    clean_em = evaluate_exact_match(all_examples, clean_store, retriever)
    poisoned_em = evaluate_exact_match(all_examples, poisoned_store, retriever)
    print("\nExact‑Match (EM) results:")
    print(f"  Clean (no poison)   : {clean_em:.4%}")
    print(f"  With poisoned client: {poisoned_em:.4%}")
    print(f"  Δ (poison – clean)  : {poisoned_em - clean_em:.4%}")

if __name__ == "__main__":
    main()
