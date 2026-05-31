#!/usr/bin/env python3
"""
poison_one_client.py  (original — requires HuggingFace + internet)
Uses HFSentenceTransformerRetriever + real MMLU dataset.

Install first:
    pip install "fed-rag[huggingface]" datasets sentence-transformers
"""

import random
from typing import Any

from datasets import load_dataset
from fed_rag.retrievers.huggingface import HFSentenceTransformerRetriever
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.attacks.data_poisoning import DataPoisoningAttack
from fed_rag.data_structures.knowledge_node import KnowledgeNode

NUM_CLIENTS      = 10
MALICIOUS_CLIENT = 0
POISONING_RATIO  = 0.3
POISON_TYPE      = "wrong_answer"
SEED             = 42
NUM_SAMPLES      = 200

print("Loading MMLU dataset from HuggingFace...")
raw = load_dataset("cais/mmlu", "all", split="test")
raw = raw.shuffle(seed=SEED).select(range(NUM_SAMPLES))

examples = [
    {
        "query":    row["question"],
        "response": row["choices"][row["answer"]],
        "topic":    row["subject"],
    }
    for row in raw
]

print("Loading sentence-transformer model (~90MB first run)...")
retriever = HFSentenceTransformerRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

rng = random.Random(SEED)
all_examples = examples[:]
rng.shuffle(all_examples)

chunk = len(all_examples) // NUM_CLIENTS
client_datasets = []
for i in range(NUM_CLIENTS):
    start = i * chunk
    end   = start + chunk if i < NUM_CLIENTS - 1 else len(all_examples)
    client_datasets.append(all_examples[start:end])

print("Dataset split: {} clients, ~{} examples each".format(NUM_CLIENTS, chunk))

attack = DataPoisoningAttack(
    poisoning_ratio=POISONING_RATIO,
    poison_type=POISON_TYPE,
    seed=SEED,
)

target_data   = client_datasets[MALICIOUS_CLIENT]
attack_result = attack.execute(target_data)
poisoned_data = attack_result.poisoned_examples

print("Client {} - poisoned {}/{} examples".format(
    MALICIOUS_CLIENT, len(poisoned_data), len(target_data)))

all_nodes = []
stores    = []

for i, data in enumerate(client_datasets):
    dataset = poisoned_data if i == MALICIOUS_CLIENT else data
    store   = InMemoryKnowledgeStore()
    nodes   = []
    for ex in dataset:
        emb  = retriever.encode_context(ex["query"])
        node = KnowledgeNode(
            text_content=ex["response"],
            embedding=emb.squeeze().tolist(),
            node_type="text",
        )
        nodes.append(node)
        all_nodes.append(node)
    store.load_nodes(nodes)
    stores.append(store)
    status = "POISONED" if i == MALICIOUS_CLIENT else "clean"
    print("  Client {:2d} ({}): {} docs indexed".format(i, status, len(nodes)))

global_store = InMemoryKnowledgeStore.from_nodes(all_nodes)
print("Global store: {} total documents".format(global_store.count))

sample_query = examples[0]["query"]
query_emb    = retriever.encode_query(sample_query)
results      = global_store.retrieve(query_emb, top_k=3)

print("Sample retrieval for: " + repr(sample_query[:60]))
for rank, (score, node) in enumerate(results, 1):
    print("  #{} (score={:.3f}): {}".format(rank, score, (node.text_content or "")[:80]))

print("Done.")
