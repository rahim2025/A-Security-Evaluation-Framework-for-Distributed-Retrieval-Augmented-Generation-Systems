import json
import logging
from fed_rag.attacks.data_poisoning import DataPoisoningAttack
from datasets import Dataset

print("\nExecuting REAL federated dataset attacking across multiple ratios (DRAG-style)...")

data = {"query": [], "response": [], "topic": []}
topics = ["math", "history", "science", "geography", "literature"]
for i in range(500):
    t = topics[i % len(topics)]
    data["query"].append(f"What is fact {i} about {t}?")
    data["response"].append(f"Fact {i}: {t} is interesting.")
    data["topic"].append(t)
    
clean_dataset = Dataset.from_dict(data)

ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
results = []

print("-" * 75)
print(f"{'Attack Ratio':<15} | {'Clean Items':<15} | {'Poisoned Items':<15} | {'Target Perf Drop':<15}")
print("-" * 75)

for r in ratios:
    attack = DataPoisoningAttack(poisoning_ratio=r, poison_type="wrong_answer", mode="replace", seed=42)
    p_ds, meta = attack.execute_hf_dataset(clean_dataset)
    
    clean_responses = clean_dataset["response"]
    poisoned_responses = p_ds["response"]
    exact_matches = sum(1 for c, p in zip(clean_responses, poisoned_responses) if c == p)
    drop = 1.0 - (exact_matches / len(clean_responses))
    
    print(f"{r:<15.2f} | {meta.num_original - meta.num_poisoned:<15} | {meta.num_poisoned:<15} | {drop:.2%}")

print("-" * 75)
