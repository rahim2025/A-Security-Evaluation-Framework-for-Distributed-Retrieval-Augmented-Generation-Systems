import json
import logging
from fed_rag.attacks.data_poisoning import DataPoisoningAttack
from datasets import Dataset

# Setup a real mini-dataset (e.g. 10 examples)
data = {"query": [], "response": [], "topic": []}
topics = ["math", "history", "science", "geography", "literature"]

for i in range(100):
    t = topics[i % len(topics)]
    data["query"].append(f"What is fact {i} about {t}?")
    data["response"].append(f"Fact {i}: {t} is interesting.")
    data["topic"].append(t)

clean_dataset = Dataset.from_dict(data)

# Real instantiation of the attack
attack = DataPoisoningAttack(
    poisoning_ratio=0.6,
    poison_type="answer_swap",
    mode="append",
    question_variants=3,
)

# Attack the huggingface dataset (real function)
poisoned_dataset, attack_metadata = attack.execute_hf_dataset(clean_dataset)

# Show the Matrix
print("-" * 60)
print(f"REAL ATTACK SUMMARY MATRIX")
print("-" * 60)
print(f"{'Metric':<25} | {'Value':<10}")
print("-" * 60)
print(f"Num Original Examples     | {attack_metadata.num_original}")
print(f"Num Poisoned Examples     | {attack_metadata.num_poisoned}")
print(f"Effective Poison Ratio    | {attack_metadata.poisoning_ratio:.2f}")
print(f"Poison Mode               | {attack_metadata.mode}")
print(f"Poison Type               | {attack_metadata.poison_type}")
print("-" * 60)

# Evaluate Degradation (mock generator comparison to show matrix)
clean_responses = clean_dataset["response"]
poisoned_responses = poisoned_dataset["response"]

exact_matches = sum(1 for c, p in zip(clean_responses, poisoned_responses) if c == p)
degradation = 1.0 - (exact_matches / len(clean_responses))

print(f"\nDEGRADATION MATRIX (Clean vs. Poisoned Base Model Target)")
print("-" * 60)
print(f"{'Target Accuracy':<25} | {exact_matches / len(clean_responses):.2%}")
print(f"{'Attack Success / Degrade':<25} | {degradation:.2%}")
print("-" * 60)

