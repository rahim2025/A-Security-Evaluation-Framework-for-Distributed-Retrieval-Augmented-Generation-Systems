"""
run_llm_evaluation.py
Upgraded FedRAG evaluation with LLM Generator step.
Compares: Retrieval-only vs LLM-augmented metrics.
"""
import argparse
import json
import os
import sys
from llm_generator import generate_answers_batch

# --- Import your existing evaluation setup ---
# Adjust this import to match your actual module name
try:
    from run_security_evaluation import run_evaluation_system_mode, run_evaluation_federated_mode
    HAS_EXISTING = True
except ImportError:
    HAS_EXISTING = False
    print("Warning: Could not import run_security_evaluation — will run standalone mode")

def run_with_llm(output_dir: str, model: str = "mistral", mode: str = "system",
                 num_clients: int = 10, num_samples: int = 20, 
                 poisoning_ratio: float = 0.3, seed: int = 0):
    """
    Runs the full pipeline:
    1. Retrieval (your existing code)
    2. LLM generation on retrieved docs
    3. Metrics on LLM output
    """
    import subprocess, sys

    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Run your existing evaluation to get retrieved docs
    retrieval_dir = os.path.join(output_dir, "retrieval_only")
    base_cmd = [
        sys.executable, "run_security_evaluation.py",
        "--mode", mode,
        "--num-clients", str(num_clients),
        "--num-samples", str(num_samples),
        "--poisoning-ratio", str(poisoning_ratio),
        "--seed", str(seed),
        "--output-dir", retrieval_dir,
        "--save-retrieved-docs",  # add this flag to your existing script
    ]
    print(f"Running retrieval step: {' '.join(base_cmd)}")
    subprocess.run(base_cmd, check=True)

    # Step 2: Load retrieved docs and run LLM generation
    retrieved_path = os.path.join(retrieval_dir, "retrieved_docs.json")
    if not os.path.exists(retrieved_path):
        print(f"ERROR: {retrieved_path} not found.")
        print("You need to add --save-retrieved-docs support to your existing script.")
        print("See Step 4 in the instructions below.")
        return

    with open(retrieved_path) as f:
        data = json.load(f)

    queries = [item["query"] for item in data]
    retrieved_docs = [item["retrieved_docs"] for item in data]
    ground_truths = [item["ground_truth"] for item in data]

    print(f"\nRunning LLM generation with model={model}...")
    llm_answers = generate_answers_batch(queries, retrieved_docs, model=model)

    # Step 3: Compute metrics on LLM answers
    from evaluate_metrics import compute_metrics  # adjust to your metric module
    results = compute_metrics(llm_answers, ground_truths)

    # Save results
    out = {
        "model": model,
        "mode": mode,
        "num_samples": len(queries),
        "poisoning_ratio": poisoning_ratio,
        "pipeline": "retrieval+llm",
        "metrics": results,
        "per_query": [
            {"query": q, "llm_answer": a, "ground_truth": g, **compute_metrics([a],[g])}
            for q, a, g in zip(queries, llm_answers, ground_truths)
        ]
    }

    out_path = os.path.join(output_dir, "llm_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n=== LLM EVALUATION RESULTS (model={model}) ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", choices=["mistral", "llama3.2", "gemma2:2b"])
    parser.add_argument("--mode", default="system", choices=["system", "federated"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--poisoning-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="logs/llm_eval")
    args = parser.parse_args()

    run_with_llm(
        output_dir=args.output_dir,
        model=args.model,
        mode=args.mode,
        num_clients=args.num_clients,
        num_samples=args.num_samples,
        poisoning_ratio=args.poisoning_ratio,
        seed=args.seed,
    )
