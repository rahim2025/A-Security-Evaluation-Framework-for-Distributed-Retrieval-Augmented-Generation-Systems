"""
Run the Data Poisoning Attack against the Reliable-dRAG system.

Usage
-----
# Default: random strategy, 50% sources poisoned, wrong_answer type
python attack/run_attack.py

# Custom options
python attack/run_attack.py --strategy targeted --targets sources_0 sources_100 \
                             --poison-type misleading --ratio 0.67 --amplify 5

# After attack, reset all sources back to clean state
python attack/run_attack.py --reset

# Evaluate attack effectiveness (compare clean vs attacked accuracy)
python attack/run_attack.py --evaluate
"""

import argparse
import json
import os
import sys
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attack.data_poisoning_attack import DataPoisoningAttack, DEFAULT_DATA_SOURCES


# ---------------------------------------------------------------------------
# Example QA pairs for evaluation (from test.ipynb)
# ---------------------------------------------------------------------------
EVAL_DATA = [
    {"question": "who got the first nobel prize in physics",
     "answers": ["wilhelm conrad röntgen"]},
    {"question": "when is the next deadpool movie being released",
     "answers": ["may 18, 2018"]},
    {"question": "which mode is used for short wave broadcast service",
     "answers": ["mfsk", "olivia"]},
    {"question": "the south west wind blows across nigeria between",
     "answers": ["till september"]},
    {"question": "who wrote the first declaration of human rights",
     "answers": ["cyrus"]},
    {"question": "who is the owner of reading football club",
     "answers": ["dai xiuli", "dai yongge"]},
    {"question": "swan lake the sleeping beauty and the nutcracker are three famous ballets by",
     "answers": ["pyotr ilyich tchaikovsky"]},
]

LLM_SERVICE_URL = "http://localhost:9000"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'polluted_token')


def load_data_points(data_dir: str):
    """Load all JSONL documents from the data directory."""
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.jsonl'):
            path = os.path.join(data_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


def query_llm(question: str) -> dict:
    """Send a query to the LLM service and return the response."""
    try:
        r = requests.post(
            f"{LLM_SERVICE_URL}/query",
            json={"query": question},
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def collect_responses(questions):
    """Query the LLM for each question and return responses."""
    responses = []
    for item in questions:
        q = item["question"]
        print(f"  Querying: {q[:60]}...")
        resp = query_llm(q)
        responses.append(resp)
        print(f"    → {resp.get('response', resp.get('error', '?'))}")
    return responses


def main():
    parser = argparse.ArgumentParser(description="Run Data Poisoning Attack on Reliable-dRAG")
    parser.add_argument('--strategy', default='random',
                        choices=['random', 'targeted', 'high_reliability'],
                        help='Peer selection strategy')
    parser.add_argument('--targets', nargs='+', default=[],
                        metavar='SOURCE_NAME',
                        help='Source names to target (used with --strategy targeted)')
    parser.add_argument('--poison-type', default='wrong_answer',
                        choices=['wrong_answer', 'misleading', 'noise', 'answer_swap'],
                        help='Type of poisoning to apply')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Fraction of data sources to poison (0.0–1.0)')
    parser.add_argument('--amplify', type=int, default=3,
                        help='Number of copies of each poisoned doc to inject')
    parser.add_argument('--variants', type=int, default=2,
                        help='Number of text variants to create per document')
    parser.add_argument('--reset', action='store_true',
                        help='Reset all data sources to clean state and exit')
    parser.add_argument('--evaluate', action='store_true',
                        help='Measure accuracy before and after the attack')
    parser.add_argument('--info', action='store_true',
                        help='Show current doc counts for all data sources')
    args = parser.parse_args()

    attack = DataPoisoningAttack(
        data_sources=DEFAULT_DATA_SOURCES,
        poisoning_ratio=args.ratio,
        attack_strategy=args.strategy,
        poison_type=args.poison_type,
        target_source_names=args.targets,
        amplification_factor=args.amplify,
        question_variants=args.variants,
        llm_service_url=LLM_SERVICE_URL,
    )

    # ------------------------------------------------------------------
    # Info mode
    # ------------------------------------------------------------------
    if args.info:
        print("\n=== Data Source Info ===")
        info = attack.get_info()
        for name, data in info.items():
            print(f"  {name}: {data}")
        return

    # ------------------------------------------------------------------
    # Reset mode
    # ------------------------------------------------------------------
    if args.reset:
        print("\n=== Resetting all data sources to clean state ===")
        results = attack.reset_all()
        for name, result in results.items():
            print(f"  {name}: {result}")
        return

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    data_points = load_data_points(DATA_DIR)
    if not data_points:
        print(f"ERROR: No data found in {DATA_DIR}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluate mode: measure accuracy before AND after
    # ------------------------------------------------------------------
    if args.evaluate:
        print("\n=== Phase 1: Collecting CLEAN responses ===")
        clean_responses = collect_responses(EVAL_DATA)

        print("\n=== Phase 2: Executing attack ===")
        result = attack.execute(data_points)
        print(f"\nAttack result: {json.dumps(result, indent=2)}")

        print("\n=== Phase 3: Collecting ATTACKED responses ===")
        attacked_responses = collect_responses(EVAL_DATA)

        print("\n=== Phase 4: Evaluating success ===")
        ground_truths = [item["answers"] for item in EVAL_DATA]
        evaluation = attack.evaluate_success(clean_responses, attacked_responses, ground_truths)
        print(json.dumps(evaluation, indent=2))

        print("\n=== Phase 5: Resetting to clean state ===")
        attack.reset_all()
        return

    # ------------------------------------------------------------------
    # Default: just run the attack
    # ------------------------------------------------------------------
    print("\n=== Executing Data Poisoning Attack ===")
    result = attack.execute(data_points)

    print("\n=== Attack Summary ===")
    print(json.dumps(result, indent=2))

    print("\n=== Post-attack data source info ===")
    info = attack.get_info()
    for name, data in info.items():
        print(f"  {name}: {data}")

    print("\nTo reset all sources back to clean state, run:")
    print("  python attack/run_attack.py --reset")


if __name__ == '__main__':
    main()
