"""
Run the KB Extraction Attack against Reliable-dRAG.

Usage
-----
python attack/kb_extraction/run_attack.py
python attack/kb_extraction/run_attack.py --source sources_0
"""

import argparse, json, os, sys, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from attack.kb_extraction.kb_extraction_attack import KBExtractionAttack, DATA_SOURCES

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'attack_logs'))


def save_log(log):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = os.path.join(LOG_DIR, f"attack_{ts}_kb_extraction.json")
    slim = json.loads(json.dumps(log, ensure_ascii=False))
    for src in slim.get("direct_extraction", {}).values():
        for doc in src.get("docs", []):
            doc["text"] = doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    return fname


def main():
    parser = argparse.ArgumentParser(description="KB Extraction Attack on Reliable-dRAG")
    parser.add_argument("--source", default=None, choices=list(DATA_SOURCES.keys()))
    parser.add_argument("--k",      type=int, default=10)
    parser.add_argument("--queries",type=int, default=None)
    args = parser.parse_args()

    sources = {args.source: DATA_SOURCES[args.source]} if args.source else DATA_SOURCES
    attack = KBExtractionAttack(data_sources=sources, k_per_query=args.k)
    if args.queries:
        attack.probe_queries = attack.probe_queries[:args.queries]

    ts = datetime.datetime.now().isoformat()
    results = attack.run()
    results.update({"timestamp": ts, "attack_type": "kb_extraction",
                    "attack_config": {"target_sources": list(sources.keys()),
                                      "k_per_query": args.k,
                                      "num_probe_queries": len(attack.probe_queries)}})
    print(f"\n  Log saved -> {save_log(results)}")
    s = results["summary"]
    print(f"\n  === Thesis Result ===")
    print(f"  Probe queries sent    : {len(attack.probe_queries)} per source")
    print(f"  Unique docs recovered : {s['total_unique_docs']}")
    print(f"  Total text extracted  : {s['total_chars_extracted']:,} chars")
    print(f"  LLM indirect leakage  : {s['llm_chars_leaked']:,} chars")

if __name__ == "__main__":
    main()
