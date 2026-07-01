"""
Run the SSM-Score Manipulation Attack against Reliable-dRAG.
Usage
-----
python attack/ssm_score/run_attack.py --info
python attack/ssm_score/run_attack.py --evaluate
python attack/ssm_score/run_attack.py --target sources_100 --rounds 5 --amplify 999999 --evaluate
python attack/ssm_score/run_attack.py --reset
"""
import argparse
import json
import os
import sys
import datetime
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from attack.ssm_score.ssm_score_attack import SSMScoreAttack, DEFAULT_DATA_SOURCES, LLM_SERVICE_URL
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'attack_logs'))
EVAL_DATA = [
    {"question": "who got the first nobel prize in physics",       "answers": ["wilhelm conrad röntgen"]},
    {"question": "when is the next deadpool movie being released",  "answers": ["may 18, 2018"]},
    {"question": "which mode is used for short wave broadcast service", "answers": ["mfsk", "olivia"]},
    {"question": "the south west wind blows across nigeria between","answers": ["till september"]},
    {"question": "who wrote the first declaration of human rights", "answers": ["cyrus"]},
    {"question": "who is the owner of reading football club",       "answers": ["dai xiuli", "dai yongge"]},
    {"question": "swan lake the sleeping beauty and the nutcracker are three famous ballets by",
     "answers": ["pyotr ilyich tchaikovsky"]},
]
def query_llm(question):
    try:
        r = requests.post(f"{LLM_SERVICE_URL}/query", json={"query": question}, timeout=120)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}
def is_correct(response, answers):
    resp = response.lower().strip()
    return any(a.lower().strip() in resp for a in answers)
def collect_responses():
    print(f"\n  {'#':<3} {'Question':<52} {'OK?'} {'Response'}")
    print(f"  {'-'*3} {'-'*52} {'-'*3} {'-'*30}")
    results = []
    for i, item in enumerate(EVAL_DATA, 1):
        resp = query_llm(item["question"])
        answer = resp.get("response", resp.get("error", "ERROR"))
        correct = is_correct(answer, item["answers"])
        print(f"  {i:<3} {item['question'][:52]:<52} {'ok' if correct else '--'}  {answer[:35]}")
        results.append({"question": item["question"], "expected": item["answers"],
                        "response": answer, "correct": correct})
    return results
def save_log(log):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = log.get("attack_config", {}).get("target_source", "unknown")
    fname = os.path.join(LOG_DIR, f"attack_{ts}_ssm_score_{target}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return fname
def main():
    parser = argparse.ArgumentParser(description="SSM-Score Manipulation Attack")
    parser.add_argument("--target",   default="sources_100", choices=list(DEFAULT_DATA_SOURCES.keys()))
    parser.add_argument("--rounds",   type=int, default=5)
    parser.add_argument("--amplify",  type=int, default=999999)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--info",     action="store_true")
    parser.add_argument("--reset",    action="store_true")
    args = parser.parse_args()
    attack = SSMScoreAttack(target_source=args.target, amplify=args.amplify, rounds=args.rounds)
    if args.info:
        print("\n=== Current Blockchain Scores ===")
        for sid, s in attack.get_current_scores().items():
            print(f"  {sid:15s}  R={s['reliability']:>12,}  U={s['usefulness']:>12,}")
        return
    if args.reset:
        print("\n=== Resetting all scores to 0 ===")
        for sid, r in attack.reset_scores().items():
            print(f"  {sid}: {r}")
        return
    ts = datetime.datetime.now().isoformat()
    if args.evaluate:
        print("\n" + "="*60)
        print("PHASE 1 -- Clean baseline")
        print("="*60)
        clean = collect_responses()
        clean_acc = sum(1 for r in clean if r["correct"]) / len(clean)
        print(f"\n  Clean accuracy: {clean_acc:.1%}")
        print("\n" + "="*60)
        print("PHASE 2 -- SSM-Score inflation")
        print("="*60)
        attack_result = attack.inflate_scores()
        print("\n" + "="*60)
        print("PHASE 3 -- Post-attack responses")
        print("="*60)
        attacked = collect_responses()
        attacked_acc = sum(1 for r in attacked if r["correct"]) / len(attacked)
        print(f"\n  Attacked accuracy: {attacked_acc:.1%}")
        degradation = ((clean_acc - attacked_acc) / clean_acc * 100) if clean_acc > 0 else 0.0
        print(f"\n  Clean:   {clean_acc:.1%}")
        print(f"  Attacked:{attacked_acc:.1%}")
        print(f"  Drop:    {degradation:.1f}%")
        print(f"  SUCCESS: {'YES' if degradation > 10 else 'NO'}")
        log = {
            "timestamp": ts,
            "attack_type": "ssm_score_manipulation",
            "attack_config": {"target_source": args.target, "rounds": args.rounds,
                              "amplify_per_round": args.amplify},
            "attack_result": attack_result,
            "evaluation": {"clean_accuracy": clean_acc, "attacked_accuracy": attacked_acc,
                           "accuracy_degradation_pct": round(degradation, 2),
                           "is_successful": degradation > 10},
            "per_question": [{"question": c["question"], "expected": c["expected"],
                               "clean_response": c["response"], "clean_correct": c["correct"],
                               "attacked_response": a["response"], "attacked_correct": a["correct"],
                               "response_changed": c["response"] != a["response"]}
                              for c, a in zip(clean, attacked)],
        }
        print(f"\n  Log saved -> {save_log(log)}")
        print("\n=== Resetting scores ===")
        for sid, r in attack.reset_scores().items():
            print(f"  {sid}: {r}")
        return
    attack_result = attack.inflate_scores()
    print(f"\n  Log saved -> {save_log({'timestamp': ts, 'attack_type': 'ssm_score_manipulation',
        'attack_config': {'target_source': args.target, 'rounds': args.rounds, 'amplify_per_round': args.amplify},
        'attack_result': attack_result})}")
if __name__ == "__main__":
    main()
