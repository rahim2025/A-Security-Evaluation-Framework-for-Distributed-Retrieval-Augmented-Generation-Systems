import argparse, datetime, json, os, random, sys
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from attack.mia.mia_attack import MIAAttack, LLM_SERVICE_URL, MEMBER_JSONL, DEFAULT_MEMBERS, DEFAULT_NONMEMBERS

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attack_logs"))

def _dry_run(n_m, n_n, seed):
    rng=random.Random(seed); np.random.seed(seed)
    ms=[rng.uniform(0,1) for _ in range(n_m)]; ns=[rng.uniform(0,1) for _ in range(n_n)]
    y_true=np.array([1]*n_m+[0]*n_n); y_scores=np.array(ms+ns)
    y_pred=(y_scores>=np.percentile(y_scores,50)).astype(int)
    auc=float(roc_auc_score(y_true,y_scores)) if len(np.unique(y_true))>1 else 0.5
    print(f"[DRY-RUN] AUC={auc:.4f} (expected ~0.50)  Acc={float(accuracy_score(y_true,y_pred)):.4f}")
    return {"auc_roc":round(auc,4),"attack_accuracy":round(float(accuracy_score(y_true,y_pred)),4),
            "precision":round(float(precision_score(y_true,y_pred,zero_division=0)),4),
            "recall":round(float(recall_score(y_true,y_pred,zero_division=0)),4),
            "f1_score":round(float(f1_score(y_true,y_pred,zero_division=0)),4),
            "privacy_risk":"DRY-RUN","n_members_tested":n_m,"n_non_members_tested":n_n,
            "mean_member_similarity":round(float(np.mean(ms)),4),
            "mean_non_member_similarity":round(float(np.mean(ns)),4),
            "similarity_delta":round(float(np.mean(ms))-float(np.mean(ns)),4),
            "confusion_matrix":{"tp":0,"tn":0,"fp":0,"fn":0}}

def save_log(log):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed=log.get("attack_config",{}).get("random_seed","xx")
    fname=os.path.join(LOG_DIR,f"attack_{ts}_mia_seed{seed}.json")
    with open(fname,"w") as f: json.dump(log,f,indent=2)
    return fname

def parse_args():
    p=argparse.ArgumentParser(description="MIA on Reliable-dRAG")
    p.add_argument("--llm_url",default=LLM_SERVICE_URL)
    p.add_argument("--member_jsonl",default=MEMBER_JSONL)
    p.add_argument("--n_members",type=int,default=DEFAULT_MEMBERS)
    p.add_argument("--n_nonmembers",type=int,default=DEFAULT_NONMEMBERS)
    p.add_argument("--seed_words",type=int,default=12)
    p.add_argument("--threshold_pct",type=int,default=50)
    p.add_argument("--api_key",default="")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--dry_run",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    print("="*56)
    print("  MIA - Membership Inference Attack | Reliable-dRAG")
    print("="*56)
    print(f"  LLM URL    : {args.llm_url}")
    print(f"  Members    : {args.n_members}   Non-members: {args.n_nonmembers}")
    print(f"  Seed       : {args.seed}   Dry-run: {args.dry_run}")
    ts=datetime.datetime.now().isoformat()
    metrics=_dry_run(args.n_members,args.n_nonmembers,args.seed) if args.dry_run else             MIAAttack(args.llm_url,args.member_jsonl,args.n_members,args.n_nonmembers,
                      args.seed_words,args.threshold_pct,args.api_key,args.seed).run()
    log={"timestamp":ts,"attack_type":"membership_inference",
         "attack_config":{"llm_url":args.llm_url,"n_members":args.n_members,
                          "n_nonmembers":args.n_nonmembers,"random_seed":args.seed,
                          "dry_run":args.dry_run},"metrics":metrics}
    lp=save_log(log)
    print(f"\n  Log -> {lp}")
    print(f"  AUC-ROC      : {metrics['auc_roc']:.4f}")
    print(f"  Privacy risk : {metrics['privacy_risk']}")
    print(f"  Sim delta    : {metrics['similarity_delta']:+.4f}")

if __name__=="__main__":
    main()
