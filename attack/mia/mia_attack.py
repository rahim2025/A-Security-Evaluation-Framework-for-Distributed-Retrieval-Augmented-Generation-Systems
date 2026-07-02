import hashlib, json, os, random
from typing import Any, Dict, List
import numpy as np, requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

LLM_SERVICE_URL = "http://localhost:9000"
_HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER_JSONL = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_0.jsonl"))
ALT_JSONL_PATHS = [
    os.path.normpath(os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_100.jsonl")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_20.jsonl")),
]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED_WORDS = 12
DEFAULT_MEMBERS = 25
DEFAULT_NONMEMBERS = 25

def _load_passages(path, n):
    passages = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw: continue
            rec = json.loads(raw)
            text = (rec.get("documents") or rec.get("html") or rec.get("text","")).strip()
            if text: passages.append(text)
            if len(passages) >= n: break
    return passages

def _non_member_passages(member_hashes, n):
    pool = []
    for path in ALT_JSONL_PATHS:
        if not os.path.exists(path): continue
        for text in _load_passages(path, n*5):
            h = hashlib.md5(text[:200].encode()).hexdigest()
            if h not in member_hashes and text not in pool: pool.append(text)
            if len(pool) >= n: break
        if len(pool) >= n: break
    for i in range(n*2):
        s = f"Synthetic non-member {i}: quantum chromodynamics describes strong interaction."
        if s not in pool: pool.append(s)
        if len(pool) >= n: break
    return pool[:n]

def _seed_phrase(text, n=SEED_WORDS): return " ".join(text.split()[:n])

def _query_llm(seed, url, api_key=""):
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = requests.post(f"{url}/query", json={"query":seed}, headers=headers, timeout=120)
        if r.status_code == 200:
            d = r.json(); return d.get("response", d.get("answer",""))
    except: pass
    return ""

def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0

class MIAAttack:
    def __init__(self, llm_service_url=LLM_SERVICE_URL, member_jsonl=MEMBER_JSONL,
                 n_members=DEFAULT_MEMBERS, n_nonmembers=DEFAULT_NONMEMBERS,
                 seed_words=SEED_WORDS, threshold_percentile=50, api_key="", random_seed=42):
        if not _ST_AVAILABLE: raise ImportError("pip install sentence-transformers")
        self.llm_url=llm_service_url; self.member_jsonl=member_jsonl
        self.n_members=n_members; self.n_nonmembers=n_nonmembers
        self.seed_words=seed_words; self.threshold_pct=threshold_percentile
        self.api_key=api_key; self.random_seed=random_seed
        print(f"  Loading {EMBEDDING_MODEL} ...")
        self._enc = SentenceTransformer(EMBEDDING_MODEL)

    def run(self):
        random.seed(self.random_seed); np.random.seed(self.random_seed)
        if not os.path.exists(self.member_jsonl):
            raise FileNotFoundError(f"Missing: {self.member_jsonl}")
        print(f"  Loading members from: {self.member_jsonl}")
        members = _load_passages(self.member_jsonl, self.n_members)
        hashes = {hashlib.md5(p[:200].encode()).hexdigest() for p in members}
        non_members = _non_member_passages(hashes, self.n_nonmembers)
        print(f"  Members: {len(members)}  Non-members: {len(non_members)}")
        print("\n  === Probing MEMBERS ===")
        ms = self._probe(members, "MEM")
        print("\n  === Probing NON-MEMBERS ===")
        ns = self._probe(non_members, "NON")
        y_true = np.array([1]*len(ms)+[0]*len(ns))
        y_scores = np.array(ms+ns)
        y_pred = (y_scores >= np.percentile(y_scores, self.threshold_pct)).astype(int)
        metrics = self._metrics(y_true, y_pred, y_scores, ms, ns)
        self._print(metrics); return metrics

    def _probe(self, passages, label):
        scores = []
        for i, p in enumerate(passages, 1):
            seed = _seed_phrase(p, self.seed_words)
            resp = _query_llm(seed, self.llm_url, self.api_key)
            if resp.strip():
                er = self._enc.encode([resp], convert_to_numpy=True)[0]
                ep = self._enc.encode([p],    convert_to_numpy=True)[0]
                sim = _cosine(er, ep)
            else: sim = 0.0
            scores.append(sim)
            print(f"    [{label}] {i:>2}/{len(passages)}  sim={sim:.4f}  seed='{seed[:40]}...'")
        return scores

    def _metrics(self, y_true, y_pred, y_scores, ms, ns):
        tp=int(np.sum((y_true==1)&(y_pred==1))); tn=int(np.sum((y_true==0)&(y_pred==0)))
        fp=int(np.sum((y_true==0)&(y_pred==1))); fn=int(np.sum((y_true==1)&(y_pred==0)))
        auc = float(roc_auc_score(y_true,y_scores)) if len(np.unique(y_true))>1 else 0.5
        mu_m=float(np.mean(ms)) if ms else 0.0; mu_n=float(np.mean(ns)) if ns else 0.0
        return {"confusion_matrix":{"tp":tp,"tn":tn,"fp":fp,"fn":fn},
                "attack_accuracy":round(float(accuracy_score(y_true,y_pred)),4),
                "precision":round(float(precision_score(y_true,y_pred,zero_division=0)),4),
                "recall":round(float(recall_score(y_true,y_pred,zero_division=0)),4),
                "f1_score":round(float(f1_score(y_true,y_pred,zero_division=0)),4),
                "auc_roc":round(auc,4),
                "privacy_risk":("CRITICAL" if auc>0.9 else "HIGH" if auc>0.75 else
                                "MEDIUM" if auc>0.6 else "LOW" if auc>0.5 else "NEGLIGIBLE"),
                "n_members_tested":int(np.sum(y_true==1)),
                "n_non_members_tested":int(np.sum(y_true==0)),
                "mean_member_similarity":round(mu_m,4),
                "mean_non_member_similarity":round(mu_n,4),
                "similarity_delta":round(mu_m-mu_n,4)}

    def _print(self, m):
        cm=m["confusion_matrix"]
        print("\n"+"="*56)
        print("  MIA Results")
        print("="*56)
        print(f"  TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
        print(f"  Accuracy      : {m['attack_accuracy']:.4f}")
        print(f"  AUC-ROC       : {m['auc_roc']:.4f}  <- thesis metric")
        print(f"  Privacy risk  : {m['privacy_risk']}")
        print(f"  Sim delta     : {m['similarity_delta']:+.4f}  (pos = members exposed)")
        print("="*56)
