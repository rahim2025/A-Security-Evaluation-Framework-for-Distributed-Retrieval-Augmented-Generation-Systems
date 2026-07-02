"""
MIA — Membership Inference Attack on Reliable-dRAG.

Attack Principle
----------------
A *member* passage is text that lives in one of the data-source knowledge bases
(data/polluted_token/sources_0.jsonl).  A *non-member* passage is text from a
different KB partition that does NOT appear in sources_0.

The attack:
  1. Extracts a short seed phrase (first SEED_WORDS words) from each passage.
  2. POSTs the seed to the LLM service (localhost:9000/query).
  3. Computes cosine similarity between the LLM response and the original
     passage using the all-MiniLM-L6-v2 sentence-transformer.
  4. Sweeps a threshold over the similarity distribution.
  5. Reports AUC-ROC, Accuracy, Precision, Recall, F1.

Hypothesis
----------
For a *member* passage the retriever returns the actual KB text as context, so
the LLM answer will closely paraphrase it → high cosine similarity.
For a *non-member* the retriever returns unrelated text → low similarity.
AUC-ROC ≈ 0.50 means no leakage; AUC-ROC > 0.70 means genuine privacy risk.

Modelled after: DRAG baseline modules/attacks/membership_inference.py
Adapted for:    Reliable-dRAG HTTP/Docker architecture (3 Flask sources + LLM)
"""

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

DATA_SOURCES = {
    "sources_0":   "http://localhost:8001",
    "sources_20":  "http://localhost:8002",
    "sources_100": "http://localhost:8003",
}
LLM_SERVICE_URL = "http://localhost:9000"

_HERE = os.path.dirname(__file__)
MEMBER_JSONL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_0.jsonl")
)
ALT_JSONL_PATHS = [
    os.path.normpath(os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_100.jsonl")),
    os.path.normpath(os.path.join(_HERE, "..", "..", "data", "polluted_token", "sources_20.jsonl")),
]

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
SEED_WORDS       = 12
DEFAULT_MEMBERS  = 25
DEFAULT_NONMEMBERS = 25


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_passages(path: str, n: int) -> List[str]:
    """Read up to n non-empty text passages from a JSONL file."""
    passages: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec  = json.loads(raw)
            text = rec.get("documents") or rec.get("html") or rec.get("text", "")
            text = text.strip()
            if text:
                passages.append(text)
            if len(passages) >= n:
                break
    return passages


def _non_member_passages(member_hashes: set, n: int) -> List[str]:
    """
    Return n non-member passages guaranteed not to overlap with member_hashes.
    Tries ALT_JSONL_PATHS first; falls back to simple synthetic sentences.
    """
    pool: List[str] = []
    for path in ALT_JSONL_PATHS:
        if not os.path.exists(path):
            continue
        for text in _load_passages(path, n * 5):
            h = hashlib.md5(text[:200].encode()).hexdigest()
            if h not in member_hashes and text not in pool:
                pool.append(text)
            if len(pool) >= n:
                break
        if len(pool) >= n:
            break

    if len(pool) < n:
        for i in range(n * 2):
            s = (
                f"Synthetic non-member sentence {i}: Quantum chromodynamics "
                f"describes the strong interaction between quarks and gluons "
                f"mediated by the exchange of eight massless gauge bosons."
            )
            if s not in pool:
                pool.append(s)
            if len(pool) >= n:
                break

    return pool[:n]


def _seed_phrase(text: str, n_words: int = SEED_WORDS) -> str:
    """Return the first n_words words of text as a probe query."""
    return " ".join(text.split()[:n_words])


def _query_llm(seed: str, url: str, api_key: str = "") -> str:
    """POST seed to the LLM service; return response text or empty string."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = requests.post(
            f"{url}/query",
            json={"query": seed},
            headers=headers,
            timeout=120,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response", data.get("answer", ""))
    except Exception:
        pass
    return ""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Core attack class
# ---------------------------------------------------------------------------

class MIAAttack:
    """
    Membership Inference Attack for Reliable-dRAG.

    Parameters
    ----------
    llm_service_url : str
        URL of the running LLM service (default localhost:9000).
    member_jsonl : str
        Path to JSONL file whose passages are in the knowledge base.
    n_members : int
        How many member passages to probe (default 25).
    n_nonmembers : int
        How many non-member passages to probe (default 25).
    seed_words : int
        Number of leading words used as the probe seed (default 12).
    threshold_percentile : int
        Percentile of the similarity distribution used as the decision
        threshold (default 50 — median split).
    api_key : str
        X-API-Key header value if auth is enabled on the sources.
    random_seed : int
        RNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        llm_service_url: str  = LLM_SERVICE_URL,
        member_jsonl: str     = MEMBER_JSONL,
        n_members: int        = DEFAULT_MEMBERS,
        n_nonmembers: int     = DEFAULT_NONMEMBERS,
        seed_words: int       = SEED_WORDS,
        threshold_percentile: int = 50,
        api_key: str          = "",
        random_seed: int      = 42,
    ):
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install: pip install sentence-transformers"
            )
        self.llm_service_url   = llm_service_url
        self.member_jsonl      = member_jsonl
        self.n_members         = n_members
        self.n_nonmembers      = n_nonmembers
        self.seed_words        = seed_words
        self.threshold_pct     = threshold_percentile
        self.api_key           = api_key
        self.random_seed       = random_seed
        print(f"  Loading sentence-transformer: {EMBEDDING_MODEL}")
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """
        Execute the full MIA pipeline.

        Returns
        -------
        dict with keys: confusion_matrix, attack_accuracy, precision, recall,
                        f1_score, auc_roc, privacy_risk,
                        n_members_tested, n_non_members_tested,
                        mean_member_similarity, mean_non_member_similarity,
                        similarity_delta
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # ---- Load passages -----------------------------------------------
        if not os.path.exists(self.member_jsonl):
            raise FileNotFoundError(
                f"Member JSONL not found: {self.member_jsonl}\n"
                "Make sure data/polluted_token/sources_0.jsonl exists."
            )
        print(f"\n  Loading members  from: {self.member_jsonl}")
        members = _load_passages(self.member_jsonl, self.n_members)
        if len(members) < self.n_members:
            print(f"  [warn] Only {len(members)} member passages available.")

        member_hashes = {hashlib.md5(p[:200].encode()).hexdigest() for p in members}
        print(f"  Loading non-members (no overlap with member set) ...")
        non_members = _non_member_passages(member_hashes, self.n_nonmembers)
        print(f"  Members: {len(members)}   Non-members: {len(non_members)}")

        # ---- Probe + score -----------------------------------------------
        print("\n  === Probing MEMBER passages ===")
        member_scores = self._probe_and_score(members, label="MEMBER")

        print("\n  === Probing NON-MEMBER passages ===")
        non_member_scores = self._probe_and_score(non_members, label="NON-MEMBER")

        # ---- Metrics -------------------------------------------------------
        y_true   = np.array([1] * len(member_scores) + [0] * len(non_member_scores))
        y_scores = np.array(member_scores + non_member_scores)

        threshold = float(np.percentile(y_scores, self.threshold_pct))
        y_pred    = (y_scores >= threshold).astype(int)

        metrics = self._compute_metrics(
            y_true, y_pred, y_scores, member_scores, non_member_scores
        )
        self._print_summary(metrics)
        return metrics

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _probe_and_score(self, passages: List[str], label: str) -> List[float]:
        """Send a seed probe for each passage; return cosine similarity scores."""
        scores: List[float] = []
        n = len(passages)
        for i, passage in enumerate(passages, 1):
            seed     = _seed_phrase(passage, self.seed_words)
            response = _query_llm(seed, self.llm_service_url, self.api_key)

            if response.strip():
                emb_r = self._encoder.encode([response], convert_to_numpy=True)[0]
                emb_p = self._encoder.encode([passage],  convert_to_numpy=True)[0]
                sim   = _cosine_similarity(emb_r, emb_p)
            else:
                sim = 0.0

            scores.append(sim)
            print(f"    [{label}] {i:>2}/{n}  "
                  f"seed='{seed[:45]}...'  sim={sim:.4f}")
        return scores

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        member_scores: List[float],
        non_member_scores: List[float],
    ) -> Dict[str, Any]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        accuracy  = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))
        auc_roc   = (
            float(roc_auc_score(y_true, y_scores))
            if len(np.unique(y_true)) > 1 else 0.5
        )

        mu_m  = float(np.mean(member_scores))     if member_scores     else 0.0
        mu_nm = float(np.mean(non_member_scores)) if non_member_scores else 0.0

        return {
            "confusion_matrix":           {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "attack_accuracy":            round(accuracy,  4),
            "precision":                  round(precision, 4),
            "recall":                     round(recall,    4),
            "f1_score":                   round(f1,        4),
            "auc_roc":                    round(auc_roc,   4),
            "privacy_risk":               self._privacy_risk(auc_roc),
            "n_members_tested":           int(np.sum(y_true == 1)),
            "n_non_members_tested":       int(np.sum(y_true == 0)),
            "mean_member_similarity":     round(mu_m,         4),
            "mean_non_member_similarity": round(mu_nm,        4),
            "similarity_delta":           round(mu_m - mu_nm, 4),
        }

    @staticmethod
    def _privacy_risk(auc_roc: float) -> str:
        if auc_roc > 0.90: return "CRITICAL — severe privacy leak"
        if auc_roc > 0.75: return "HIGH — significant privacy risk"
        if auc_roc > 0.60: return "MEDIUM — moderate privacy concern"
        if auc_roc > 0.50: return "LOW — slight privacy risk"
        return "NEGLIGIBLE — attack ineffective (AUC ≈ random)"

    @staticmethod
    def _print_summary(m: Dict[str, Any]) -> None:
        cm = m["confusion_matrix"]
        print("\n" + "=" * 62)
        print("  MIA — Results")
        print("=" * 62)
        print(f"  Members tested       : {m['n_members_tested']}")
        print(f"  Non-members tested   : {m['n_non_members_tested']}")
        print(f"  Confusion matrix     : TP={cm['tp']}  TN={cm['tn']}  "
              f"FP={cm['fp']}  FN={cm['fn']}")
        print(f"  Attack accuracy      : {m['attack_accuracy']:.4f}")
        print(f"  Precision            : {m['precision']:.4f}")
        print(f"  Recall               : {m['recall']:.4f}")
        print(f"  F1 score             : {m['f1_score']:.4f}")
        print(f"  AUC-ROC              : {m['auc_roc']:.4f}  <- thesis metric")
        print(f"  Privacy risk         : {m['privacy_risk']}")
        print(f"\n  Mean member sim      : {m['mean_member_similarity']:.4f}")
        print(f"  Mean non-member sim  : {m['mean_non_member_similarity']:.4f}")
        print(f"  Similarity delta     : {m['similarity_delta']:+.4f}")
        print("=" * 62)
