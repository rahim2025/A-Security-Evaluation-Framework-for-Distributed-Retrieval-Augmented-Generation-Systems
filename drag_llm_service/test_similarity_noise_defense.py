"""
drag_llm_service/test_similarity_noise_defense.py

Unit tests for the similarity-only Laplace noise defense
(drag_llm_service/src/retriever/defense.py, `add_laplace_noise_to_similarity`
and its wiring into `select_top_k_with_defense`).

No live service, Docker, or model download required -- `_FakeReranker`
below stands in for `Reranker` with hand-picked, fixed embeddings, so
these tests run against plain numpy arithmetic only. Matches this
project's existing test convention (drag_llm_service/test_service.py):
plain `test_*` functions, run as a script -- no pytest dependency
required, though these are also pytest-discoverable if pytest is
available (`assert`-based, `test_*` naming).

Usage
-----
  python drag_llm_service/test_similarity_noise_defense.py
  # or, if pytest is installed:
  pytest drag_llm_service/test_similarity_noise_defense.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# drag_llm_service has no top-level __init__.py -- it is not a dotted-import
# package. server.py's own convention (app/server.py) is to put
# drag_llm_service/ itself on sys.path and import "src.retriever...";
# matched here rather than inventing a different import scheme.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.retriever.defense import (  # noqa: E402
    DEFAULT_SIMILARITY_SENSITIVITY,
    _derive_seed,
    add_laplace_noise_to_similarity,
    select_top_k_with_defense,
)


class _FakeReranker:
    """
    Minimal stand-in for `Reranker` exposing only what
    `select_top_k_with_defense` actually calls: `method`, `hybrid_alpha`,
    `orig_score_weight`, `embed_query_and_candidates()`, `_bm25_scores()`.
    Fixed, hand-picked embeddings -- no SentenceTransformer model load, no
    network, fully deterministic.
    """

    def __init__(self, cand_embeddings: np.ndarray, query_embedding: np.ndarray):
        self.method = "dense"
        self.hybrid_alpha = 0.3
        self.orig_score_weight = 0.0
        self._cand_embeddings = cand_embeddings
        self._query_embedding = query_embedding

    def embed_query_and_candidates(self, query, texts):
        # Same contract as Reranker.embed_query_and_candidates: returns
        # (query_vector, candidate_matrix), already L2-normalized so the
        # dot product below is a plain cosine similarity.
        return self._query_embedding, self._cand_embeddings

    def _bm25_scores(self, query, texts):
        return np.zeros(len(texts), dtype=np.float32)


def _make_candidates(n: int, sources=None):
    sources = sources or [f"src{i % 3}" for i in range(n)]
    return [
        {"id": f"c{i}", "text": f"candidate text {i}", "score": 0.0,
         "meta": {"source": sources[i]}}
        for i in range(n)
    ]


def _make_orthogonal_like_embeddings(n: int, dim: int = 8, seed: int = 0):
    """n unit vectors + a query unit vector, giving a spread of raw cosine
    similarities in a controlled, reproducible way (not all identical, not
    degenerate for _minmax)."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(n, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    q = rng.normal(size=(dim,)).astype(np.float32)
    q /= np.linalg.norm(q)
    return mat, q


# ── 1. defense disabled -> identical existing behavior ──────────────────────

def test_disabled_is_exact_passthrough():
    """`enabled=False` must return the exact same array values as calling
    without the defense existing at all -- a true no-op, not 'small noise'."""
    raw = np.array([0.1, -0.2, 0.9, -1.0, 0.0], dtype=np.float32)
    out = add_laplace_noise_to_similarity(raw, query="q", enabled=False, epsilon=0.01, seed=1)
    assert np.array_equal(out, raw), "disabled defense must not modify the input at all"


def test_disabled_end_to_end_matches_pre_defense_pipeline():
    """select_top_k_with_defense with similarity_noise_enabled absent/false must
    produce bit-identical rerank_score/order to a hand-computed reference that
    replicates the pre-noise fusion math exactly."""
    n = 6
    cand_embs, q_emb = _make_orthogonal_like_embeddings(n, seed=7)
    candidates = _make_candidates(n)
    reranker = _FakeReranker(cand_embs, q_emb)

    cfg = {"dedup_enabled": False, "consensus_weight": 0.0}  # isolate similarity-noise effect only
    result = select_top_k_with_defense(
        reranker, "some query", candidates, top_k=n,
        reliability_scores=None, reliability_weight=0.0, cfg=cfg,
    )

    # Reference: exactly what the pipeline computed before this defense existed.
    dense = cand_embs @ q_emb
    lo, hi = float(np.min(dense)), float(np.max(dense))
    expected_fused = (dense - lo) / (hi - lo)
    expected_order = np.argsort(-expected_fused)

    got_order = [int(c["id"][1:]) for c in result]
    assert got_order == list(expected_order), (
        "disabled similarity-noise defense changed the selection order vs. "
        "the pre-defense fusion math"
    )
    for c, idx in zip(result, expected_order):
        assert abs(c["rerank_score"] - float(expected_fused[idx])) < 1e-6


# ── 2. defense enabled -> similarity scores are perturbed ───────────────────

def test_enabled_perturbs_scores():
    raw = np.array([0.1, -0.2, 0.9, -1.0, 0.0], dtype=np.float32)
    out = add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=0.5, seed=1)
    assert not np.array_equal(out, raw), "enabled defense should perturb at least one score"
    assert out.shape == raw.shape


def test_smaller_epsilon_yields_larger_expected_perturbation():
    """Sanity check on the epsilon -> scale relationship (b = sensitivity/epsilon):
    a much smaller epsilon (bigger scale) should produce a much larger mean
    absolute perturbation over many candidates, on average."""
    raw = np.zeros(2000, dtype=np.float32)
    small_eps_out = add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=0.05, seed=1)
    large_eps_out = add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=5.0, seed=1)
    assert np.mean(np.abs(small_eps_out)) > np.mean(np.abs(large_eps_out)), (
        "smaller epsilon (larger Laplace scale) should perturb similarity "
        "scores more, on average, than a larger epsilon"
    )


def test_invalid_epsilon_rejected():
    raw = np.array([0.1, 0.2], dtype=np.float32)
    for bad_eps in (0.0, -1.0):
        try:
            add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=bad_eps, seed=1)
        except ValueError:
            continue
        raise AssertionError(f"epsilon={bad_eps} should have raised ValueError")


# ── 3. reliability scores remain unchanged ───────────────────────────────────

def test_reliability_scores_unaffected_by_similarity_noise():
    n = 6
    cand_embs, q_emb = _make_orthogonal_like_embeddings(n, seed=3)
    candidates = _make_candidates(n, sources=["a", "b", "c", "a", "b", "c"])
    reranker = _FakeReranker(cand_embs, q_emb)
    reliability_scores = {"a": 10.0, "b": 50.0, "c": 90.0}

    cfg_no_noise = {"dedup_enabled": False, "consensus_weight": 0.0,
                     "similarity_noise_enabled": False}
    cfg_noise = {"dedup_enabled": False, "consensus_weight": 0.0,
                 "similarity_noise_enabled": True, "similarity_noise_epsilon": 0.2,
                 "similarity_noise_seed": 42}

    result_no_noise = select_top_k_with_defense(
        reranker, "q", candidates, top_k=n,
        reliability_scores=reliability_scores, reliability_weight=0.5, cfg=cfg_no_noise,
    )
    result_noise = select_top_k_with_defense(
        reranker, "q", candidates, top_k=n,
        reliability_scores=reliability_scores, reliability_weight=0.5, cfg=cfg_noise,
    )

    rel_by_id_no_noise = {c["id"]: reliability_scores[c["meta"]["source"]] for c in result_no_noise}
    rel_by_id_noise = {c["id"]: reliability_scores[c["meta"]["source"]] for c in result_noise}
    assert rel_by_id_no_noise == rel_by_id_noise, (
        "the raw reliability score attached to each source must be identical "
        "regardless of the similarity-noise defense -- reliability is never "
        "touched by this defense"
    )
    # Also confirm the defense function itself never sees/returns anything
    # resembling a reliability score -- it only takes/returns a similarity array.
    import inspect
    sig = inspect.signature(add_laplace_noise_to_similarity)
    assert "reliability" not in " ".join(sig.parameters.keys()).lower()


# ── 4. noise is applied before top-k selection ───────────────────────────────

def test_noise_can_change_topk_order_and_matches_direct_computation():
    """With a tiny epsilon (huge noise) on two near-tied candidates, the
    resulting order must match exactly what applying the SAME noise function
    directly to the raw dense scores (then re-running the unchanged
    _minmax/argsort) would produce -- proving noise flows into ranking
    before top-k, not after or not at all."""
    n = 4
    cand_embs, q_emb = _make_orthogonal_like_embeddings(n, seed=99)
    candidates = _make_candidates(n)
    reranker = _FakeReranker(cand_embs, q_emb)

    cfg = {"dedup_enabled": False, "consensus_weight": 0.0,
           "similarity_noise_enabled": True, "similarity_noise_epsilon": 0.05,
           "similarity_noise_seed": 123}
    result = select_top_k_with_defense(
        reranker, "the query text", candidates, top_k=n,
        reliability_scores=None, reliability_weight=0.0, cfg=cfg,
    )

    # Independently recompute the expected pipeline: raw dense -> noise
    # (same function, same params) -> minmax -> argsort. If the defense
    # weren't wired in before top-k, this would not match.
    dense = cand_embs @ q_emb
    noisy = add_laplace_noise_to_similarity(
        dense, query="the query text", enabled=True, epsilon=0.05, seed=123,
    )
    lo, hi = float(np.min(noisy)), float(np.max(noisy))
    expected_fused = (noisy - lo) / (hi - lo) if hi != lo else np.zeros_like(noisy)
    expected_order = list(np.argsort(-expected_fused))

    got_order = [int(c["id"][1:]) for c in result]
    assert got_order == [int(i) for i in expected_order]


# ── 5. configuration correctly enables/disables the defense ─────────────────

def test_config_flag_controls_enable_disable():
    n = 5
    cand_embs, q_emb = _make_orthogonal_like_embeddings(n, seed=11)
    candidates = _make_candidates(n)
    reranker = _FakeReranker(cand_embs, q_emb)
    dense = cand_embs @ q_emb

    for flag, expect_noise in [(False, False), (True, True)]:
        cfg = {"dedup_enabled": False, "consensus_weight": 0.0,
               "similarity_noise_enabled": flag, "similarity_noise_epsilon": 0.1,
               "similarity_noise_seed": 5}
        result = select_top_k_with_defense(
            reranker, "cfg-test query", candidates, top_k=n,
            reliability_scores=None, reliability_weight=0.0, cfg=cfg,
        )
        lo, hi = float(np.min(dense)), float(np.max(dense))
        clean_fused_by_id = {
            f"c{i}": float((dense[i] - lo) / (hi - lo)) for i in range(n)
        }
        matches_clean = all(
            abs(c["rerank_score"] - clean_fused_by_id[c["id"]]) < 1e-6 for c in result
        )
        if expect_noise:
            assert not matches_clean, "similarity_noise_enabled=true should perturb rerank_score"
        else:
            assert matches_clean, "similarity_noise_enabled=false should exactly match the clean pipeline"

    # Absent key entirely must default to disabled (matches config.yaml's
    # documented default and requirement 7: "when the defense is disabled,
    # retrieval behavior must remain exactly unchanged").
    cfg_absent = {"dedup_enabled": False, "consensus_weight": 0.0}
    result_absent = select_top_k_with_defense(
        reranker, "cfg-test query", candidates, top_k=n,
        reliability_scores=None, reliability_weight=0.0, cfg=cfg_absent,
    )
    lo, hi = float(np.min(dense)), float(np.max(dense))
    for c in result_absent:
        i = int(c["id"][1:])
        expected = float((dense[i] - lo) / (hi - lo))
        assert abs(c["rerank_score"] - expected) < 1e-6


# ── 6. reproducibility under the repository's seed mechanism ────────────────

def test_same_seed_and_query_gives_identical_noise():
    raw = np.linspace(-1.0, 1.0, 50).astype(np.float32)
    out1 = add_laplace_noise_to_similarity(raw, query="reproducibility query", enabled=True,
                                            epsilon=0.3, seed=42)
    out2 = add_laplace_noise_to_similarity(raw, query="reproducibility query", enabled=True,
                                            epsilon=0.3, seed=42)
    assert np.array_equal(out1, out2), "same seed + same query must reproduce identical noise"


def test_different_seed_gives_different_noise():
    raw = np.linspace(-1.0, 1.0, 50).astype(np.float32)
    out1 = add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=0.3, seed=1)
    out2 = add_laplace_noise_to_similarity(raw, query="q", enabled=True, epsilon=0.3, seed=2)
    assert not np.array_equal(out1, out2)


def test_different_query_gives_different_noise_same_seed():
    """Same base seed, different query text -> different derived sub-seed ->
    different noise (not one frozen vector for the whole process)."""
    raw = np.linspace(-1.0, 1.0, 50).astype(np.float32)
    out1 = add_laplace_noise_to_similarity(raw, query="query A", enabled=True, epsilon=0.3, seed=42)
    out2 = add_laplace_noise_to_similarity(raw, query="query B", enabled=True, epsilon=0.3, seed=42)
    assert not np.array_equal(out1, out2)


def test_derive_seed_is_pure_and_stable():
    """_derive_seed must not depend on process-level hash randomization --
    call it twice in this same process (the strongest guarantee testable
    without spawning a second interpreter) and confirm purity."""
    a = _derive_seed(42, "some query")
    b = _derive_seed(42, "some query")
    assert a == b
    c = _derive_seed(43, "some query")
    assert a != c


def test_sensitivity_constant_matches_cosine_range():
    """Documents/locks the sensitivity-derivation contract: cosine similarity
    lives in [-1, 1], so DEFAULT_SIMILARITY_SENSITIVITY must be its range, 2.0."""
    assert DEFAULT_SIMILARITY_SENSITIVITY == 2.0


_ALL_TESTS = [
    test_disabled_is_exact_passthrough,
    test_disabled_end_to_end_matches_pre_defense_pipeline,
    test_enabled_perturbs_scores,
    test_smaller_epsilon_yields_larger_expected_perturbation,
    test_invalid_epsilon_rejected,
    test_reliability_scores_unaffected_by_similarity_noise,
    test_noise_can_change_topk_order_and_matches_direct_computation,
    test_config_flag_controls_enable_disable,
    test_same_seed_and_query_gives_identical_noise,
    test_different_seed_gives_different_noise,
    test_different_query_gives_different_noise_same_seed,
    test_derive_seed_is_pure_and_stable,
    test_sensitivity_constant_matches_cosine_range,
]


if __name__ == "__main__":
    failures = 0
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_ALL_TESTS) - failures}/{len(_ALL_TESTS)} passed")
    sys.exit(1 if failures else 0)
