"""
Defense against the SSM Grounding-Farming attack (attack/ssm_score/grounding_farming_attack.py).

Attack recap
-------------
drag_llm_service/app/server.py (`_query_analyze`) decides whether a source
gets positive on-chain credit using a naive literal-substring check:

    norm_response = _norm_join(_normalize_for_match(response_text))
    selected_contexts_norm = [_norm_join(_normalize_for_match(c["text"])) for c in selected]
    grounded_by_sources = [norm_response in ctx for ctx in selected_contexts_norm]

This asks only "does the model's answer text appear somewhere inside this
source's chunk?" -- not whether that chunk had anything to do with
*producing* the answer. A source can trip this on queries it did not
meaningfully help answer by returning broad, high-coverage filler content
engineered to contain many plausible short-answer phrasings (see
.claude/ssm_grounding_farming_plan.md). Every resulting score-update
transaction is submitted by the real orchestrator, correctly authorized,
and individually within the on-chain rate/delta bounds -- defense/ssm_defense
(the transaction-layer defense for the sibling key-forgery SSM variant)
cannot see this at all, because nothing about it is a forged transaction.
The forgery is semantic, not cryptographic: the premise "this source's
content actually grounded the answer" is false, even though every byte on
the wire is honest.

Defense principle
------------------
Require a *second*, independent signal before honoring a substring match as
real grounding: was this candidate also a genuinely well-ranked, on-topic
pick for THIS specific query -- not merely one that happened to scrape into
the top-k? The retrieval/selection pipeline already computes two such
signals for every candidate, for free, with no extra inference cost:

  - rerank_score   : Reranker.rerank() / select_top_k_with_defense()'s fused
                      dense+BM25(+reliability) relevance score for this
                      query, min-max normalized to [0,1] within this
                      query's own candidate pool.
  - consensus_score : select_top_k_with_defense()'s cosine similarity to the
                      embedding centroid of candidates from OTHER sources
                      (drag_llm_service/src/retriever/defense.py) -- only
                      present when retrieval.defense.enabled is true.

Grounding-farming's own validation data (.claude/ssm_grounding_farming_plan.md
§4a) found its broad-coverage filler survives retrieval at a low rate (5-30%)
and, when it does survive, typically ranks near the bottom of the returned
top-k rather than at the top -- exactly what low rerank_score/consensus_score
would catch. Legitimate, genuinely on-topic content does not pay this cost:
it is retrieved because it is actually relevant, so it scores well above a
"barely qualified" threshold as a matter of course.

This costs nothing extra at query time (both scores are already computed),
does not touch generation, and does not change behavior for any source that
is honestly grounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GroundingVerdict:
    claimed: bool
    verified: bool
    reason: Optional[str] = None
    rerank_score: Optional[float] = None
    consensus_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "claimed": self.claimed,
            "verified": self.verified,
            "reason": self.reason,
            "rerank_score": self.rerank_score,
            "consensus_score": self.consensus_score,
        }


@dataclass
class GroundingVerificationReport:
    verdicts: List[GroundingVerdict] = field(default_factory=list)

    @property
    def downgraded_count(self) -> int:
        return sum(1 for v in self.verdicts if v.claimed and not v.verified)

    def to_dict(self) -> dict:
        return {
            "downgraded_count": self.downgraded_count,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def verify_grounding(
    grounded_by_sources: List[bool],
    selected: List[Dict[str, Any]],
    min_rerank_score: float = 0.35,
    min_consensus_score: Optional[float] = 0.35,
) -> List[bool]:
    """
    Downgrade a substring-match "grounded" claim to False unless the matched
    candidate was ALSO a well-ranked, on-topic pick for this query.

    Parameters
    ----------
    grounded_by_sources : the existing naive substring-match result, one bool
                          per entry in `selected` (same order).
    selected             : the candidates chosen by the retrieval/reranking
                          pipeline for this query, each optionally carrying
                          "rerank_score" and "consensus_score" (both already
                          computed by Reranker.rerank /
                          select_top_k_with_defense -- see module docstring).
    min_rerank_score     : [0,1]. A claimed-grounded candidate scoring below
                          this on the query-relevance axis is treated as a
                          coincidental substring hit, not real grounding.
    min_consensus_score  : [0,1] or None. Same idea on the cross-source
                          topical-agreement axis. None disables this check
                          (e.g. if retrieval.defense.enabled is false and no
                          consensus_score is ever present). Skipped per
                          candidate when that candidate carries no
                          consensus_score (consistent with the rest of the
                          pipeline, which treats "no signal" as neutral
                          rather than a reason to reject).

    Returns
    -------
    A new list of bools, same length/order as `grounded_by_sources` -- only
    ever turns True into False, never False into True.
    """
    verified: List[bool] = []
    for is_grounded, candidate in zip(grounded_by_sources, selected):
        if not is_grounded:
            verified.append(False)
            continue

        rerank_score = candidate.get("rerank_score")
        consensus_score = candidate.get("consensus_score")

        ok = True
        if rerank_score is not None and rerank_score < min_rerank_score:
            ok = False
        if ok and min_consensus_score is not None and consensus_score is not None \
                and consensus_score < min_consensus_score:
            ok = False

        verified.append(ok)
    return verified


def verify_grounding_detailed(
    grounded_by_sources: List[bool],
    selected: List[Dict[str, Any]],
    min_rerank_score: float = 0.35,
    min_consensus_score: Optional[float] = 0.35,
) -> GroundingVerificationReport:
    """Same logic as verify_grounding(), but returns a full report (reasons,
    underlying scores) for evaluation/logging rather than just the bools."""
    report = GroundingVerificationReport()
    for is_grounded, candidate in zip(grounded_by_sources, selected):
        rerank_score = candidate.get("rerank_score")
        consensus_score = candidate.get("consensus_score")

        if not is_grounded:
            report.verdicts.append(GroundingVerdict(
                claimed=False, verified=False, reason="no_substring_match",
                rerank_score=rerank_score, consensus_score=consensus_score,
            ))
            continue

        reason = None
        ok = True
        if rerank_score is not None and rerank_score < min_rerank_score:
            ok = False
            reason = f"rerank_score {rerank_score:.3f} < {min_rerank_score}"
        if ok and min_consensus_score is not None and consensus_score is not None \
                and consensus_score < min_consensus_score:
            ok = False
            reason = f"consensus_score {consensus_score:.3f} < {min_consensus_score}"

        report.verdicts.append(GroundingVerdict(
            claimed=True, verified=ok,
            reason=reason if not ok else "verified",
            rerank_score=rerank_score, consensus_score=consensus_score,
        ))
    return report
