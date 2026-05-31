"""Network-level defenses for distributed / federated RAG systems.

These defenses mirror DRAG's cross-peer validation and response-perturbation
mechanisms but are expressed in FedRAG's store-agnostic abstractions so they
can be reused across architectures.
"""

from __future__ import annotations

import json
import random
import string
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from fed_rag.data_structures.knowledge_node import KnowledgeNode


class CrossPeerValidation:
    """Validate an answer by comparing responses from multiple peers / clients.

    This is a port of DRAG's ``CrossPeerValidation`` defence to FedRAG's
    client-agnostic interface.  The caller provides a list of peer answers
    (strings) and the validator returns whether the candidate matches the
    majority view.
    """

    def __init__(
        self,
        min_agreement_ratio: float = 0.6,
        voting_method: str = "majority",
        min_peers: int = 3,
        use_similarity: bool = True,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.min_agreement_ratio = min_agreement_ratio
        self.voting_method = voting_method
        self.min_peers = min_peers
        self.use_similarity = use_similarity
        self.similarity_threshold = similarity_threshold

        # statistics
        self.total_validations = 0
        self.blocked_answers = 0
        self.passed_answers = 0
        self._confidence_sum = 0.0

    def validate(
        self,
        candidate_answer: str,
        peer_answers: Sequence[str],
    ) -> tuple[bool, float, dict[str, Any]]:
        """Validate *candidate_answer* against *peer_answers*.

        Returns:
            ``(is_valid, confidence, details_dict)``
        """
        self.total_validations += 1

        if len(peer_answers) < self.min_peers:
            self.passed_answers += 1
            self._confidence_sum += 0.5
            return (
                True,
                0.5,
                {
                    "reason": f"insufficient_peers({len(peer_answers)}<{self.min_peers})",
                    "validation_skipped": True,
                },
            )

        is_valid, confidence, details = self._majority_voting(
            candidate_answer, list(peer_answers)
        )

        if is_valid:
            self.passed_answers += 1
        else:
            self.blocked_answers += 1
        self._confidence_sum += confidence

        details["num_peers"] = len(peer_answers)
        return is_valid, confidence, details

    def get_stats(self) -> dict[str, Any]:
        total = self.total_validations
        return {
            "name": "CrossPeerValidation",
            "total_validations": total,
            "blocked_answers": self.blocked_answers,
            "passed_answers": self.passed_answers,
            "block_rate": self.blocked_answers / max(1, total),
            "avg_confidence": self._confidence_sum / max(1, total),
        }

    # ------------------------------------------------------------------
    # voting internals
    # ------------------------------------------------------------------

    def _majority_voting(
        self, candidate: str, peer_answers: list[str]
    ) -> tuple[bool, float, dict[str, Any]]:
        groups = self._group_similar(peer_answers) if self.use_similarity else {
            ans: [ans] for ans in set(peer_answers)
        }
        group_counts = {key: len(members) for key, members in groups.items()}
        if not group_counts:
            return False, 0.0, {"reason": "no_valid_groups"}

        majority_key = max(group_counts, key=group_counts.get)
        majority_count = group_counts[majority_key]
        agreement = majority_count / len(peer_answers)

        candidate_in_majority = self._similar(
            candidate, majority_key
        ) if self.use_similarity else candidate.strip().lower() == majority_key.strip().lower()

        has_clear_majority = agreement > 0.5 or (
            majority_count >= 2 and agreement >= self.min_agreement_ratio
        )

        if not has_clear_majority:
            return True, 0.5, {
                "reason": "no_clear_majority",
                "agreement_ratio": agreement,
                "distribution": dict(group_counts),
                "candidate_accepted_by_default": True,
            }

        if candidate_in_majority and agreement >= self.min_agreement_ratio:
            return True, agreement, {
                "reason": "majority_agreement",
                "agreement_ratio": agreement,
                "majority_answer": majority_key,
                "distribution": dict(group_counts),
            }
        return False, 1.0 - agreement, {
            "reason": "differs_from_majority",
            "agreement_ratio": agreement,
            "majority_answer": majority_key,
            "candidate_answer": candidate,
            "distribution": dict(group_counts),
        }

    def _group_similar(self, answers: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for answer in answers:
            placed = False
            for key in groups:
                if self._similar(answer, key):
                    groups[key].append(answer)
                    placed = True
                    break
            if not placed:
                groups[answer] = [answer]
        return groups

    def _similar(self, a: str, b: str) -> bool:
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()
        if a_norm == b_norm:
            return True
        return SequenceMatcher(None, a_norm, b_norm).ratio() >= self.similarity_threshold


class ResponsePerturbation:
    """Perturb retrieved knowledge to thwart exact-match extraction attacks.

    This is the FedRAG port of DRAG's ``ResponsePerturbation`` defence.  It
    mutates the *answer* field inside a ``KnowledgeNode``'s metadata before
    the node is returned to the caller, while leaving the node text content
    (used by the generator) untouched.
    """

    NOISE_CHARS = string.ascii_letters + string.digits + " .,;:!?"

    def __init__(
        self,
        perturbation_level: float = 0.15,
        mode: str = "noise",
        seed: int | None = None,
    ) -> None:
        self.perturbation_level = float(perturbation_level)
        self.mode = mode
        self._rng = random.Random(seed)

        self._total_perturbed = 0
        self._total_skipped = 0

    def perturb_node(self, node: KnowledgeNode) -> KnowledgeNode:
        """Return a shallow copy of *node* with a perturbed answer in metadata."""
        answer = node.metadata.get("answer", "") if node.metadata else ""
        if not answer:
            self._total_skipped += 1
            return node

        new_meta = dict(node.metadata) if node.metadata else {}
        new_meta["answer"] = self._perturb_text(answer)
        self._total_perturbed += 1

        return KnowledgeNode(
            node_type=node.node_type,
            text_content=node.text_content,
            node_id=node.node_id,
            embedding=node.embedding,
            metadata=new_meta,
        )

    def perturb_json(self, payload: str) -> str:
        """Perturb the ``answer`` field of a JSON string (DRAG-compat mode)."""
        if not payload:
            self._total_skipped += 1
            return payload
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            self._total_skipped += 1
            return payload

        original = data.get("answer", "")
        if not original:
            self._total_skipped += 1
            return payload

        data["answer"] = self._perturb_text(original)
        self._total_perturbed += 1
        return json.dumps(data)

    def get_stats(self) -> dict[str, Any]:
        total = self._total_perturbed + self._total_skipped
        return {
            "name": "ResponsePerturbation",
            "mode": self.mode,
            "perturbation_level": self.perturbation_level,
            "perturbed_count": self._total_perturbed,
            "skipped_count": self._total_skipped,
            "perturbation_rate": self._total_perturbed / max(1, total),
        }

    # ------------------------------------------------------------------
    # text perturbation strategies
    # ------------------------------------------------------------------

    def _perturb_text(self, text: str) -> str:
        if self.mode == "mask":
            return self._mask(text)
        if self.mode == "truncate":
            return self._truncate(text)
        return self._noise(text)

    def _noise(self, text: str) -> str:
        chars = list(text)
        n = max(1, int(len(chars) * self.perturbation_level))
        positions = self._rng.sample(range(len(chars)), min(n, len(chars)))
        for pos in positions:
            replacement = self._rng.choice(self.NOISE_CHARS)
            while replacement == chars[pos] and len(self.NOISE_CHARS) > 1:
                replacement = self._rng.choice(self.NOISE_CHARS)
            chars[pos] = replacement
        return "".join(chars)

    def _mask(self, text: str) -> str:
        chars = list(text)
        n = max(1, int(len(chars) * self.perturbation_level))
        for pos in self._rng.sample(range(len(chars)), min(n, len(chars))):
            chars[pos] = "*"
        return "".join(chars)

    def _truncate(self, text: str) -> str:
        keep = max(1, int(len(text) * (1.0 - self.perturbation_level)))
        return text[:keep]
