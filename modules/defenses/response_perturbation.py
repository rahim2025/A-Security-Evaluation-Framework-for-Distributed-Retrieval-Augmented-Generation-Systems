import json
import random
import string
from typing import Dict, Any
from loguru import logger


class ResponsePerturbation:
    """
    Response-perturbation defense against Knowledge Base Extraction attacks.

    The attack extracts exact Datapoint JSON from the ``relevant_knowledge``
    field of RAGAnswer.  This defense intercepts that JSON and corrupts the
    ``answer`` field before it leaves the network, breaking exact-match
    extraction while leaving the LLM-generated answer (already computed from
    the original data) completely untouched.

    Attack vector mitigated:
        - exact_matches  : 26  → 0   (any character change breaks exact match)
        - avg_ss_extracted_only : 1.0 → <0.7  (semantic drift from noise)
        - avg_eed_extracted_only: 0.0 → >0.2  (character-level edit distance rises)
        - crr            : 0.52 → lower (fewer chunks pass SS / EED thresholds)

    Perturbation is applied to the JSON payload only; the peer's LLM-generated
    answer that the end-user actually receives is never touched.
    """

    NOISE_CHARS = string.ascii_letters + string.digits + " .,;:!?"

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', True)
        self.perturbation_level = float(config.get('perturbation_level', 0.15))
        self.mode = config.get('mode', 'noise')   # 'noise' | 'mask' | 'truncate'

        # Statistics
        self._total_perturbed = 0
        self._total_skipped = 0

        logger.info("ResponsePerturbation initialized:")
        logger.info(f"  - Mode              : {self.mode}")
        logger.info(f"  - Perturbation level: {self.perturbation_level:.0%}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def perturb(self, relevant_knowledge_json: str) -> str:
        """
        Return a perturbed copy of the relevant_knowledge JSON string.

        Only the ``answer`` field is modified so that question-based routing
        and semantic search still work correctly for legitimate users.

        Args:
            relevant_knowledge_json: JSON string representing a Datapoint.

        Returns:
            Modified JSON string with answer field perturbed.
        """
        if not self.enabled or not relevant_knowledge_json:
            self._total_skipped += 1
            return relevant_knowledge_json

        try:
            dp = json.loads(relevant_knowledge_json)
        except (json.JSONDecodeError, ValueError):
            self._total_skipped += 1
            return relevant_knowledge_json

        original_answer = dp.get('answer', '')
        if not original_answer:
            self._total_skipped += 1
            return relevant_knowledge_json

        dp['answer'] = self._perturb_answer(original_answer)
        self._total_perturbed += 1

        logger.debug(
            f"[Perturbation] answer: '{original_answer[:40]}' → '{dp['answer'][:40]}'"
        )
        return json.dumps(dp)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _perturb_answer(self, answer: str) -> str:
        """Apply the configured perturbation strategy to an answer string."""
        if self.mode == 'mask':
            return self._mask(answer)
        elif self.mode == 'truncate':
            return self._truncate(answer)
        else:  # default: 'noise'
            return self._noise(answer)

    def _noise(self, text: str) -> str:
        """Replace a fraction of characters with random printable characters."""
        chars = list(text)
        n = max(1, int(len(chars) * self.perturbation_level))
        positions = random.sample(range(len(chars)), min(n, len(chars)))
        for pos in positions:
            replacement = random.choice(self.NOISE_CHARS)
            # Avoid replacing with the same character (no-op)
            while replacement == chars[pos] and len(self.NOISE_CHARS) > 1:
                replacement = random.choice(self.NOISE_CHARS)
            chars[pos] = replacement
        return ''.join(chars)

    def _mask(self, text: str) -> str:
        """Replace a fraction of characters with '*'."""
        chars = list(text)
        n = max(1, int(len(chars) * self.perturbation_level))
        positions = random.sample(range(len(chars)), min(n, len(chars)))
        for pos in positions:
            chars[pos] = '*'
        return ''.join(chars)

    def _truncate(self, text: str) -> str:
        """Truncate the answer to (1 - perturbation_level) of its length."""
        keep = max(1, int(len(text) * (1.0 - self.perturbation_level)))
        return text[:keep]

    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        total = self._total_perturbed + self._total_skipped
        return {
            'name': 'ResponsePerturbation',
            'enabled': self.enabled,
            'mode': self.mode,
            'perturbation_level': self.perturbation_level,
            'total_responses': total,
            'perturbed_count': self._total_perturbed,
            'skipped_count': self._total_skipped,
            'perturbation_rate': self._total_perturbed / max(1, total),
        }
