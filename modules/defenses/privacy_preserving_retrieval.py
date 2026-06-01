from typing import Any, Dict

from loguru import logger


class PrivacyPreservingRetrieval:
    """
    Privacy-Preserving Retrieval defense against Membership Inference Attacks.

    Injects calibrated Laplace noise into per-peer retrieval similarity scores at
    query time.  This degrades the fine-grained membership signal an adversary can
    extract (semantic similarity, which drives 60 % of the MIA score) while
    preserving RAG utility at small noise scales because the confidence-threshold
    gate in peer.py still passes genuine top-matches most of the time.

    Usage (via simulator.py):
        defense = PrivacyPreservingRetrieval(noise_scale=0.1)
        defense.apply(network)   # before evaluation
        ...
        defense.remove(network)  # optional cleanup / restore
    """

    def __init__(self, noise_scale: float = 0.1, noise_mechanism: str = "laplace"):
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0.0")
        self.noise_scale = noise_scale
        self.noise_mechanism = noise_mechanism
        self._original_scales: Dict[int, float] = {}
        self._peers_protected = 0

    def apply(self, network) -> None:
        """Set noise_scale on every peer KB in the network."""
        self._original_scales.clear()
        count = 0
        for i, peer in enumerate(network.peers):
            if peer is not None:
                self._original_scales[i] = getattr(peer.knowledge_base, "noise_scale", 0.0)
                peer.knowledge_base.noise_scale = self.noise_scale
                count += 1
        self._peers_protected = count
        logger.info(
            f"PrivacyPreservingRetrieval applied: mechanism={self.noise_mechanism}, "
            f"noise_scale={self.noise_scale}, peers={count}"
        )

    def remove(self, network) -> None:
        """Restore original noise_scale values on all peers."""
        for i, peer in enumerate(network.peers):
            if peer is not None:
                peer.knowledge_base.noise_scale = self._original_scales.get(i, 0.0)
        self._original_scales.clear()
        logger.info("PrivacyPreservingRetrieval removed from network.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": "PrivacyPreservingRetrieval",
            "noise_mechanism": self.noise_mechanism,
            "noise_scale": self.noise_scale,
            "peers_protected": self._peers_protected,
        }
