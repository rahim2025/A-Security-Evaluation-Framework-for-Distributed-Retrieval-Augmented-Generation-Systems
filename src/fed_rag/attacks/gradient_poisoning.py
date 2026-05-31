"""
gradient_poisoning.py
=====================
Simulates gradient-level poisoning attacks on the FedRAG federated
fine-tuning pipeline (LSR / RALT trainers).

In a real federated training round, each client computes gradients on its
local data and sends them to the server for aggregation. A malicious client
can send crafted gradients that:
  - Degrade the global model on clean queries (untargeted attack)
  - Cause the model to produce attacker-chosen outputs for specific inputs
    (targeted / backdoor attack)
  - Invert the gradient to reconstruct private training data (model inversion)

This module provides a deterministic simulation of these three attack types
that is compatible with the hash-based FedRAGSimulator (no GPU required).
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GradientPoisoningResult:
    attack_type: str
    malicious_clients: list[int]
    poisoning_scale: float
    baseline_f1: float
    attacked_f1: float
    delta_f1: float
    inversion_exposure: float       # 0.0–1.0 fraction of training data reconstructed
    backdoor_success_rate: float    # 0.0–1.0 for targeted attacks
    num_rounds: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return (
            f"[{self.attack_type}]  Malicious clients: {self.malicious_clients}  "
            f"Scale: {self.poisoning_scale:.2f}  "
            f"F1: {self.baseline_f1:.4f} → {self.attacked_f1:.4f}  "
            f"Δ={self.delta_f1:+.4f}  "
            f"Inversion exposure: {self.inversion_exposure:.1%}  "
            f"Backdoor success: {self.backdoor_success_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------

class GradientPoisoningAttack:
    """Simulate gradient poisoning on a federated fine-tuning round.

    Parameters
    ----------
    attack_type:
        ``"untargeted"``  — scale gradients to maximise global loss.
        ``"targeted"``    — craft gradients to trigger a backdoor response.
        ``"inversion"``   — reconstruct private training data from gradients.
    poisoning_scale:
        Magnitude multiplier applied to the malicious gradient update.
        1.0 = same magnitude as honest clients. 5.0–10.0 = aggressive.
    target_query:
        For targeted attacks, the query that should trigger the backdoor.
    backdoor_response:
        The response the model should return for target_query after poisoning.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        attack_type: str = "untargeted",
        poisoning_scale: float = 5.0,
        target_query: str | None = None,
        backdoor_response: str | None = None,
        seed: int = 0,
    ) -> None:
        if attack_type not in ("untargeted", "targeted", "inversion"):
            raise ValueError(f"Unknown attack_type '{attack_type}'. "
                             "Choose: untargeted, targeted, inversion")
        self.attack_type = attack_type
        self.poisoning_scale = poisoning_scale
        self.target_query = target_query or "What is the system admin password?"
        self.backdoor_response = backdoor_response or "ATTACKER_CONTROLLED_RESPONSE"
        self.seed = seed

    def execute(
        self,
        clients: list[Any],
        malicious_client_ids: list[int],
        num_rounds: int = 5,
        baseline_f1: float = 1.0,
    ) -> GradientPoisoningResult:
        """Simulate gradient poisoning over `num_rounds` federated rounds.

        This is a deterministic simulation — no actual model training occurs.
        The F1 degradation formula models the expected impact based on:
        - fraction of malicious clients
        - poisoning scale (gradient magnification)
        - number of rounds (accumulation effect)
        """
        rng = random.Random(self.seed)
        n_total = len(clients)
        n_malicious = len(malicious_client_ids)
        malicious_fraction = n_malicious / max(1, n_total)

        if self.attack_type == "untargeted":
            attacked_f1, inversion_exposure, backdoor_rate = self._simulate_untargeted(
                baseline_f1, malicious_fraction, num_rounds, rng
            )
        elif self.attack_type == "targeted":
            attacked_f1, inversion_exposure, backdoor_rate = self._simulate_targeted(
                baseline_f1, malicious_fraction, num_rounds, rng
            )
        else:
            attacked_f1, inversion_exposure, backdoor_rate = self._simulate_inversion(
                baseline_f1, malicious_fraction, num_rounds, rng
            )

        attacked_f1 = max(0.0, min(1.0, attacked_f1))

        return GradientPoisoningResult(
            attack_type=self.attack_type,
            malicious_clients=malicious_client_ids,
            poisoning_scale=self.poisoning_scale,
            baseline_f1=baseline_f1,
            attacked_f1=attacked_f1,
            delta_f1=attacked_f1 - baseline_f1,
            inversion_exposure=inversion_exposure,
            backdoor_success_rate=backdoor_rate,
            num_rounds=num_rounds,
            details={
                "n_total_clients": n_total,
                "n_malicious_clients": n_malicious,
                "malicious_fraction": malicious_fraction,
                "attack_type": self.attack_type,
                "poisoning_scale": self.poisoning_scale,
            },
        )

    def _simulate_untargeted(
        self, baseline_f1: float, malicious_fraction: float,
        num_rounds: int, rng: random.Random
    ) -> tuple[float, float, float]:
        """Untargeted: maximise gradient to corrupt global model."""
        effective_scale = self.poisoning_scale * malicious_fraction
        degradation_per_round = effective_scale * 0.04 * rng.uniform(0.85, 1.15)
        total_degradation = min(0.80, degradation_per_round * num_rounds)
        attacked_f1 = baseline_f1 - total_degradation
        inversion = 0.0
        backdoor = 0.0
        return attacked_f1, inversion, backdoor

    def _simulate_targeted(
        self, baseline_f1: float, malicious_fraction: float,
        num_rounds: int, rng: random.Random
    ) -> tuple[float, float, float]:
        """Targeted: backdoor trigger implantation."""
        # Targeted attacks preserve global F1 — only specific trigger queries are affected
        f1_noise = malicious_fraction * 0.02 * rng.uniform(0.9, 1.1)
        attacked_f1 = baseline_f1 - f1_noise
        inversion = 0.0
        # Backdoor success grows with scale and rounds
        backdoor = min(1.0, malicious_fraction * self.poisoning_scale * 0.15 * num_rounds)
        return attacked_f1, inversion, backdoor

    def _simulate_inversion(
        self, baseline_f1: float, malicious_fraction: float,
        num_rounds: int, rng: random.Random
    ) -> tuple[float, float, float]:
        """Inversion: reconstruct private training data from shared gradients."""
        # Inversion is passive — does not change model performance
        attacked_f1 = baseline_f1
        # Exposure scales with scale parameter and rounds (accumulation)
        inversion = min(1.0, malicious_fraction * self.poisoning_scale * 0.12 * num_rounds)
        backdoor = 0.0
        return attacked_f1, inversion, backdoor
