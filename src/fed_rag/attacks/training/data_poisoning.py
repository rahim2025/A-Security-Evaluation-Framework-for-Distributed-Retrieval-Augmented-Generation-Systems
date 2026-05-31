"""Training-time data-poisoning attack.

Wraps the existing ``DataPoisoningAttack`` so it can be injected into a
client's local training loop via ``BaseTrainingAttack``.
"""

from __future__ import annotations

from typing import Any

from fed_rag.attacks.data_poisoning import DataPoisoningAttack
from fed_rag.attacks.training import BaseTrainingAttack


class TrainingDataPoisoningAttack(BaseTrainingAttack):
    """Poison local training data before federated local training.

    Supports both HuggingFace ``Dataset`` objects and plain ``list[dict]``
    datasets.  When the input is neither, the attack returns it unchanged
    so that PyTorch ``DataLoader`` datasets can be handled by custom
    subclasses if needed.
    """

    def __init__(
        self,
        poisoning_ratio: float = 0.1,
        poison_type: str = "wrong_answer",
        mode: str = "append",
        amplification_factor: int = 1,
        question_variants: int = 1,
        seed: int | None = None,
    ) -> None:
        self._attack = DataPoisoningAttack(
            poisoning_ratio=poisoning_ratio,
            poison_type=poison_type,
            mode=mode,
            amplification_factor=amplification_factor,
            question_variants=question_variants,
            seed=seed,
        )
        self._config = {
            "poisoning_ratio": poisoning_ratio,
            "poison_type": poison_type,
            "mode": mode,
            "amplification_factor": amplification_factor,
            "question_variants": question_variants,
            "seed": seed,
        }

    def on_before_local_training(self, net: Any, dataset: Any) -> Any:
        """Poison dataset before training.

        Args:
            net: The local model (unused).
            dataset: Training data.

        Returns:
            Poisoned dataset, or the original if poisoning is not
            applicable to the supplied type.
        """
        try:
            from datasets import Dataset as HFDataset

            if isinstance(dataset, HFDataset):
                poisoned_ds, _ = self._attack.execute_hf_dataset(dataset)
                return poisoned_ds
        except Exception:
            pass

        if isinstance(dataset, list):
            result = self._attack.execute(dataset)
            return result.poisoned_examples

        return dataset

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)
