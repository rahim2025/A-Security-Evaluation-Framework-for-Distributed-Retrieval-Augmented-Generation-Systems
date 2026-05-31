"""Training-time attack utilities for federated RAG.

These attacks are injected into the federated learning loop (local training
phase) so that corrupted data, gradients, or model weights propagate through
FedAvg and degrade the global retriever.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class BaseTrainingAttack(ABC):
    """Base class for training-phase attacks in federated RAG.

    Subclasses override one or more hooks:

    - ``on_weights_ready`` – manipulate model weights before they are sent
      to the server (gradient/model-poisoning) or after they are received
      from the server (restore a backdoor).
    - ``on_before_local_training`` – poison local data before the client
      runs its local training loop.
    - ``on_after_local_training`` – tamper with the model state after local
      training finishes (e.g. implant a backdoor).
    """

    def on_weights_ready(
        self, weights: Sequence[np.ndarray]
    ) -> Sequence[np.ndarray]:
        """Hook called on NumPy model weights.

        Args:
            weights: List of NumPy arrays representing the current model.

        Returns:
            Manipulated weights (default: identity).
        """
        return weights

    def on_before_local_training(
        self, net: Any, dataset: Any
    ) -> Any:
        """Hook called before local training starts.

        Args:
            net: The local model (torch.nn.Module, SentenceTransformer, …).
            dataset: Training data (list, HuggingFace ``Dataset``, PyTorch
                ``Dataset``, etc.).

        Returns:
            Potentially poisoned dataset.  Implementations that do not
            touch data should return the input unchanged.
        """
        return dataset

    def on_after_local_training(self, net: Any) -> None:
        """Hook called after local training finishes.

        Args:
            net: The local model.
        """
        return

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable dict describing this attack."""
        ...
