"""Gradient and model-weight manipulation attacks for federated training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fed_rag.attacks.training import BaseTrainingAttack


class GradientManipulationAttack(BaseTrainingAttack):
    """Manipulate outgoing model weights (gradient / model poisoning).

    This attack is applied **after** local training finishes, just before
    the client uploads its weights to the central server.  It can also be
    applied to weights **received** from the server if the client restore
    hook is configured to call ``on_weights_ready`` on incoming parameters.

    Supported operations (can be combined):
    - **scale** – multiply all weights by a scalar.
    - **noise_std** – add Gaussian noise.
    - **flip_sign** – negate all weights (Byzantine ``-1`` attack).
    """

    def __init__(
        self,
        scale: float = 1.0,
        noise_std: float = 0.0,
        flip_sign: bool = False,
        seed: int | None = None,
    ) -> None:
        self.scale = scale
        self.noise_std = noise_std
        self.flip_sign = flip_sign
        self._rng = np.random.default_rng(seed)
        self._config = {
            "scale": scale,
            "noise_std": noise_std,
            "flip_sign": flip_sign,
            "seed": seed,
        }

    def on_weights_ready(
        self, weights: Sequence[np.ndarray]
    ) -> Sequence[np.ndarray]:
        manipulated: list[np.ndarray] = []
        for w in weights:
            arr = np.array(w, dtype=np.float32)
            if self.flip_sign:
                arr = -arr
            arr = arr * self.scale
            if self.noise_std > 0.0:
                arr = arr + self._rng.normal(
                    0.0, self.noise_std, size=arr.shape
                )
            manipulated.append(arr)
        return manipulated

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)
