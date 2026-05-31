"""
data_split.py — IID and non-IID data partition helpers.
Import and use instead of manual slicing.
"""
import numpy as np
from typing import List


def iid_split(dataset: list, n_clients: int, seed: int = 0) -> List[list]:
    """Equal random partition — every client gets same distribution."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))
    chunks = np.array_split(idx, n_clients)
    return [[dataset[i] for i in chunk] for chunk in chunks]


def noniid_split(dataset: list, n_clients: int, alpha: float = 0.5, seed: int = 0) -> List[list]:
    """
    Dirichlet non-IID split.
    alpha=0.5 → moderately heterogeneous (realistic federated setting).
    alpha=0.1 → highly heterogeneous (each client has very different topics).
    alpha=100 → nearly IID.

    Requires dataset items to have a "label" or "category" field.
    Falls back to IID if no label field found.
    """
    rng = np.random.default_rng(seed)

    # Group by label
    label_map: dict = {}
    for i, item in enumerate(dataset):
        lbl = item.get("label") or item.get("category") or (hash(item["answer"]) % 20)
        label_map.setdefault(lbl, []).append(i)

    client_indices: List[list] = [[] for _ in range(n_clients)]

    for lbl, indices in label_map.items():
        rng.shuffle(indices)
        # Draw client proportions from Dirichlet
        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        proportions = (proportions * len(indices)).astype(int)
        # Fix rounding so we don't drop items
        diff = len(indices) - proportions.sum()
        proportions[proportions.argmax()] += diff
        # Assign
        cursor = 0
        for c, count in enumerate(proportions):
            client_indices[c].extend(indices[cursor:cursor + count])
            cursor += count

    return [[dataset[i] for i in sorted(idxs)] for idxs in client_indices]
