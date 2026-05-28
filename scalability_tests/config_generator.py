"""
config_generator.py
-------------------
Produces CLI argument lists for each scalability experiment.
Every experiment overrides only the parameters relevant to that dimension;
everything else falls back to the default YAML files.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExperimentConfig:
    """One concrete configuration to evaluate."""
    experiment_id: str
    dimension: str        # 'volume' | 'network' | 'attack' | 'dataset'
    label: str            # human-readable x-axis label
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    data_config: str = "config/data/mmlu.yaml"  # --config path for dataset


def build_cli_args(exp: ExperimentConfig) -> List[str]:
    """Convert an ExperimentConfig into a list of CLI arguments."""
    args = ["--config", exp.data_config]
    for key, value in exp.cli_overrides.items():
        args += [f"--{key}", str(value)]
    return args


# ---------------------------------------------------------------------------
# Dimension 1 — Dataset Volume
# Vary the number of samples while keeping everything else at defaults.
# ---------------------------------------------------------------------------
VOLUME_SAMPLES = [10, 25, 50, 100, 200]

def volume_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_id=f"vol_{n}",
            dimension="volume",
            label=str(n),
            cli_overrides={"data.num_samples": n},
        )
        for n in VOLUME_SAMPLES
    ]


# ---------------------------------------------------------------------------
# Dimension 2 — Network Size (number of peers)
# Vary num_peers; num_peer_attachments is kept at 4 (default).
# ---------------------------------------------------------------------------
PEER_COUNTS = [5, 10, 20, 30, 50]

def network_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_id=f"net_{p}",
            dimension="network",
            label=str(p),
            cli_overrides={
                "rag.num_peers": p,
                "data.num_samples": 30,   # fixed small set so tests are fast
            },
        )
        for p in PEER_COUNTS
    ]


# ---------------------------------------------------------------------------
# Dimension 3 — Attack Intensity (poisoning ratio)
# Gradually increase the fraction of malicious peers.
# ---------------------------------------------------------------------------
POISON_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def attack_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_id=f"atk_{int(r*100)}pct",
            dimension="attack",
            label=f"{int(r*100)}%",
            cli_overrides={
                "data.num_samples": 30,
                "security.enable_attack": True,
                "security.poisoning_ratio": r,
                "security.attack_strategy": "high_degree",
                "security.poison_type": "answer_swap",
            },
        )
        for r in POISON_RATIOS
    ]


# ---------------------------------------------------------------------------
# Dimension 4 — Dataset Diversity (swap full datasets)
# Run the same fixed-size experiment on three different datasets.
# ---------------------------------------------------------------------------
DATASET_CONFIGS = [
    ("mmlu",    "config/data/mmlu.yaml"),
    ("medical", "config/data/medical.yaml"),
    ("news",    "config/data/news.yaml"),
]

def dataset_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_id=f"ds_{name}",
            dimension="dataset",
            label=name,
            cli_overrides={"data.num_samples": 30},
            data_config=cfg_path,
        )
        for name, cfg_path in DATASET_CONFIGS
    ]


def all_experiments() -> List[ExperimentConfig]:
    return (
        volume_experiments()
        + network_experiments()
        + attack_experiments()
        + dataset_experiments()
    )
