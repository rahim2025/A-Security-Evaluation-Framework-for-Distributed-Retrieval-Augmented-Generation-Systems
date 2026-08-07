"""
attack/selective_forward_sim/config_loader.py

Tiny YAML config loader shared by the SFA simulation attack/defense
runners. Existing attack/defense modules in this repo (Mia_attack,
selective_forward, ssm_score, ...) parameterize purely via argparse
flags + module-level constants; this module is the one place in the
repo that reads config/*.yaml, so CLI flags for selective_forward_sim
can override YAML defaults instead of duplicating them.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CONFIG_DIR = os.path.join(ROOT, "config")


def load_yaml(name: str) -> Dict[str, Any]:
    """Load a YAML file by name (resolved against config/) or absolute path."""
    path = name if os.path.isabs(name) else os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(cfg: Dict[str, Any], *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
