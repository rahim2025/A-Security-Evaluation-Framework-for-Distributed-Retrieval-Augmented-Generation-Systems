"""Common functions that can be shared across different tasks.
"""
import os
import gzip
import json
import jsonlines
from typing import Dict, Iterable, List, Text
import torch
import re

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def enumerate_resume(dataset, results_path):
    """
    generator that returns the item and the index in the dataset.
    if the results_path exists, it will skip all items that have been processed before.
    """
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item

def _normalize_for_match(s: str) -> str:
    return "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in str(s)).split()

def _norm_join(tokens: List[str]) -> str:
    return " ".join([t for t in tokens if t])

def move_to_device(obj, device):
    """
    """
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(v, device) for v in obj])
    elif isinstance(obj, set):
        return set([move_to_device(v, device) for v in obj])
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj