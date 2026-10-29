"""Minimal checkpointing: params + observation statistics + config as a pickle of NumPy trees.

Ported from crazyflow-gate-ppo (``gate_ppo/checkpoint.py``). :func:`checkpoint_hash` gives the
content hash that every demonstration episode records for reproducibility (spec §3.1, item 6).
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def save_checkpoint(path: str | Path, train_state: Any, config: dict, extra: dict | None = None):
    """Write ``train_state.params`` / ``obs_stats`` (as NumPy) plus the run config to ``path``."""
    payload = {
        "params": jax.tree.map(np.asarray, train_state.params),
        "obs_stats": {
            "mean": np.asarray(train_state.obs_stats.mean),
            "var": np.asarray(train_state.obs_stats.var),
            "count": float(train_state.obs_stats.count),
        },
        "n_updates": int(train_state.n_updates),
        "config": config,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def checkpoint_hash(path: str | Path) -> str:
    """SHA-256 of the checkpoint file (first 16 hex digits)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
