"""Shared helpers: project paths, config loading, seeding, JSON IO."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def p(rel: str | os.PathLike) -> Path:
    """Resolve a project-relative path."""
    return ROOT / rel


def load_meta() -> dict:
    with open(p("configs/watershed_meta.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cfg() -> dict:
    with open(p("configs/experiment_config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prof = os.environ.get("XS_PROFILE")
    if prof:
        cfg["profile"] = prof
    return cfg


def budget(cfg: dict, section: str) -> dict:
    return cfg[section][cfg["profile"]]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _Enc(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, rel: str) -> Path:
    path = p(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=_Enc)
    return path


def load_json(rel: str):
    with open(p(rel), encoding="utf-8") as f:
        return json.load(f)
