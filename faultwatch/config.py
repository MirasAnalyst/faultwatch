"""Load an asset config (YAML). One config per equipment type / dataset."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED = ["name", "dataset", "experiment", "sensors"]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path.name}: missing required keys {missing}")
    cfg.setdefault("regime_features", [])
    cfg.setdefault("asset_id", None)
    cfg.setdefault("time", None)
    cfg.setdefault("normalization", {})
    cfg.setdefault("detector", {})
    cfg.setdefault("seed", 42)
    cfg["_root"] = str(path.resolve().parent.parent)
    return cfg
