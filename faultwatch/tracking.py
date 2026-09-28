"""MLflow tracking and model registry (optional: `python run.py ... --mlflow`).

Each asset run logs its config as params, every numeric result as a metric,
the reports folder as artifacts, and registers the scoring bundle as
`faultwatch-<asset>` so a new version appears whenever the asset is retrained.
Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

DEFAULT_URI = "sqlite:///mlflow.db"


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, f"{key}.")
        elif isinstance(v, list) and v and all(isinstance(x, dict) and "method" in x for x in v):
            for row in v:          # comparison tables: one metric per method
                yield from _flatten({k2: v2 for k2, v2 in row.items() if k2 != "method"},
                                    f"{key}.{row['method']}.")
        else:
            yield key, v


def _clean(key: str) -> str:
    return re.sub(r"[^\w\-./ ]", "_", key)[:250]


class FaultWatchModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so a registered model can be served with `mlflow models serve`."""

    def load_context(self, context):
        from faultwatch.serve import Scorer
        self.scorer = Scorer.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        out = self.scorer.score(model_input)
        if "top_sensors" in out:
            out["top_sensors"] = out["top_sensors"].map(json.dumps)
        return out


def log_run(cfg: dict, metrics: dict, out_dir: Path, bundle_path: Path, register: bool = True):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment("faultwatch")
    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.set_tags({"asset": cfg["name"], "experiment_type": cfg["experiment"],
                         "dataset": cfg["dataset"]})
        params = {_clean(k): str(v)[:500] for k, v in _flatten(
            {k: v for k, v in cfg.items() if not k.startswith("_")})}
        mlflow.log_params(params)
        nums = {_clean(k): float(v) for k, v in _flatten(metrics)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        mlflow.log_metrics(nums)
        mlflow.log_artifacts(str(out_dir), artifact_path="reports")
        root = Path(__file__).resolve().parent
        mlflow.pyfunc.log_model(
            name="model", python_model=FaultWatchModel(),
            artifacts={"bundle": str(bundle_path)}, code_paths=[str(root)],
            registered_model_name=f"faultwatch-{cfg['name']}" if register else None)
        return run.info.run_id
