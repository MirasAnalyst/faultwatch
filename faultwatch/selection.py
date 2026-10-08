"""Model selection by grouped cross-validation on the training machines.

Folds are split by machine / degradation state / recording, never by row, so
a candidate cannot win by recognising a machine it has already seen. The
comparison is reported for every asset; the configured model is replaced by
the CV winner only when the config says `model_selection: {apply: true}`.

    model_selection:
      candidates: [gbm, xgb, catboost, logistic]
      folds: 5
      apply: false
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .models import make_classifier, make_regressor


def _score(task, y, p):
    if task == "classifier":
        return float(f1_score(y, p, average="macro"))
    return float(-np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))   # higher is better


def compare_models(task: str, X: pd.DataFrame, y: pd.Series, groups, candidates, folds=5,
                   seed=42) -> pd.DataFrame:
    """One row per candidate: mean and spread of the CV metric (macro-F1 for
    classifiers, RMSE for regressors) and fit time."""
    make = make_classifier if task == "classifier" else make_regressor
    groups = np.asarray(groups)
    k = min(folds, len(np.unique(groups)))
    rows = []
    for kind in candidates:
        scores, t0 = [], time.perf_counter()
        for fi, vi in GroupKFold(n_splits=k).split(X, y, groups):
            m = make(kind, seed).fit(X.iloc[fi], y.iloc[fi])
            scores.append(_score(task, y.iloc[vi], m.predict(X.iloc[vi])))
        metric = "macro_f1" if task == "classifier" else "rmse"
        s = np.array(scores) if task == "classifier" else -np.array(scores)
        rows.append({"model": kind, f"cv_{metric}": round(float(s.mean()), 4),
                     f"cv_{metric}_std": round(float(s.std()), 4),
                     "fit_seconds": round((time.perf_counter() - t0) / k, 2)})
    df = pd.DataFrame(rows)
    col = df.columns[1]
    return df.sort_values(col, ascending=(task != "classifier")).reset_index(drop=True)


def choose(task: str, X, y, groups, cfg: dict, default: str):
    """Returns (model kind to use, comparison table or None)."""
    ms = cfg.get("model_selection")
    if not ms:
        return default, None
    table = compare_models(task, X, y, groups, ms.get("candidates", ["gbm", "xgb", "catboost"]),
                           ms.get("folds", 5), cfg.get("seed", 42))
    best = table["model"].iloc[0]
    return (best if ms.get("apply", False) else default), table


def as_metrics(table: pd.DataFrame | None, used: str) -> dict | None:
    if table is None:
        return None
    return {"used": used, "winner_by_cv": table["model"].iloc[0],
            "candidates": table.to_dict("records")}
