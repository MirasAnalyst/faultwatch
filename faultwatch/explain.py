"""SHAP explanations for the tree models."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .plotting import style_axes, INK, INK_2, SERIES


def shap_importance(model, X: pd.DataFrame, max_rows=2000, seed=42) -> pd.Series:
    Xs = X.sample(min(len(X), max_rows), random_state=seed)
    if hasattr(model, "coef_"):       # linear model
        sv = shap.LinearExplainer(model, Xs).shap_values(Xs)
    else:
        sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):          # older shap, multiclass
        sv = np.stack(sv, axis=-1)
    sv = np.abs(np.asarray(sv))
    if sv.ndim == 3:                  # (rows, features, classes)
        sv = sv.mean(axis=2)
    return pd.Series(sv.mean(axis=0), index=X.columns).sort_values(ascending=False)


def _pretty(name: str) -> str:
    if name.startswith("z_"):
        return f"{name[2:]} (deviation)"
    if name.startswith("slope_"):
        return f"{name[6:]} (trend)"
    return name


def plot_importance(imp: pd.Series, title: str, path, top_n=10):
    top = imp.head(top_n)[::-1]
    top.index = [_pretty(i) for i in top.index]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1.2))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("mean |SHAP value|", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
