"""Supervised layers on top of the residuals: fault classification,
severity estimation and remaining-useful-life (RUL) regression."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

GBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1)


def classifier(seed=42):
    return lgb.LGBMClassifier(random_state=seed, class_weight="balanced", **GBM_PARAMS)


def regressor(seed=42):
    return lgb.LGBMRegressor(random_state=seed, **GBM_PARAMS)


def add_trend_features(Z: pd.DataFrame, asset: pd.Series, time: pd.Series, window: int):
    """Rolling slope of each residual over the last `window` samples per asset -
    degradation is a trend, not just a level."""
    cols = list(Z.columns)
    df = Z.assign(_a=asset.values, _t=time.values).sort_values(["_a", "_t"])
    t = df["_t"].astype(float)
    out = {}
    g = df.groupby("_a")
    tm = g["_t"].rolling(window, min_periods=3).mean().reset_index(level=0, drop=True)
    tt = (t * t).groupby(df["_a"]).rolling(window, min_periods=3).mean().reset_index(level=0, drop=True)
    var_t = (tt - tm ** 2).replace(0, np.nan)
    for c in cols:
        xm = g[c].rolling(window, min_periods=3).mean().reset_index(level=0, drop=True)
        xt = (df[c] * t).groupby(df["_a"]).rolling(window, min_periods=3).mean().reset_index(level=0, drop=True)
        out[f"slope_{c.removeprefix('z_')}"] = ((xt - xm * tm) / var_t).fillna(0.0)
    return pd.DataFrame(out, index=df.index).loc[Z.index]


def rul_features(Z: pd.DataFrame, df: pd.DataFrame, detector, asset_id: str, time: str,
                 trend_window: int) -> pd.DataFrame:
    """Inputs to the remaining-life model: residual levels, residual trends,
    the health score and machine age. Shared by training and serving."""
    X = pd.concat([Z, add_trend_features(Z, df[asset_id], df[time], trend_window)], axis=1)
    X["health_score"] = np.log1p(detector.score(Z))
    X["cycle"] = df[time].values
    return X


def nasa_score(y_true, y_pred):
    """Asymmetric PHM08 score: predicting failure too late is punished more."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
