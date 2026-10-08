"""Supervised layers on top of the residuals: fault classification,
severity estimation and remaining-useful-life (RUL) regression.

Model families: LightGBM ('gbm', default), XGBoost ('xgb'), CatBoost
('catboost') and a linear baseline. `faultwatch.selection` compares them
with grouped cross-validation on the training machines only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

GBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1)
MODEL_KINDS = ("gbm", "xgb", "catboost", "logistic")


def classifier(seed=42):
    return lgb.LGBMClassifier(random_state=seed, class_weight="balanced", **GBM_PARAMS)


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Lets XGBoost / CatBoost take string class labels and the same class
    balancing as the LightGBM classifier (`class_weight="balanced"`).
    `estimator_` is the fitted booster, for SHAP."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.utils.class_weight import compute_sample_weight
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = clone(self.estimator).fit(
            X, codes, sample_weight=compute_sample_weight("balanced", codes))
        return self

    def predict_proba(self, X):
        return np.asarray(self.estimator_.predict_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _xgb(task, seed):
    import xgboost as xgb
    kw = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=-1)
    return xgb.XGBClassifier(**kw) if task == "classifier" else xgb.XGBRegressor(**kw)


def _catboost(task, seed):
    import catboost as cb
    kw = dict(iterations=400, learning_rate=0.08, depth=6, random_seed=seed, verbose=0,
              allow_writing_files=False, thread_count=-1)
    return cb.CatBoostClassifier(**kw) if task == "classifier" else cb.CatBoostRegressor(**kw)


def make_classifier(kind="gbm", seed=42):
    """'gbm' (default), 'xgb', 'catboost', or 'logistic' - a linear model on a
    few physically meaningful features, for assets with only a handful of
    fault examples."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    if kind == "xgb":
        return EncodedClassifier(_xgb("classifier", seed))
    if kind == "catboost":
        return EncodedClassifier(_catboost("classifier", seed))
    if kind == "gbm":
        return classifier(seed)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def regressor(seed=42):
    return lgb.LGBMRegressor(random_state=seed, **GBM_PARAMS)


def make_regressor(kind="gbm", seed=42):
    """'gbm' (default), 'xgb', 'catboost' or 'linear' (standardized ridge)."""
    if kind == "linear":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "xgb":
        return _xgb("regressor", seed)
    if kind == "catboost":
        return _catboost("regressor", seed)
    if kind == "gbm":
        return regressor(seed)
    raise ValueError(f"unknown model kind {kind!r}")


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
