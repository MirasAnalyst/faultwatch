"""Health-score detector trained on healthy data only.

Default is a Mahalanobis distance on the regime-normalized residuals
(the classic multivariate statistical process control T^2 approach, with a
shrinkage covariance so it stays stable with few healthy samples).
It needs no failure examples - important because most real assets have
very few recorded failures.

The alarm threshold is set on a held-out slice of healthy data at a chosen
false-alarm quantile, and an alarm must persist for `persistence`
consecutive samples before it fires (standard alarm-management practice to
avoid chattering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest


class HealthDetector:
    def __init__(self, method="mahalanobis", quantile=0.99, persistence=1, seed=42):
        self.method = method
        self.quantile = quantile
        self.persistence = persistence
        self.seed = seed

    def fit(self, Z_healthy: pd.DataFrame, calib_frac=0.3):
        Z = Z_healthy.to_numpy()
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(Z))
        n_cal = max(1, int(len(Z) * calib_frac))
        cal, fit = Z[idx[:n_cal]], Z[idx[n_cal:]]
        self.columns_ = list(Z_healthy.columns)
        if self.method == "mahalanobis":
            lw = LedoitWolf().fit(fit)
            self.mu_ = lw.location_
            self.prec_ = np.linalg.inv(lw.covariance_)
        elif self.method == "iforest":
            self.model_ = IsolationForest(n_estimators=300, random_state=self.seed).fit(fit)
        else:
            raise ValueError(f"unknown detector method {self.method}")
        self.threshold_ = float(np.quantile(self._score(cal), self.quantile))
        return self

    def _score(self, Z: np.ndarray) -> np.ndarray:
        if self.method == "mahalanobis":
            d = Z - self.mu_
            return np.einsum("ij,jk,ik->i", d, self.prec_, d)
        return -self.model_.score_samples(Z)

    def score(self, Z: pd.DataFrame) -> np.ndarray:
        return self._score(Z[self.columns_].to_numpy())

    def contributions(self, Z: pd.DataFrame) -> pd.DataFrame:
        """Per-sensor share of the Mahalanobis score (rows sum to the score).
        Answers the operator's first question: *which* sensors raised the alarm."""
        if self.method != "mahalanobis":
            raise NotImplementedError("contributions only defined for mahalanobis")
        d = Z[self.columns_].to_numpy() - self.mu_
        c = d * (d @ self.prec_)
        return pd.DataFrame(c, index=Z.index, columns=[s.removeprefix("z_") for s in self.columns_])


def first_alarm(flags: np.ndarray, times: np.ndarray, persistence: int):
    """Time of the first alarm that has stayed on for `persistence` samples."""
    run = 0
    for f, t in zip(flags, times):
        run = run + 1 if f else 0
        if run >= persistence:
            return t
    return None
