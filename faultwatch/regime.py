"""Operating-mode (regime) normalization.

Raw sensor values move far more with load than with degradation: a gas
turbine at full power runs hundreds of degrees hotter than at idle. So we
learn what each sensor *should* read at the current operating point, from
healthy data only, and work with the standardized residual:

    z = (measured - expected_at_this_load) / healthy_spread

Degradation shows up as a persistent shift in z; load changes do not.

Optional extras for run-to-failure fleets:
  * per-asset baseline - subtract each machine's own early-life residual,
    so unit-to-unit manufacturing differences are not flagged as faults;
  * smoothing - rolling mean over time to suppress sensor noise.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DISCRETE_MAX_REGIMES = 50


class RegimeNormalizer:
    def __init__(self, sensors, regime_features=(), asset_id=None, time=None,
                 per_asset_baseline=False, baseline_window=30, smoothing_window=1,
                 regime_decimals=None):
        self.sensors = list(sensors)
        self.regime_features = list(regime_features)
        self.asset_id = asset_id
        self.time = time
        self.per_asset_baseline = per_asset_baseline
        self.baseline_window = baseline_window
        self.smoothing_window = smoothing_window
        self.regime_decimals = regime_decimals
        self.mode_ = None

    # ---- expected value at the operating point -------------------------
    def _regime_key(self, df):
        r = df[self.regime_features]
        if self.regime_decimals is not None:
            r = r.round(self.regime_decimals)
        return r.astype(str).agg("|".join, axis=1)

    def _expected(self, df):
        if not self.regime_features:
            return pd.DataFrame(np.tile(self.global_mean_.values, (len(df), 1)),
                                index=df.index, columns=self.sensors)
        if self.mode_ == "discrete":
            key = self._regime_key(df)
            exp = self.table_mean_.reindex(key.values)
            exp.index = df.index
            return exp.fillna(self.global_mean_)
        X = df[self.regime_features]
        return pd.DataFrame({s: m.predict(X) for s, m in self.models_.items()}, index=df.index)

    def _spread(self, df):
        if self.mode_ == "discrete":
            sd = self.table_std_.reindex(self._regime_key(df).values)
            sd.index = df.index
            return sd.fillna(self.global_std_)
        return pd.DataFrame(np.tile(self.global_std_.values, (len(df), 1)),
                            index=df.index, columns=self.sensors)

    # ---- fit / transform ------------------------------------------------
    def fit(self, healthy: pd.DataFrame):
        h = healthy
        self.global_mean_ = h[self.sensors].mean()
        if self.regime_features:
            n_regimes = self._regime_key(h).nunique()
            self.mode_ = "discrete" if n_regimes <= DISCRETE_MAX_REGIMES else "continuous"
        else:
            self.mode_ = "none"

        if self.mode_ == "discrete":
            key = self._regime_key(h)
            self.table_mean_ = h[self.sensors].groupby(key.values).mean()
        elif self.mode_ == "continuous":
            self.models_ = {}
            for s in self.sensors:
                m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                      min_child_samples=20, verbose=-1)
                m.fit(h[self.regime_features], h[s])
                self.models_[s] = m

        raw_resid = h[self.sensors] - self._expected(h)
        resid = self._asset_adjust(h, raw_resid)
        self.global_std_ = resid.std().replace(0, 1e-9)
        if self.mode_ == "discrete":
            key = self._regime_key(h)
            self.table_std_ = resid.groupby(key.values).std().replace(0, np.nan)
        return self

    def _asset_adjust(self, df, resid):
        if not (self.per_asset_baseline and self.asset_id and self.time):
            return resid
        order = df[self.time].groupby(df[self.asset_id]).rank(method="first")
        early = resid[order <= self.baseline_window]
        base = early.groupby(df.loc[early.index, self.asset_id]).mean()
        return resid - base.reindex(df[self.asset_id].values).set_axis(df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        resid = df[self.sensors] - self._expected(df)
        resid = self._asset_adjust(df, resid)
        z = resid / self._spread(df)
        if self.smoothing_window > 1 and self.asset_id and self.time:
            z = z.assign(_a=df[self.asset_id].values, _t=df[self.time].values)
            z = z.sort_values(["_a", "_t"])
            z[self.sensors] = (z.groupby("_a")[self.sensors]
                                 .rolling(self.smoothing_window, min_periods=1).mean()
                                 .reset_index(level=0, drop=True))
            z = z.drop(columns=["_a", "_t"]).loc[df.index]
        return z.add_prefix("z_")
