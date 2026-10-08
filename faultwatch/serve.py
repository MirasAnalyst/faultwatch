"""Trained-model bundles and online scoring.

Each experiment saves one bundle per asset type to `models/<name>.joblib`:
the regime normalizer, health detector and whichever supervised models the
asset has (fault classifier, severity or remaining-life regressors), plus a
healthy reference used by the drift monitor. The API, the dashboard and the
MLflow registry all load models through `Scorer`, so they score exactly like
the offline experiments.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models import rul_features

MODELS_DIR = "models"
PSI_BINS = 10
DRIFT_PSI = 0.25            # a residual whose distribution moved this much -> stale baseline
DRIFT_OUT_OF_ENVELOPE = 0.05  # share of samples at operating points never seen in training


def _psi_edges(x: np.ndarray) -> np.ndarray:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, PSI_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def psi(ref_edges: np.ndarray, ref_share: np.ndarray, x: np.ndarray) -> float:
    """Population stability index of `x` against a reference histogram."""
    cur = np.histogram(x, ref_edges)[0] / max(len(x), 1)
    r, c = np.clip(ref_share, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def healthy_reference(df_healthy: pd.DataFrame, Z_train: pd.DataFrame, detector, regime_features) -> dict:
    """What 'normal' looked like at training time, for the drift monitor:
    residuals of training samples the detector calls normal (the same filter
    `Scorer.drift` applies to new data) and the healthy operating envelope."""
    ref = {"z": {}, "envelope": {}}
    Z_ok = Z_train[detector.score(Z_train) <= detector.threshold_]
    for c in Z_ok.columns:
        x = Z_ok[c].dropna().to_numpy()
        edges = _psi_edges(x)
        ref["z"][c] = (edges, np.histogram(x, edges)[0] / len(x))
    for f in regime_features:
        ref["envelope"][f] = tuple(df_healthy[f].quantile([0.005, 0.995]).astype(float))
    return ref


def save_bundle(cfg: dict, root: Path, **parts) -> Path:
    path = Path(root) / MODELS_DIR / f"{cfg['name']}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    joblib.dump({"cfg": clean, **parts}, path)
    return path


class Scorer:
    """Score raw sensor rows for one asset type."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.cfg = bundle["cfg"]
        self.norm = bundle["normalizer"]
        self.det = bundle["detector"]

    @classmethod
    def load(cls, path) -> Scorer:
        return cls(joblib.load(path))

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def required_columns(self) -> list[str]:
        cols = list(self.cfg["sensors"]) + list(self.cfg["regime_features"])
        if (self.b.get("rul_model") is not None or self.norm.per_asset_baseline
                or self.norm.smoothing_window > 1):
            cols += [self.cfg["asset_id"], self.cfg["time"]]
        cols += list(self.b.get("component_features", []))
        return list(dict.fromkeys(cols))

    def _check(self, df: pd.DataFrame):
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns for {self.name}: {missing}")

    def score(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """One output row per input row: health score, alarm, the sensors
        driving it and, when the asset has them, diagnosis / severity / RUL."""
        self._check(df)
        df = df.reset_index(drop=True)
        Z = self.norm.transform(df)
        s = self.det.score(Z)
        out = pd.DataFrame({"health_score": s, "threshold": self.det.threshold_,
                            "alarm": s > self.det.threshold_})
        if self.det.method == "mahalanobis":
            c = self.det.contributions(Z).clip(lower=0)
            share = c.div(c.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            out["top_sensors"] = [
                {k: round(float(v), 3) for k, v in row.nlargest(top_n).items()}
                for _, row in share.iterrows()]
        a, t = self.cfg.get("asset_id"), self.cfg.get("time")
        if self.det.persistence > 1 and a in df and t in df:
            # alarm must stay on for `persistence` consecutive samples of the same machine
            order = df.sort_values([a, t]).index
            al, asset = out.loc[order, "alarm"], df.loc[order, a]
            streak = ((al != al.shift()) | (asset != asset.shift())).cumsum()
            run = al.astype(int).groupby(streak).cumsum()
            out["alarm_confirmed"] = (al & (run >= self.det.persistence)).reindex(out.index)
        if self.b.get("classifier") is not None:
            X = pd.concat([Z, df[self.cfg["regime_features"]]], axis=1)[self.b["classifier_features"]]
            proba = self.b["classifier"].predict_proba(X)
            classes = self.b["classifier"].classes_
            out["diagnosis"] = classes[proba.argmax(axis=1)]
            out["diagnosis_confidence"] = proba.max(axis=1).round(3)
            for comp, m in self.b.get("severity_models", {}).items():
                out[f"severity_{comp}"] = m.predict(X)
        for comp, m in self.b.get("component_models", {}).items():
            # graded condition of each component, and P(component at a critical grade)
            P = pd.DataFrame(m.predict_proba(df[self.b["component_features"]]), columns=m.classes_)
            out[f"condition_{comp}"] = P.columns[P.to_numpy().argmax(axis=1)]
            out[f"p_critical_{comp}"] = P[self.b["component_critical"][comp]].sum(axis=1).round(3).values
        if self.b.get("rul_model") is not None:
            X = rul_features(Z, df, self.det, a, t, self.b["trend_window"])[self.b["rul_features"]]
            out["rul"] = np.clip(self.b["rul_model"].predict(X), 0, self.b["rul_cap"]).round(1)
        return out

    def drift(self, df: pd.DataFrame) -> dict:
        """Is the healthy baseline still valid? Looks only at samples the
        detector calls normal, so a real fault is not mistaken for drift."""
        self._check(df)
        df = df.reset_index(drop=True)
        ref = self.b["reference"]
        Z = self.norm.transform(df)
        normal = self.det.score(Z) <= self.det.threshold_
        sensor_psi = {c.removeprefix("z_"): round(psi(*ref["z"][c], Z.loc[normal, c].dropna().to_numpy()), 3)
                      for c in Z.columns} if normal.sum() >= 50 else {}
        outside = np.zeros(len(df), dtype=bool)
        for f, (lo, hi) in ref["envelope"].items():
            outside |= ((df[f] < lo) | (df[f] > hi)).to_numpy()
        worst = max(sensor_psi.items(), key=lambda kv: kv[1]) if sensor_psi else (None, 0.0)
        reasons = []
        if worst[1] > DRIFT_PSI:
            reasons.append(f"{worst[0]} residual shifted on normal data (PSI {worst[1]:.2f})")
        if outside.mean() > DRIFT_OUT_OF_ENVELOPE:
            reasons.append(f"{outside.mean():.0%} of samples outside the trained operating envelope")
        return {"samples": int(len(df)), "samples_judged_normal": int(normal.sum()),
                "sensor_psi": sensor_psi, "out_of_envelope_share": round(float(outside.mean()), 3),
                "retrain_recommended": bool(reasons), "reasons": reasons}


POLICY_KEYS = ("safety", "copilot")   # engineering policy, editable without retraining


def load_all(root: Path | str = ".") -> dict[str, Scorer]:
    """Every trained bundle under models/. The FMECA (`safety`) and copilot
    vocabulary are taken from the current configs/*.yaml, so a safety engineer
    can change a severity or redundancy without retraining the model."""
    import yaml
    scorers = {p.stem: Scorer.load(p) for p in sorted((Path(root) / MODELS_DIR).glob("*.joblib"))}
    for f in sorted((Path(root) / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text()) or {}
        s = scorers.get(cfg.get("name"))
        if s is not None:
            for k in POLICY_KEYS:
                if k in cfg:
                    s.cfg[k] = cfg[k]
    return scorers
