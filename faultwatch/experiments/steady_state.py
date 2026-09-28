"""Experiment type 1: labeled degradation states observed across many loads.

Fits: regime normalizer + health detector (healthy data only), fault
classifier, severity regressors. Compares the detector with the alarms a
plant typically has today.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit

from ..anomaly import HealthDetector
from ..data import LOADERS
from ..explain import plot_importance, shap_importance
from ..models import classifier, regressor
from ..plotting import INK, INK_2, SERIES, style_axes
from ..regime import RegimeNormalizer

BINS = [0.2, 0.4, 0.6, 0.8, 1.0001]
BIN_LABELS = ["20-40%", "40-60%", "60-80%", "80-100%"]


def label_states(df, cfg):
    sev = pd.DataFrame({
        comp: (spec["new"] - df[spec["column"]]) / (spec["new"] - spec["worst"])
        for comp, spec in cfg["components"].items()
    }).clip(0, 1)
    faulty = sev >= cfg["fault_min_severity"]
    n = faulty.sum(axis=1)
    cls = np.where(n == 0, "healthy",
                   np.where(n == 1, faulty.idxmax(axis=1), "combined"))
    healthy = (sev < cfg["healthy_max_severity"]).all(axis=1)
    return sev.max(axis=1), pd.Series(cls, index=df.index), healthy


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    df = LOADERS[cfg["dataset"]](root / cfg["data_dir"])
    sensors, regime = cfg["sensors"], cfg["regime_features"]
    severity, fault_class, healthy = label_states(df, cfg)
    df = df.assign(severity=severity, fault_class=fault_class, healthy=healthy)

    # Split by degradation state so the same machine condition never appears
    # in both train and test (it appears once per load level).
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_fraction"], random_state=cfg["seed"])
    tr_idx, te_idx = next(gss.split(df, groups=df[cfg["asset_id"]]))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    h_tr = tr[tr.healthy]

    # ---- 1. anomaly detection ------------------------------------------
    norm = RegimeNormalizer(sensors, regime).fit(h_tr)
    Z_tr, Z_te = norm.transform(tr), norm.transform(te)
    det_cfg = cfg["detector"]
    det = HealthDetector(det_cfg.get("method", "mahalanobis"), det_cfg.get("quantile", 0.99),
                         seed=cfg["seed"]).fit(Z_tr[tr.healthy])

    # comparison: same detector without operating-mode awareness
    flat = RegimeNormalizer(sensors, []).fit(h_tr)
    det_flat = HealthDetector("mahalanobis", det_cfg.get("quantile", 0.99),
                              seed=cfg["seed"]).fit(flat.transform(h_tr))

    bs = cfg["baseline_alarm_sensor"]
    fixed_thr = h_tr[bs].mean() + 3 * h_tr[bs].std()

    alarms = {
        f"Fixed {bs} high alarm": (te[bs] > fixed_thr).to_numpy(),
        f"Load-aware {bs} alarm": (np.abs(Z_te[f"z_{bs}"]) > 3).to_numpy(),
        "Multivariate, not load-aware": det_flat.score(flat.transform(te)) > det_flat.threshold_,
        "FaultWatch (multivariate, load-aware)": det.score(Z_te) > det.threshold_,
    }
    te_bins = pd.cut(te.severity, BINS, labels=BIN_LABELS, right=False)
    det_rows = []
    for name, a in alarms.items():
        row = {"method": name,
               "false_alarm_rate": float(a[te.healthy.to_numpy()].mean())}
        for b in BIN_LABELS:
            m = (te_bins == b).to_numpy()
            row[f"detect_{b}"] = float(a[m].mean())
        row["detect_all_faults"] = float(a[(te.severity >= 0.2).to_numpy()].mean())
        det_rows.append(row)
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(out / "detection_comparison.csv", index=False)
    _plot_detection(det_df, out / "detection_vs_severity.png", cfg.get("short_name", cfg["name"]))

    # ---- 2. fault classification (which component) -----------------------
    X_tr = pd.concat([Z_tr, tr[regime]], axis=1)
    X_te = pd.concat([Z_te, te[regime]], axis=1)
    clf = classifier(cfg["seed"]).fit(X_tr, tr.fault_class)
    pred = clf.predict(X_te)
    labels = list(clf.classes_)
    cm = pd.DataFrame(confusion_matrix(te.fault_class, pred, labels=labels),
                      index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm.to_csv(out / "fault_confusion_matrix.csv")
    imp = shap_importance(clf, X_te, seed=cfg["seed"])
    plot_importance(imp, "What the fault classifier looks at (SHAP)", out / "shap_fault_classifier.png")

    # ---- 3. severity (how bad) ------------------------------------------
    sev_metrics = {}
    for comp, spec in cfg["components"].items():
        reg = regressor(cfg["seed"]).fit(X_tr, tr[spec["column"]])
        p = reg.predict(X_te)
        y = te[spec["column"]]
        sev_metrics[comp] = {
            "mae_coefficient": float(mean_absolute_error(y, p)),
            "mae_pct_of_range": float(mean_absolute_error(y, p) / (spec["new"] - spec["worst"]) * 100),
            "r2": float(r2_score(y, p)),
        }

    # ---- 4. an operator-facing example alert ------------------------------
    example = _example_alert(te, Z_te, det, clf, X_te, cfg)

    metrics = {
        "dataset": cfg["description"],
        "rows": {"train": len(tr), "test": len(te), "healthy_train": int(tr.healthy.sum()),
                 "healthy_test": int(te.healthy.sum())},
        "detection": det_rows,
        "fault_classification": {
            "accuracy": float(accuracy_score(te.fault_class, pred)),
            "macro_f1": float(f1_score(te.fault_class, pred, average="macro")),
            "classes": labels,
        },
        "severity_estimation": sev_metrics,
        "top_shap_features": imp.head(5).round(4).to_dict(),
        "example_alert": example,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _example_alert(te, Z_te, det, clf, X_te, cfg):
    score = det.score(Z_te)
    cand = te[(te.fault_class == "turbine") & (score > det.threshold_)]
    if cand.empty:
        cand = te[score > det.threshold_]
    i = cand.sort_values("severity").index[len(cand) // 2]
    contrib = det.contributions(Z_te.loc[[i]]).iloc[0]
    share = (contrib.clip(lower=0) / contrib.clip(lower=0).sum()).sort_values(ascending=False)
    proba = pd.Series(clf.predict_proba(X_te.loc[[i]])[0], index=clf.classes_)
    return {
        "operating_point": {f: float(te.at[i, f]) for f in cfg["regime_features"]},
        "health_score": round(float(score[te.index.get_loc(i)]), 1),
        "alarm_threshold": round(det.threshold_, 1),
        "top_sensors": {k: f"{v:.0%}" for k, v in share.head(3).items()},
        "sensor_deviation_sigma": {k: round(float(Z_te.at[i, f"z_{k}"]), 1) for k in share.head(3).index},
        "diagnosis": {k: f"{v:.0%}" for k, v in proba.sort_values(ascending=False).items()},
        "truth": {"class": te.at[i, "fault_class"],
                  **{spec["column"]: float(te.at[i, spec["column"]]) for spec in cfg["components"].values()}},
    }


def _plot_detection(det_df, path, title):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(BIN_LABELS))
    for k, (_, r) in enumerate(det_df.iterrows()):
        y = [r[f"detect_{b}"] * 100 for b in BIN_LABELS]
        ax.plot(x, y, marker="o", markersize=6, color=SERIES[k],
                label=f"{r['method']}  (false alarms {r['false_alarm_rate']:.1%})")
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("Degradation severity (share of the way to worst state)")
    ax.set_ylabel("Faults detected (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Faults detected, by severity - {title}", loc="left", color=INK, fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=1,
              labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
