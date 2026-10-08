"""Experiment type 3: a library of recorded faults with known labels.

Typical sources: a test rig with seeded defects (bearing vibration) or a
plant simulator with scripted disturbances (Tennessee Eastman). The loader
returns one row per sample with columns

    sensors..., regime features..., fault (label), faulty (bool: is the fault
    active at this sample), split (train/test), run (recording id), sample (index)

Fits: health detector on healthy training samples only, a fault classifier,
and optionally a severity regressor. Compares the detector with the alarms a
plant typically has today at the same false-alarm budget.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from ..anomaly import HealthDetector, first_alarm
from ..data import LOADERS
from ..explain import plot_importance, shap_importance
from ..models import make_classifier, regressor
from ..plotting import INK, INK_2, SERIES, style_axes
from ..regime import RegimeNormalizer
from ..selection import as_metrics, choose
from ..serve import healthy_reference, save_bundle
from ..stats import compare_alarms


def univariate_limits(h: pd.DataFrame, sensors, target_fa: float):
    """Fixed high/low limits on every sensor - a DCS alarm on each tag - with
    the per-tag tail widened until the combined false-alarm rate on healthy
    data matches the multivariate detector's budget, so the comparison is fair."""
    lo_q, hi_q = 0.0, 0.5
    for _ in range(40):
        q = (lo_q + hi_q) / 2
        lo, hi = h[sensors].quantile(q), h[sensors].quantile(1 - q)
        fa = ((h[sensors] < lo) | (h[sensors] > hi)).any(axis=1).mean()
        if fa > target_fa:
            hi_q = q          # too many false alarms: widen limits
        else:
            lo_q = q
    q = lo_q
    return h[sensors].quantile(q), h[sensors].quantile(1 - q)


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    df = LOADERS[cfg["dataset"]](root / cfg["data_dir"], **cfg.get("loader_args", {}))
    sensors, regime = cfg["sensors"], cfg["regime_features"]
    healthy_label = cfg.get("healthy_label", "normal")
    df["fault_class"] = np.where(df["faulty"], df["fault"], healthy_label)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    h_tr = tr[~tr.faulty]
    det_cfg = cfg["detector"]
    q, pers = det_cfg.get("quantile", 0.99), det_cfg.get("persistence", 1)

    # ---- 1. anomaly detection ------------------------------------------
    norm = RegimeNormalizer(sensors, regime).fit(h_tr)
    Z_tr, Z_te = norm.transform(tr), norm.transform(te)
    calib = det_cfg.get("calibration", "random")
    det = HealthDetector(det_cfg.get("method", "mahalanobis"), q, pers, cfg["seed"], calib).fit(Z_tr[~tr.faulty])

    alarms = {}
    b = cfg.get("baseline_alarm", {})
    if "sensor" in b:
        s = b["sensor"]
        thr = h_tr[s].mean() + 3 * h_tr[s].std()
        alarms[b.get("label", f"Fixed {s} high alarm")] = (te[s] > thr).to_numpy()
    lo, hi = univariate_limits(h_tr, sensors, 1 - q)
    alarms["High/low limits on every sensor"] = ((te[sensors] < lo) | (te[sensors] > hi)).any(axis=1).to_numpy()
    if regime:
        flat = RegimeNormalizer(sensors, []).fit(h_tr)
        det_flat = HealthDetector("mahalanobis", q, pers, cfg["seed"], calib).fit(flat.transform(h_tr))
        alarms["Multivariate, not load-aware"] = det_flat.score(flat.transform(te)) > det_flat.threshold_
    alarms["FaultWatch"] = det.score(Z_te) > det.threshold_

    healthy_te = (~te.faulty).to_numpy()
    natural = lambda c: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", c)]
    classes = sorted((c for c in te.fault_class.unique() if c != healthy_label), key=natural)
    det_rows, per_class = [], {}
    for name, a in alarms.items():
        row = {"method": name, "false_alarm_rate": float(a[healthy_te].mean()),
               "detection_rate": float(a[~healthy_te].mean())}
        delays = []
        for _, g in te.assign(_a=a).groupby("run"):
            if g.faulty.any() and not g.faulty.all():           # runs with a known onset
                onset = g.loc[g.faulty, "sample"].min()
                after = g[g["sample"] >= onset]
                ta = first_alarm(after["_a"].to_numpy(), after["sample"].to_numpy(), pers)
                delays.append(np.nan if ta is None else ta - onset)
        if delays:
            row["runs_detected_pct"] = round(100 * float(np.mean(~np.isnan(delays))), 1)
            row["median_delay_samples"] = float(np.nanmedian(delays)) if not np.all(np.isnan(delays)) else None
        det_rows.append(row)
        per_class[name] = {c: float(a[(te.fault_class == c).to_numpy()].mean()) for c in classes}
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(out / "detection_comparison.csv", index=False)

    # ---- 2. fault classification -------------------------------------------
    X_tr = pd.concat([Z_tr, tr[regime]], axis=1)
    X_te = pd.concat([Z_te, te[regime]], axis=1)
    ccfg = cfg.get("classifier", {})
    if ccfg.get("features"):          # a physics-chosen subset generalises better with few fault examples
        cols = [f"z_{f}" for f in ccfg["features"]]
        X_tr, X_te = X_tr[cols], X_te[cols]
    # CV groups: whole recordings, or - when every recording holds a single fault
    # (Tennessee Eastman) - contiguous time blocks within each recording, so a
    # held-out fold never shares a time window with the fitting folds
    nb = cfg.get("model_selection", {}).get("time_blocks")
    groups = (tr["run"].astype(str) + "_" + (tr["sample"] * nb // (tr.groupby("run")["sample"].transform("max") + 1))
              .astype(str)) if nb else tr["run"]
    kind, sel = choose("classifier", X_tr, tr.fault_class, groups, cfg, ccfg.get("model", "gbm"))
    clf = make_classifier(kind, cfg["seed"]).fit(X_tr, tr.fault_class)
    pred = clf.predict(X_te)
    labels = list(clf.classes_)
    cm = pd.DataFrame(confusion_matrix(te.fault_class, pred, labels=labels),
                      index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm.to_csv(out / "fault_confusion_matrix.csv")
    recall = {c: float((pred[(te.fault_class == c).to_numpy()] == c).mean()) for c in labels}
    imp = shap_importance(clf, X_te, seed=cfg["seed"])
    plot_importance(imp, "What the fault classifier looks at (SHAP)", out / "shap_fault_classifier.png")
    fa = {r["method"]: r["false_alarm_rate"] for r in det_rows}
    _plot_classes(per_class, recall, classes, fa, out / "detection_and_diagnosis_by_fault.png",
                  cfg.get("short_name", cfg["name"]))

    # ---- 3. severity (optional) ---------------------------------------------
    sev_metrics, sev_models = {}, {}
    for name, col in cfg.get("severity", {}).items():
        m = tr.faulty.to_numpy()
        reg = sev_models[name] = regressor(cfg["seed"]).fit(X_tr[m], tr.loc[m, col])
        mt = te.faulty.to_numpy()
        p = reg.predict(X_te[mt])
        sev_metrics[name] = {"mae": float(mean_absolute_error(te.loc[mt, col], p)),
                             "range": [float(df[col].min()), float(df[col].max())]}

    # ---- 4. per-sample predictions + uncertainty (recording = independent unit)
    units = pd.DataFrame({"run": te["run"].astype(str).to_numpy(), "sample": te["sample"].to_numpy(),
                          "fault": (~healthy_te), **alarms})
    units.assign(fault_class=te.fault_class.to_numpy(), predicted_class=pred,
                 health_score=det.score(Z_te)).to_csv(out / "predictions.csv.gz", index=False)
    uncertainty = compare_alarms(units, list(alarms), "FaultWatch", "fault", cluster="run",
                                 seed=cfg["seed"])

    metrics = {
        "dataset": cfg["description"],
        "samples": {"train": len(tr), "test": len(te), "healthy_train": int((~tr.faulty).sum()),
                    "healthy_test": int(healthy_te.sum())},
        "detection": det_rows,
        "detection_by_fault": per_class,
        "fault_classification": {
            "accuracy": float(accuracy_score(te.fault_class, pred)),
            "macro_f1": float(f1_score(te.fault_class, pred, average="macro")),
            "recall_by_class": recall,
        },
        "top_shap_features": imp.head(5).round(4).to_dict(),
    }
    if sev_metrics:
        metrics["severity_estimation"] = sev_metrics
    metrics["uncertainty"] = uncertainty
    if sel is not None:
        metrics["model_selection"] = as_metrics(sel, kind)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    keep = list(dict.fromkeys(regime + sensors + ["fault_class", "run", "sample"]
                              + list(cfg.get("severity", {}).values())))
    demo = te.sample(frac=1, random_state=cfg["seed"]).groupby("fault_class").head(60)[keep]
    save_bundle(cfg, root, normalizer=norm, detector=det, classifier=clf,
                classifier_features=list(X_tr.columns), severity_models=sev_models,
                reference=healthy_reference(h_tr, Z_tr[~tr.faulty], det, regime),
                demo=demo.sort_values(["fault_class", "run", "sample"]).reset_index(drop=True))
    return metrics


def _plot_classes(per_class, recall, classes, fa, path, title):
    """Two small multiples on the same 0-100% scale: detection, then diagnosis."""
    base = [k for k in per_class if k != "FaultWatch"]
    best = max(base, key=lambda k: np.mean(list(per_class[k].values())))
    y = np.arange(len(classes))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 0.34 * len(classes) + 1.8), sharey=True)
    a1.scatter([per_class[best][c] * 100 for c in classes], y, s=46, color=SERIES[1],
               label=f"{best} ({fa[best]:.0%} false alarms)", zorder=3, edgecolor="white", linewidth=1.5)
    a1.scatter([per_class["FaultWatch"][c] * 100 for c in classes], y, s=46, color=SERIES[0],
               label=f"FaultWatch ({fa['FaultWatch']:.0%} false alarms)", zorder=4,
               edgecolor="white", linewidth=1.5)
    a1.set_title("Fault samples detected (%)", loc="left", color=INK, fontsize=10)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left",
              bbox_to_anchor=(0, -0.3 / len(classes) ** 0.5), ncol=1)
    vals = [recall.get(c, np.nan) * 100 for c in classes]
    a2.barh(y, vals, color=SERIES[0], height=0.55)
    for yi, v in zip(y, vals):
        a2.text(v + 1.5 if v < 88 else v - 1.5, yi, f"{v:.0f}", va="center", fontsize=8,
                ha="left" if v < 88 else "right", color=INK_2 if v < 88 else "white")
    a2.set_title("Correctly diagnosed (%)", loc="left", color=INK, fontsize=10)
    a1.set_yticks(y, classes)
    for ax in (a1, a2):
        ax.set_xlim(-3, 103)
        style_axes(ax, grid_axis="x")
    fig.suptitle(f"Detection and diagnosis by fault type - {title}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
