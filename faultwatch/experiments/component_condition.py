"""Experiment type 5: graded condition of several components at once.

Typical source: a test rig or a planned-maintenance history where each
component's condition is graded (e.g. "valve: small lag / severe lag / close
to total failure") and several components degrade at the same time. That is
the normal situation on a hydraulic power unit: a leaking pump, a lagging
valve and a gas-depleted accumulator overlap.

Fits, per component, a condition-grade classifier on per-cycle features, and
an unsupervised health score on normal-wear cycles. Everything is scored
out-of-fold with folds grouped by test-rig segment (a block of consecutive
cycles at one setting), so no segment is in both train and test.

The safety question it answers: is any component at a grade that threatens
the function (for steering gear: loss of steering)? That is compared with the
alarms a hydraulic power unit typically has today (low pressure, high oil
temperature, low flow) and with limits on every sensor.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, KFold

from ..anomaly import HealthDetector
from ..data import LOADERS
from ..explain import shap_importance
from ..models import make_classifier
from ..plotting import INK, INK_2, SERIES, style_axes
from ..regime import RegimeNormalizer
from ..selection import as_metrics, choose
from ..serve import healthy_reference, save_bundle
from ..stats import bootstrap_ci, compare_alarms


def expand_features(cfg) -> list[str]:
    return [f"{s}_{f}" for s in cfg["sensors"] for f in cfg["features"]]


def _oof(make, X, y, folds):
    """Out-of-fold class probabilities."""
    classes = np.unique(y)
    P = np.zeros((len(X), len(classes)))
    for fi, vi in folds:
        m = make().fit(X.iloc[fi], y.iloc[fi])
        P[np.ix_(vi, np.searchsorted(classes, m.classes_))] = m.predict_proba(X.iloc[vi])
    return pd.DataFrame(P, columns=classes, index=X.index)


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    df = LOADERS[cfg["dataset"]](root / cfg["data_dir"])
    feats = expand_features(cfg)
    det_feats = [f"{s}_{f}" for s in cfg["sensors"] for f in cfg["detector_features"]]
    comps, seed = cfg["components"], cfg["seed"]
    group = df[cfg["group"]]
    X = df[feats]

    # tiers: normal wear (no alarm wanted) / degraded / critical (alarm needed)
    critical = pd.DataFrame({c: df[s["column"]].isin(s["critical"]) for c, s in comps.items()})
    normal_wear = pd.concat([df[s["column"]].isin(s["normal"]) for s in comps.values()], axis=1).all(axis=1)
    df["critical"] = critical.any(axis=1)
    df["tier"] = np.where(df.critical, "critical", np.where(normal_wear, "normal_wear", "degraded"))

    folds = list(GroupKFold(n_splits=cfg.get("folds", 5)).split(X, groups=group))

    # ---- 1. condition grade per component (out-of-fold) ----------------------
    grading, p_crit, pred_grade, final, selections, shap_top = {}, {}, {}, {}, {}, {}
    for c, s in comps.items():
        y = df[s["column"]].astype(str)
        kind, sel = choose("classifier", X, y, group, cfg, cfg.get("model", "gbm"))
        P = _oof(lambda: make_classifier(kind, seed), X, y, folds)
        pred = P.columns[P.to_numpy().argmax(axis=1)].to_numpy()
        p_crit[c] = P[[str(g) for g in s["critical"]]].sum(axis=1)
        pred_grade[c] = pred
        # the same model scored with a random (row) split - the leakage trap
        leaky = _oof(lambda: make_classifier(kind, seed), X, y,
                     list(KFold(cfg.get("folds", 5), shuffle=True, random_state=seed).split(X)))
        leaky_acc = float(accuracy_score(y, leaky.columns[leaky.to_numpy().argmax(axis=1)]))
        order = [str(g) for g in s["grades"]]
        cm = confusion_matrix(y, pred, labels=order)
        grading[c] = {
            "grades_best_to_worst": order,
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "recall_by_grade": {g: float(cm[i, i] / max(cm[i].sum(), 1)) for i, g in enumerate(order)},
            "critical_recall": float((pred[critical[c].to_numpy()] == y[critical[c]].to_numpy()).mean()),
            "accuracy_if_split_by_random_cycles": leaky_acc,
        }
        pd.DataFrame(cm, index=[f"true_{g}" for g in order], columns=[f"pred_{g}" for g in order]) \
            .to_csv(out / f"confusion_{c}.csv")
        final[c] = make_classifier(kind, seed).fit(X, y)
        shap_top[c] = shap_importance(final[c], X, seed=seed).head(5).round(4).to_dict()
        if sel is not None:
            selections[c] = as_metrics(sel, kind)

    # ---- 2. unsupervised health score on normal-wear cycles (out-of-fold) -----
    health = np.full(len(df), np.nan)
    hd = cfg["detector"]
    thr = {}
    for k, (fi, vi) in enumerate(folds):
        ref = df.iloc[fi][normal_wear.iloc[fi].to_numpy()]
        norm = RegimeNormalizer(det_feats, []).fit(ref)
        det = HealthDetector(hd.get("method", "mahalanobis"), hd.get("quantile", 0.99), 1, seed,
                             hd.get("calibration", "random")).fit(norm.transform(ref))
        health[vi] = det.score(norm.transform(df.iloc[vi]))
        thr[k] = det.threshold_
    fold_of = np.zeros(len(df), dtype=int)
    for k, (_, vi) in enumerate(folds):
        fold_of[vi] = k
    health_alarm = health > np.array([thr[k] for k in fold_of])

    # ---- 3. today's alarms, thresholds set on normal-wear cycles of the fitting folds
    base = {b["label"]: b for b in cfg["baseline_alarms"]}
    conv = {name: np.zeros(len(df), dtype=bool) for name in base}
    conv["Limits on every sensor (mean ± 3σ)"] = np.zeros(len(df), dtype=bool)
    for fi, vi in folds:
        ref = df.iloc[fi][normal_wear.iloc[fi].to_numpy()]
        val = df.iloc[vi]
        for name, b in base.items():
            mu, sd = ref[b["feature"]].mean(), ref[b["feature"]].std()
            conv[name][vi] = (val[b["feature"]] > mu + 3 * sd if b["direction"] == "high"
                              else val[b["feature"]] < mu - 3 * sd).to_numpy()
        means = [f"{s}_mean" for s in cfg["sensors"]]
        mu, sd = ref[means].mean(), ref[means].std()
        conv["Limits on every sensor (mean ± 3σ)"][vi] = ((val[means] - mu).abs() > 3 * sd).any(axis=1).to_numpy()

    fw_alarm = pd.DataFrame(p_crit).max(axis=1).to_numpy() >= cfg.get("critical_probability", 0.5)
    alarms = {**conv, "FaultWatch health score (unsupervised)": health_alarm,
              "FaultWatch component grading": fw_alarm}

    units = pd.DataFrame({"segment": group.to_numpy(), "fault": df.critical.to_numpy(),
                          "tier": df.tier.to_numpy(), **alarms})
    scored = units[units.tier != "degraded"]
    uncertainty = compare_alarms(scored, list(alarms), "FaultWatch component grading", "fault",
                                 cluster="segment", seed=seed)
    det_rows = []
    for name, a in alarms.items():
        det_rows.append({"method": name,
                         "critical_detected": float(a[df.critical].mean()),
                         "false_alarm_rate_normal_wear": float(a[(df.tier == "normal_wear").to_numpy()].mean()),
                         "alarm_rate_degraded": float(a[(df.tier == "degraded").to_numpy()].mean())})
    pd.DataFrame(det_rows).to_csv(out / "detection_comparison.csv", index=False)

    preds = df[["cycle", "segment", "tier"] + [s["column"] for s in comps.values()]].copy()
    for c in comps:
        preds[f"pred_{c}"] = pred_grade[c]
        preds[f"p_critical_{c}"] = p_crit[c].round(4).to_numpy()
    preds["health_score"] = health
    for name, a in alarms.items():
        preds[name] = a
    preds.to_csv(out / "predictions.csv.gz", index=False)
    _plot(grading, det_rows, out / "condition_by_component.png", cfg.get("short_name", cfg["name"]))

    metrics = {
        "dataset": cfg["description"],
        "cycles": {"total": len(df), "segments": int(group.nunique()),
                   "normal_wear": int(normal_wear.sum()), "degraded": int((df.tier == "degraded").sum()),
                   "critical": int(df.critical.sum())},
        "component_grading": grading,
        "critical_detection": det_rows,
        "critical_detection_ci": {
            "FaultWatch": bootstrap_ci(units[units.fault], lambda d: d["FaultWatch component grading"].mean(),
                                       cluster="segment", seed=seed)},
        "uncertainty": uncertainty,
        "top_shap_features": shap_top,
    }
    if selections:
        metrics["model_selection"] = selections
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- 4. serving bundle (models refit on all cycles) ----------------------
    ref = df[normal_wear]
    norm = RegimeNormalizer(det_feats, []).fit(ref)
    det = HealthDetector(hd.get("method", "mahalanobis"), hd.get("quantile", 0.99), 1, seed,
                         hd.get("calibration", "random")).fit(norm.transform(ref))
    demo_rows = preds.groupby("tier", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 80), random_state=seed)).index
    demo = df.loc[demo_rows, ["cycle", "segment", "tier"] + feats + [s["column"] for s in comps.values()]]
    save_bundle({**cfg, "sensors": det_feats}, root, normalizer=norm, detector=det,
                component_models=final, component_features=feats,
                component_critical={c: [str(g) for g in s["critical"]] for c, s in comps.items()},
                reference=healthy_reference(ref, norm.transform(ref), det, []),
                demo=demo.sort_values("cycle").reset_index(drop=True))
    return metrics


def _plot(grading, det_rows, path, title):
    comps = list(grading)
    fig, axes = plt.subplots(1, len(comps) + 1, figsize=(12, 3.6),
                             gridspec_kw={"width_ratios": [1] * len(comps) + [1.6]})
    for ax, c in zip(axes, comps):
        g = grading[c]["recall_by_grade"]
        y = np.arange(len(g))[::-1]
        ax.barh(y, [v * 100 for v in g.values()], color=SERIES[0], height=0.55)
        ax.set_yticks(y, list(g))
        ax.set_xlim(0, 100)
        ax.set_title(f"{c}\n(grade, best → worst)", loc="left", color=INK, fontsize=9)
        style_axes(ax, grid_axis="x")
    axes[0].set_xlabel("Correctly graded (%)")
    ax = axes[-1]
    d = pd.DataFrame(det_rows)
    y = np.arange(len(d))[::-1]
    ax.scatter(d.critical_detected * 100, y, color=SERIES[0], s=42, label="critical cycles alarmed", zorder=3)
    ax.scatter(d.false_alarm_rate_normal_wear * 100, y, color=SERIES[1], s=42, label="normal-wear cycles alarmed",
               zorder=3)
    ax.set_yticks(y, [m.replace("FaultWatch ", "FaultWatch\n") for m in d.method], fontsize=7.5)
    ax.yaxis.tick_right()
    ax.set_xlim(-3, 103)
    ax.set_title("Safety-critical condition alarm (%)", loc="left", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, -0.12))
    style_axes(ax, grid_axis="x")
    fig.suptitle(f"Component condition, scored on test-rig segments never seen in training - {title}",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
