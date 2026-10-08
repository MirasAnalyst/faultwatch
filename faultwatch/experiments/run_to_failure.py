"""Experiment type 2: a fleet of machines, each recorded from healthy to failure.

Fits: per-machine baselined residuals + health detector (early life only),
remaining-useful-life model, and a maintenance schedule optimizer.
Detection is scored out-of-fold (engines never seen in fitting).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..anomaly import HealthDetector, first_alarm
from ..data import LOADERS
from ..explain import plot_importance, shap_importance
from ..models import make_regressor, nasa_score, rul_features
from ..plotting import INK, INK_2, NEUTRAL, SERIES, style_axes
from ..regime import RegimeNormalizer
from ..schedule import evaluate_plan, plan_costs, plan_maintenance
from ..selection import as_metrics, choose
from ..serve import healthy_reference, save_bundle
from ..stats import bootstrap_ci, mcnemar_exact, wilcoxon_paired


def _normalizer(cfg, per_asset=None, smoothing=None):
    n = cfg["normalization"]
    return RegimeNormalizer(
        cfg["sensors"], cfg["regime_features"], cfg["asset_id"], cfg["time"],
        per_asset_baseline=n.get("per_asset_baseline", False) if per_asset is None else per_asset,
        baseline_window=n.get("baseline_window", 30),
        smoothing_window=n.get("smoothing_window", 1) if smoothing is None else smoothing)


def _alarm_table(df, flags, cfg, persistence):
    a, t = cfg["asset_id"], cfg["time"]
    warmup = cfg["normalization"].get("smoothing_window", 1)
    flags = np.asarray(flags) & (df[t].to_numpy() > warmup)   # ignore filter warm-up
    rows = []
    for unit, g in df.assign(_f=flags).sort_values([a, t]).groupby(a):
        life = int(g["life"].iloc[0])
        ta = first_alarm(g["_f"].to_numpy(), g[t].to_numpy(), persistence)
        rows.append({"asset": unit, "life": life, "alarm_at": ta,
                     "lead": None if ta is None else life - ta})
    r = pd.DataFrame(rows)
    r["false_early"] = r["alarm_at"].notna() & (r["alarm_at"] <= cfg["early_life_fraction"] * r["life"])
    r["missed"] = r["alarm_at"].isna()
    return r


def _summarize(name, r):
    good = r[~r.false_early & ~r.missed]
    return {"method": name,
            "engines": len(r),
            "warned_in_time_pct": round(100 * len(good) / len(r), 1),
            "median_warning_cycles": float(good["lead"].median()) if len(good) else None,
            "min_warning_cycles": float(good["lead"].min()) if len(good) else None,
            "warned_20plus_cycles_pct": round(100 * (good["lead"] >= 20).sum() / len(r), 1),
            "false_early_alarm_pct": round(100 * r.false_early.mean(), 1),
            "missed_pct": round(100 * r.missed.mean(), 1)}


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    train, test, test_rul = LOADERS[cfg["dataset"]](root / cfg["data_dir"], cfg.get("subset", "FD001"))
    a, t = cfg["asset_id"], cfg["time"]
    det_cfg = cfg["detector"]
    q, pers = det_cfg.get("quantile", 0.99), det_cfg.get("persistence", 1)
    healthy = train[t] <= cfg["healthy_window"]
    bs = cfg["baseline_alarm_sensor"]

    # ---- 1. out-of-fold early-warning evaluation --------------------------
    variants = {
        f"Fixed {bs} high alarm": None,
        "Multivariate, fleet baseline": dict(per_asset=False),
        "FaultWatch (per-engine baseline)": dict(),
    }
    tables = {k: [] for k in variants}
    oof_score = pd.Series(np.nan, index=train.index)
    oof_thr = {}
    for fi, vi in GroupKFold(n_splits=5).split(train, groups=train[a]):
        fit, val = train.iloc[fi], train.iloc[vi]
        h = fit[healthy.iloc[fi].to_numpy()]
        for name, kw in variants.items():
            if kw is None:
                thr = h[bs].mean() + 3 * h[bs].std()
                flags = (val[bs] > thr).to_numpy()
            else:
                norm = _normalizer(cfg, **kw).fit(h)
                det = HealthDetector(det_cfg.get("method", "mahalanobis"), q, pers, cfg["seed"])
                det.fit(norm.transform(fit)[healthy.iloc[fi].to_numpy()])
                s = det.score(norm.transform(val))
                flags = s > det.threshold_
                if name.startswith("FaultWatch"):
                    oof_score.iloc[vi] = s
                    for u in val[a].unique():
                        oof_thr[u] = det.threshold_
            tables[name].append(_alarm_table(val, flags, cfg, pers))
    tables = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    warn_rows = [_summarize(k, v) for k, v in tables.items()]
    tables["FaultWatch (per-engine baseline)"].to_csv(out / "alarms_per_engine.csv", index=False)
    pd.concat([v.assign(method=k) for k, v in tables.items()]).to_csv(
        out / "alarms_all_methods.csv", index=False)
    warn_unc = _warning_uncertainty(tables, "FaultWatch (per-engine baseline)", cfg["seed"])
    _plot_health(train, oof_score, oof_thr, tables["FaultWatch (per-engine baseline)"], cfg,
                 out / "health_score_examples.png")

    # ---- 2. remaining useful life ------------------------------------------
    norm = _normalizer(cfg).fit(train[healthy])
    det = HealthDetector(det_cfg.get("method", "mahalanobis"), q, pers, cfg["seed"])
    Z_tr = norm.transform(train)
    det.fit(Z_tr[healthy])
    Z_te = norm.transform(test)

    tw = cfg["rul"]["trend_window"]
    X_tr = rul_features(Z_tr, train, det, a, t, tw)
    X_te = rul_features(Z_te, test, det, a, t, tw)
    cap = cfg["rul"]["cap"]
    y_tr = train["RUL"].clip(upper=cap)

    kind, sel = choose("regressor", X_tr, y_tr, train[a], cfg, cfg["rul"].get("model", "gbm"))
    regressor = lambda seed: make_regressor(kind, seed)

    # out-of-fold RUL error on training engines -> uncertainty used by the planner
    oof = np.zeros(len(train))
    for fi, vi in GroupKFold(n_splits=5).split(train, groups=train[a]):
        oof[vi] = regressor(cfg["seed"]).fit(X_tr.iloc[fi], y_tr.iloc[fi]).predict(X_tr.iloc[vi])
    near = (train["RUL"] <= 2 * cfg["maintenance"]["horizon"]).to_numpy()
    cv_rmse_all = float(np.sqrt(np.mean((oof - y_tr) ** 2)))
    cv_rmse_near = float(np.sqrt(np.mean((oof[near] - y_tr[near]) ** 2)))

    rul_model = regressor(cfg["seed"]).fit(X_tr, y_tr)
    last = test.groupby(a)[t].idxmax()
    pred = pd.Series(np.clip(rul_model.predict(X_te.loc[last.values]), 0, cap), index=last.index)
    truth = test_rul.set_index("unit")["RUL"].loc[pred.index]
    const = float(y_tr.mean())
    rul_metrics = {
        "test_engines": int(len(pred)),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "rmse_truth_capped": float(np.sqrt(np.mean((pred - truth.clip(upper=cap)) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "nasa_score": nasa_score(truth, pred),
        "baseline_constant_rmse": float(np.sqrt(np.mean((const - truth) ** 2))),
        "cv_rmse_train": cv_rmse_all,
        "cv_rmse_near_failure": cv_rmse_near,
    }
    pd.DataFrame({"engine": pred.index, "rul_pred": pred.round(1).values, "rul_true": truth.values}) \
        .to_csv(out / "rul_test_predictions.csv", index=False)
    _plot_rul(pred, truth, out / "rul_pred_vs_true.png")
    imp = shap_importance(rul_model, X_te, seed=cfg["seed"])
    plot_importance(imp, "What drives the remaining-life estimate (SHAP)", out / "shap_rul_model.png")

    # ---- 3. maintenance plan for the test fleet ----------------------------
    m = cfg["maintenance"]
    costs = dict(preventive_cost=m["preventive_cost"], failure_cost=m["failure_cost"],
                 wasted_life_cost=m["wasted_life_cost"])
    assets = pd.DataFrame({"asset": pred.index, "rul_pred": pred.values})
    plan = plan_maintenance(assets, m["horizon"], m["capacity_per_day"],
                            rul_uncertainty=cv_rmse_near, **costs)
    plan.assign(rul_true=truth.values).to_csv(out / "maintenance_plan.csv", index=False)
    oracle = plan_maintenance(pd.DataFrame({"asset": pred.index, "rul_pred": truth.values}),
                              m["horizon"], m["capacity_per_day"], rul_uncertainty=0.5, **costs)
    no_plan = plan.assign(service_day=np.nan)
    per_engine = pd.DataFrame({
        "fw": plan_costs(plan, truth, m["horizon"], **costs)["cost"],
        "rtf": plan_costs(no_plan, truth, m["horizon"], **costs)["cost"]})
    saving = bootstrap_ci(per_engine, lambda d: d["rtf"].sum() - d["fw"].sum(), seed=cfg["seed"])
    maint = {
        "planning_horizon": m["horizon"],
        "engines_failing_in_horizon": int((truth <= m["horizon"]).sum()),
        "rul_uncertainty_cycles": round(cv_rmse_near, 1),
        "solver_status": plan["status"].iloc[0],
        "run_to_failure": evaluate_plan(no_plan, truth, m["horizon"], **costs),
        "faultwatch_plan": evaluate_plan(plan, truth, m["horizon"], **costs),
        "perfect_foresight": evaluate_plan(oracle, truth, m["horizon"], **costs),
        "saving_vs_run_to_failure": saving,
    }
    rul_metrics["rmse_ci"] = bootstrap_ci(pd.DataFrame({"p": pred.values, "t": truth.values}),
                                          lambda d: float(np.sqrt(np.mean((d.p - d.t) ** 2))),
                                          seed=cfg["seed"])

    metrics = {"dataset": cfg["description"],
               "engines": {"train": int(train[a].nunique()), "test": int(test[a].nunique())},
               "early_warning": warn_rows, "rul": rul_metrics,
               "top_shap_features": imp.head(5).round(3).to_dict(),
               "maintenance": maint, "uncertainty": warn_unc}
    if sel is not None:
        metrics["model_selection"] = as_metrics(sel, kind)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cols = list(dict.fromkeys([a, t] + cfg["regime_features"] + cfg["sensors"]))
    demo = test[cols].merge(test_rul.rename(columns={"unit": a, "RUL": "rul_at_end"}), on=a)
    demo["true_rul"] = demo["rul_at_end"] + demo.groupby(a)[t].transform("max") - demo[t]
    save_bundle(cfg, root, normalizer=norm, detector=det, rul_model=rul_model,
                rul_features=list(X_tr.columns), rul_cap=cap, trend_window=tw,
                reference=healthy_reference(train[healthy], Z_tr, det, cfg["regime_features"]),
                demo=demo.drop(columns="rul_at_end"), maintenance=m,
                rul_uncertainty=cv_rmse_near)
    return metrics


def _warning_uncertainty(tables, reference, seed):
    """Engines are independent units: bootstrap them for CIs, and compare
    methods engine by engine (McNemar on 'warned in time', Wilcoxon on lead)."""
    ok = {k: (~v.false_early & ~v.missed).to_numpy() for k, v in tables.items()}
    lead = {k: np.where(ok[k], v["lead"].astype(float), 0.0) for k, v in tables.items()}
    out = {}
    for k, v in tables.items():
        d = pd.DataFrame({"ok": ok[k], "lead": v["lead"].astype(float)})
        r = {"warned_in_time": bootstrap_ci(d, lambda x: x.ok.mean(), seed=seed),
             "median_warning_cycles": bootstrap_ci(
                 d, lambda x: x.loc[x.ok, "lead"].median() if x.ok.any() else np.nan, seed=seed)}
        if k != reference:
            r["vs_reference"] = {"reference": reference,
                                 "warned_in_time": mcnemar_exact(ok[reference], ok[k]),
                                 "lead_cycles_missed_as_zero": wilcoxon_paired(lead[reference], lead[k])}
        out[k] = r
    return out


def _plot_health(train, score, thr, alarms, cfg, path):
    a, t = cfg["asset_id"], cfg["time"]
    ok = alarms[~alarms.false_early & ~alarms.missed].sort_values("life")
    picks = ok.iloc[[len(ok) // 6, len(ok) // 2, 5 * len(ok) // 6]]["asset"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, u in enumerate(picks):
        g = train[train[a] == u]
        g = g[g[t] > cfg["normalization"].get("smoothing_window", 1)]
        s = score.loc[g.index]
        x = g[t] - g["life"]
        ax.plot(x, s, color=SERIES[k], linewidth=1.6, label=f"Engine {u} (life {int(g['life'].iloc[0])} cycles)")
        row = alarms[alarms.asset == u].iloc[0]
        xa = row["alarm_at"] - row["life"]
        ax.plot([xa], [s[g[t] == row["alarm_at"]].iloc[0]], marker="o", markersize=8,
                color=SERIES[k], markeredgecolor="white", markeredgewidth=2, zorder=5)
    th = float(np.median(list(thr.values())))
    ax.axhline(th, color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[0], th * 1.15, "alarm threshold", color=INK_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Cycles before failure")
    ax.set_ylabel("Health score (log scale)")
    ax.set_title("Health score rises well before failure (dots = first alarm)", loc="left", color=INK, fontsize=12)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rul(pred, truth, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(truth, pred, s=40, color=SERIES[0], edgecolor="white", linewidth=1)
    lim = max(truth.max(), pred.max()) + 5
    ax.plot([0, lim], [0, lim], color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax.text(lim * 0.62, lim * 0.55, "perfect prediction", color=INK_2, fontsize=9, rotation=38)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True remaining life (cycles)")
    ax.set_ylabel("Predicted remaining life (cycles)")
    ax.set_title("Remaining-life predictions, 100 unseen engines", loc="left", color=INK, fontsize=11)
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
