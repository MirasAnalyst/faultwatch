"""Experiment type 4: per-machine normal-behaviour models scored on logged events.

Typical source: SCADA historian data plus a fault logbook. Each dataset is one
machine: a normal-operation training period followed by a prediction period
that contains one event, either a logged anomaly leading up to a failure or
normal operation (the false-alarm test). This follows the CARE to Compare
wind-turbine benchmark.

For every dataset a separate model is fitted on that machine's own training
period (each machine needs its own healthy baseline). Settings are fixed in
the config in advance, not tuned on the events.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..anomaly import HealthDetector
from ..data import LOADERS
from ..plotting import INK, INK_2, NEUTRAL, SERIES, style_axes
from ..regime import RegimeNormalizer
from ..serve import healthy_reference, save_bundle
from ..stats import compare_alarms

HOURS_PER_SAMPLE = 1 / 6          # 10-minute SCADA averages


def confirmed(flags: np.ndarray, persistence: int) -> np.ndarray:
    """True where the alarm has been on for `persistence` consecutive samples."""
    run, out = 0, np.zeros(len(flags), dtype=bool)
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= persistence
    return out


def _fit(df, cfg, regime):
    n = cfg["normalization"]
    tr = df[(df.train_test == "train") & df.normal_status].dropna(subset=cfg["sensors"] + regime)
    norm = RegimeNormalizer(cfg["sensors"], regime, asset_id="asset_id", time="id",
                            smoothing_window=n.get("smoothing_window", 1)).fit(tr)
    d = cfg["detector"]
    Z_tr = norm.transform(tr)
    det = HealthDetector(d.get("method", "mahalanobis"), d.get("quantile", 0.999), d.get("persistence", 1),
                         cfg["seed"], d.get("calibration", "blocked")).fit(Z_tr)
    return tr, Z_tr, norm, det


def _score_event(df, ev, cfg):
    """Returns per-method alarm results and the FaultWatch trace for one dataset."""
    regime, sensors = cfg["regime_features"], cfg["sensors"]
    pers = cfg["detector"].get("persistence", 1)
    tr, Z_tr, norm, det = _fit(df, cfg, regime)
    # Status codes are used only to clean the training history. In the
    # prediction period they are not used: in CARE Wind Farm A the labelled
    # fault windows are coded 'downtime' while the turbines are producing, so
    # filtering on status would leak the label (and delete the events).
    pred = df[df.train_test == "prediction"].dropna(subset=sensors + regime)
    sw = cfg["normalization"].get("smoothing_window", 1)
    history = df[df.train_test == "train"].dropna(subset=sensors + regime).tail(sw - 1)

    def with_history(n):              # rolling smoothing starts from real history, not cold
        return n.transform(pd.concat([history, pred])).loc[pred.index]

    Z = with_history(norm)
    score = det.score(Z)

    flat = RegimeNormalizer(sensors, [], asset_id="asset_id", time="id", smoothing_window=sw).fit(tr)
    det_flat = HealthDetector("mahalanobis", det.quantile, pers, cfg["seed"], det.calibration).fit(flat.transform(tr))
    limit = tr[sensors].max()

    raw = {
        "High-temperature limits (training max per sensor)": (pred[sensors] > limit).any(axis=1).to_numpy(),
        "Multivariate, not power-aware": det_flat.score(with_history(flat)) > det_flat.threshold_,
        "FaultWatch": score > det.threshold_,
    }
    in_window = ((pred["id"] >= ev.event_start_id) & (pred["id"] <= ev.event_end_id)).to_numpy()
    before = (pred["id"] < ev.event_start_id).to_numpy()
    end_time = pd.Timestamp(ev.event_end)
    # truly normal samples: the whole prediction period of a normal dataset, or
    # the part before the labelled window of an anomaly dataset (after the
    # logged failure the machine is still damaged, so it is excluded)
    normal = before if ev.event_label == "anomaly" else np.ones(len(pred), dtype=bool)
    rows = {}
    for name, flags in raw.items():
        c = confirmed(flags, pers)
        r = {"alarm_any": bool(c.any()),
             "sample_alarm_rate_normal": float(flags[normal].mean()) if normal.any() else np.nan}
        if ev.event_label == "anomaly":
            hit = c & in_window
            r["detected"] = bool(hit.any())
            r["false_before_event"] = bool((c & before).any())
            if hit.any():
                first = pred["time_stamp"].to_numpy()[hit.argmax()]
                r["lead_hours"] = float((end_time - pd.Timestamp(first)) / pd.Timedelta(hours=1))
        rows[name] = r

    # which sensors drove the first FaultWatch alarm inside the event window
    c = confirmed(raw["FaultWatch"], pers)
    top = {}
    if ev.event_label == "anomaly" and (c & in_window).any():
        i = (c & in_window).argmax()
        contrib = det.contributions(Z.iloc[[i]]).iloc[0].clip(lower=0)
        top = (contrib / contrib.sum()).nlargest(3).round(2).to_dict()
    trace = pd.DataFrame({"time_stamp": pred["time_stamp"].values, "id": pred["id"].values,
                          "health_score": score, "threshold": det.threshold_})
    return rows, top, trace, (tr, Z_tr, norm, det, pred)


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    datasets, events = LOADERS[cfg["dataset"]](root / cfg["data_dir"])
    per_event, traces, keep = [], {}, None
    for ev in events.itertuples():
        rows, top, trace, fitted = _score_event(datasets[ev.event_id], ev, cfg)
        per_event.append({"event_id": int(ev.event_id), "asset": int(ev.asset), "label": ev.event_label,
                          "description": ev.event_description if isinstance(ev.event_description, str) else "",
                          "top_sensors_at_first_alarm": top,
                          **{f"{m}|{k}": v for m, r in rows.items() for k, v in r.items()}})
        traces[int(ev.event_id)] = trace
        if int(ev.event_id) == cfg.get("demo_event"):
            keep = (ev, fitted)
        print(f"   event {ev.event_id:>3} {ev.event_label:8s} {per_event[-1]['description'][:28]:28s} "
              f"FaultWatch: {rows['FaultWatch']}")
    pe = pd.DataFrame(per_event)
    pe.to_csv(out / "events.csv", index=False)

    methods = ["High-temperature limits (training max per sensor)", "Multivariate, not power-aware", "FaultWatch"]
    anom, norm_ev = pe[pe.label == "anomaly"], pe[pe.label == "normal"]
    summary = []
    for m in methods:
        det = anom[f"{m}|detected"].astype(bool)
        lead = anom.loc[det, f"{m}|lead_hours"]
        summary.append({
            "method": m,
            "anomaly_events": len(anom),
            "detected": int(det.sum()),
            "median_lead_hours": float(lead.median()) if len(lead) else None,
            "anomaly_events_with_alarm_before_event": int(anom[f"{m}|false_before_event"].astype(bool).sum()),
            "normal_events": len(norm_ev),
            "normal_events_with_false_alarm": int(norm_ev[f"{m}|alarm_any"].astype(bool).sum()),
            "sample_alarm_rate_on_normal_data": float(pe[f"{m}|sample_alarm_rate_normal"].mean(skipna=True)),
        })
    # events are the independent units: an alarm counts as a detection on a
    # failure event (inside its window) and as a false alarm on a normal one
    units = pd.DataFrame({"event_id": pe.event_id, "fault": pe.label == "anomaly"})
    for m in methods:
        units[m] = np.where(units.fault, pe[f"{m}|detected"].fillna(False).astype(bool),
                            pe[f"{m}|alarm_any"].astype(bool))
    uncertainty = compare_alarms(units, methods, "FaultWatch", "fault", cluster=None, seed=cfg["seed"])
    outcome = {r["event_id"]: (f"caught {r['FaultWatch|lead_hours'] / 24:.1f} days ahead"
                               if r.get("FaultWatch|detected") else "missed")
               for r in per_event if r["label"] == "anomaly"}
    _plot_traces(traces, events, cfg, outcome, out / "health_score_events.png")

    metrics = {"dataset": cfg["description"], "events": {"anomaly": len(anom), "normal": len(norm_ev)},
               "event_detection": summary,
               "root_cause_check": [{"event_id": r.event_id, "description": r.description,
                                     "top_sensors_at_first_alarm": r.top_sensors_at_first_alarm}
                                    for r in anom.itertuples()],
               "uncertainty": uncertainty}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if keep is not None:              # serve the model of one turbine for the API / dashboard demo
        ev, (tr, Z_tr, norm, det, pred) = keep
        demo = pred[["time_stamp", "asset_id", "id"] + cfg["regime_features"] + cfg["sensors"]].copy()
        demo["event_window"] = (demo["id"] >= ev.event_start_id) & (demo["id"] <= ev.event_end_id)
        save_bundle({**cfg, "asset_id": "asset_id", "time": "id",
                     "demo_description": f"Turbine {ev.asset}, event {ev.event_id}: {ev.event_description}"},
                    root, normalizer=norm, detector=det,
                    reference=healthy_reference(tr, Z_tr, det, cfg["regime_features"]),
                    demo=demo.reset_index(drop=True))
    return metrics


def _plot_traces(traces, events, cfg, outcome, path):
    """Health score through the prediction period for a few anomaly events."""
    picks = [e for e in cfg.get("plot_events", []) if e in traces][:3]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(8.5, 2.3 * len(picks) + 0.6), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, eid in zip(axes, picks):
        t, ev = traces[eid], events.set_index("event_id").loc[eid]
        x = (t["time_stamp"] - pd.Timestamp(ev.event_end)) / pd.Timedelta(days=1)
        s = np.maximum(t["health_score"], 1e-2)
        xs = (pd.Timestamp(ev.event_start) - pd.Timestamp(ev.event_end)) / pd.Timedelta(days=1)
        ax.axvspan(xs, 0, color=SERIES[1], alpha=0.10, lw=0)
        ax.plot(x, s, color=SERIES[0], linewidth=1.2)
        ax.axhline(t["threshold"].iloc[0], color=NEUTRAL, linestyle="--", linewidth=1.1)
        ax.set_yscale("log")
        ax.set_title(f"Turbine {ev.asset} - {ev.event_description}: {outcome.get(eid, '')}",
                     loc="left", color=INK, fontsize=9.5)
        ax.set_ylabel("Health score", fontsize=9)
        style_axes(ax)
    axes[-1].set_xlabel("Days before the logged failure")
    axes[0].text(0.01, 0.92, "dashed = alarm threshold   shaded = labelled pre-failure window",
                 transform=axes[0].transAxes, color=INK_2, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
