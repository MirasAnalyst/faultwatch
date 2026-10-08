"""Production monitoring: is the model still right, fast, affordable, fair,
and actually used?

The drift monitor (`Scorer.drift`) asks whether *normal* has changed. This
module adds what happens after an alarm leaves the model:

  * service health   - latency p50/p95, calls, compute and LLM cost
  * adoption         - share of alerts acknowledged, time to acknowledge,
                       share acted on (work order / inspection) vs dismissed
  * accuracy         - delayed labels from work orders: was a fault found?
                       (alert precision; recall needs failures without an alert)
  * slice parity     - false-alarm and detection rates per ship class /
                       asset type / operating regime, flagged when one slice
                       is much worse (the bias check for an equipment model)
  * retrain policy   - turns drift, accuracy decay and adoption collapse into
                       one recommended action

Everything is stored in a small SQLite file (monitoring/monitor.db) so the
API, the dashboard and a batch job can share it.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import wilson_ci

CPU_USD_PER_HOUR = 0.05          # one vCPU-hour, for a cost-per-call estimate
SCHEMA = """
create table if not exists calls (ts real, endpoint text, asset text, rows int, latency_ms real,
    cpu_ms real, alarms int, cost_usd real, tokens int, provider text);
create table if not exists alerts (alert_id text primary key, ts real, asset text, machine text,
    health_score real, diagnosis text, effect text, risk text, slice text,
    acked_ts real, action text, acked_by text);
create table if not exists labels (alert_id text, ts real, outcome text, note text);
"""
ACTIONS = ("inspected", "work_order", "deferred", "false_alarm")
OUTCOMES = ("fault_confirmed", "no_fault_found")


class MonitorStore:
    def __init__(self, path: str | Path = "monitoring/monitor.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        """Commit on success and always close (sqlite3's own context manager only commits)."""
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    # ---- writes ---------------------------------------------------------------
    def log_call(self, endpoint: str, asset: str = "", rows: int = 0, latency_ms: float = 0.0,
                 cpu_ms: float = 0.0, alarms: int = 0, cost_usd: float | None = None, tokens: int = 0,
                 provider: str = ""):
        cost = cpu_ms / 3.6e6 * CPU_USD_PER_HOUR if cost_usd is None else cost_usd
        with self._db() as c:
            c.execute("insert into calls values (?,?,?,?,?,?,?,?,?,?)",
                      (time.time(), endpoint, asset, rows, latency_ms, cpu_ms, alarms, cost, tokens, provider))

    def record_alerts(self, asset: str, alerts: pd.DataFrame) -> list[str]:
        """alerts: rows that raised an alarm, with optional columns machine,
        health_score, diagnosis, effect, risk, slice. Returns alert ids."""
        ids = [uuid.uuid4().hex[:12] for _ in range(len(alerts))]
        now = time.time()
        with self._db() as c:
            for i, r in zip(ids, alerts.to_dict("records")):
                c.execute("insert into alerts (alert_id, ts, asset, machine, health_score, diagnosis, effect, "
                          "risk, slice) values (?,?,?,?,?,?,?,?,?)",
                          (i, r.get("ts", now), asset, str(r.get("machine", "")), float(r.get("health_score", 0)),
                           str(r.get("diagnosis", "")), str(r.get("effect", "")), str(r.get("risk", "")),
                           str(r.get("slice", ""))))
        return ids

    def ack(self, alert_id: str, action: str, by: str = "", ts: float | None = None) -> bool:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        with self._db() as c:
            n = c.execute("update alerts set acked_ts=?, action=?, acked_by=? where alert_id=?",
                          (ts or time.time(), action, by, alert_id)).rowcount
        return n == 1

    def label(self, alert_id: str, outcome: str, note: str = "") -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        with self._db() as c:
            c.execute("insert into labels values (?,?,?,?)", (alert_id, time.time(), outcome, note))

    # ---- reads ----------------------------------------------------------------
    def frames(self):
        with self._db() as c:
            return (pd.read_sql("select * from calls", c), pd.read_sql("select * from alerts", c),
                    pd.read_sql("select * from labels", c))

    def summary(self, since_days: float | None = None) -> dict:
        calls, alerts, labels = self.frames()
        if since_days:
            t0 = time.time() - since_days * 86400
            calls, alerts = calls[calls.ts >= t0], alerts[alerts.ts >= t0]
        out = {"service": _service(calls), "adoption": _adoption(alerts),
               "accuracy": _accuracy(alerts, labels), "slices": _slices(alerts, labels)}
        out["retrain_policy"] = retrain_policy(None, out)
        return out


def _service(calls: pd.DataFrame) -> dict:
    if calls.empty:
        return {"calls": 0}
    by = calls.groupby("endpoint")
    return {"calls": int(len(calls)),
            "latency_ms_p50": float(calls.latency_ms.quantile(0.5)),
            "latency_ms_p95": float(calls.latency_ms.quantile(0.95)),
            "cost_usd_total": float(calls.cost_usd.sum()),
            "llm_tokens_total": int(calls.tokens.sum()),
            "by_endpoint": {k: {"calls": int(len(g)), "p95_ms": float(g.latency_ms.quantile(0.95)),
                                "cost_usd": float(g.cost_usd.sum())} for k, g in by}}


def _adoption(alerts: pd.DataFrame) -> dict:
    if alerts.empty:
        return {"alerts": 0}
    acked = alerts.acked_ts.notna()
    tta = (alerts.acked_ts - alerts.ts)[acked] / 3600
    return {"alerts": int(len(alerts)),
            "ack_rate": float(acked.mean()),
            "median_hours_to_ack": float(tta.median()) if len(tta) else None,
            "acted_on_share": float(alerts.action.isin(["inspected", "work_order"]).sum() / max(acked.sum(), 1)),
            "dismissed_as_false_alarm_share": float((alerts.action == "false_alarm").sum() / max(acked.sum(), 1))}


def _accuracy(alerts: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if labels.empty:
        return {"labelled_alerts": 0}
    last = labels.sort_values("ts").groupby("alert_id").tail(1)
    tp = int((last.outcome == "fault_confirmed").sum())
    n = len(last)
    lo, hi = wilson_ci(tp, n)
    return {"labelled_alerts": n, "precision": tp / n, "precision_ci": [lo, hi]}


def _slices(alerts: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if alerts.empty or labels.empty:
        return {}
    last = labels.sort_values("ts").groupby("alert_id").tail(1)
    j = alerts.merge(last, on="alert_id")
    j["slice"] = j["slice"].replace("", "all")
    rows = {}
    for s, g in j.groupby("slice"):
        fp = int((g.outcome == "no_fault_found").sum())
        rows[s] = {"labelled": int(len(g)), "false_alarm_share": fp / len(g),
                   "ci": list(wilson_ci(fp, len(g)))}
    return {"per_slice": rows, **_disparity({k: v["false_alarm_share"] for k, v in rows.items()},
                                            {k: v["labelled"] for k, v in rows.items()})}


def _disparity(rates: dict, n: dict, min_n: int = 10, ratio: float = 2.0, gap: float = 0.1) -> dict:
    ok = {k: v for k, v in rates.items() if n.get(k, 0) >= min_n}
    if len(ok) < 2:
        return {"disparity_flag": False, "reason": "fewer than two slices with enough labels"}
    worst, best = max(ok, key=ok.get), min(ok, key=ok.get)
    flag = ok[worst] - ok[best] > gap and ok[worst] > ratio * max(ok[best], 1e-9)
    return {"disparity_flag": bool(flag), "worst_slice": worst, "best_slice": best,
            "worst_rate": ok[worst], "best_rate": ok[best]}


def slice_parity(preds: pd.DataFrame, slice_col: str, alarm_col: str, fault_col: str) -> pd.DataFrame:
    """Offline bias check on validation predictions: false-alarm rate and
    detection rate per slice (e.g. operating regime, ship class), with Wilson
    intervals, plus a flag on slices that are much worse than the best."""
    rows = []
    for s, g in preds.groupby(slice_col):
        h, f = g[~g[fault_col].astype(bool)], g[g[fault_col].astype(bool)]
        fa, det = int(h[alarm_col].sum()), int(f[alarm_col].sum())
        rows.append({"slice": s, "healthy": len(h), "faulty": len(f),
                     "false_alarm_rate": fa / len(h) if len(h) else np.nan,
                     "fa_ci_high": wilson_ci(fa, len(h))[1] if len(h) else np.nan,
                     "detection_rate": det / len(f) if len(f) else np.nan,
                     "det_ci_low": wilson_ci(det, len(f))[0] if len(f) else np.nan})
    df = pd.DataFrame(rows)
    best_det = df.detection_rate.max()
    best_fa = df.false_alarm_rate.min()
    df["flag"] = ((df.detection_rate < best_det - 0.15) | (df.false_alarm_rate > best_fa + 0.05))
    return df


def retrain_policy(drift: dict | None, summary: dict, min_precision: float = 0.5,
                   min_ack_rate: float = 0.6) -> list[dict]:
    """One place that turns monitoring signals into an action.

    retrain       - the healthy baseline moved (sensor recalibration, overhaul)
    recalibrate   - alarms are mostly wrong but the baseline is fine: raise the
                    threshold / persistence, or relabel
    process review- alarms are right but nobody acts on them: the problem is
                    the workflow (routing, shift handover), not the model
    """
    recs = []
    if drift and drift.get("retrain_recommended"):
        recs.append({"action": "retrain", "why": "; ".join(drift.get("reasons", []))})
    acc = summary.get("accuracy", {})
    if acc.get("labelled_alerts", 0) >= 20 and acc["precision_ci"][1] < min_precision:
        recs.append({"action": "recalibrate",
                     "why": f"alert precision {acc['precision']:.0%} (upper CI {acc['precision_ci'][1]:.0%})"})
    ad = summary.get("adoption", {})
    if ad.get("alerts", 0) >= 20 and ad.get("ack_rate", 1) < min_ack_rate:
        recs.append({"action": "process_review", "why": f"only {ad['ack_rate']:.0%} of alerts acknowledged"})
    sl = summary.get("slices", {})
    if sl.get("disparity_flag"):
        recs.append({"action": "investigate_slice",
                     "why": f"false alarms {sl['worst_rate']:.0%} on {sl['worst_slice']} vs "
                            f"{sl['best_rate']:.0%} on {sl['best_slice']}"})
    return recs or [{"action": "none", "why": "all monitored signals within limits"}]


def to_json(d) -> str:
    return json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o))
