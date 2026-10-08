"""FaultWatch scoring API.

    uvicorn api:app --reload            # after `python run.py configs/*.yaml`
    open http://127.0.0.1:8000/docs

Endpoints
    GET  /assets                    asset types with a trained model, and the columns each needs
    POST /assets/{name}/score       health score, alarm, driving sensors, diagnosis / condition / RUL per row
    POST /assets/{name}/risk        safety-risk register: failure mode -> ship-level effect, likelihood, action
    POST /assets/{name}/drift       is the healthy baseline still valid? (retraining trigger)
    POST /assets/{name}/plan        maintenance schedule from the latest RUL of each machine
    POST /alerts/{id}/ack           operator acknowledges an alert (adoption tracking)
    POST /alerts/{id}/label         work-order outcome for an alert (delayed accuracy labels)
    GET  /monitoring                latency, cost, adoption, precision, slice parity, retrain advice
    POST /copilot/ask               cited answer from machinery-casualty investigation reports
    POST /copilot/alert             the same, starting from a FaultWatch alert
    POST /copilot/triage            structured classification of a failure narrative
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from faultwatch.monitor import ACTIONS, OUTCOMES, MonitorStore
from faultwatch.safety import risk_register
from faultwatch.schedule import plan_maintenance
from faultwatch.serve import Scorer, load_all

ROOT = os.environ.get("FAULTWATCH_ROOT", ".")
app = FastAPI(title="FaultWatch", version="0.3.0",
              description="Shipboard machinery safety analytics: early fault detection, diagnosis, "
                          "remaining life, safety-risk register and an incident copilot.")
SCORERS: dict[str, Scorer] = load_all(ROOT)
MONITOR = MonitorStore(Path(ROOT) / "monitoring" / "monitor.db")


class Rows(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1,
                                       description="Sensor readings, one object per sample")


class Ack(BaseModel):
    action: Literal[ACTIONS]
    by: str = ""


class Label(BaseModel):
    outcome: Literal[OUTCOMES]
    note: str = ""


class Question(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)


class AlertQuestion(BaseModel):
    asset: str
    alert: dict[str, Any] = Field(..., description="diagnosis / component, top_sensors, effect, note")


class Narrative(BaseModel):
    narrative: str = Field(..., min_length=50, max_length=20000)


@app.middleware("http")
async def track(request: Request, call_next):
    t0, c0 = time.perf_counter(), time.process_time()
    response = await call_next(request)
    path = request.url.path
    if path.startswith(("/assets/", "/copilot/")) and request.method == "POST":
        parts = path.strip("/").split("/")
        MONITOR.log_call(endpoint="/".join([parts[0], parts[-1]]), asset=parts[1] if parts[0] == "assets" else "",
                         latency_ms=1000 * (time.perf_counter() - t0), cpu_ms=1000 * (time.process_time() - c0),
                         rows=int(response.headers.get("x-rows", 0)), alarms=int(response.headers.get("x-alarms", 0)),
                         cost_usd=float(response.headers["x-llm-cost"]) if "x-llm-cost" in response.headers else None,
                         tokens=int(response.headers.get("x-llm-tokens", 0)),
                         provider=response.headers.get("x-llm-provider", ""))
    return response


def _get(name: str) -> Scorer:
    if name not in SCORERS:
        raise HTTPException(404, f"no trained model for '{name}'. Known: {sorted(SCORERS)}")
    return SCORERS[name]


def _frame(s: Scorer, body: Rows) -> pd.DataFrame:
    df = pd.DataFrame(body.rows)
    missing = [c for c in s.required_columns if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing columns: {missing}")
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict("records")


def _machine(s: Scorer, df: pd.DataFrame):
    a = s.cfg.get("asset_id")
    return df[a] if a and a in df else None


@app.get("/health")
def health():
    return {"status": "ok", "models": len(SCORERS)}


@app.get("/assets")
def assets():
    out = {}
    for name, s in SCORERS.items():
        outputs = ["health_score", "alarm", "top_sensors"]
        outputs += ["diagnosis"] if s.b.get("classifier") is not None else []
        outputs += [f"condition_{c}" for c in s.b.get("component_models", {})]
        outputs += ["rul"] if s.b.get("rul_model") is not None else []
        out[name] = {"description": s.cfg.get("description"), "experiment": s.cfg["experiment"],
                     "required_columns": s.required_columns, "alarm_threshold": round(s.det.threshold_, 2),
                     "safety_system": s.cfg.get("safety", {}).get("system"), "outputs": outputs}
    return out


@app.post("/assets/{name}/score")
def score(name: str, body: Rows, record: bool = False):
    """`record=true` stores each confirmed alarm so operators can acknowledge it
    (`/alerts/{id}/ack`) and work orders can label it (`/alerts/{id}/label`)."""
    from fastapi.responses import JSONResponse
    s = _get(name)
    df = _frame(s, body)
    out = s.score(df)
    alarm = out.get("alarm_confirmed", out["alarm"]).fillna(False).astype(bool)
    result = {"asset": name, "results": _records(out), "alarms": int(alarm.sum())}
    if record and alarm.any():
        m = _machine(s, df)
        rec = out[alarm].assign(machine=m[alarm].values if m is not None else "")
        if "diagnosis" in rec:
            rec = rec.assign(diagnosis=rec["diagnosis"])
        result["alert_ids"] = MONITOR.record_alerts(name, rec)
    return JSONResponse(result, headers={"x-rows": str(len(df)), "x-alarms": str(int(alarm.sum()))})


@app.post("/assets/{name}/risk")
def risk(name: str, body: Rows, horizon: float = 30):
    """Safety-risk register from the latest row of each machine."""
    s = _get(name)
    df = _frame(s, body)
    reg = risk_register(s.score(df), s.b, horizon, machine=_machine(s, df))
    return {"asset": name, "system": s.cfg.get("safety", {}).get("system"), "horizon": horizon,
            "register": _records(reg)}


@app.post("/assets/{name}/drift")
def drift(name: str, body: Rows):
    s = _get(name)
    return s.drift(_frame(s, body))


@app.post("/assets/{name}/plan")
def plan(name: str, body: Rows):
    """Send the recent history of each machine; returns the service day per machine."""
    s = _get(name)
    if s.b.get("rul_model") is None:
        raise HTTPException(400, f"'{name}' has no remaining-life model to plan from")
    df = _frame(s, body)
    a, t = s.cfg["asset_id"], s.cfg["time"]
    out = s.score(df)
    latest = df.assign(rul=out["rul"].values).sort_values(t).groupby(a).tail(1)
    m = s.b["maintenance"]
    p = plan_maintenance(latest[[a, "rul"]].rename(columns={a: "asset", "rul": "rul_pred"}),
                         m["horizon"], m["capacity_per_day"],
                         preventive_cost=m["preventive_cost"], failure_cost=m["failure_cost"],
                         wasted_life_cost=m["wasted_life_cost"],
                         rul_uncertainty=s.b.get("rul_uncertainty", 10.0))
    return {"asset": name, "horizon": m["horizon"], "solver_status": p["status"].iloc[0],
            "plan": _records(p.drop(columns="status").sort_values("service_day"))}


@app.post("/alerts/{alert_id}/ack")
def ack(alert_id: str, body: Ack):
    if not MONITOR.ack(alert_id, body.action, body.by):
        raise HTTPException(404, f"unknown alert {alert_id}")
    return {"alert_id": alert_id, "action": body.action}


@app.post("/alerts/{alert_id}/label")
def label(alert_id: str, body: Label):
    MONITOR.label(alert_id, body.outcome, body.note)
    return {"alert_id": alert_id, "outcome": body.outcome}


@app.get("/monitoring")
def monitoring(since_days: float | None = None):
    return MONITOR.summary(since_days)


@lru_cache(maxsize=1)
def _copilot():
    from faultwatch.genai.copilot import Copilot
    corpus = Path(ROOT) / "corpus"
    if not (corpus / "chunks.jsonl.gz").exists():
        raise HTTPException(503, "incident corpus not built: python scripts/download_data.py incidents")
    return Copilot(corpus_dir=corpus)


def _llm_headers(r: dict) -> dict:
    return {"x-llm-cost": str(r.get("cost_usd", 0.0)), "x-llm-tokens": str(r.get("tokens", 0)),
            "x-llm-provider": str(r.get("provider", ""))}


@app.post("/copilot/ask")
def copilot_ask(body: Question):
    from fastapi.responses import JSONResponse
    r = _copilot().answer(body.question)
    r.pop("evidence", None)
    return JSONResponse(r, headers=_llm_headers(r))


@app.post("/copilot/alert")
def copilot_alert(body: AlertQuestion):
    from fastapi.responses import JSONResponse
    cfg = SCORERS[body.asset].cfg if body.asset in SCORERS else {}
    r = _copilot().lessons_for_alert(body.alert, cfg)
    r.pop("evidence", None)
    return JSONResponse(r, headers=_llm_headers(r))


@app.post("/copilot/triage")
def copilot_triage(body: Narrative):
    return _copilot().triage(body.narrative)
