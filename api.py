"""FaultWatch scoring API.

    uvicorn api:app --reload            # after `python run.py configs/*.yaml`
    open http://127.0.0.1:8000/docs

Endpoints
    GET  /assets                    asset types with a trained model, and the columns each needs
    POST /assets/{name}/score       health score, alarm, driving sensors, diagnosis / RUL per row
    POST /assets/{name}/drift       is the healthy baseline still valid? (retraining trigger)
    POST /assets/{name}/plan        maintenance schedule from the latest RUL of each machine
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from faultwatch.schedule import plan_maintenance
from faultwatch.serve import Scorer, load_all

ROOT = os.environ.get("FAULTWATCH_ROOT", ".")
app = FastAPI(title="FaultWatch", version="0.2.0",
              description="Early fault detection for marine, power and process equipment.")
SCORERS: dict[str, Scorer] = load_all(ROOT)


class Rows(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1,
                                       description="Sensor readings, one object per sample")


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


@app.get("/health")
def health():
    return {"status": "ok", "models": len(SCORERS)}


@app.get("/assets")
def assets():
    return {name: {"description": s.cfg.get("description"),
                   "experiment": s.cfg["experiment"],
                   "required_columns": s.required_columns,
                   "alarm_threshold": round(s.det.threshold_, 2),
                   "outputs": ["health_score", "alarm", "top_sensors"]
                   + (["diagnosis"] if s.b.get("classifier") is not None else [])
                   + (["rul"] if s.b.get("rul_model") is not None else [])}
            for name, s in SCORERS.items()}


@app.post("/assets/{name}/score")
def score(name: str, body: Rows):
    s = _get(name)
    df = _frame(s, body)
    out = s.score(df)
    return {"asset": name, "results": _records(out),
            "alarms": int(out.get("alarm_confirmed", out["alarm"]).sum())}


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
