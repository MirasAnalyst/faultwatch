"""Maintenance scheduling as a small mixed-integer program (PuLP / CBC).

Given each machine's predicted remaining life *and its uncertainty*, choose
which day to service it (or not at all) within the planning horizon,
subject to crew capacity.

Trade-off being optimized, per machine and candidate day d:
  * risk     - P(machine fails before day d) x cost of an in-service failure,
               with remaining life ~ Normal(prediction, model error)
  * waste    - useful life thrown away by servicing early
  * capacity - crews can only service so many machines per day
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
from scipy.stats import norm


def plan_maintenance(assets: pd.DataFrame, horizon: int, capacity_per_day: int,
                     preventive_cost: float, failure_cost: float,
                     wasted_life_cost: float, rul_uncertainty: float) -> pd.DataFrame:
    """assets: columns [asset, rul_pred]. Returns one row per asset with its
    planned service day (NaN = not scheduled in this horizon)."""
    a = assets.copy()
    sigma = max(rul_uncertainty, 1e-6)
    # consider machines with a meaningful chance of failing inside the horizon
    a["p_fail_in_horizon"] = norm.cdf((horizon - a["rul_pred"]) / sigma)
    at_risk = a[a["p_fail_in_horizon"] >= 0.01]
    days = range(1, horizon + 1)

    prob = pulp.LpProblem("maintenance", pulp.LpMinimize)
    x = {(e, d): pulp.LpVariable(f"x_{e}_{d}", cat="Binary")
         for e in at_risk["asset"] for d in days}
    skip = {e: pulp.LpVariable(f"skip_{e}", cat="Binary") for e in at_risk["asset"]}

    cost = []
    for _, r in at_risk.iterrows():
        e, rul = r["asset"], r["rul_pred"]
        for d in days:
            p_fail = norm.cdf((d - rul) / sigma)          # fails before we get there
            c = (preventive_cost * (1 - p_fail) + failure_cost * p_fail
                 + wasted_life_cost * max(rul - d, 0))
            cost.append(x[e, d] * float(c))
        cost.append(skip[e] * float(failure_cost * r["p_fail_in_horizon"]))
        prob += pulp.lpSum(x[e, d] for d in days) + skip[e] == 1
    for d in days:
        prob += pulp.lpSum(x[e, d] for e in at_risk["asset"]) <= capacity_per_day
    prob += pulp.lpSum(cost)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    plan = {e: next((d for d in days if x[e, d].value() > 0.5), None) for e in at_risk["asset"]}
    a["service_day"] = a["asset"].map(plan)
    a["status"] = pulp.LpStatus[prob.status]
    return a


def evaluate_plan(plan: pd.DataFrame, rul_true: pd.Series, horizon: int,
                  preventive_cost: float, failure_cost: float, wasted_life_cost: float):
    """Score a plan against the true remaining life."""
    p = plan.assign(rul_true=rul_true.values)
    served = p["service_day"].notna() & (p["service_day"] <= p["rul_true"])  # serviced before it fails
    failed = (~served) & (p["rul_true"] <= horizon)
    wasted = (p["rul_true"] - p["service_day"]).clip(lower=0).fillna(0)
    cost = served * (preventive_cost + wasted_life_cost * wasted) + failed * failure_cost
    return {"serviced_in_time": int(served.sum()), "failures": int(failed.sum()),
            "total_cost": float(cost.sum())}
