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


def plan_costs(plan: pd.DataFrame, rul_true: pd.Series, horizon: int,
               preventive_cost: float, failure_cost: float, wasted_life_cost: float) -> pd.DataFrame:
    """Per-machine outcome of a plan against the true remaining life."""
    p = plan.assign(rul_true=np.asarray(rul_true))
    served = p["service_day"].notna() & (p["service_day"] <= p["rul_true"])  # serviced before it fails
    failed = (~served) & (p["rul_true"] <= horizon)
    wasted = (p["rul_true"] - p["service_day"]).clip(lower=0).fillna(0)
    cost = served * (preventive_cost + wasted_life_cost * wasted) + failed * failure_cost
    return pd.DataFrame({"asset": p["asset"].values, "served": served.values,
                         "failed": failed.values, "cost": cost.astype(float).values})


def evaluate_plan(plan: pd.DataFrame, rul_true: pd.Series, horizon: int,
                  preventive_cost: float, failure_cost: float, wasted_life_cost: float):
    """Score a plan against the true remaining life."""
    c = plan_costs(plan, rul_true, horizon, preventive_cost, failure_cost, wasted_life_cost)
    return {"serviced_in_time": int(c["served"].sum()), "failures": int(c["failed"].sum()),
            "total_cost": float(c["cost"].sum())}


def plan_maintenance_itinerary(units: pd.DataFrame, horizon: int, port_days, required_online,
                               n_installed: int, capacity_per_day: int, preventive_cost: float,
                               failure_cost: float, wasted_life_cost: float, rul_uncertainty: float,
                               safety_cost: float = 0.0, max_failure_risk: float | None = None,
                               service_at_sea: bool = True) -> pd.DataFrame:
    """Service plan for the units of one ship (e.g. its gas-turbine or diesel
    generator sets), aware of the itinerary and of redundancy.

    units:           columns [asset, rul_pred] (days)
    port_days:       days 1..horizon alongside (service allowed without restriction)
    required_online: {day: units needed for that day's load} (port / sea / manoeuvring)
    service_at_sea:  allow service at sea only while an N+1 margin remains, i.e.
                     units out <= installed - required - 1 on that day
    safety_cost:     extra cost of an in-service failure for its safety
                     consequence (a failed unit removes margin; with a second
                     failure or a high-demand day it becomes a blackout)
    max_failure_risk: hard cap on P(unit fails before its service). Units over
                     the cap in the horizon cannot be skipped; days over the cap
                     are not allowed (unless no day meets it - then the
                     earliest allowed day is forced).
    """
    a = units.copy()
    sigma = max(rul_uncertainty, 1e-6)
    a["p_fail_in_horizon"] = norm.cdf((horizon - a["rul_pred"]) / sigma)
    at_risk = a[a["p_fail_in_horizon"] >= 0.01]
    port = set(port_days)
    slots = {}
    for d in range(1, horizon + 1):
        if d in port:
            slots[d] = capacity_per_day
        elif service_at_sea:
            slots[d] = min(capacity_per_day, max(0, n_installed - required_online[d] - 1))
        else:
            slots[d] = 0
    days = [d for d in range(1, horizon + 1) if slots[d] > 0]

    prob = pulp.LpProblem("itinerary_maintenance", pulp.LpMinimize)
    x, skip, cost = {}, {}, []
    for _, r in at_risk.iterrows():
        e, rul = r["asset"], r["rul_pred"]
        p_day = {d: norm.cdf((d - rul) / sigma) for d in days}
        ok = [d for d in days if max_failure_risk is None or p_day[d] <= max_failure_risk]
        if not ok and days:
            ok = [days[0]]                     # already over the cap: earliest slot
        for d in ok:
            x[e, d] = pulp.LpVariable(f"x_{e}_{d}", cat="Binary")
            c = (preventive_cost * (1 - p_day[d]) + (failure_cost + safety_cost) * p_day[d]
                 + wasted_life_cost * max(rul - d, 0))
            cost.append(x[e, d] * float(c))
        may_skip = max_failure_risk is None or r["p_fail_in_horizon"] <= max_failure_risk or not ok
        terms = [x[e, d] for d in ok]
        if may_skip:
            skip[e] = pulp.LpVariable(f"skip_{e}", cat="Binary")
            cost.append(skip[e] * float((failure_cost + safety_cost) * r["p_fail_in_horizon"]))
            terms.append(skip[e])
        prob += pulp.lpSum(terms) == 1
    for d in days:
        prob += pulp.lpSum(v for (e, dd), v in x.items() if dd == d) <= slots[d]
    prob += pulp.lpSum(cost)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    plan = {e: d for (e, d), v in x.items() if v.value() is not None and v.value() > 0.5}
    a["service_day"] = a["asset"].map(plan)
    a["at_sea"] = a["service_day"].map(lambda d: d == d and d not in port)
    a["status"] = pulp.LpStatus[status]
    return a
