"""Experiment type 6: ship power-plant blackout risk under maintenance policies.

A scenario model, not a dataset. It asks: on a fleet of cruise ships, each
with N generator sets, how many blackouts and how many days without an N+1
margin does each maintenance policy lead to over the next month?

What is real and what is assumed:
  * Real: each generator set is a C-MAPSS test engine, with its true remaining
    life and the remaining life FaultWatch predicted for it (the 100 unseen
    engines of the run-to-failure experiment, read from its report), and
    the out-of-fold model error the planner uses.
  * Assumed (all in configs/ship_power_plant.yaml): ships, sets per ship,
    a 7-day itinerary of port / sea / manoeuvring days, units needed for each,
    heavy-weather days, repair and service durations, and costs.

Per day and ship, available = installed - failed (in repair) - in service.
  * blackout:           available < required (the load cannot be carried)
  * no-margin day:      available == required (one more trip = blackout)

Policies (each plans once, at day 0, from what it knows then):
  * run to failure:     repair after failure only
  * time-based:         service sets whose running age passes a fixed limit,
                        at the first port call (age limit = a low percentile
                        of fleet life, the classic OEM-interval approach)
  * FaultWatch plan:    the existing fleet MILP (crew capacity, ignores the itinerary)
  * FaultWatch, itinerary-aware: MILP per ship with port windows, N+1 rule
                        at sea, safety cost and a failure-risk cap
  * perfect foresight:  itinerary-aware MILP given the true remaining life

Monte Carlo over random fleets (which engines sit on which ship), itinerary
phase and heavy weather gives a distribution for every KPI; the result is
reported per 100 ship-voyages with 95% intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import LOADERS
from ..plotting import INK, INK_2, NEUTRAL, SERIES, style_axes
from ..schedule import plan_maintenance, plan_maintenance_itinerary

POLICIES = ["Run to failure", "Time-based (age limit)", "FaultWatch plan (fleet MILP)",
            "FaultWatch, itinerary-aware", "Perfect foresight"]


def _units(cfg, root):
    """Generator sets: true and predicted remaining life, and running age."""
    src = cfg["source"]
    rep = root / "reports" / src["asset"]
    pred = pd.read_csv(rep / "rul_test_predictions.csv")
    met = json.loads((rep / "metrics.json").read_text())
    sigma = met["maintenance"]["rul_uncertainty_cycles"]
    train, test, _ = LOADERS["cmapss"](root / src["data_dir"], src.get("subset", "FD001"))
    age = test.groupby("unit")["cycle"].max()
    life = train.groupby("unit")["cycle"].max()
    u = pred.rename(columns={"engine": "asset"}).assign(age=lambda d: d.asset.map(age).values)
    return u, sigma, life


def _itinerary(cfg, rng, horizon):
    pat = cfg["itinerary"]["pattern"]
    off = rng.integers(len(pat))
    phase = [pat[(d + off) % len(pat)] for d in range(horizon + 1)]
    req = {}
    for d in range(1, horizon + 1):
        kind = phase[d]
        need = cfg["demand"][kind]
        if kind == "sea" and (phase[d - 1] == "port" or phase[(d + 1) % len(phase)] == "port"):
            need = max(need, cfg["demand"]["manoeuvring"])      # leaving / arriving
        if kind == "sea" and rng.random() < cfg["itinerary"]["heavy_weather_prob"]:
            need = max(need, cfg["demand"]["heavy_weather"])
        req[d] = need
    port = {d for d in range(1, horizon + 1) if phase[d] == "port"}
    return port, req


def _simulate(ship, service, req, horizon, cfg):
    """Day-by-day availability for one ship. `service`: {asset: day or nan}."""
    rep_days, svc_days = cfg["repair_days"], cfg["service_days"]
    out_until = {}
    failed, served = set(), set()
    blackouts = no_margin = failures = services = 0
    for d in range(1, horizon + 1):
        down = 0
        for _, u in ship.iterrows():
            a = u.asset
            sd = service.get(a, np.nan)
            if a not in failed and a not in served and sd == sd and d == sd:
                served.add(a)
                services += 1
                out_until[a] = d + svc_days - 1
            if a not in failed and a not in served and d >= u.rul_true:
                failed.add(a)
                failures += 1
                out_until[a] = d + rep_days - 1
            if out_until.get(a, 0) >= d:
                down += 1
        avail = len(ship) - down
        blackouts += avail < req[d]
        no_margin += avail == req[d]
    return blackouts, no_margin, failures, services


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    units, sigma, fleet_life = _units(cfg, root)
    H, N = cfg["horizon"], cfg["sets_per_ship"]
    m = cfg["maintenance"]
    costs = dict(preventive_cost=m["preventive_cost"], failure_cost=m["failure_cost"],
                 wasted_life_cost=m["wasted_life_cost"])
    age_limit = float(np.quantile(fleet_life, cfg["time_based"]["life_quantile"]))
    rng = np.random.default_rng(cfg["seed"])
    n_ships = len(units) // N
    voyages_per_ship = H / len(cfg["itinerary"]["pattern"])
    rows = []
    for rep in range(cfg["replicates"]):
        perm = rng.permutation(len(units))[: n_ships * N]
        fleet = units.iloc[perm].assign(ship=np.repeat(np.arange(n_ships), N)).reset_index(drop=True)
        trips = {s: _itinerary(cfg, rng, H) for s in range(n_ships)}

        fleet_plan = plan_maintenance(fleet[["asset", "rul_pred"]], H, m["fleet_capacity_per_day"],
                                      rul_uncertainty=sigma, **costs).set_index("asset")["service_day"]
        res = {p: np.zeros(5) for p in POLICIES}
        for s in range(n_ships):
            ship = fleet[fleet.ship == s]
            port, req = trips[s]
            first_port = min(port) if port else 1
            plans = {"Run to failure": {},
                     "Time-based (age limit)": {
                         u.asset: first_port for u in ship.itertuples() if u.age + H >= age_limit},
                     "FaultWatch plan (fleet MILP)": fleet_plan.loc[ship.asset].to_dict()}
            kw = dict(horizon=H, port_days=port, required_online=req, n_installed=N,
                      capacity_per_day=m["ship_capacity_per_day"], safety_cost=m["blackout_cost"]
                      * m["p_blackout_given_failure"], max_failure_risk=m.get("max_failure_risk"), **costs)
            plans["FaultWatch, itinerary-aware"] = plan_maintenance_itinerary(
                ship[["asset", "rul_pred"]], rul_uncertainty=sigma, **kw).set_index("asset")["service_day"].to_dict()
            plans["Perfect foresight"] = plan_maintenance_itinerary(
                ship[["asset", "rul_true"]].rename(columns={"rul_true": "rul_pred"}), rul_uncertainty=0.5,
                **kw).set_index("asset")["service_day"].to_dict()
            for p, plan in plans.items():
                b, nm, f, sv = _simulate(ship, plan, req, H, cfg)
                cost = sv * m["preventive_cost"] + f * m["failure_cost"] + b * m["blackout_cost"]
                res[p] += [b, nm, f, sv, cost]
        ship_voyages = n_ships * voyages_per_ship
        for p, (b, nm, f, sv, cost) in res.items():
            rows.append({"replicate": rep, "policy": p,
                         "blackouts_per_100_voyages": 100 * b / ship_voyages,
                         "no_margin_days_per_100_voyages": 100 * nm / ship_voyages,
                         "in_service_failures": f, "planned_services": sv, "cost": cost})
        print(f"   replicate {rep + 1}/{cfg['replicates']}", end="\r")
    sims = pd.DataFrame(rows)
    sims.to_csv(out / "simulations.csv", index=False)

    def summary(col):
        g = sims.groupby("policy")[col]
        return {p: {"mean": float(g.get_group(p).mean()),
                    "p2_5": float(g.get_group(p).quantile(0.025)),
                    "p97_5": float(g.get_group(p).quantile(0.975))} for p in POLICIES}
    piv = sims.pivot(index="replicate", columns="policy", values="blackouts_per_100_voyages")
    avoided = piv["Run to failure"] - piv["FaultWatch, itinerary-aware"]
    vs_fleet = piv["FaultWatch plan (fleet MILP)"] - piv["FaultWatch, itinerary-aware"]
    metrics = {
        "dataset": cfg["description"],
        "scenario": {"ships": n_ships, "generator_sets_per_ship": N, "horizon_days": H,
                     "voyages_per_ship": round(voyages_per_ship, 2), "replicates": cfg["replicates"],
                     "rul_model_error_days": sigma, "time_based_age_limit_cycles": age_limit,
                     "note": "simulation: itinerary, demand, durations and costs are assumptions (see config)"},
        "blackouts_per_100_voyages": summary("blackouts_per_100_voyages"),
        "no_margin_days_per_100_voyages": summary("no_margin_days_per_100_voyages"),
        "in_service_failures": summary("in_service_failures"),
        "planned_services": summary("planned_services"),
        "cost": summary("cost"),
        "blackouts_avoided_vs_run_to_failure": {
            "mean": float(avoided.mean()), "p2_5": float(avoided.quantile(0.025)),
            "p97_5": float(avoided.quantile(0.975))},
        "itinerary_awareness_gain": {
            "blackouts_mean": float(vs_fleet.mean()), "p2_5": float(vs_fleet.quantile(0.025)),
            "p97_5": float(vs_fleet.quantile(0.975))},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot(sims, out / "blackout_risk_by_policy.png")
    return metrics


def _plot(sims, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    y = np.arange(len(POLICIES))[::-1]
    for ax, (col, title) in zip(axes, [("blackouts_per_100_voyages", "Blackouts per 100 ship-voyages"),
                                       ("no_margin_days_per_100_voyages", "Days with no N+1 margin\nper 100 voyages"),
                                       ("cost", "Cost per month ($M)")]):
        for k, p in enumerate(POLICIES):
            v = sims.loc[sims.policy == p, col] / (1e6 if col == "cost" else 1)
            lo, hi = v.quantile([0.025, 0.975])
            color = (SERIES[0] if p.startswith("FaultWatch, itin")
                     else NEUTRAL if p == "Perfect foresight" else SERIES[1])
            ax.plot([lo, hi], [y[k]] * 2, color=color, linewidth=2, alpha=0.5)
            ax.scatter([v.mean()], [y[k]], color=color, s=40, zorder=3, edgecolor="white", linewidth=1.2)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        style_axes(ax, grid_axis="x")
    axes[0].set_yticks(y, POLICIES, fontsize=8.5)
    fig.text(0.01, 0.01, "dot = mean, line = 95% of simulated months (random fleets, itineraries, weather). "
             "Simulation - assumptions in configs/ship_power_plant.yaml", color=INK_2, fontsize=7.5)
    fig.suptitle("Ship power plant: blackout risk under each maintenance policy", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
