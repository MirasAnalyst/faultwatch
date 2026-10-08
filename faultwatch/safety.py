"""From equipment condition to ship safety: an FMECA-style risk register.

The models upstream answer *engineering* questions (is it degrading, which
component, how long is left). A safety team needs the next step: if this
fails, what happens to the ship, how likely is that before we can act, and
what should be done now?

Each asset config carries a `safety:` block:

    safety:
      system: steering_gear
      redundancy: 2                 # independent units installed
      required: 1                   # units needed to keep the function
      failure_modes:
        pump_leak: {effect: loss_of_steering, severity: 4, note: ...}

`failure_modes` keys are what the models output: a component (graded
condition), a diagnosis class, or `degradation` for remaining-life models.

Likelihood: P(the failure mode occurs within the horizon) per machine, from
remaining life +/- model error, a graded critical-condition probability, or
a confirmed alarm with its diagnosis. Redundancy turns per-unit likelihood
into P(the ship loses the function) - a k-out-of-n system fails only when
more than n-k units are down together.

The risk matrix is the usual likelihood band x severity class; the bands
and actions are set here once and are deliberately simple so an engineer
can audit every ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EFFECTS = {
    "loss_of_propulsion": "Loss of propulsion",
    "blackout": "Blackout (loss of main power)",
    "loss_of_steering": "Loss of steering",
    "fire_risk": "Machinery-space fire risk",
    "loss_of_redundancy": "Loss of redundancy (no margin left)",
    "process_upset": "Process upset / release",
    "unspecified": "Degraded equipment (effect not mapped)",
}
SEVERITY = {1: "minor", 2: "significant", 3: "serious", 4: "catastrophic"}
LIKELIHOOD_BANDS = [0.001, 0.01, 0.1, 0.5]          # -> bands 1 (rare) .. 5 (likely)
ACTIONS = {"high": "Act before the next sea passage: service now or restrict operation",
           "medium": "Plan service at the next port call; increase monitoring",
           "low": "Monitor; no change to the plan"}


def likelihood_band(p) -> np.ndarray:
    return np.searchsorted(LIKELIHOOD_BANDS, np.asarray(p, float), side="right") + 1


def risk_class(p, severity) -> np.ndarray:
    idx = likelihood_band(p) * np.asarray(severity)
    return np.where(idx >= 12, "high", np.where(idx >= 6, "medium", "low"))


def k_out_of_n_failure(p_units, required: int) -> float:
    """P(fewer than `required` units survive), units failing independently with
    probabilities `p_units` (Poisson-binomial). The installed redundancy is
    len(p_units)."""
    dist = np.array([1.0])                       # dist[j] = P(j units failed)
    for p in p_units:
        dist = np.convolve(dist, [1 - p, p])
    n = len(p_units)
    return float(dist[n - required + 1:].sum()) if required <= n else 1.0


def p_fail_from_rul(rul, sigma, horizon) -> np.ndarray:
    """P(remaining life < horizon) with RUL ~ Normal(prediction, model error)."""
    return norm.cdf((horizon - np.asarray(rul, float)) / max(float(sigma), 1e-6))


def unit_likelihoods(scored: pd.DataFrame, bundle: dict, horizon: float) -> pd.DataFrame:
    """Long table: one row per (scored row, failure mode) with the
    probability that the failure mode occurs within the horizon."""
    modes = bundle["cfg"].get("safety", {}).get("failure_modes", {})
    rows = []
    if "rul" in scored:                                       # remaining-life asset
        p = p_fail_from_rul(scored["rul"], bundle.get("rul_uncertainty", 10.0), horizon)
        rows.append(pd.DataFrame({"row": scored.index, "failure_mode": "degradation", "p_fail": p}))
    for c in [c.removeprefix("p_critical_") for c in scored if c.startswith("p_critical_")]:
        rows.append(pd.DataFrame({"row": scored.index, "failure_mode": c, "p_fail": scored[f"p_critical_{c}"]}))
    if "diagnosis" in scored and "rul" not in scored:          # classifier asset
        alarm = scored.get("alarm_confirmed", scored["alarm"]).astype(bool)
        p = np.where(alarm, scored["diagnosis_confidence"], 0.0)
        rows.append(pd.DataFrame({"row": scored.index, "failure_mode": scored["diagnosis"], "p_fail": p}))
    if not rows:                                               # detector-only asset
        alarm = scored.get("alarm_confirmed", scored["alarm"]).astype(bool)
        rows.append(pd.DataFrame({"row": scored.index, "failure_mode": "degradation",
                                  "p_fail": np.where(alarm, 0.5, 0.0)}))
    long = pd.concat(rows, ignore_index=True)
    healthy = {"healthy", "normal", bundle["cfg"].get("healthy_label", "normal")}
    long = long[~long.failure_mode.isin(healthy)]
    spec = long.failure_mode.map(lambda m: modes.get(m, modes.get("*", {})))
    long["effect"] = spec.map(lambda s: s.get("effect", "unspecified"))
    long["severity"] = spec.map(lambda s: int(s.get("severity", 2)))
    long["note"] = spec.map(lambda s: s.get("note", ""))
    return long


def risk_register(scored: pd.DataFrame, bundle: dict, horizon: float, machine: pd.Series | None = None,
                  latest_only: bool = True) -> pd.DataFrame:
    """Ranked risk register, one row per (machine, failure mode).

    scored:  output of `Scorer.score`
    machine: machine id per scored row (e.g. engine / pump number); None = each row
    latest_only: use each machine's last row (the current condition)
    """
    s = scored.copy()
    s["machine"] = machine.values if machine is not None else s.index
    if latest_only:
        s = s.groupby("machine", sort=False).tail(1)
    long = unit_likelihoods(s, bundle, horizon)
    long["machine"] = s.loc[long.row, "machine"].values
    reg = (long.sort_values("p_fail", ascending=False)
               .groupby(["machine", "failure_mode"], as_index=False).first())
    reg["likelihood_band"] = likelihood_band(reg.p_fail)
    reg["risk"] = risk_class(reg.p_fail, reg.severity)
    reg["effect_label"] = reg.effect.map(EFFECTS)
    reg["action"] = reg.risk.map(ACTIONS)
    order = {"high": 0, "medium": 1, "low": 2}
    reg = reg.assign(_o=reg.risk.map(order)).sort_values(["_o", "severity", "p_fail"],
                                                         ascending=[True, False, False])
    return reg.drop(columns=["_o", "row"]).reset_index(drop=True)


def system_risk(register: pd.DataFrame, bundle: dict, system_units: dict) -> pd.DataFrame:
    """P(ship loses the function) per system, from the per-unit likelihoods of
    the units that make it up and the installed redundancy.

    system_units: {system/ship name: [machine ids]}"""
    sc = bundle["cfg"].get("safety", {})
    required = int(sc.get("required", 1))
    worst = register.groupby("machine")["p_fail"].max()
    rows = []
    for name, units in system_units.items():
        p = [float(worst.get(u, 0.0)) for u in units]
        p_loss = k_out_of_n_failure(p, required)
        sev = int(register[register.machine.isin(units)]["severity"].max()) if len(register) else 1
        rows.append({"system": name, "units": len(units), "required": required,
                     "worst_unit_p_fail": max(p) if p else 0.0, "p_function_lost": p_loss,
                     "risk": risk_class([p_loss], [sev])[0]})
    return pd.DataFrame(rows).sort_values("p_function_lost", ascending=False).reset_index(drop=True)
