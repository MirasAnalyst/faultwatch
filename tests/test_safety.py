"""Safety-consequence layer, itinerary-aware planning and pilot-design helpers."""
import numpy as np
import pandas as pd

from faultwatch import safety
from faultwatch.experiments.rollout import _panel, _wedge_assign, parallel_ab, poisson_twfe
from faultwatch.schedule import plan_maintenance_itinerary


def test_k_out_of_n():
    assert safety.k_out_of_n_failure([0.1, 0.1], required=1) == np.float64(0.1 * 0.1)
    assert abs(safety.k_out_of_n_failure([0.1, 0.1], required=2) - (1 - 0.9 ** 2)) < 1e-12
    # 2-of-4 sets needed: the function is lost only when 3+ sets are down
    p = safety.k_out_of_n_failure([0.5] * 4, required=2)
    assert abs(p - 5 / 16) < 1e-12


def test_risk_matrix():
    got = safety.risk_class([0.6, 0.6, 0.05, 0.005, 0.0005], [4, 2, 4, 4, 4])
    assert list(got) == ["high", "medium", "high", "medium", "low"]


def _bundle(**safety_cfg):
    return {"cfg": {"name": "x", "safety": {"failure_modes": safety_cfg}}, "rul_uncertainty": 5.0}


def test_register_from_remaining_life_ranks_urgent_first():
    scored = pd.DataFrame({"rul": [3.0, 80.0, 25.0], "alarm": [True, False, True]})
    reg = safety.risk_register(scored, _bundle(degradation={"effect": "blackout", "severity": 4}), horizon=30,
                               machine=pd.Series(["GT1", "GT2", "GT3"]))
    assert reg.machine.iloc[0] == "GT1" and reg.risk.iloc[0] == "high"
    assert reg.set_index("machine").loc["GT2", "risk"] == "low"
    assert (reg.effect_label == "Blackout (loss of main power)").all()


def test_register_from_component_grades_and_diagnosis():
    scored = pd.DataFrame({"p_critical_pump_leak": [0.9], "p_critical_cooler": [0.02], "alarm": [True]})
    reg = safety.risk_register(scored, _bundle(pump_leak={"effect": "loss_of_steering", "severity": 4},
                                               cooler={"effect": "fire_risk", "severity": 3}), horizon=30)
    assert reg.failure_mode.tolist() == ["pump_leak", "cooler"]
    assert reg.effect.iloc[0] == "loss_of_steering"
    scored = pd.DataFrame({"diagnosis": ["turbine", "healthy"], "diagnosis_confidence": [0.8, 0.9],
                           "alarm": [True, False]})
    reg = safety.risk_register(scored, _bundle(turbine={"effect": "loss_of_propulsion", "severity": 3}),
                               horizon=30, latest_only=False)
    assert reg.failure_mode.tolist() == ["turbine"]          # healthy rows never enter the register


def test_itinerary_plan_services_in_port_and_keeps_margin_at_sea():
    units = pd.DataFrame({"asset": ["G1", "G2", "G3", "G4"], "rul_pred": [6.0, 9.0, 60.0, 80.0]})
    port = {1, 3, 4, 6}
    req = {d: (1 if d in port else 3) for d in range(1, 15)}     # sea days need 3 of 4 -> no slack
    plan = plan_maintenance_itinerary(units, 14, port, req, n_installed=4, capacity_per_day=1,
                                      preventive_cost=10, failure_cost=1000, wasted_life_cost=1,
                                      rul_uncertainty=1.0, max_failure_risk=0.1)
    days = plan.set_index("asset")["service_day"]
    assert days["G1"] in port and days["G2"] in port and days["G1"] != days["G2"]
    assert days["G1"] < 6 and np.isnan(days["G4"])
    assert not plan["at_sea"].any()


def test_pilot_panel_and_analysis_recover_effect():
    rng = np.random.default_rng(0)
    rates = {"events": {"control": 2.0, "treated": 1.0}}
    a = _wedge_assign(rng, 30, 12, 3)
    assert sorted(set(a)) == [3, 6, 9]
    p = _panel(rng, 30, 12, rates, a, frailty_cv=0.3, season_amp=0.3)
    r = poisson_twfe(p, "events")
    assert r["ci_low"] < 0.5 < r["ci_high"] and r["p_value"] < 0.001
    treated = np.arange(15)
    p2 = _panel(rng, 30, 12, rates, np.where(np.arange(30) < 15, 0, np.inf), frailty_cv=0.3,
                season_amp=0.3, pre_months=12)
    assert parallel_ab(p2, "events", treated)["p_value"] < 0.01
