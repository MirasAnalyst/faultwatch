"""Fast unit tests on synthetic data - no downloads needed."""
import numpy as np
import pandas as pd

from faultwatch.anomaly import HealthDetector, first_alarm
from faultwatch.regime import RegimeNormalizer
from faultwatch.schedule import evaluate_plan, plan_maintenance


def _plant(n=3000, fault=0.0, seed=0):
    """Two sensors that scale with load; a fault shifts sensor b only."""
    rng = np.random.default_rng(seed)
    load = rng.choice([0.3, 0.6, 1.0], n)
    return pd.DataFrame({
        "load": load,
        "a": 100 * load + rng.normal(0, 1, n),
        "b": 50 * load + rng.normal(0, 1, n) + fault,
    })


def test_regime_normalization_removes_load_effect():
    h = _plant()
    z = RegimeNormalizer(["a", "b"], ["load"]).fit(h).transform(h)
    assert abs(z.mean()).max() < 0.1
    assert (z.std() - 1).abs().max() < 0.1


def test_detector_flags_fault_but_not_healthy():
    h, f = _plant(seed=1), _plant(fault=5.0, seed=2)
    norm = RegimeNormalizer(["a", "b"], ["load"]).fit(h)
    det = HealthDetector(quantile=0.99).fit(norm.transform(h))
    healthy_rate = (det.score(norm.transform(_plant(seed=3))) > det.threshold_).mean()
    fault_rate = (det.score(norm.transform(f)) > det.threshold_).mean()
    assert healthy_rate < 0.03
    assert fault_rate > 0.9
    contrib = det.contributions(norm.transform(f)).mean()
    assert contrib["b"] > contrib["a"]          # blames the right sensor


def test_first_alarm_needs_persistence():
    flags = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1])
    t = np.arange(len(flags))
    assert first_alarm(flags, t, 1) == 0
    assert first_alarm(flags, t, 3) == 7
    assert first_alarm(np.zeros(5), np.arange(5), 1) is None


def test_scheduler_respects_capacity_and_prioritizes_urgent():
    assets = pd.DataFrame({"asset": [1, 2, 3, 4], "rul_pred": [3.0, 4.0, 25.0, 200.0]})
    kw = dict(preventive_cost=10, failure_cost=1000, wasted_life_cost=1)
    plan = plan_maintenance(assets, horizon=30, capacity_per_day=1, rul_uncertainty=2, **kw)
    days = plan.set_index("asset")["service_day"]
    assert days[1] < days[3] and days[2] < days[3]
    assert np.isnan(days[4])                     # far from failure: not scheduled
    assert plan["service_day"].value_counts().max() <= 1
    res = evaluate_plan(plan, pd.Series([3, 4, 25, 200]), 30, **kw)
    assert res["failures"] == 0
