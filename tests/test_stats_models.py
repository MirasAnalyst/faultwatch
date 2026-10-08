"""Model families, grouped model selection and the statistics toolkit."""
import numpy as np
import pandas as pd
import pytest

from faultwatch import stats
from faultwatch.explain import shap_importance
from faultwatch.models import make_classifier, make_regressor
from faultwatch.selection import choose, compare_models


def _clf_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(np.where(X.a > 0.5, "compressor", np.where(X.b > 0.5, "turbine", "healthy")))
    groups = np.repeat(np.arange(n // 20), 20)
    return X, y, groups


@pytest.mark.parametrize("kind", ["gbm", "xgb", "catboost", "logistic"])
def test_every_classifier_takes_string_labels(kind):
    X, y, _ = _clf_data()
    m = make_classifier(kind).fit(X, y)
    assert set(m.predict(X)) <= set(y)
    assert m.predict_proba(X).shape == (len(X), 3)
    assert (m.predict(X) == y).mean() > 0.85
    imp = shap_importance(m, X, max_rows=200)
    assert set(imp.index[:2]) == {"a", "b"}          # SHAP finds the two informative inputs


@pytest.mark.parametrize("kind", ["gbm", "xgb", "catboost", "linear"])
def test_every_regressor_fits(kind):
    X, _, _ = _clf_data()
    y = 3 * X.a - 2 * X.b
    p = make_regressor(kind).fit(X, y).predict(X)
    assert np.corrcoef(p, y)[0, 1] > 0.9


def test_model_selection_is_grouped_and_report_only_by_default():
    X, y, g = _clf_data()
    table = compare_models("classifier", X, y, g, ["gbm", "logistic"], folds=3)
    assert list(table.columns[:2]) == ["model", "cv_macro_f1"]
    kind, t = choose("classifier", X, y, g, {"model_selection": {"candidates": ["gbm", "logistic"],
                                                                  "folds": 3}}, default="logistic")
    assert kind == "logistic" and t is not None       # apply: false keeps the configured model
    kind, _ = choose("classifier", X, y, g, {}, default="gbm")
    assert kind == "gbm"


def test_cluster_bootstrap_is_wider_than_row_bootstrap():
    """Rows within a machine are correlated: treating them as independent
    understates uncertainty."""
    rng = np.random.default_rng(0)
    machine_effect = rng.normal(0, 1, 20)
    df = pd.DataFrame({"m": np.repeat(np.arange(20), 50)})
    df["x"] = machine_effect[df.m] + rng.normal(0, 0.1, len(df))
    rows = stats.bootstrap_ci(df, lambda d: d.x.mean(), n_boot=400)
    clus = stats.bootstrap_ci(df, lambda d: d.x.mean(), cluster="m", n_boot=400)
    assert (clus["ci_high"] - clus["ci_low"]) > 3 * (rows["ci_high"] - rows["ci_low"])


def test_paired_tests():
    a = np.array([1] * 30 + [0] * 5, bool)
    b = np.array([1] * 10 + [0] * 25, bool)
    r = stats.mcnemar_exact(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["p_value"] < 1e-4
    assert stats.mcnemar_exact(a, a)["p_value"] == 1.0
    assert stats.paired_permutation_test(np.full(20, 0.3))["p_value"] < 0.001
    assert stats.paired_permutation_test(np.r_[np.full(10, 1.0), np.full(10, -1.0)])["p_value"] > 0.5
    assert stats.wilcoxon_paired(np.arange(30) + 5.0, np.arange(30))["p_value"] < 1e-4
    mw = stats.mann_whitney(np.zeros(50) + np.arange(50), np.arange(50) + 100)
    assert mw["auc"] == 1.0


def test_power_and_wilson():
    n = stats.n_per_arm_two_proportions(0.10, 0.05)
    assert 400 < n < 500                              # textbook value ~435
    assert stats.n_per_arm_poisson_rates(0.1, 0.5) > stats.n_per_arm_poisson_rates(0.1, 0.3)
    lo, hi = stats.wilson_ci(0, 12)
    assert lo == 0 and 0.2 < hi < 0.3


def test_cuped_reduces_variance_and_keeps_mean():
    rng = np.random.default_rng(1)
    pre = rng.normal(10, 3, 500)
    y = pre + rng.normal(0, 1, 500)
    adj = stats.cuped(y, pre)
    assert abs(adj.mean() - y.mean()) < 1e-9
    assert adj.var() < 0.2 * y.var()


def test_did_recovers_effect_in_staggered_rollout():
    rng = np.random.default_rng(2)
    rows = []
    adopt = {u: (u % 4) * 3 + 3 for u in range(24)}       # four waves
    for u in range(24):
        base = rng.normal(5, 1)
        for t in range(15):
            d = int(t >= adopt[u])
            rows.append({"u": u, "t": t, "d": d, "y": base + 0.1 * t - 2.0 * d + rng.normal(0, 0.5)})
    r = stats.did_estimate(pd.DataFrame(rows), "y", "d", "u", "t")
    assert r["ci_low"] < -2.0 < r["ci_high"] and r["p_value"] < 1e-6


def test_compare_alarms_table():
    rng = np.random.default_rng(3)
    units = pd.DataFrame({"run": np.repeat(np.arange(10), 30), "fault": np.tile([True] * 20 + [False] * 10, 10)})
    units["fw"] = np.where(units.fault, rng.random(300) < 0.9, rng.random(300) < 0.02)
    units["limit"] = np.where(units.fault, rng.random(300) < 0.4, rng.random(300) < 0.02)
    r = stats.compare_alarms(units, ["limit", "fw"], "fw", "fault", cluster="run", n_boot=200)
    assert r["fw"]["detection"]["ci_low"] > r["limit"]["detection"]["ci_high"]
    assert r["limit"]["vs_reference"]["p_value"] < 0.01
