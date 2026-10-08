"""The Spark pipeline must produce the same features and scores as the pandas code
that trained the models. Skipped when Java or PySpark is not available."""
import glob
import os
import shutil

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
if not (os.environ.get("JAVA_HOME") or shutil.which("java")
        or glob.glob("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")):
    pytest.skip("no Java runtime for Spark", allow_module_level=True)

from faultwatch.anomaly import HealthDetector  # noqa: E402
from faultwatch.models import add_trend_features  # noqa: E402
from faultwatch.regime import RegimeNormalizer  # noqa: E402
from faultwatch.serve import healthy_reference, save_bundle  # noqa: E402
from pipelines.features_spark import get_spark, quality_report, residual_features, score_with_bundle  # noqa: E402

SENSORS = ["a", "b", "c"]


def _fleet(n_units=6, life=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        offset = rng.normal(0, 2, 3)                          # unit-to-unit differences
        for t in range(1, life + 1):
            wear = max(0, t - 40) * 0.05
            rows.append({"unit": u, "cycle": t, **{s: 10 * (k + 1) + offset[k] + rng.normal(0, 1) + wear * (k == 1)
                                                   for k, s in enumerate(SENSORS)}})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def spark():
    s = get_spark("faultwatch-test")
    yield s
    s.stop()


def test_features_match_pandas(spark):
    df = _fleet()
    healthy = df[df.cycle <= 30]
    norm = RegimeNormalizer(SENSORS, [], "unit", "cycle", per_asset_baseline=True, baseline_window=30,
                            smoothing_window=5).fit(healthy)
    Z = norm.transform(df)
    T = add_trend_features(Z, df["unit"], df["cycle"], 10)
    expected = pd.concat([df[["unit", "cycle"]], Z, T], axis=1).sort_values(["unit", "cycle"])

    got = residual_features(spark.createDataFrame(df), SENSORS, norm.global_mean_, norm.global_std_, "unit", "cycle",
                            per_asset_baseline=True, baseline_window=30, smoothing_window=5, trend_window=10)
    got = got.toPandas().sort_values(["unit", "cycle"])
    for c in [f"z_{s}" for s in SENSORS] + [f"slope_{s}" for s in SENSORS]:
        np.testing.assert_allclose(got[c].to_numpy(), expected[c].to_numpy(), rtol=1e-6, atol=1e-8, err_msg=c)


def test_quality_report_catches_problems(spark):
    df = _fleet(3, 40)
    df.loc[5, "a"] = np.nan
    df.loc[10:40, "c"] = 30.0                                   # flat-lined sensor on unit 1
    df = pd.concat([df, df.iloc[[0]]])                           # duplicated reading
    rep = quality_report(spark.createDataFrame(df), SENSORS, ["unit", "cycle"], ranges={"b": (0, 15)},
                         max_null_share=0.0)
    by = {c["check"]: c for c in rep["checks"]}
    assert not rep["passed"]
    assert not by["duplicate_keys"]["passed"] and not by["null_share"]["passed"]
    assert not by["physical_range"]["passed"]
    assert by["stuck_sensor"]["warning"] and by["stuck_sensor"]["detail"]["machines_with_flatline"] == {"c": 1}


def test_distributed_scoring_matches_scorer(spark, tmp_path):
    from faultwatch.serve import Scorer
    df = _fleet(4, 60, seed=3)
    healthy = df[df.cycle <= 30]
    norm = RegimeNormalizer(SENSORS, [], "unit", "cycle", per_asset_baseline=True, smoothing_window=3).fit(healthy)
    det = HealthDetector(quantile=0.99, persistence=2).fit(norm.transform(healthy))
    cfg = {"name": "toy_fleet", "experiment": "run_to_failure", "dataset": "x", "sensors": SENSORS,
           "regime_features": [], "asset_id": "unit", "time": "cycle"}
    path = save_bundle(cfg, tmp_path, normalizer=norm, detector=det,
                       reference=healthy_reference(healthy, norm.transform(healthy), det, []))
    got = score_with_bundle(spark.createDataFrame(df), str(path), "unit").toPandas().sort_values(["unit", "cycle"])
    exp = Scorer.load(path).score(df.sort_values(["unit", "cycle"]).reset_index(drop=True))
    np.testing.assert_allclose(got["health_score"].to_numpy(), exp["health_score"].to_numpy(), rtol=1e-9)
    assert (got["alarm_confirmed"].to_numpy() == exp["alarm_confirmed"].to_numpy()).all()
