"""Scoring bundle, drift monitor and API on a synthetic asset - no downloads needed."""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from test_core import _plant

from faultwatch.anomaly import HealthDetector
from faultwatch.regime import RegimeNormalizer
from faultwatch.serve import Scorer, healthy_reference, save_bundle

CFG = {"name": "toy_pump", "experiment": "steady_state", "dataset": "synthetic",
       "sensors": ["a", "b"], "regime_features": ["load"], "asset_id": None, "time": None}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fw")
    h = _plant(seed=1)
    norm = RegimeNormalizer(["a", "b"], ["load"]).fit(h)
    Z = norm.transform(h)
    det = HealthDetector(quantile=0.99).fit(Z)
    save_bundle(CFG, root, normalizer=norm, detector=det,
                reference=healthy_reference(h, Z, det, ["load"]))
    return root


def test_scorer_flags_fault_and_blames_sensor(root):
    s = Scorer.load(root / "models" / "toy_pump.joblib")
    ok, bad = s.score(_plant(seed=3)), s.score(_plant(fault=5.0, seed=4))
    assert ok["alarm"].mean() < 0.03 and bad["alarm"].mean() > 0.9
    assert max(bad["top_sensors"].iloc[0], key=bad["top_sensors"].iloc[0].get) == "b"


def test_drift_monitor(root):
    s = Scorer.load(root / "models" / "toy_pump.joblib")
    assert not s.drift(_plant(seed=5))["retrain_recommended"]
    shifted = _plant(fault=1.5, seed=6)            # small shift that stays under the alarm
    assert s.drift(shifted)["retrain_recommended"]
    new_load = _plant(seed=7).assign(load=1.4)     # operating point never trained on
    assert s.drift(new_load)["out_of_envelope_share"] == 1.0


def test_api(root, monkeypatch):
    monkeypatch.setenv("FAULTWATCH_ROOT", str(root))
    sys.modules.pop("api", None)
    api = importlib.import_module("api")
    c = TestClient(api.app)
    assert "toy_pump" in c.get("/assets").json()
    rows = _plant(n=20, fault=5.0, seed=8).to_dict("records")
    r = c.post("/assets/toy_pump/score", json={"rows": rows})
    assert r.status_code == 200 and r.json()["alarms"] >= 18
    assert c.post("/assets/toy_pump/score", json={"rows": [{"a": 1}]}).status_code == 422
    assert c.post("/assets/nope/score", json={"rows": rows}).status_code == 404
    assert c.post("/assets/toy_pump/plan", json={"rows": rows}).status_code == 400
