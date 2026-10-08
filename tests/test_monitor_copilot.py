"""Monitoring store, retrain policy, copilot guardrails and the new API endpoints."""
import gzip
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from faultwatch.genai.copilot import Copilot, alert_to_question, scrub
from faultwatch.genai.index import Retriever
from faultwatch.genai.llm import ExtractiveProvider
from faultwatch.monitor import MonitorStore, retrain_policy, slice_parity

DOCS = {
    "NTSB-MIR9901": ("Engine room fire aboard passenger vessel", "Fire/Explosion", [
        "A fuel oil leak from a cracked high-pressure fuel injection pipe on the No. 2 diesel generator "
        "sprayed onto the uninsulated turbocharger exhaust casing and ignited. The engine room fire spread "
        "to cable trays.",
        "The fuel injection pipe had no double-walled jacket and the hot surface insulation was missing after "
        "maintenance. Probable cause: failure to restore exhaust insulation, allowing fuel spray ignition."]),
    "NTSB-MIR9902": ("Loss of propulsion and blackout of cruise ship", "Machinery damage", [
        "In heavy seas the lube oil sump level of the main diesel generators dropped and low lube oil pressure "
        "trips shut down all engines, causing a blackout and loss of propulsion near a lee shore.",
        "The sump levels were kept at the low end of the allowed range. Ship motion caused suction loss. "
        "Lessons learned: keep lube oil sump levels high before heavy weather and test the blackout recovery."]),
    "NTSB-MIR9903": ("Steering gear failure and grounding", "Grounding/Stranding", [
        "The steering gear hydraulic pump developed internal leakage and the rudder failed to respond; "
        "the vessel grounded. The hydraulic power unit oil temperature was high for weeks before.",
        "The second steering pump was not started because the crew had not practised the changeover. "
        "Probable cause: loss of steering due to hydraulic pump wear and the failure to start the standby unit."]),
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    chunks, man = [], []
    for doc, (title, ctype, texts) in DOCS.items():
        man.append({"doc_id": doc, "source": "ntsb", "title": title, "casualty_type": ctype, "date": "",
                    "url": f"https://example/{doc}", "license": "public domain"})
        for i, t in enumerate(texts):
            chunks.append({"doc_id": doc, "section": "Analysis", "text": t, "chunk_id": f"{doc}#{i:03d}"})
    pd.DataFrame(man).to_csv(d / "manifest.csv", index=False)
    with gzip.open(d / "chunks.jsonl.gz", "wt") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return d


def test_retrieval_finds_the_right_report(corpus):
    r = Retriever(corpus, dense="lsa")
    assert r.top_docs("diesel generators tripped on low lube oil pressure in rough weather", k=1) == ["NTSB-MIR9902"]
    assert r.top_docs("fuel spray on hot exhaust ignited", k=1, mode="bm25") == ["NTSB-MIR9901"]


def test_copilot_cites_only_retrieved_passages_and_refuses_off_topic(corpus):
    cp = Copilot(Retriever(corpus, dense="lsa"), ExtractiveProvider())
    a = cp.answer("steering hydraulic pump leakage, rudder not responding - similar cases?")
    assert not a["insufficient_evidence"] and a["citations"]
    assert a["invalid_citations"] == []
    assert any(c["doc_id"] == "NTSB-MIR9903" for c in a["citations"])
    off = cp.answer("what time does the pool deck open for guests")
    assert off["insufficient_evidence"] and off["provider"] == "guardrail"


def test_alert_question_and_triage(corpus):
    q = alert_to_question({"component": "pump_leak", "top_sensors": {"TS1_max": 0.5}, "effect": "loss_of_steering"},
                          {"copilot": {"system_terms": "steering gear hydraulic power unit",
                                       "sensor_terms": {"TS1_max": "oil temperature"}}})
    assert "steering gear" in q and "oil temperature" in q and "loss of steering" in q
    cp = Copilot(Retriever(corpus, dense="lsa"), ExtractiveProvider())
    t = cp.triage("A fuel leak sprayed onto the hot turbocharger and started a fire in the engine room.",
                  exclude_doc=None)
    assert t["event_type"] == "fire_explosion"
    assert scrub("call +1 305 555 0100 or ops@ship.com") == "call [phone] or [email]"


def test_monitor_adoption_accuracy_and_policy(tmp_path):
    m = MonitorStore(tmp_path / "m.db")
    m.log_call("assets/score", "gt", rows=10, latency_ms=12.0, cpu_ms=5.0, alarms=2)
    ids = m.record_alerts("gt", pd.DataFrame({"machine": ["A"] * 15 + ["B"] * 15, "health_score": 50.0,
                                              "slice": ["Radiance"] * 15 + ["Oasis"] * 15}))
    for i in ids[:5]:
        m.ack(i, "work_order")
    for k, i in enumerate(ids):                     # Radiance alarms mostly wrong, Oasis mostly right
        m.label(i, "no_fault_found" if (k < 15 and k % 5) else "fault_confirmed")
    s = m.summary()
    assert s["service"]["calls"] == 1 and s["adoption"]["ack_rate"] == pytest.approx(5 / 30)
    assert s["accuracy"]["labelled_alerts"] == 30
    assert s["slices"]["disparity_flag"] and s["slices"]["worst_slice"] == "Radiance"
    actions = {r["action"] for r in s["retrain_policy"]}
    assert {"process_review", "investigate_slice"} <= actions
    assert retrain_policy({"retrain_recommended": True, "reasons": ["x"]}, {})[0]["action"] == "retrain"
    with pytest.raises(ValueError):
        m.ack(ids[0], "ignored")


def test_slice_parity_flags_bad_regime():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"regime": np.repeat(["low", "high"], 400), "fault": np.tile([True, False], 400)})
    df["alarm"] = np.where(df.fault, rng.random(800) < np.where(df.regime == "low", 0.95, 0.5), rng.random(800) < 0.02)
    t = slice_parity(df, "regime", "alarm", "fault").set_index("slice")
    assert t.loc["high", "flag"] and not t.loc["low", "flag"]


def test_api_new_endpoints(corpus, tmp_path, monkeypatch):
    from test_serve import CFG, _plant

    from faultwatch.anomaly import HealthDetector
    from faultwatch.regime import RegimeNormalizer
    from faultwatch.serve import healthy_reference, save_bundle
    h = _plant(seed=1)
    norm = RegimeNormalizer(["a", "b"], ["load"]).fit(h)
    det = HealthDetector(quantile=0.99).fit(norm.transform(h))
    cfg = {**CFG, "safety": {"failure_modes": {"degradation": {"effect": "loss_of_propulsion", "severity": 3}}}}
    save_bundle(cfg, tmp_path, normalizer=norm, detector=det,
                reference=healthy_reference(h, norm.transform(h), det, ["load"]))
    (tmp_path / "corpus").symlink_to(corpus)
    monkeypatch.setenv("FAULTWATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("FAULTWATCH_LLM", "extractive")
    sys.modules.pop("api", None)
    api = importlib.import_module("api")
    c = TestClient(api.app)
    rows = _plant(n=20, fault=5.0, seed=8).to_dict("records")
    r = c.post("/assets/toy_pump/score?record=true", json={"rows": rows}).json()
    assert len(r["alert_ids"]) == r["alarms"] >= 18
    assert c.post(f"/alerts/{r['alert_ids'][0]}/ack", json={"action": "inspected"}).status_code == 200
    assert c.post("/alerts/nope/ack", json={"action": "inspected"}).status_code == 404
    assert c.post(f"/alerts/{r['alert_ids'][0]}/label", json={"outcome": "fault_confirmed"}).status_code == 200
    reg = c.post("/assets/toy_pump/risk", json={"rows": rows}).json()["register"]
    assert reg and reg[0]["effect"] == "loss_of_propulsion"
    mon = c.get("/monitoring").json()
    assert mon["service"]["calls"] >= 2 and mon["adoption"]["alerts"] >= 18 and mon["accuracy"]["precision"] == 1.0
    a = c.post("/copilot/ask", json={"question": "blackout after low lube oil pressure trips in heavy seas"}).json()
    assert a["citations"] and a["citations"][0]["doc_id"] == "NTSB-MIR9902"
    t = c.post("/copilot/triage", json={"narrative": "Fuel sprayed from a cracked injection pipe onto the "
                                                     "exhaust and ignited, starting an engine room fire."}).json()
    assert t["event_type"] == "fire_explosion"
