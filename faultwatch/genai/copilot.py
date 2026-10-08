"""Machinery Incident Copilot.

Two workflows:

1. Alert -> lessons. A FaultWatch alert (asset, diagnosis, the sensors driving
   it, the FMECA effect) becomes a retrieval query. The answer gives likely
   causes, checks and the barriers that failed in similar past casualties,
   every sentence cited to a report passage.

2. Failure-report triage. A free-text narrative becomes a structured record
   (event type, system, ship-level effect, severity) for the safety register.

Guardrails: answers must cite retrieved passages (citations are validated
against what was retrieved); weak retrieval returns INSUFFICIENT_EVIDENCE
instead of an answer; contact details are scrubbed from inputs; the prompts
frame the output as decision support for an engineer.
"""
from __future__ import annotations

import re
from collections import Counter

from ..safety import EFFECTS
from . import prompts
from .index import Retriever, tokenize
from .llm import ExtractiveProvider, get_provider

TAG_RE = re.compile(r"\[([A-Z]+-[\w.\-]+#\d{3})\]")
PII_RE = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
          (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "[phone]")]
MIN_EVIDENCE_OVERLAP = 2      # query terms that must appear in the best passage
# Scope gate: the copilot answers machinery / machinery-safety questions only. A question
# must name at least one of these (stems match by prefix), so guest-services or general
# questions are refused even when they share everyday words with an accident report.
DOMAIN_STEMS = (
    "engine", "generat", "propuls", "propel", "steer", "rudder", "fuel", "oil", "lube", "lubric", "fire",
    "explos", "blackout", "power", "electric", "switchboard", "breaker", "voltage", "pump", "valve", "bearing",
    "turbin", "turbo", "compressor", "hydraul", "boiler", "burner", "exhaust", "crank", "cylinder", "piston",
    "shaft", "gear", "clutch", "governor", "thruster", "azipod", "pod", "alarm", "trip", "pressure",
    "temperat", "vibrat", "leak", "flood", "bilge", "cooling", "seawater", "machinery", "overhaul",
    "maintenance", "battery", "lithium", "co2", "extinguish", "ventilat", "grounding", "collision",
    "allision", "contact", "drift", "lost", "loss", "failure", "fault", "manoeuv", "maneuv", "berth", "weather",
)


def in_scope(question: str) -> bool:
    return any(t.startswith(DOMAIN_STEMS) for t in tokenize(question))


def scrub(text: str) -> str:
    for rx, repl in PII_RE:
        text = rx.sub(repl, text)
    return text


def alert_to_question(alert: dict, cfg: dict | None = None) -> str:
    """Turn a scored alert into the question an engineer would ask.

    alert keys (all optional): asset_name, diagnosis, top_sensors {name: share},
    effect, note, component."""
    cfg = cfg or {}
    terms = cfg.get("copilot", {}).get("sensor_terms", {})
    system = cfg.get("copilot", {}).get("system_terms", cfg.get("short_name", alert.get("asset_name", "machinery")))
    sensors = [terms.get(s, s.replace("_", " ")) for s in (alert.get("top_sensors") or {})]
    parts = [f"{system}:"]
    if alert.get("diagnosis") or alert.get("component"):
        parts.append(f"{alert.get('diagnosis') or alert.get('component')} fault suspected".replace("_", " "))
    if sensors:
        parts.append("abnormal " + ", ".join(sensors[:3]))
    if alert.get("effect"):
        parts.append(f"risk of {EFFECTS.get(alert['effect'], alert['effect']).lower()}")
    if alert.get("note"):
        parts.append(alert["note"])
    return "; ".join(parts) + ". What caused similar failures, what should be checked, and which barriers failed?"


class Copilot:
    def __init__(self, retriever: Retriever | None = None, provider=None, corpus_dir="corpus", k: int = 6):
        self.r = retriever or Retriever(corpus_dir)
        self.llm = provider or get_provider()
        self.k = k

    def _evidence(self, hits):
        return "\n".join(f"[{h.chunk_id}] ({h.title} - {h.section}) {h.text}" for h in hits.itertuples())

    def answer(self, question: str) -> dict:
        question = scrub(question)
        hits = self.r.search(question, k=self.k)
        qterms = set(tokenize(question))
        best_overlap = max((len(qterms & set(tokenize(t))) for t in hits.text), default=0)
        base = {"question": question, "retrieved": hits[["chunk_id", "doc_id", "title", "section", "url"]]
                .to_dict("records")}
        if not in_scope(question) or best_overlap < MIN_EVIDENCE_OVERLAP:
            why = ("outside the copilot's scope (machinery and machinery-safety questions only)"
                   if not in_scope(question) else "no investigation report in the corpus addresses this")
            return {**base, "answer": f"INSUFFICIENT_EVIDENCE: {why}.",
                    "insufficient_evidence": True, "citations": [], "invalid_citations": [],
                    "provider": "guardrail", "cost_usd": 0.0, "latency_s": 0.0, "tokens": 0}
        evidence = self._evidence(hits)
        r = self.llm.complete(prompts.ANSWER_SYSTEM,
                              prompts.ANSWER_USER.format(question=question, evidence=evidence))
        cited = list(dict.fromkeys(TAG_RE.findall(r.text)))
        allowed = set(hits.chunk_id)
        meta = hits.set_index("chunk_id")
        return {**base, "answer": r.text, "insufficient_evidence": r.text.startswith("INSUFFICIENT_EVIDENCE"),
                "citations": [{"tag": t, "doc_id": meta.at[t, "doc_id"], "title": meta.at[t, "title"],
                               "url": meta.at[t, "url"]} for t in cited if t in allowed],
                "invalid_citations": [t for t in cited if t not in allowed],
                "evidence": evidence, "provider": r.provider, "model": r.model,
                "tokens": r.input_tokens + r.output_tokens, "cost_usd": r.cost_usd, "latency_s": r.latency_s,
                "prompt_version": prompts.PROMPT_VERSION}

    def lessons_for_alert(self, alert: dict, cfg: dict | None = None) -> dict:
        return self.answer(alert_to_question(alert, cfg))

    def triage(self, narrative: str, exclude_doc: str | None = None) -> dict:
        """Structured classification of a casualty narrative. With an LLM: a JSON
        call. Offline: retrieval k-nearest-neighbour vote over reports whose
        event type is known (the baseline the LLM must beat)."""
        narrative = scrub(narrative)
        if isinstance(self.llm, ExtractiveProvider):
            hits = self.r.search(narrative, k=15, per_doc=1, exclude_docs={exclude_doc} if exclude_doc else None)
            types = self.r.manifest.set_index("doc_id")["casualty_type"]
            labelled = [d for d in hits.doc_id if types.get(d, "")][:7]
            # class priors of the labelled corpus: a vote is divided by its class's share,
            # otherwise the most common event type (fires) wins every close call
            prior = Counter(prompts.ntsb_event_class(t) for t in types[types != ""])
            total = sum(prior.values())
            votes = Counter()
            for rank, d in enumerate(labelled):              # nearer reports count more
                c = prompts.ntsb_event_class(types[d])
                votes[c] += (1 / (rank + 1)) / (prior[c] / total)
            votes = {k: round(v, 2) for k, v in votes.most_common()}
            ev = next(iter(votes), "other")
            return {"event_type": ev, "system": "other", "ship_level_effect": "unspecified", "severity": 2,
                    "rationale": f"nearest-report vote {votes}", "provider": "knn", "cost_usd": 0.0,
                    "latency_s": 0.0}
        r = self.llm.complete(
            prompts.TRIAGE_SYSTEM.format(event_types=prompts.EVENT_TYPES, systems=prompts.SYSTEMS,
                                         effects=list(EFFECTS)),
            prompts.TRIAGE_USER.format(narrative=narrative[:6000]), json_mode=True, max_tokens=400)
        out = r.json()
        if out.get("event_type") not in prompts.EVENT_TYPES:
            out["event_type"] = "other"
        return {**out, "provider": r.provider, "cost_usd": r.cost_usd, "latency_s": r.latency_s}
