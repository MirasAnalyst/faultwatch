"""Copilot evaluation: retrieval, grounded answers, refusals and triage.

    python -m faultwatch.genai.evals --offline              # free, deterministic (CI)
    python -m faultwatch.genai.evals --dense st             # local PyTorch embeddings
    FAULTWATCH_LLM=azure_openai python -m faultwatch.genai.evals --judge   # real LLM + LLM judge

Gold set: evals/gold_questions.jsonl, one JSON object per line
    {"id": ..., "question": ..., "gold": [doc_id, ...]}     # gold [] = off-topic, must refuse
Questions describe a failure mechanism the way an engineer would, without
naming the vessel, so a hit means retrieval found the mechanism, not the name.

Metrics
  retrieval   hit@1/3/5 and MRR@10 of the gold report, per mode (BM25 / dense / hybrid)
  answers     citation validity (every cited tag was retrieved), gold-report cited,
              faithfulness (share of answer sentences supported by the cited passage:
              token overlap offline, an LLM judge with --judge), refusal accuracy on
              off-topic questions, false refusals on answerable ones, latency, cost
  triage      event-type accuracy and macro-F1 on NTSB reports whose structured
              'casualty type' is the label (narrative stripped of that field and title),
              against a majority-class baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import prompts
from .copilot import TAG_RE, Copilot
from .index import Retriever, tokenize
from .llm import ExtractiveProvider, get_provider

ROOT = Path(__file__).resolve().parents[2]


def load_gold(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def retrieval_metrics(r: Retriever, gold: pd.DataFrame, mode: str) -> dict:
    rows = []
    for g in gold[gold.gold.map(len) > 0].itertuples():
        docs = r.top_docs(g.question, k=10, mode=mode)
        rank = next((i + 1 for i, d in enumerate(docs) if d in g.gold), None)
        rows.append({"id": g.id, "rank": rank})
    d = pd.DataFrame(rows)
    rr = d["rank"].map(lambda x: 1 / x if x else 0.0)
    return {"questions": len(d), **{f"hit@{k}": float((d["rank"] <= k).mean()) for k in (1, 3, 5)},
            "mrr@10": float(rr.mean()), "_ranks": d}


def _supported(sentence: str, evidence: dict[str, str]) -> bool:
    tags = TAG_RE.findall(sentence)
    words = set(tokenize(TAG_RE.sub("", sentence)))
    if not words or not tags:
        return False
    best = max((len(words & set(tokenize(evidence.get(t, "")))) / len(words) for t in tags), default=0)
    return best >= 0.6


def answer_metrics(cp: Copilot, gold: pd.DataFrame, judge=None) -> tuple[dict, pd.DataFrame]:
    rows, judge_cost, judge_failures = [], 0.0, 0
    for g in gold.itertuples():
        t0 = time.perf_counter()
        a = cp.answer(g.question)
        ev = {h["chunk_id"]: t for h, t in zip(a["retrieved"], cp.r.search(g.question, k=cp.k).text)}
        # a claim ends at its citation tag(s): split after ']' or at line breaks
        sentences = [s for s in re.split(r"(?<=\])\s+(?=[A-Z0-9-])|\n", a["answer"]) if len(s) > 30]
        # short uncited lines are headings ("1. Likely causes"), not claims
        sentences = [s for s in sentences if TAG_RE.search(s) or len(s.split()) >= 8]
        if judge is not None and a.get("evidence") and not a["insufficient_evidence"]:
            jr = judge.complete(prompts.JUDGE_SYSTEM, prompts.JUDGE_USER.format(answer=a["answer"],
                                                                                evidence=a["evidence"]),
                                json_mode=True, max_tokens=1000)
            judge_cost += jr.cost_usd
            try:
                j = jr.json()
                n_sup, n_uns = int(j["supported_sentences"]), int(j["unsupported_sentences"])
                sup = n_sup / (n_sup + n_uns) if n_sup + n_uns else np.nan
            except (ValueError, KeyError, TypeError):
                sup = np.nan                      # unparseable verdict: missing, not "unsupported"
                judge_failures += 1
        else:
            sup = np.mean([_supported(s, ev) for s in sentences]) if sentences else np.nan
        cited_docs = {c["doc_id"] for c in a["citations"]}
        rows.append({"id": g.id, "answerable": bool(g.gold), "refused": a["insufficient_evidence"],
                     "citations": len(a["citations"]), "invalid_citations": len(a["invalid_citations"]),
                     "gold_cited": bool(cited_docs & set(g.gold)) if g.gold else None,
                     "faithfulness": sup, "latency_s": time.perf_counter() - t0,
                     "cost_usd": a.get("cost_usd", 0.0), "provider": a.get("provider")})
    d = pd.DataFrame(rows)
    on, off = d[d.answerable], d[~d.answerable]
    answered = on[~on.refused]
    return {"provider": cp.llm.name, "model": getattr(cp.llm, "model", ""),
            "citation_validity": float((answered.invalid_citations == 0).mean()) if len(answered) else None,
            "answers_with_citations": float((answered.citations > 0).mean()) if len(answered) else None,
            "gold_report_cited": float(answered.gold_cited.mean()) if len(answered) else None,
            "faithfulness": float(answered.faithfulness.mean()) if len(answered) else None,
            "false_refusal_rate": float(on.refused.mean()) if len(on) else None,
            "off_topic_refusal_rate": float(off.refused.mean()) if len(off) else None,
            "latency_s_p50": float(d.latency_s.median()), "latency_s_p95": float(d.latency_s.quantile(0.95)),
            "cost_usd_per_answer": float(d.cost_usd.mean()),
            **({"judge_cost_usd": judge_cost, "judge_failures": judge_failures} if judge is not None else {})}, d


def triage_metrics(cp: Copilot, limit: int | None = None) -> dict:
    man = cp.r.manifest
    lab = man[(man.source == "ntsb") & (man.casualty_type != "")]
    if limit:
        lab = lab.sample(min(limit, len(lab)), random_state=0)
    y, p, cost = [], [], 0.0
    for doc in lab.itertuples():
        ch = cp.r.chunks[cp.r.chunks.doc_id == doc.doc_id].head(2)
        text = " ".join(ch.text)
        text = re.sub(r"(?:Accident|Casualty)\s+[Tt]ype\s+[A-Za-z/ ,\-]+?(?=\s+Location|\s{2}|$)", " ", text)
        text = text.replace(doc.title, " ")
        y.append(prompts.ntsb_event_class(doc.casualty_type))
        t = cp.triage(text, exclude_doc=doc.doc_id)
        p.append(t["event_type"])
        cost += t.get("cost_usd", 0.0)
    y, p = np.array(y), np.array(p)
    majority = pd.Series(y).mode()[0] if len(y) else None
    return {"reports": int(len(y)), "method": "llm" if not isinstance(cp.llm, ExtractiveProvider) else "knn_retrieval",
            "accuracy": float(accuracy_score(y, p)) if len(y) else None,
            "macro_f1": float(f1_score(y, p, average="macro")) if len(y) else None,
            "majority_class": majority, "majority_baseline_accuracy": float((y == majority).mean()) if len(y) else None,
            "label_distribution": pd.Series(y).value_counts().to_dict(), "cost_usd_total": cost}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--gold", default=str(ROOT / "evals" / "gold_questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "incident_copilot"))
    ap.add_argument("--offline", action="store_true", help="extractive answers, LSA dense retrieval (no network)")
    ap.add_argument("--dense", default=None, help="st | lsa | azure (default: lsa offline, else auto)")
    ap.add_argument("--judge", action="store_true", help="grade faithfulness with the LLM (needs a key)")
    ap.add_argument("--triage-limit", type=int, default=None)
    ap.add_argument("--min-hit-at-5", type=float, default=None, help="fail (exit 1) below this hybrid hit@5")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args(argv)

    dense = a.dense or ("lsa" if a.offline else "auto")
    r = Retriever(a.corpus, dense=dense)
    gold = load_gold(Path(a.gold))
    provider = ExtractiveProvider() if a.offline else get_provider()
    cp = Copilot(r, provider)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    retrieval, ranks = {}, []
    styles = sorted(gold.get("style", pd.Series(["all"] * len(gold))).fillna("all").unique())
    for mode in ("bm25", "dense", "hybrid"):
        m = retrieval_metrics(r, gold, mode)
        ranks.append(m.pop("_ranks").assign(mode=mode))
        retrieval[mode] = m
        if "style" in gold:                          # technical wording vs how a watchkeeper talks
            for st in styles:
                g = gold[gold["style"] == st]
                if (g.gold.map(len) > 0).any():
                    sm = retrieval_metrics(r, g, mode)
                    sm.pop("_ranks")
                    retrieval[mode][f"by_style.{st}"] = sm
    answers, per_q = answer_metrics(cp, gold, judge=provider if a.judge and not a.offline else None)
    triage = triage_metrics(cp, a.triage_limit)
    metrics = {"corpus": {"reports": int(len(r.manifest)), "chunks": int(len(r.chunks)),
                          "by_source": r.manifest.source.value_counts().to_dict()},
               "gold_questions": {"answerable": int((gold.gold.map(len) > 0).sum()),
                                  "off_topic": int((gold.gold.map(len) == 0).sum())},
               "dense_backend": r.dense_name, "prompt_version": prompts.PROMPT_VERSION,
               "retrieval": retrieval, "answers": answers, "triage": triage}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    pd.concat(ranks).to_csv(out / "retrieval_ranks.csv", index=False)
    per_q.to_csv(out / "answers_per_question.csv", index=False)
    print(json.dumps(metrics, indent=2, default=str))
    if a.mlflow:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("faultwatch-copilot")
        with mlflow.start_run(run_name=f"copilot-{provider.name}-{r.dense_name}"):
            mlflow.log_params({"provider": provider.name, "dense": r.dense_name,
                               "prompt_version": prompts.PROMPT_VERSION})
            for mode, m in retrieval.items():
                mlflow.log_metrics({f"{mode}.{k.replace('@', '_at_')}": v for k, v in m.items()
                                    if isinstance(v, float)})
            mlflow.log_metrics({f"answers.{k}": v for k, v in answers.items() if isinstance(v, float)})
            mlflow.log_metrics({f"triage.{k}": v for k, v in triage.items() if isinstance(v, float)})
            mlflow.log_artifacts(str(out), artifact_path="reports")
    if a.min_hit_at_5 is not None and retrieval["hybrid"]["hit@5"] < a.min_hit_at_5:
        print(f"FAIL: hybrid hit@5 {retrieval['hybrid']['hit@5']:.2f} < {a.min_hit_at_5}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
