# Capability map

Where each part of the end-to-end data-science lifecycle lives in this repo.

| Capability | Artifact | Evidence |
|---|---|---|
| Problem framing and value | `README.md` (questions, baselines), `docs/decision_memo.md` | Every result compared with today's alarm; $ and safety-event deltas |
| Predictive modeling | `faultwatch/models.py`, `selection.py`, experiments | LightGBM / XGBoost / CatBoost / linear, grouped CV comparison per asset |
| Forecasting | remaining-life regression (`run_to_failure`) | RMSE 12.3 cycles (CI 10.3-14.4) on 100 unseen engines |
| Classification / ranking | fault diagnosis, component grading, risk register ranking | per-asset metrics, `safety.py` |
| Prescriptive / optimization | `schedule.py` (two MILPs, PuLP/CBC), `experiments/power_plant.py` (Monte Carlo) | Blackouts 2.7 → 0 per 100 voyages (simulated) |
| GenAI / RAG | `faultwatch/genai/` (hybrid retrieval, providers, guardrails, triage) | `reports/incident_copilot/`, `evals/` |
| Prompt engineering and evaluation | `genai/prompts.py`, `genai/evals.py`, `deploy/azure_foundry/` | citation validity, faithfulness, refusals, wording-style retrieval |
| Experimentation and causal inference | `faultwatch/stats.py`, `experiments/rollout.py`, `docs/pilot_design.md` | bootstrap CIs, McNemar, permutation, Wilcoxon, power, CUPED, DiD, RI |
| Explainability | `explain.py` (SHAP), detector contributions | SHAP charts per asset, sensor shares per alert |
| Production deployment | `api.py`, `Dockerfile`, `deploy/azureml/`, `databricks.yml` | one `Scorer` for API, dashboard, Spark and Azure ML |
| Data platform | `pipelines/features_spark.py`, `notebooks/` | DQ gate; Spark features parity-tested to 1e-6 |
| MLOps | `tracking.py` (MLflow, registry incl. Unity Catalog), CI | `.github/workflows/ci.yml` |
| Monitoring | `serve.py` (drift), `monitor.py` (latency, cost, adoption, precision, slices) | `GET /monitoring`, dashboard Model health |
| Stakeholder communication | `docs/decision_memo.md`, `docs/model_cards/`, dashboard | plain-language limits in every section |
