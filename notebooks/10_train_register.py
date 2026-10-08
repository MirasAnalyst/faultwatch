# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Train and register
# MAGIC Refit each asset type from its config, log runs to the workspace MLflow, and register a new
# MAGIC version in Unity Catalog. Promotion to `@champion` is a reviewed step, not automatic.

# COMMAND ----------
dbutils.widgets.text("catalog", "marine_ops")
dbutils.widgets.text("schema", "faultwatch")
dbutils.widgets.text("configs", "configs/turbofan_cmapss.yaml,configs/naval_gas_turbine.yaml")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
configs = dbutils.widgets.get("configs").split(",")

# COMMAND ----------
import os
from pathlib import Path

import mlflow
from faultwatch.config import load_config
from faultwatch.experiments import EXPERIMENTS
from faultwatch.serve import MODELS_DIR
from faultwatch.tracking import log_run

os.environ["MLFLOW_TRACKING_URI"] = "databricks"
mlflow.set_registry_uri("databricks-uc")
for c in configs:
    cfg = load_config(c)
    out = Path("/tmp/reports") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    metrics = EXPERIMENTS[cfg["experiment"]](cfg, out)
    bundle = Path(cfg["_root"]) / MODELS_DIR / f"{cfg['name']}.joblib"
    print(cfg["name"], log_run(cfg, metrics, out, bundle,
                               registered_name=f"{catalog}.{schema}.faultwatch_{cfg['name']}"))
