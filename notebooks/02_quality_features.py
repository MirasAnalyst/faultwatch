# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Data quality gate and features
# MAGIC Stops the job if the extract fails schema / null / range / duplicate checks, so a broken
# MAGIC sensor feed never reaches the operators as a wave of false alarms.

# COMMAND ----------
dbutils.widgets.text("catalog", "marine_ops")
dbutils.widgets.text("schema", "faultwatch")
dbutils.widgets.text("asset", "turbofan_cmapss_fd001")
catalog, schema, asset = (dbutils.widgets.get(k) for k in ("catalog", "schema", "asset"))

# COMMAND ----------
import json

import mlflow
from faultwatch.tracking import bundle_from_registry
from faultwatch.serve import Scorer
from pipelines.features_spark import quality_report, residual_features

mlflow.set_registry_uri("databricks-uc")
bundle_path = bundle_from_registry(f"models:/{catalog}.{schema}.faultwatch_{asset}@champion")
scorer = Scorer.load(bundle_path)
cfg, norm = scorer.cfg, scorer.norm
sdf = spark.table(f"{catalog}.{schema}.sensor_raw")

# COMMAND ----------
dq = quality_report(sdf, cfg["sensors"], [cfg["asset_id"], cfg["time"]], cfg.get("physical_ranges"))
spark.createDataFrame([{"asset": asset, "report": json.dumps(dq, default=str), "passed": dq["passed"]}]) \
    .write.mode("append").saveAsTable(f"{catalog}.{schema}.quality_reports")
if not dq["passed"]:
    raise RuntimeError(f"data-quality gate failed: {[c['check'] for c in dq['checks'] if not c['passed']]}")

# COMMAND ----------
n = cfg["normalization"]
feats = residual_features(sdf, cfg["sensors"], norm.global_mean_, norm.global_std_, cfg["asset_id"], cfg["time"],
                          n.get("per_asset_baseline", False), n.get("baseline_window", 30),
                          n.get("smoothing_window", 1), cfg.get("rul", {}).get("trend_window", 20))
feats.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.features_{asset}")
