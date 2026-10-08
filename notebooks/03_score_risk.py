# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Score the fleet and refresh the safety-risk register
# MAGIC The registered model bundle runs per machine with `applyInPandas`: the same `Scorer`
# MAGIC the API uses, so batch and online scores agree.

# COMMAND ----------
dbutils.widgets.text("catalog", "marine_ops")
dbutils.widgets.text("schema", "faultwatch")
dbutils.widgets.text("asset", "turbofan_cmapss_fd001")
dbutils.widgets.text("horizon_days", "30")
catalog, schema, asset = (dbutils.widgets.get(k) for k in ("catalog", "schema", "asset"))
horizon = float(dbutils.widgets.get("horizon_days"))

# COMMAND ----------
import joblib
import mlflow
from faultwatch.tracking import bundle_from_registry
from faultwatch.safety import risk_register
from pipelines.features_spark import score_with_bundle

mlflow.set_registry_uri("databricks-uc")
path = bundle_from_registry(f"models:/{catalog}.{schema}.faultwatch_{asset}@champion")
bundle_file = path
bundle = joblib.load(bundle_file)
cfg = bundle["cfg"]
scores = score_with_bundle(spark.table(f"{catalog}.{schema}.sensor_raw"), bundle_file, cfg["asset_id"])
scores.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.scores_{asset}")

# COMMAND ----------
latest = scores.toPandas().sort_values(cfg["time"]).groupby(cfg["asset_id"]).tail(1)
register = risk_register(latest, bundle, horizon, machine=latest[cfg["asset_id"]], latest_only=False)
spark.createDataFrame(register.astype(str)).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.risk_register_{asset}")
display(register.head(20))
