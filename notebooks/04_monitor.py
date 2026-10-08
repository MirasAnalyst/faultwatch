# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Monitor
# MAGIC Drift of the healthy baseline, alarm volume, and the retrain / recalibrate / process-review
# MAGIC recommendation. Adoption and precision come from the alert acknowledgements and work-order
# MAGIC outcomes the API records (`/alerts/{id}/ack`, `/alerts/{id}/label`).

# COMMAND ----------
dbutils.widgets.text("catalog", "marine_ops")
dbutils.widgets.text("schema", "faultwatch")
dbutils.widgets.text("asset", "turbofan_cmapss_fd001")
catalog, schema, asset = (dbutils.widgets.get(k) for k in ("catalog", "schema", "asset"))

# COMMAND ----------
import json

import mlflow
from faultwatch.tracking import bundle_from_registry
from faultwatch.monitor import retrain_policy
from faultwatch.serve import Scorer

mlflow.set_registry_uri("databricks-uc")
path = bundle_from_registry(f"models:/{catalog}.{schema}.faultwatch_{asset}@champion")
scorer = Scorer.load(path)
recent = spark.table(f"{catalog}.{schema}.sensor_raw").toPandas()
drift = scorer.drift(recent)
alarms = spark.table(f"{catalog}.{schema}.scores_{asset}").selectExpr("avg(cast(alarm as double)) as rate").first()["rate"]
actions = retrain_policy(drift, {})
spark.createDataFrame([{"asset": asset, "drift": json.dumps(drift, default=str), "alarm_rate": float(alarms),
                        "actions": json.dumps(actions)}]).write.mode("append").saveAsTable(f"{catalog}.{schema}.monitoring")
if any(a["action"] == "retrain" for a in actions):
    dbutils.jobs.taskValues.set("retrain", True)       # picked up by an alert / the retrain job
display(actions)
