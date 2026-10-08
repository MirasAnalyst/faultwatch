# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest
# MAGIC Land the latest historian extract (one row per machine per sample) as a Delta table.
# MAGIC In this repo the source is the C-MAPSS test fleet; on board it would be the vessel
# MAGIC historian export synced ashore (e.g. a Kongsberg / Wärtsilä / ABB data link).

# COMMAND ----------
dbutils.widgets.text("catalog", "marine_ops")
dbutils.widgets.text("schema", "faultwatch")
dbutils.widgets.text("source", "/Volumes/marine_ops/faultwatch/landing/historian/")
catalog, schema, source = (dbutils.widgets.get(k) for k in ("catalog", "schema", "source"))
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# COMMAND ----------
from pyspark.sql import functions as F

raw = (spark.read.option("header", True).option("inferSchema", True).csv(source)
       .withColumn("_ingested_at", F.current_timestamp()))
(raw.write.mode("append").option("mergeSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.sensor_raw"))
display(raw.groupBy("unit").count().orderBy("unit"))
