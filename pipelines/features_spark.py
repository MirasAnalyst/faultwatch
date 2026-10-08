"""Spark feature + batch-scoring pipeline (runs locally or as a Databricks job).

    python pipelines/features_spark.py --config configs/turbofan_cmapss.yaml            # local[*]
    databricks bundle run faultwatch_daily                                              # on Databricks

Stages
  1. ingest      raw historian extract -> Spark DataFrame (CSV/parquet locally, Delta on Databricks)
  2. quality     data-quality report: schema, nulls, physical ranges, duplicate keys,
                 stuck sensors, freshness. Hard failures stop the job before scoring.
  3. features    regime residuals, per-machine early-life baseline, smoothing and
                 rolling trend slopes in native Spark window functions - the same
                 numbers as faultwatch.regime / faultwatch.models (parity-tested)
  4. score       the fitted model bundle applied per machine with applyInPandas,
                 so the exact Scorer used by the API runs distributed across the fleet
  5. write       features + scores to parquet / Delta, partitioned by run date

Normalization parameters are fitted offline on healthy history
(`python run.py ...`) and read from the model bundle; Spark only applies them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_spark(app: str = "faultwatch"):
    """Local session unless running on Databricks (where `spark` already exists)."""
    if not os.environ.get("JAVA_HOME"):
        for cand in glob.glob("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home") + \
                glob.glob("/usr/lib/jvm/*17*"):
            os.environ["JAVA_HOME"] = cand
            break
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)          # workers use this interpreter
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    return (SparkSession.builder.appName(app).master(os.environ.get("SPARK_MASTER", "local[*]"))
            .config("spark.sql.session.timeZone", "UTC").config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "8").getOrCreate())


# ---------------------------------------------------------------------------
# 2. data quality
def quality_report(sdf, sensors, keys, ranges=None, stuck_window=20, max_null_share=0.05):
    """Returns {checks: [...], passed: bool}. `ranges` = {sensor: (lo, hi)} physical limits."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    checks = []
    missing = [c for c in sensors + keys if c not in sdf.columns]
    checks.append({"check": "schema", "passed": not missing, "detail": {"missing_columns": missing}})
    if missing:
        return {"checks": checks, "passed": False}
    n = sdf.count()
    nulls = sdf.select([F.avg(F.col(c).isNull().cast("double")).alias(c) for c in sensors]).first().asDict()
    bad = {c: round(v, 4) for c, v in nulls.items() if v > max_null_share}
    checks.append({"check": "null_share", "passed": not bad, "detail": {"over_limit": bad, "rows": n}})
    dup = sdf.groupBy(*keys).count().filter("count > 1").count()
    checks.append({"check": "duplicate_keys", "passed": dup == 0, "detail": {"duplicated_keys": dup}})
    if ranges:
        out = sdf.select([F.sum(((F.col(s) < lo) | (F.col(s) > hi)).cast("int")).alias(s)
                          for s, (lo, hi) in ranges.items() if s in sdf.columns]).first().asDict()
        bad = {s: int(v) for s, v in out.items() if v}
        checks.append({"check": "physical_range", "passed": not bad, "detail": {"rows_out_of_range": bad}})
    if len(keys) == 2:
        w = Window.partitionBy(keys[0]).orderBy(keys[1]).rowsBetween(-(stuck_window - 1), 0)
        full = F.count(F.lit(1)).over(w) >= stuck_window              # judge full windows only
        st = sdf.select(keys[0], *[(full & (F.stddev(s).over(w) == 0)).alias(s) for s in sensors])
        stuck = st.groupBy(keys[0]).agg(*[F.max(F.col(s).cast("int")).alias(s) for s in sensors])
        cnt = stuck.select([F.sum(s).alias(s) for s in sensors]).first().asDict()
        bad = {s: int(v) for s, v in cnt.items() if v}
        # a flat-lined sensor is a warning, not a stop: the detector still sees the other sensors
        checks.append({"check": "stuck_sensor", "passed": True, "warning": bool(bad),
                       "detail": {"machines_with_flatline": bad, "window": stuck_window}})
        last = sdf.groupBy(keys[0]).agg(F.max(keys[1]).alias("last"))
        lag = last.select((F.max("last") - F.min("last")).alias("spread")).first()["spread"]
        checks.append({"check": "freshness", "passed": True, "warning": bool(lag and lag > 0),
                       "detail": {"latest_minus_oldest_last_reading": lag}})
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


# ---------------------------------------------------------------------------
# 3. features (native Spark)
def residual_features(sdf, sensors, mean, std, asset, time, per_asset_baseline=True, baseline_window=30,
                      smoothing_window=1, trend_window=20):
    """z-residuals, per-machine early-life baseline, rolling smoothing and
    rolling trend slope - mirrors RegimeNormalizer (no regime features) and
    models.add_trend_features."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    by_time = Window.partitionBy(asset).orderBy(time)
    df = sdf.withColumn("_rn", F.row_number().over(by_time))
    for s in sensors:
        df = df.withColumn(f"_r_{s}", F.col(s) - F.lit(float(mean[s])))
    if per_asset_baseline:
        base = (df.filter(F.col("_rn") <= baseline_window).groupBy(asset)
                  .agg(*[F.avg(f"_r_{s}").alias(f"_b_{s}") for s in sensors]))
        df = df.join(base, on=asset, how="left")
        for s in sensors:
            df = df.withColumn(f"_r_{s}", F.col(f"_r_{s}") - F.coalesce(F.col(f"_b_{s}"), F.lit(0.0)))
    smooth = by_time.rowsBetween(-(smoothing_window - 1), 0)
    for s in sensors:
        z = F.col(f"_r_{s}") / F.lit(float(std[s]))
        df = df.withColumn(f"z_{s}", F.avg(z).over(smooth) if smoothing_window > 1 else z)
    tw = by_time.rowsBetween(-(trend_window - 1), 0)
    t = F.col(time).cast("double")
    df = (df.withColumn("_tm", F.avg(t).over(tw)).withColumn("_tt", F.avg(t * t).over(tw))
            .withColumn("_n", F.count(t).over(tw)))
    var_t = F.col("_tt") - F.col("_tm") ** 2
    for s in sensors:
        xm, xt = F.avg(F.col(f"z_{s}")).over(tw), F.avg(F.col(f"z_{s}") * t).over(tw)
        slope = (xt - xm * F.col("_tm")) / var_t
        df = df.withColumn(f"slope_{s}", F.when((F.col("_n") >= 3) & (var_t != 0), slope).otherwise(0.0))
    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(*drop)


# ---------------------------------------------------------------------------
# 4. distributed scoring with the fitted bundle
def score_with_bundle(sdf, bundle_path: str, group_col: str):
    """Run faultwatch.serve.Scorer on each machine's history in parallel."""
    import pandas as pd
    from pyspark.sql import types as T

    from faultwatch.serve import Scorer
    scorer = Scorer.load(bundle_path)
    cols = scorer.required_columns
    probe = scorer.score(sdf.limit(50).toPandas()[cols])
    out_cols = [c for c in probe.columns if c != "top_sensors"]
    schema = T.StructType([T.StructField(group_col, sdf.schema[group_col].dataType),
                           T.StructField(scorer.cfg.get("time") or "_row", T.DoubleType())] +
                          [T.StructField(c, T.StringType() if probe[c].dtype == object else
                                         T.BooleanType() if probe[c].dtype == bool else T.DoubleType())
                           for c in out_cols] + [T.StructField("top_sensors", T.StringType())])
    time_col = scorer.cfg.get("time")

    def _score(pdf: pd.DataFrame) -> pd.DataFrame:
        s = Scorer.load(bundle_path)
        pdf = pdf.sort_values(time_col).reset_index(drop=True) if time_col else pdf.reset_index(drop=True)
        o = s.score(pdf[cols])
        res = pd.DataFrame({group_col: pdf[group_col].values,
                            (time_col or "_row"): (pdf[time_col] if time_col else pdf.index).astype(float).values})
        for c in out_cols:
            res[c] = o[c].values
        res["top_sensors"] = o["top_sensors"].map(json.dumps) if "top_sensors" in o else None
        return res

    return sdf.groupBy(group_col).applyInPandas(_score, schema=schema)


# ---------------------------------------------------------------------------
def main(argv=None):
    import joblib
    import pandas as pd

    from faultwatch.config import load_config
    from faultwatch.data import CMAPSS_COLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "turbofan_cmapss.yaml"))
    ap.add_argument("--input", default=None, help="raw sensor file (default: the C-MAPSS test fleet)")
    ap.add_argument("--out", default=str(ROOT / "data" / "features"))
    ap.add_argument("--format", default="parquet", choices=["parquet", "delta"])
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    bundle_path = ROOT / "models" / f"{cfg['name']}.joblib"
    b = joblib.load(bundle_path)
    norm = b["normalizer"]
    spark = get_spark()
    src = a.input or str(ROOT / cfg["data_dir"] / f"test_{cfg.get('subset', 'FD001')}.txt")
    raw = pd.read_csv(src, sep=r"\s+", header=None, names=CMAPSS_COLS)
    sdf = spark.createDataFrame(raw)
    asset, time = cfg["asset_id"], cfg["time"]

    # plausible physical envelope: the config's limits, else the whole training history
    # (healthy and worn) widened by 25% of its span - a reading outside it is a sensor fault
    ranges = cfg.get("physical_ranges")
    if not ranges:
        hist = pd.read_csv(ROOT / cfg["data_dir"] / f"train_{cfg.get('subset', 'FD001')}.txt", sep=r"\s+",
                           header=None, names=CMAPSS_COLS)[cfg["sensors"]]
        span = hist.max() - hist.min()
        ranges = {s: (float(hist[s].min() - 0.25 * span[s]), float(hist[s].max() + 0.25 * span[s]))
                  for s in cfg["sensors"]}
    dq = quality_report(sdf, cfg["sensors"], [asset, time], ranges)
    out = Path(a.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality_report.json").write_text(json.dumps(dq, indent=2, default=str))
    print(json.dumps(dq, indent=2, default=str))
    if not dq["passed"]:
        raise SystemExit("data-quality checks failed - not scoring")

    n = cfg["normalization"]
    feats = residual_features(sdf, cfg["sensors"], norm.global_mean_, norm.global_std_, asset, time,
                              n.get("per_asset_baseline", False), n.get("baseline_window", 30),
                              n.get("smoothing_window", 1), cfg["rul"]["trend_window"])
    scores = score_with_bundle(sdf, str(bundle_path), asset)
    fmt = a.format
    feats.write.mode("overwrite").format(fmt).save(str(out / "features"))
    scores.write.mode("overwrite").format(fmt).save(str(out / "scores"))
    latest = scores.orderBy(time).groupBy(asset).agg({"rul": "last"}).count()
    print(f"-> {out}/features, {out}/scores ({latest} machines)")
    spark.stop()


if __name__ == "__main__":
    main()
