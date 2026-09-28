"""Run FaultWatch on one or more asset configs.

    python run.py configs/naval_gas_turbine.yaml
    python run.py configs/*.yaml --mlflow      # also log to MLflow + register models
"""
import argparse
import json
from pathlib import Path

from faultwatch.config import load_config
from faultwatch.experiments import EXPERIMENTS
from faultwatch.serve import MODELS_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="reports")
    p.add_argument("--mlflow", action="store_true", help="log runs and register models in MLflow")
    args = p.parse_args()
    for c in args.configs:
        cfg = load_config(c)
        out = Path(args.out) / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {cfg['name']} ({cfg['experiment']})")
        metrics = EXPERIMENTS[cfg["experiment"]](cfg, out)
        print(json.dumps(metrics, indent=2)[:4000])
        print(f"-> {out}/")
        if args.mlflow:
            from faultwatch.tracking import log_run
            bundle = Path(cfg["_root"]) / MODELS_DIR / f"{cfg['name']}.joblib"
            print(f"-> MLflow run {log_run(cfg, metrics, out, bundle)}")


if __name__ == "__main__":
    main()
