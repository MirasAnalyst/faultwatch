"""Run FaultWatch on one asset config.

    python run.py configs/naval_gas_turbine.yaml
    python run.py configs/turbofan_cmapss.yaml
"""
import argparse
import json
from pathlib import Path

from faultwatch.config import load_config
from faultwatch.experiments import run_to_failure, steady_state

EXPERIMENTS = {"steady_state": steady_state.run, "run_to_failure": run_to_failure.run}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="reports")
    args = p.parse_args()
    for c in args.configs:
        cfg = load_config(c)
        out = Path(args.out) / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {cfg['name']} ({cfg['experiment']})")
        metrics = EXPERIMENTS[cfg["experiment"]](cfg, out)
        print(json.dumps(metrics, indent=2)[:4000])
        print(f"-> {out}/")


if __name__ == "__main__":
    main()
