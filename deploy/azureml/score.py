"""Azure ML scoring script (custom-container deployment).

The same Scorer as the FastAPI service and the Spark batch job, so online,
batch and offline scores agree.
"""
import json
import os
from pathlib import Path

import pandas as pd

from faultwatch.serve import Scorer

scorer = None


def init():
    global scorer
    root = Path(os.environ["AZUREML_MODEL_DIR"])
    scorer = Scorer.load(next(root.rglob("*.joblib")))


def run(raw: str) -> str:
    rows = json.loads(raw)["rows"]
    out = scorer.score(pd.DataFrame(rows))
    alarm = out.get("alarm_confirmed", out["alarm"])
    return json.dumps({"asset": scorer.name, "alarms": int(alarm.fillna(False).sum()),
                       "results": json.loads(out.to_json(orient="records"))})
