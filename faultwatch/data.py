"""Dataset loaders. Each returns plain pandas DataFrames; the rest of the
pipeline only sees column names declared in the asset config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NAVAL_COLS = [
    "lever_pos", "ship_speed", "gt_torque", "gt_rpm", "gg_rpm",
    "prop_torque_stbd", "prop_torque_port", "hpt_exit_temp", "comp_in_temp",
    "comp_out_temp", "hpt_exit_press", "comp_in_press", "comp_out_press",
    "exhaust_press", "turbine_inj_ctrl", "fuel_flow", "kMc", "kMt",
]

CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_naval(data_dir: str | Path) -> pd.DataFrame:
    """UCI 'Condition Based Maintenance of Naval Propulsion Plants'.

    Frigate CODLAG gas turbine, simulated at 9 load levels across a grid of
    compressor (kMc 0.95-1.0) and turbine (kMt 0.975-1.0) decay states.
    """
    df = pd.read_csv(Path(data_dir) / "data.txt", sep=r"\s+", header=None, names=NAVAL_COLS)
    df["kMc"] = df["kMc"].round(3)
    df["kMt"] = df["kMt"].round(3)
    # One "asset state" = one (compressor, turbine) decay combination seen at all loads.
    df["state_id"] = df.groupby(["kMc", "kMt"]).ngroup()
    return df


def load_cmapss(data_dir: str | Path, subset: str = "FD001"):
    """NASA C-MAPSS turbofan run-to-failure data.

    Returns (train, test, test_rul). `train` gets a RUL column (cycles to failure).
    `test_rul` is the true remaining life at each test engine's last cycle.
    """
    d = Path(data_dir)
    train = pd.read_csv(d / f"train_{subset}.txt", sep=r"\s+", header=None, names=CMAPSS_COLS)
    test = pd.read_csv(d / f"test_{subset}.txt", sep=r"\s+", header=None, names=CMAPSS_COLS)
    rul = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])
    rul["unit"] = np.arange(1, len(rul) + 1)
    life = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = life - train["cycle"]
    train["life"] = life
    return train, test, rul


LOADERS = {"naval": load_naval, "cmapss": load_cmapss}
