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


# ---------------------------------------------------------------------------
# CWRU bearing vibration (Case Western Reserve University Bearing Data Center)
# 12 kHz drive-end accelerometer; SKF 6205 deep-groove bearing, 0-3 HP motor load.
CWRU_FILES = {  # file id -> (fault, fault diameter in inches, motor load HP)
    97: ("normal", 0.0, 0), 98: ("normal", 0.0, 1), 99: ("normal", 0.0, 2), 100: ("normal", 0.0, 3),
    **{f: ("inner_race", 0.007, i) for i, f in enumerate([105, 106, 107, 108])},
    **{f: ("ball", 0.007, i) for i, f in enumerate([118, 119, 120, 121])},
    **{f: ("outer_race", 0.007, i) for i, f in enumerate([130, 131, 132, 133])},
    **{f: ("inner_race", 0.014, i) for i, f in enumerate([169, 170, 171, 172])},
    **{f: ("ball", 0.014, i) for i, f in enumerate([185, 186, 187, 188])},
    **{f: ("outer_race", 0.014, i) for i, f in enumerate([197, 198, 199, 200])},
    **{f: ("inner_race", 0.021, i) for i, f in enumerate([209, 210, 211, 212])},
    **{f: ("ball", 0.021, i) for i, f in enumerate([222, 223, 224, 225])},
    **{f: ("outer_race", 0.021, i) for i, f in enumerate([234, 235, 236, 237])},
}
BEARING_6205 = {"bpfi": 5.4152, "bpfo": 3.5848, "bsf": 2.3568}   # x shaft speed
CWRU_FS = 12000


def vibration_features(x: np.ndarray, fs: float, shaft_hz: float, window: int) -> pd.DataFrame:
    """Condition-monitoring features per non-overlapping window, the same ones a
    vibration analyst reads: overall level, impulsiveness, where the energy
    sits, and envelope-spectrum peaks at the bearing defect frequencies."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    from scipy.stats import kurtosis, skew

    env = np.abs(hilbert(sosfiltfilt(butter(4, [2000, 5000], "bandpass", fs=fs, output="sos"), x)))
    bands = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 6000)]
    fr = np.fft.rfftfreq(window, 1 / fs)
    han = np.hanning(window)
    rows = []
    for k in range(len(x) // window):
        w = x[k * window:(k + 1) * window]
        e = env[k * window:(k + 1) * window]
        rms = np.sqrt(np.mean(w ** 2))
        P = np.abs(np.fft.rfft((w - w.mean()) * han)) ** 2
        E = np.abs(np.fft.rfft((e - e.mean()) * han))
        floor = np.median(E[(fr > 10) & (fr < 500)])
        r = {"rms": np.log(rms), "crest": np.log(np.abs(w).max() / rms),
             "kurtosis": np.log(kurtosis(w, fisher=False)), "skewness": skew(w)}
        for lo, hi in bands:
            r[f"band_{lo // 1000}_{hi // 1000}k"] = np.log(P[(fr >= lo) & (fr < hi)].sum() / P.sum())
        for name, order in BEARING_6205.items():          # 1x + 2x defect frequency
            amp = sum(E[np.abs(fr - h * order * shaft_hz) <= 2 * fs / window].max() for h in (1, 2))
            r[f"env_{name}"] = np.log(amp / floor)
        rows.append(r)
    return pd.DataFrame(rows)


def load_cwru(data_dir: str | Path, window: int = 4096, test_fraction: float = 0.3,
              test_defect_size: float | None = 0.014) -> pd.DataFrame:
    """One row per vibration window.

    Splitting windows of one recording between train and test (common in
    papers) lets a model recognise the individual test bearing rather than the
    fault. So every recording of `test_defect_size` - physically different
    bearings - is held out entirely. The rig has a single healthy bearing, so
    its recordings are split in time (last `test_fraction` held out).
    With test_defect_size=None every recording is split in time."""
    import scipy.io as sio
    from scipy.signal import decimate

    parts = []
    for fid, (fault, size, load) in CWRU_FILES.items():
        m = sio.loadmat(Path(data_dir) / f"{fid}.mat")
        x = next(m[k] for k in m if k.endswith("DE_time")).ravel()
        rpm = float(next((m[k] for k in m if k.endswith("RPM")), [[1797 - 25 * load]])[0][0])
        if fault == "normal":
            # The normal-baseline recordings are sampled at 48 kHz, the fault
            # recordings at 12 kHz. Mixing them unconverted makes 'normal'
            # trivially separable by sampling rate alone.
            x = decimate(x, 4, ftype="fir")
        f = vibration_features(x, CWRU_FS, rpm / 60, window)
        if test_defect_size is not None and fault != "normal":
            f["split"] = "test" if np.isclose(size, test_defect_size) else "train"
        else:
            n_test = int(round(len(f) * test_fraction))
            f["split"] = ["train"] * (len(f) - n_test) + ["test"] * n_test
        f = f.assign(run=fid, sample=np.arange(len(f)), fault=fault, fault_size_in=size,
                     load_hp=load, faulty=fault != "normal")
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Tennessee Eastman process (Downs & Vogel 1993; Braatz group data set)
TEP_VARS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
TEP_ONSET = 160   # test runs: fault introduced after 8 h (3-min samples)


def load_tep(data_dir: str | Path) -> pd.DataFrame:
    """Training runs d00-d21 (d00 normal; d01-d21 recorded after fault onset)
    and test runs d00_te-d21_te (fault introduced at sample 160)."""
    d = Path(data_dir)
    parts = []
    for i in range(22):
        for split, suffix in (("train", ""), ("test", "_te")):
            a = np.loadtxt(d / f"d{i:02d}{suffix}.dat")
            if a.shape[0] == 52 and a.shape[1] != 52:
                a = a.T                              # d00.dat is stored transposed
            f = pd.DataFrame(a, columns=TEP_VARS)
            onset = TEP_ONSET if split == "test" else 0
            f = f.assign(run=f"{split}_{i:02d}", sample=np.arange(len(f)), split=split,
                         fault="normal" if i == 0 else f"IDV{i}",
                         faulty=(i > 0) & (np.arange(len(f)) >= onset))
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


LOADERS = {"naval": load_naval, "cmapss": load_cmapss, "cwru": load_cwru, "tep": load_tep}
