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


# ---------------------------------------------------------------------------
# Wind turbine SCADA: CARE to Compare, Wind Farm A (EDP open data, onshore, Portugal)
CARE_A_COLUMNS = {   # 10-minute averages; names from feature_description.csv
    "sensor_0_avg": "ambient_temp", "wind_speed_3_avg": "wind_speed", "power_30_avg": "power",
    "sensor_52_avg": "rotor_rpm", "sensor_18_avg": "generator_rpm", "sensor_5_avg": "pitch_angle",
    "sensor_6_avg": "hub_controller_temp", "sensor_7_avg": "nacelle_controller_temp",
    "sensor_8_avg": "choke_coil_temp", "sensor_9_avg": "vcp_board_temp",
    "sensor_10_avg": "converter_cooling_water_temp",
    "sensor_11_avg": "gearbox_hss_bearing_temp", "sensor_12_avg": "gearbox_oil_temp",
    "sensor_13_avg": "generator_bearing_de_temp", "sensor_14_avg": "generator_bearing_nde_temp",
    "sensor_15_avg": "stator_winding_1_temp", "sensor_16_avg": "stator_winding_2_temp",
    "sensor_17_avg": "stator_winding_3_temp", "sensor_19_avg": "split_ring_chamber_temp",
    "sensor_20_avg": "busbar_temp", "sensor_21_avg": "grid_inverter_igbt_temp",
    "sensor_35_avg": "rotor_inverter_igbt_1_temp", "sensor_36_avg": "rotor_inverter_igbt_2_temp",
    "sensor_37_avg": "rotor_inverter_igbt_3_temp", "sensor_38_avg": "transformer_l1_temp",
    "sensor_39_avg": "transformer_l2_temp", "sensor_40_avg": "transformer_l3_temp",
    "sensor_41_avg": "hydraulic_oil_temp", "sensor_43_avg": "nacelle_temp",
    "sensor_53_avg": "nose_cone_temp",
}
NORMAL_STATUS = (0, 2)   # normal production, idling


def load_care(data_dir: str | Path, lag_samples: int = 6):
    """Returns (datasets, events). `datasets` maps event_id -> DataFrame for one
    turbine: a normal-operation training period followed by a prediction
    period that contains the event. `events` is event_info.csv."""
    d = Path(data_dir)
    events = pd.read_csv(d / "event_info.csv", sep=";", encoding="latin1")
    datasets = {}
    for eid in events["event_id"]:
        f = pd.read_csv(d / "datasets" / f"{eid}.csv", sep=";",
                        usecols=["time_stamp", "asset_id", "id", "train_test", "status_type_id", *CARE_A_COLUMNS])
        f = f.rename(columns=CARE_A_COLUMNS).sort_values("id").reset_index(drop=True)
        f["time_stamp"] = pd.to_datetime(f["time_stamp"])
        # temperatures lag load: include the last hour of power as an operating-point input
        f["power_1h"] = f["power"].rolling(lag_samples, min_periods=1).mean()
        f["normal_status"] = f["status_type_id"].isin(NORMAL_STATUS)
        datasets[int(eid)] = f
    return datasets, events


# ---------------------------------------------------------------------------
# Hydraulic power unit: UCI "Condition monitoring of hydraulic systems" (ZeMA, Helwig et al. 2015)
# Stand-in for the hydraulic power units behind ship steering gear and fin stabilizers.
HYDRAULIC_SENSORS = {   # file -> (quantity, sampling rate Hz)
    "PS1": ("pressure", 100), "PS2": ("pressure", 100), "PS3": ("pressure", 100),
    "PS4": ("pressure", 100), "PS5": ("pressure", 100), "PS6": ("pressure", 100),
    "EPS1": ("motor_power", 100), "FS1": ("flow", 10), "FS2": ("flow", 10),
    "TS1": ("temperature", 1), "TS2": ("temperature", 1), "TS3": ("temperature", 1),
    "TS4": ("temperature", 1), "VS1": ("vibration", 1),
}
# CE, CP and SE are "virtual" sensors computed by the rig from the others
# (cooling efficiency/power, efficiency factor); they are left out so every
# input is something a real hydraulic power unit would measure.
HYDRAULIC_COMPONENTS = ["cooler", "valve", "pump_leak", "accumulator"]


def hydraulic_cycle_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cycle summary of one sensor (rows = 60 s load cycles): level,
    spread, extremes, and the trend across the cycle."""
    t = np.linspace(-0.5, 0.5, x.shape[1])
    q = max(1, x.shape[1] // 10)
    return {"mean": x.mean(1), "std": x.std(1), "min": x.min(1), "max": x.max(1),
            "slope": (x - x.mean(1, keepdims=True)) @ t / (t @ t),
            "start": x[:, :q].mean(1), "end": x[:, -q:].mean(1)}


def load_hydraulic(data_dir: str | Path) -> pd.DataFrame:
    """One row per 60 s load cycle: engineered features for each sensor
    (`PS1_mean`, `TS1_slope`, ...), the four component conditions, the
    stable flag, the cycle number and a `segment` id.

    The rig was run in blocks: component settings were held for a stretch of
    consecutive cycles, then changed. A `segment` is one such block. Cycles of
    one segment are near-copies of each other (oil temperature drifts slowly),
    so they must never be split between train and test."""
    d = Path(data_dir)
    cache = d / "cycle_features.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cols = {}
    for name in HYDRAULIC_SENSORS:
        x = pd.read_csv(d / f"{name}.txt", sep="\t", header=None, dtype=np.float32).to_numpy(np.float64)
        for stat, v in hydraulic_cycle_features(x).items():
            cols[f"{name}_{stat}"] = v
    df = pd.DataFrame(cols)
    prof = pd.read_csv(d / "profile.txt", sep="\t", header=None,
                       names=HYDRAULIC_COMPONENTS + ["unstable"])
    df = pd.concat([df, prof], axis=1)
    df["cycle"] = np.arange(len(df))
    changed = df[HYDRAULIC_COMPONENTS].diff().abs().sum(axis=1) > 0
    df["segment"] = changed.cumsum()
    df.to_csv(cache, index=False)
    return df


LOADERS = {"naval": load_naval, "cmapss": load_cmapss, "cwru": load_cwru, "tep": load_tep,
           "care": load_care, "hydraulic": load_hydraulic}
