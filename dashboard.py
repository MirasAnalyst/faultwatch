"""FaultWatch operator dashboard.

    streamlit run dashboard.py          # after `python run.py configs/*.yaml`

Scores each asset type's held-out demo data through the same `Scorer` the API
uses, so what the operator sees here is what the API would return.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from faultwatch.plotting import GRID, INK_2, NEUTRAL, SERIES
from faultwatch.schedule import plan_maintenance
from faultwatch.serve import load_all

CRITICAL = "#d03b3b"
st.set_page_config(page_title="FaultWatch", layout="wide")


@st.cache_resource
def scorers():
    return load_all(".")


@st.cache_data
def scored(name: str) -> pd.DataFrame:
    s = scorers()[name]
    demo = s.b["demo"]
    out = s.score(demo)
    out["top_sensors"] = out["top_sensors"].map(lambda d: ", ".join(f"{k} {v:.0%}" for k, v in d.items()))
    return pd.concat([demo.reset_index(drop=True), out], axis=1)


def axis_style(chart):
    return (chart.configure_axis(gridColor=GRID, domainColor=GRID, labelColor=INK_2, titleColor=INK_2)
                 .configure_view(stroke=None))


def status_badge(alarm: bool) -> str:
    return (":red[**▲ ALARM**]" if alarm else ":green[**● Normal**]")


def contributions_chart(shares: dict):
    df = pd.DataFrame({"sensor": list(shares), "share": list(shares.values())})
    df = df[df["share"] >= 0.005]
    bars = alt.Chart(df).mark_bar(color=SERIES[0], cornerRadiusEnd=4, height=18).encode(
        x=alt.X("share:Q", axis=alt.Axis(format="%", tickCount=5, title="share of health score")),
        y=alt.Y("sensor:N", sort="-x", title=None),
        tooltip=["sensor", alt.Tooltip("share:Q", format=".0%")])
    labels = bars.mark_text(align="left", dx=4, color=INK_2).encode(text=alt.Text("share:Q", format=".0%"))
    return axis_style((bars + labels).properties(height=40 * len(df) + 20))


# ---------------------------------------------------------------------------
S = scorers()
st.title("FaultWatch")
if not S:
    st.error("No trained models found. Run `python scripts/download_data.py` then "
             "`python run.py configs/*.yaml` first.")
    st.stop()

name = st.sidebar.selectbox("Asset type", list(S), format_func=lambda n: S[n].cfg.get("description", n))
s = S[name]
st.caption(f"{s.cfg.get('description', name)} · held-out data never seen in training · "
           f"alarm threshold {s.det.threshold_:.0f}")
df = scored(name)
tab_live, tab_drift, tab_val = st.tabs(["Live condition", "Baseline drift", "Validation results"])

# ---- live condition --------------------------------------------------------
with tab_live:
    if s.b.get("rul_model") is not None:                      # fleet run-to-failure asset
        a, t = s.cfg["asset_id"], s.cfg["time"]
        latest = df.sort_values(t).groupby(a).tail(1).set_index(a)
        m = s.b["maintenance"]
        plan = plan_maintenance(latest["rul"].rename("rul_pred").rename_axis("asset").reset_index(),
                                m["horizon"], m["capacity_per_day"],
                                preventive_cost=m["preventive_cost"], failure_cost=m["failure_cost"],
                                wasted_life_cost=m["wasted_life_cost"],
                                rul_uncertainty=s.b.get("rul_uncertainty", 10.0)).set_index("asset")
        latest["service_day"] = plan["service_day"]
        alarm_col = "alarm_confirmed" if "alarm_confirmed" in latest else "alarm"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines", len(latest))
        c2.metric("In alarm", int(latest[alarm_col].sum()))
        c3.metric(f"Service due ≤ {m['horizon']} days", int(latest["service_day"].notna().sum()))
        c4.metric("Median remaining life", f"{latest['rul'].median():.0f} cycles")

        st.subheader("Fleet")
        view = latest.assign(status=np.where(latest[alarm_col], "▲ alarm", "● normal"))[
            ["status", "health_score", "rul", "service_day", "top_sensors", t]].sort_values("rul")
        view.columns = ["Status", "Health score", "RUL (cycles)", "Service day", "Driving sensors", "Age (cycles)"]
        st.dataframe(view.style.format({"Health score": "{:.0f}", "RUL (cycles)": "{:.0f}", "Service day": "{:.0f}"}),
                     width="stretch", height=280)

        unit = st.selectbox("Machine", view.index, format_func=lambda u: f"{a} {u}")
        g = df[df[a] == unit].sort_values(t)
        row = g.iloc[-1]
        st.markdown(f"### {a} {unit} &nbsp; {status_badge(bool(row[alarm_col]))}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Health score", f"{row['health_score']:.0f}")
        k1.caption(f"alarm above {s.det.threshold_:.0f}")
        k2.metric("Remaining life", f"{row['rul']:.0f} cycles")
        if "true_rul" in g:
            k2.caption(f"actual: {row['true_rul']:.0f} cycles")
        sd = latest.at[unit, "service_day"]
        k3.metric("Planned service", "not in horizon" if pd.isna(sd) else f"day {sd:.0f}")
        k3.caption(f"crew capacity {m['capacity_per_day']} machines/day")

        left, right = st.columns(2)
        with left:
            gp = g[g[t] > s.norm.smoothing_window]                  # skip filter warm-up
            decades = [10.0 ** k for k in range(int(np.floor(np.log10(max(gp["health_score"].min(), 1e-3)))),
                                                 int(np.ceil(np.log10(gp["health_score"].max()))) + 1)]
            base = alt.Chart(gp).encode(x=alt.X(f"{t}:Q", title="Cycle"))
            line = base.mark_line(color=SERIES[0], strokeWidth=2).encode(
                y=alt.Y("health_score:Q", scale=alt.Scale(type="log"),
                        axis=alt.Axis(values=decades, format="~s"), title="Health score (log)"),
                tooltip=[t, alt.Tooltip("health_score:Q", format=".0f"), "top_sensors"])
            thr = alt.Chart(pd.DataFrame({"y": [s.det.threshold_]})).mark_rule(
                color=NEUTRAL, strokeDash=[5, 4]).encode(y="y:Q")
            thr_lbl = alt.Chart(pd.DataFrame({"y": [s.det.threshold_], "x": [gp[t].min()]})).mark_text(
                align="left", dy=-7, color=INK_2, text="alarm threshold").encode(x="x:Q", y="y:Q")
            st.markdown("**Health score**")
            st.altair_chart(axis_style((line + thr + thr_lbl).properties(height=280)), width="stretch")
        with right:
            long = g.melt(id_vars=[t], value_vars=["rul", "true_rul"], var_name="series", value_name="cycles")
            long["series"] = long["series"].map({"rul": "Predicted", "true_rul": "Actual"})
            dom = ["Predicted", "Actual"]
            ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=alt.X(f"{t}:Q", title="Cycle"), y=alt.Y("cycles:Q", title="Remaining life (cycles)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[SERIES[0], SERIES[1]]),
                                legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=dom, range=[[1, 0], [5, 4]]), legend=None),
                tooltip=[t, "series", alt.Tooltip("cycles:Q", format=".0f")])
            st.markdown("**Remaining life: predicted vs actual**")
            st.altair_chart(axis_style(ch.properties(height=280)), width="stretch")
        st.markdown("**What is driving the health score now**")
        st.altair_chart(contributions_chart(s.score(g[s.required_columns], top_n=6)["top_sensors"].iloc[-1]),
                        width="stretch")

    else:                                                     # labeled-states asset
        truth = "fault_class" if "fault_class" in df else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", len(df))
        c2.metric("In alarm", f"{df['alarm'].mean():.0%}")
        if truth:
            c3.metric("Diagnosis matches truth", f"{(df['diagnosis'] == df[truth]).mean():.1%}")
        classes = sorted(df[truth].unique()) if truth else []
        pick = st.radio("Show samples whose true condition is", ["all"] + classes, horizontal=True)
        sub = df if pick == "all" else df[df[truth] == pick]
        extra = [c for c in ("severity", "fault_size_in", "run", "sample") if c in df]
        cols = s.cfg["regime_features"] + extra + ([truth] if truth else []) + \
            ["alarm", "health_score", "diagnosis", "diagnosis_confidence", "top_sensors"]
        st.dataframe(sub[cols].style.format({"health_score": "{:.1f}", "severity": "{:.0%}",
                                            "diagnosis_confidence": "{:.0%}"}),
                     width="stretch", height=260)
        i = st.number_input("Inspect sample (row number above)", 0, len(sub) - 1, min(len(sub) - 1, len(sub) // 2))
        row = sub.iloc[int(i)]
        st.markdown(f"### Sample {int(i)} &nbsp; {status_badge(bool(row['alarm']))}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Health score", f"{row['health_score']:.1f}")
        k1.caption(f"alarm above {s.det.threshold_:.1f}")
        k2.metric("Diagnosis", row["diagnosis"])
        k2.caption(f"{row['diagnosis_confidence']:.0%} confidence")
        if truth:
            k3.metric("Truth", row[truth])
            if "severity" in row:
                k3.caption(f"severity {row['severity']:.0%} of the way to worst state")
        sev = {c.removeprefix("severity_"): row[c] for c in df.columns if c.startswith("severity_")}
        if sev:
            comps = s.cfg.get("components", {})
            st.markdown("**Estimated wear** &nbsp; " + " · ".join(
                f"{k} {np.clip((comps[k]['new'] - v) / (comps[k]['new'] - comps[k]['worst']), 0, 1):.0%} "
                f"of the way to worst (coefficient {v:.4f}, actual {row[comps[k]['column']]:.3f})"
                if k in comps else f"{k}: {v:.3g}" for k, v in sev.items()))
        full = s.score(sub.iloc[[int(i)]][s.required_columns], top_n=6)["top_sensors"].iloc[0]
        st.markdown("**What is driving the health score**")
        st.altair_chart(contributions_chart(full), width="stretch")

# ---- drift -----------------------------------------------------------------
with tab_drift:
    st.markdown("Checks whether the healthy baseline still describes this machine. It looks only at "
                "samples the detector calls normal, so a developing fault is not mistaken for drift. "
                "Try a simulated sensor recalibration:")
    sensor = st.selectbox("Sensor", s.cfg["sensors"])
    shift = st.slider("Recalibration offset (healthy standard deviations)", 0.0, 3.0, 0.0, 0.25)
    demo = s.b["demo"].copy()
    if s.cfg.get("time") in demo and s.cfg.get("asset_id") in demo:
        a, t = s.cfg["asset_id"], s.cfg["time"]
        late = demo[t] > demo.groupby(a)[t].transform("max") * 0.5   # step change mid-history
    else:
        late = pd.Series(True, index=demo.index)
    demo.loc[late, sensor] += shift * float(s.norm.global_std_[sensor])
    rep = s.drift(demo)
    st.markdown("### " + (":orange[**Retraining recommended**]" if rep["retrain_recommended"]
                          else ":green[**Baseline still valid**]"))
    for r in rep["reasons"]:
        st.markdown(f"- {r}")
    psi = pd.Series(rep["sensor_psi"], name="PSI").sort_values(ascending=False).rename_axis("sensor").reset_index()
    bars = alt.Chart(psi).mark_bar(cornerRadiusEnd=4, height=14, color=SERIES[0]).encode(
        x=alt.X("PSI:Q", title="Population stability index (normal samples vs training)"),
        y=alt.Y("sensor:N", sort="-x", title=None), tooltip=["sensor", alt.Tooltip("PSI:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"x": [0.25]})).mark_rule(color=NEUTRAL, strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart(axis_style((bars + rule).properties(height=22 * len(psi) + 30)), width="stretch")
    st.caption(f"Dashed line = retraining trigger (PSI 0.25). {rep['samples_judged_normal']} of "
               f"{rep['samples']} samples judged normal; {rep['out_of_envelope_share']:.1%} outside the "
               "trained operating envelope.")

# ---- validation ------------------------------------------------------------
with tab_val:
    rep_dir = Path("reports") / name
    mfile = rep_dir / "metrics.json"
    if mfile.exists():
        metrics = json.loads(mfile.read_text())
        for key in ("detection", "early_warning"):
            if key in metrics:
                st.markdown("**Alarm comparison on held-out data**")
                st.dataframe(pd.DataFrame(metrics[key]).set_index("method"), width="stretch")
        for img in sorted(rep_dir.glob("*.png")):
            st.image(str(img), caption=img.stem.replace("_", " "), width="stretch")
        with st.expander("All metrics (JSON)"):
            st.json(metrics)
    else:
        st.info("No report yet for this asset.")
