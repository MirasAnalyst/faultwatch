"""Experiment type 7: how would we *prove* FaultWatch works on real ships?

A pilot-design study. It simulates the fleet-level outcomes a pilot would
measure, then analyses them exactly as the real pilot would be analysed, to
answer: which design, how many ships, how many months?

Outcomes per ship-month (Poisson counts, with ship-to-ship variation and a
seasonal fleet-wide trend that a naive before/after comparison would confuse
with the program's effect):
  * in-service generator-set failures  (the safety outcome; rare)
  * days without an N+1 margin         (the leading indicator; more frequent)

True rates under current practice (time-based maintenance) and under
FaultWatch are read from the power-plant scenario (reports/ship_power_plant)
when it exists, else from the config.

Designs:
  * stepped wedge - ships randomized into waves that switch to FaultWatch one
    after another (everyone is treated by the end, which operations prefer).
    Analysis: two-way fixed effects (ship + month), SE clustered by ship; the
    rate ratio from a fixed-effects Poisson model; and randomization inference
    that re-draws the wave assignment (exact, valid with few ships).
    The power study uses the linear two-way model, which stays stable with
    sparse counts, and also reports each design's false-positive rate under
    no effect - with few ships, clustered standard errors can be too small.
  * parallel A/B - half the ships randomized to FaultWatch for the whole
    pilot. Analysis: per-ship rate ratio, with CUPED on the pre-pilot year's
    counts to remove persistent ship-to-ship differences.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plotting import INK, INK_2, SERIES, style_axes
from ..stats import cuped, did_estimate, randomization_inference


def _true_rates(cfg, root):
    """Events per ship-month under each policy."""
    r = dict(cfg["rates"])
    src = root / "reports" / cfg.get("rates_from", "ship_power_plant") / "metrics.json"
    if src.exists():
        m = json.loads(src.read_text())
        sc = m["scenario"]
        ship_months = sc["ships"] * sc["horizon_days"] / 30.4
        voy_per_month = sc["voyages_per_ship"] * 30.4 / sc["horizon_days"]
        f, nm = m["in_service_failures"], m["no_margin_days_per_100_voyages"]
        r = {"failures": {"control": f["Time-based (age limit)"]["mean"] / ship_months,
                          "treated": f["FaultWatch, itinerary-aware"]["mean"] / ship_months},
             "no_margin_days": {"control": nm["Time-based (age limit)"]["mean"] / 100 * voy_per_month,
                                "treated": nm["FaultWatch, itinerary-aware"]["mean"] / 100 * voy_per_month},
             "source": str(src.relative_to(root))}
    return r


def _panel(rng, n_ships, months, rates, assign_month, frailty_cv, season_amp, pre_months=0):
    """Simulated ship-month counts. assign_month[s] = first treated month (inf = never)."""
    frail = rng.gamma(1 / frailty_cv ** 2, frailty_cv ** 2, n_ships)          # mean 1
    rows = []
    for s in range(n_ships):
        for t in range(-pre_months, months):
            season = 1 + season_amp * np.sin(2 * np.pi * t / 12)
            d = int(t >= assign_month[s])
            lam = {k: v["treated" if d else "control"] * frail[s] * season for k, v in rates.items()
                   if isinstance(v, dict)}
            rows.append({"ship": s, "month": t, "treated": d,
                         **{k: rng.poisson(l) for k, l in lam.items()}})
    return pd.DataFrame(rows)


def _wedge_assign(rng, n_ships, months, waves):
    starts = np.linspace(months / (waves + 1), months * waves / (waves + 1), waves).round().astype(int)
    return starts[rng.permutation(np.arange(n_ships) % waves)]


def poisson_twfe(panel, outcome):
    """Rate ratio from a ship + month fixed-effects Poisson model, SE clustered by ship."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = panel[panel.month >= 0]
    keep = df.groupby("ship")[outcome].transform("sum") > 0          # all-zero ships carry no information
    df = df[keep]
    if df[outcome].sum() == 0 or df.treated.nunique() < 2:
        return {"rate_ratio": np.nan, "p_value": 1.0}
    try:
        fit = smf.glm(f"{outcome} ~ treated + C(ship) + C(month)", data=df,
                      family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df["ship"]})
        b, p = fit.params["treated"], fit.pvalues["treated"]
        lo, hi = fit.conf_int().loc["treated"]
        return {"rate_ratio": float(np.exp(b)), "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)),
                "p_value": float(p)}
    except Exception:            # separation with very few events
        return {"rate_ratio": np.nan, "p_value": 1.0}


def _twfe_p(panel, outcome):
    df = panel[panel.month >= 0]
    if df[outcome].sum() == 0:
        return 1.0
    return did_estimate(df, outcome, "treated", "ship", "month")["p_value"]


def parallel_ab(panel, outcome, treated_ships, use_cuped=True):
    """Per-ship monthly rate in the pilot, CUPED-adjusted with the pre-pilot rate."""
    from scipy import stats
    pilot = panel[panel.month >= 0].groupby("ship")[outcome].mean()
    pre = panel[panel.month < 0].groupby("ship")[outcome].mean().reindex(pilot.index).fillna(0)
    y = cuped(pilot.to_numpy(), pre.to_numpy()) if use_cuped and pre.var() > 0 else pilot.to_numpy()
    t = pilot.index.isin(treated_ships)
    r = stats.ttest_ind(y[t], y[~t], equal_var=False)
    ratio = pilot[t].mean() / pilot[~t].mean() if pilot[~t].mean() > 0 else np.nan
    return {"rate_ratio": float(ratio), "p_value": float(r.pvalue) if np.isfinite(r.pvalue) else 1.0}


def run(cfg: dict, out: Path) -> dict:
    root = Path(cfg["_root"])
    rates = _true_rates(cfg, root)
    rng = np.random.default_rng(cfg["seed"])
    sim = cfg["simulation"]
    kw = dict(frailty_cv=sim["ship_frailty_cv"], season_amp=sim["seasonal_amplitude"])

    # ---- 1. one pilot as it would actually be run and analysed --------------
    pilot = cfg["pilot"]
    assign = _wedge_assign(rng, pilot["ships"], pilot["months"], pilot["waves"])
    panel = _panel(rng, pilot["ships"], pilot["months"], rates, assign, **kw)
    panel.to_csv(out / "example_pilot_panel.csv", index=False)
    example = {}
    for outcome in ("failures", "no_margin_days"):
        est = poisson_twfe(panel, outcome)
        naive_before = panel[(panel.treated == 0)][outcome].mean()
        naive_after = panel[(panel.treated == 1)][outcome].mean()

        def refit(a, outcome=outcome):
            p = panel.copy()
            p["treated"] = (p.month.to_numpy() >= a[p.ship.to_numpy()]).astype(int)
            return np.log(max(poisson_twfe(p, outcome)["rate_ratio"], 1e-6))
        ri = randomization_inference(refit, assign,
                                     lambda g: _wedge_assign(g, pilot["ships"], pilot["months"], pilot["waves"]),
                                     n_perm=cfg["randomization_inference_draws"], seed=cfg["seed"])
        example[outcome] = {"twfe_poisson": est, "randomization_inference_p": ri["p_value"],
                            "naive_after_vs_before_ratio": float(naive_after / naive_before) if naive_before else None,
                            "events": int(panel.loc[panel.month >= 0, outcome].sum())}

    # ---- 2. power: which design, how many ships, how many months -------------
    power_rows = []
    for outcome in ("failures", "no_margin_days"):
        for ships in cfg["power"]["ships"]:
            for months in cfg["power"]["months"]:
                hits = {"stepped_wedge": 0, "parallel_ab_cuped": 0, "parallel_ab": 0}
                for _ in range(cfg["power"]["trials"]):
                    a = _wedge_assign(rng, ships, months, pilot["waves"])
                    p = _panel(rng, ships, months, rates, a, **kw)
                    hits["stepped_wedge"] += _twfe_p(p, outcome) < 0.05
                    treated = rng.permutation(ships)[: ships // 2]
                    a2 = np.where(np.isin(np.arange(ships), treated), 0, np.inf)
                    p2 = _panel(rng, ships, months, rates, a2, pre_months=12, **kw)
                    hits["parallel_ab_cuped"] += parallel_ab(p2, outcome, treated)["p_value"] < 0.05
                    hits["parallel_ab"] += parallel_ab(p2, outcome, treated, use_cuped=False)["p_value"] < 0.05
                for design, h in hits.items():
                    power_rows.append({"outcome": outcome, "design": design, "ships": ships, "months": months,
                                       "power": h / cfg["power"]["trials"]})
    # false-positive rate of each test when FaultWatch has no effect
    null_rates = {k: ({"control": v["control"], "treated": v["control"]} if isinstance(v, dict) else v)
                  for k, v in rates.items()}
    for outcome in ("failures", "no_margin_days"):
        for ships in cfg["power"]["ships"]:
            months = pilot["months"]
            fp = {"stepped_wedge": 0, "parallel_ab_cuped": 0, "parallel_ab": 0}
            for _ in range(cfg["power"]["trials"]):
                a = _wedge_assign(rng, ships, months, pilot["waves"])
                fp["stepped_wedge"] += _twfe_p(_panel(rng, ships, months, null_rates, a, **kw), outcome) < 0.05
                treated = rng.permutation(ships)[: ships // 2]
                a2 = np.where(np.isin(np.arange(ships), treated), 0, np.inf)
                p2 = _panel(rng, ships, months, null_rates, a2, pre_months=12, **kw)
                fp["parallel_ab_cuped"] += parallel_ab(p2, outcome, treated)["p_value"] < 0.05
                fp["parallel_ab"] += parallel_ab(p2, outcome, treated, use_cuped=False)["p_value"] < 0.05
            for design, h in fp.items():
                power_rows.append({"outcome": outcome, "design": design, "ships": ships, "months": months,
                                   "power": np.nan, "false_positive_rate": h / cfg["power"]["trials"]})
    power = pd.DataFrame(power_rows)
    power.to_csv(out / "power.csv", index=False)

    def smallest(outcome, design, target=0.8):
        ok = power[(power.outcome == outcome) & (power.design == design) & (power.power >= target)]
        ok = ok[ok.power.notna()]
        if ok.empty:
            return None
        r = ok.assign(sm=ok.ships * ok.months).sort_values(["sm", "months"]).iloc[0]
        return {"ships": int(r.ships), "months": int(r.months), "power": float(r.power)}

    metrics = {
        "dataset": cfg["description"],
        "true_rates_per_ship_month": rates,
        "true_rate_ratio": {k: v["treated"] / v["control"] for k, v in rates.items() if isinstance(v, dict)},
        "example_pilot": {"ships": pilot["ships"], "months": pilot["months"], "waves": pilot["waves"],
                          "results": example},
        "smallest_pilot_for_80pct_power": {
            o: {d: smallest(o, d) for d in ("stepped_wedge", "parallel_ab_cuped", "parallel_ab")}
            for o in ("failures", "no_margin_days")},
        "false_positive_rate_at_alpha_0.05": {
            o: {d: power[(power.outcome == o) & (power.design == d) & power.false_positive_rate.notna()]
                .set_index("ships")["false_positive_rate"].to_dict()
                for d in ("stepped_wedge", "parallel_ab_cuped", "parallel_ab")}
            for o in ("failures", "no_margin_days")},
        "power_trials_per_cell": cfg["power"]["trials"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    _plot(power, out / "pilot_power.png")
    return metrics


def _plot(power, path):
    outcomes = [("failures", "In-service generator failures"), ("no_margin_days", "Days without N+1 margin")]
    designs = [("stepped_wedge", "Stepped wedge (TWFE Poisson)"), ("parallel_ab_cuped", "Parallel A/B + CUPED"),
               ("parallel_ab", "Parallel A/B")]
    months = sorted(power.months.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (o, title) in zip(axes, outcomes):
        for k, (d, label) in enumerate(designs):
            g = power[(power.outcome == o) & (power.design == d) & (power.months == months[-1])
                      & power.power.notna()]
            ax.plot(g.ships, g.power * 100, marker="o", color=SERIES[k], label=label, markersize=5)
        ax.axhline(80, color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(f"{title}\n({months[-1]}-month pilot)", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Ships in the pilot")
        ax.set_ylim(0, 102)
        style_axes(ax)
    axes[0].set_ylabel("Power to detect the effect (%)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("How big must a pilot be to prove the effect? (simulated, p < 0.05)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
