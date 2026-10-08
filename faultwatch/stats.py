"""Statistics for "did it really work?": confidence intervals, paired tests,
power analysis and difference-in-differences.

Rules this module follows:
  * Resample the unit that is independent - the machine, the recording, the
    event - never the individual sample. Samples from one machine are
    strongly correlated, and treating them as independent gives intervals
    that are far too narrow.
  * Compare two alarm methods on the *same* units (paired tests), which is
    much more powerful than comparing two independent rates.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# confidence intervals
def bootstrap_ci(data, stat: Callable, cluster=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """Percentile bootstrap CI of `stat(data)`.

    data:    a DataFrame (or array) of units
    cluster: column name (or array) of the independent unit; whole clusters
             are resampled. None = rows are independent.
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    rng = np.random.default_rng(seed)
    point = float(stat(df))
    if cluster is None:
        idx_of = [np.array([i]) for i in range(len(df))]
    else:
        keys = df[cluster] if isinstance(cluster, str) else pd.Series(np.asarray(cluster), index=df.index)
        idx_of = [np.flatnonzero((keys == k).to_numpy()) for k in pd.unique(keys)]
    n = len(idx_of)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        rows = np.concatenate([idx_of[i] for i in pick])
        v = stat(df.iloc[rows])
        if v is not None and np.isfinite(v):
            draws.append(float(v))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws else (np.nan, np.nan)
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "level": 1 - alpha, "units": n}


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion (good for small n and rates near 0/1)."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (0.0 if successes == 0 else float(max(0, c - h)),
            1.0 if successes == n else float(min(1, c + h)))


# ---------------------------------------------------------------------------
# paired comparisons of two methods on the same units
def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test on paired binary outcomes (e.g. 'engine warned in
    time' under method A vs method B). Only discordant pairs carry information."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "both": int((a & b).sum()),
            "neither": int((~a & ~b).sum()), "p_value": p}


def paired_permutation_test(diffs, n_perm: int = 10000, seed: int = 42) -> dict:
    """Sign-flip permutation test of mean(diffs) = 0. `diffs` are per-unit
    differences (method A minus method B), one per independent unit."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return {"mean_diff": 0.0, "p_value": 1.0, "units": len(d)}
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return {"mean_diff": float(d.mean()), "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "units": int(len(d))}


def wilcoxon_paired(a, b) -> dict:
    """Wilcoxon signed-rank test on paired continuous outcomes (e.g. warning lead)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 2 or np.allclose(d, 0):
        return {"median_diff": 0.0, "p_value": 1.0, "pairs": int(ok.sum())}
    r = stats.wilcoxon(a[ok], b[ok], zero_method="zsplit")
    return {"median_diff": float(np.median(d)), "p_value": float(r.pvalue), "pairs": int(ok.sum())}


def mann_whitney(x, y) -> dict:
    """Two independent samples (e.g. health scores of healthy vs faulty machines).
    Also returns the AUC = P(score of a random y > random x), an effect size
    that stakeholders can read."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.mannwhitneyu(y, x, alternative="two-sided")
    return {"auc": float(r.statistic / (len(x) * len(y))), "p_value": float(r.pvalue),
            "n_x": len(x), "n_y": len(y)}


# ---------------------------------------------------------------------------
# experiment design
def n_per_arm_two_proportions(p_control: float, p_treat: float, alpha: float = 0.05,
                              power: float = 0.8) -> int:
    """Units per arm to detect p_control -> p_treat with a two-sided z-test."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p_control + p_treat) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar))
           + zb * np.sqrt(p_control * (1 - p_control) + p_treat * (1 - p_treat))) ** 2
    return int(np.ceil(num / (p_control - p_treat) ** 2))


def n_per_arm_poisson_rates(rate_control: float, rate_ratio: float, alpha: float = 0.05,
                            power: float = 0.8) -> float:
    """Exposure (e.g. ship-months) per arm to detect a change in an event rate
    (events per unit exposure) by `rate_ratio` - a Wald test on the log rate ratio."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    r1 = rate_control * rate_ratio
    return float((za + zb) ** 2 * (1 / rate_control + 1 / r1) / np.log(rate_ratio) ** 2)


def cuped(y, x_pre) -> np.ndarray:
    """CUPED variance reduction: subtract the part of the outcome explained by a
    pre-period covariate (e.g. each ship's failure rate in the prior year)."""
    y, x = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1) if np.var(x) > 0 else 0.0
    return y - theta * (x - x.mean())


# ---------------------------------------------------------------------------
# difference-in-differences
def did_estimate(panel: pd.DataFrame, outcome: str, treated: str, unit: str, period: str) -> dict:
    """Two-way fixed-effects DiD: outcome ~ treated + unit FE + period FE, with
    standard errors clustered by unit. `treated` is 1 once a unit has adopted.
    Valid for a staggered rollout when the effect is constant over time (which
    holds in the simulation; with real data, check event-study plots)."""
    import statsmodels.formula.api as smf
    df = panel[[outcome, treated, unit, period]].dropna().copy()
    df.columns = ["y", "d", "u", "t"]
    fit = smf.ols("y ~ d + C(u) + C(t)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["u"]})
    lo, hi = fit.conf_int().loc["d"]
    return {"effect": float(fit.params["d"]), "se": float(fit.bse["d"]),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(fit.pvalues["d"]),
            "units": int(df["u"].nunique()), "periods": int(df["t"].nunique())}


def randomization_inference(estimate_fn: Callable[[np.ndarray], float], assignment: np.ndarray,
                            reassign: Callable[[np.random.Generator], np.ndarray],
                            n_perm: int = 500, seed: int = 42) -> dict:
    """Exact-style p-value for a randomized rollout: recompute the estimate
    under re-drawn random assignments. Needs no distributional assumptions and
    stays valid with few units (a pilot on a handful of ships)."""
    rng = np.random.default_rng(seed)
    obs = estimate_fn(assignment)
    null = np.array([estimate_fn(reassign(rng)) for _ in range(n_perm)])
    return {"estimate": float(obs),
            "p_value": float((1 + (np.abs(null) >= abs(obs)).sum()) / (n_perm + 1)),
            "permutations": n_perm}


def compare_alarms(units: pd.DataFrame, methods: list[str], reference: str, positive: str,
                   cluster: str | None = None, n_boot: int = 1000, seed: int = 42) -> dict:
    """Uncertainty for an alarm comparison table.

    units:     one row per sample (or event); a boolean column per method
               (alarm raised) and a boolean `positive` column (fault present)
    cluster:   independent unit to resample (machine, recording); None = rows
    reference: the method every other one is tested against (FaultWatch)

    Per method: detection rate on faults and false-alarm rate on healthy
    rows with bootstrap CIs; and a paired test of detection vs the reference
    (sign-flip permutation on per-cluster differences, or exact McNemar when
    rows are independent)."""
    pos, neg = units[units[positive]], units[~units[positive]]
    out = {}
    for m in methods:
        rate = lambda d, m=m: float(d[m].mean()) if len(d) else np.nan
        r = {"detection": bootstrap_ci(pos, rate, cluster, n_boot, seed=seed),
             "false_alarm": bootstrap_ci(neg, rate, cluster, n_boot, seed=seed) if len(neg) else None}
        if m != reference:
            if cluster:
                per = pos.groupby(cluster)[[reference, m]].mean()
                r["vs_reference"] = paired_permutation_test(per[reference] - per[m], seed=seed)
            else:
                r["vs_reference"] = mcnemar_exact(pos[reference], pos[m])
            r["vs_reference"]["reference"] = reference
        out[m] = r
    return out


def fmt_ci(r: dict, pct: bool = False, digits: int = 1) -> str:
    """'84 (95% CI 78-90)' for README tables and the dashboard."""
    k = 100 if pct else 1
    u = "%" if pct else ""
    return (f"{r['estimate'] * k:.{digits}f}{u} "
            f"(95% CI {r['ci_low'] * k:.{digits}f}-{r['ci_high'] * k:.{digits}f}{u})")
