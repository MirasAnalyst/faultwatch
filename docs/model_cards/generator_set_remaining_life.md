# Model card: generator-set remaining life and service planning (`turbofan_cmapss_fd001`)

**Intended use:** per-machine remaining-life estimate with uncertainty, early warning,
and a service plan that respects crew capacity (fleet MILP) or port calls and the N+1
rule (itinerary MILP). Feeds the safety register (blackout risk) and the power-plant
scenario.

**Models:** per-machine early-life baseline + smoothing → Mahalanobis health score;
LightGBM regressor on residual levels, rolling trends, health score and age (target
capped at 125 cycles). CatBoost is equivalent in grouped CV (10.62 vs 10.65 RMSE).

**Data:** NASA C-MAPSS FD001, **simulated** turbofan run-to-failure (100 training
engines, 100 test engines cut off before failure).

**Performance:** warning in time on 100% of engines (out-of-fold), median 84 cycles
(95% CI 76-90) vs 11 for the exhaust alarm (paired McNemar p = 0.03, Wilcoxon on lead
p < 1e-17); RUL RMSE 12.3 (10.3-14.4), MAE 9.7; maintenance plan saves $5.3M
(3.6-7.4) vs run-to-failure on assumed costs.

**Limits:** one operating condition (FD001); costs and the cycle-to-day reading are
illustrative; the RUL error used by the planner (10.6 cycles) is from cross-validation
near end of life.

**Monitoring:** drift on normal residuals; precision from work-order labels; the
planner's uncertainty should be refreshed from live residuals after each retrain.

**Safety mapping:** degradation → blackout (severity 4); 4 sets installed, 2 required.
