# Model card: steering-gear hydraulic power unit condition (`steering_hydraulics`)

**Intended use:** grade the condition of cooler, valve, pump (internal leakage) and
accumulator of a hydraulic power unit every load cycle, and flag any component at a
critical grade. Decision support for the steering-gear maintenance plan.

**Models:** per-cycle features (level, spread, extremes, trend, start / end) of 14
physical sensors; one LightGBM grade classifier per component; an auxiliary
unsupervised health score on normal-wear cycles.

**Data:** UCI condition monitoring of hydraulic systems, a **real test rig** (ZeMA),
2,205 cycles, four components degraded simultaneously, CC BY 4.0. Evaluation is
out-of-fold by rig segment (194 blocks of constant settings); the serving model leaves
out 15% of segments so the dashboard demo is unseen.

**Performance (unseen segments):** grading accuracy pump 99.8%, valve 93%, cooler 92%,
accumulator 84%; critical grade recognised 99 / 97 / 85 / 69%. Any-critical alarm 85%
at 16% on normal-wear cycles; the low-flow alarm reaches 90% at 0% but cannot say
which component.

**Limits:** a single rig; the accumulator is the weakest component; only 62
normal-wear cycles, so a healthy-only baseline is not usable here (66% false alarms).
Logistic regression would grade the cooler better (CV 0.97 vs 0.93) - adopting it
needs a fresh hold-out.

**Safety mapping:** pump leakage / valve → loss of steering (severity 4); accumulator →
loss of steering (3); cooler → fire risk (3); two power units installed (SOLAS).
