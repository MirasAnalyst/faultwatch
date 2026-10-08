# Pilot design: proving the effect on real ships

Numbers below come from `configs/fleet_rollout.yaml` (`reports/fleet_rollout/`),
which simulates ship-month outcomes with the effect sizes from the power-plant
scenario and analyses them exactly as the real pilot would be.

## Unit and assignment
* **Unit of randomization: the ship.** Machines on one ship share crew,
  itinerary and maintenance culture; randomizing machines would contaminate
  the comparison.
* **Stratify** by ship class (plant type: COGES / diesel-electric) and by the
  previous year's failure rate, then randomize within strata.

## Outcomes
| Role | Metric | Why |
|---|---|---|
| Primary | Days at sea without an N+1 generator margin, per ship-month | Leading indicator of blackout risk; frequent enough to measure |
| Secondary | In-service generator / steering-gear failures | The safety outcome itself; too rare to power alone |
| Guardrails | Planned maintenance hours, alarms per watch, unplanned port delays | Catch "safer but unworkable" |
| Adoption | Share of alerts acknowledged within the watch; acted on vs dismissed | A correct alert nobody acts on changes nothing |

## Designs and power (simulated, alpha 0.05, 100 trials per cell)
| Design | Analysis | Smallest pilot with >= 80% power (N+1-margin days) |
|---|---|---|
| Parallel A/B (half the ships) | per-ship rate, t-test; CUPED with the prior year's rate | 16 ships x 6 months |
| Stepped wedge (4 waves, all treated by the end) | two-way fixed effects (ship, month), SE clustered by ship; randomization inference | 24 ships x 6 months |

* For **in-service failures** no design reached 80% power, even 64 ships x 18 months.
* False-positive rates under no effect stayed near 5% for every design and fleet size.
* A naive before/after comparison is biased by the fleet-wide season (hurricane
  season, heat load); both designs above difference it out.

## Analysis plan (fixed before the pilot starts)
1. Primary: rate ratio of N+1-margin days, TWFE Poisson (stepped wedge) or
   per-ship rate comparison with CUPED (parallel), 95% CI.
2. Confirm with randomization inference over the actual assignment mechanism
   (valid with few ships; `faultwatch.stats.randomization_inference`).
3. Guardrails: one-sided tests for harm; stop if planned-maintenance hours rise
   more than 20% or alarms per watch double.
4. Report by ship class (slice parity) and adoption alongside the effect.

## Stopping rules
* **Safety:** any blackout or loss of steering on a pilot ship triggers an
  immediate review of that ship's alerts and plans, regardless of arm.
* **Futility** at month 3 only if the CI excludes a 20% improvement in the
  primary metric. No early stopping for efficacy.
