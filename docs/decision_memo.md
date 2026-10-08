# Decision memo: a machinery-health pilot for generator sets and steering gear

*Sample memo for a hypothetical cruise operator; every number comes from this repository.*

**To:** VP Marine Operations / Fleet Safety · **From:** Data Science · **Decision needed:** approve a 6-month pilot on 16-24 ships

## The problem
A blackout or loss of steering at sea is a safety event before it is a cost: a
ship that loses power near a lee shore (*Viking Sky*, 2019) or in a port
approach (*Dali*, 2024) has minutes, not hours. Today our generator sets are
serviced on running-hour intervals and protected by fixed alarm limits. Both
are blind to *how* a particular machine is degrading, and neither knows the
itinerary: a set can be taken down for service at sea on the day the ship
needs every set for manoeuvring.

## What we built and what it shows
FaultWatch scores machinery from the sensors we already log, maps each failure
mode to its ship-level effect, and plans service around port calls and the N+1
rule. On public data that stands in for our machinery:

* Gas-turbine degradation is detected in **95%** of cases vs **0%** for the
  exhaust-temperature alarm; generator sets get **84 cycles' warning vs 11**,
  with no false early alarms.
* In a month-long simulation of 25 ships x 4 generator sets built on those
  real predictions, itinerary-aware planning cut **blackouts per 100 voyages
  from 2.7 (interval-based) to 0** and days without an N+1 margin from 12 to
  1.4, while planned services fell from 59 to 29.
* On steering-gear hydraulics, the existing low-flow alarm already catches
  most critical states; FaultWatch adds **which component** (pump leakage
  99.8%, valve 93%), which decides the repair and the risk.

## What we do not know yet
These are public, mostly simulated datasets. Our ships differ: sensors drop
out, crews intervene, failures are rare. The simulation's itinerary, loads and
costs are assumptions. Only a pilot on our own fleet can show the effect.

## Recommendation
Run a **6-month pilot on 16-24 ships**, randomized:

* **Primary metric: days a ship sails without an N+1 power margin.** In-service
  failures are too rare to measure in a pilot of any affordable size (no design
  reached 80% power even at 64 ships for 18 months); N+1-margin days are
  frequent enough and move with them.
* **Guardrails:** in-service failures, planned-maintenance hours, alarm volume
  per watch, alerts acknowledged within the watch.
* **Design:** parallel A/B (16 ships) is the most powerful; a stepped wedge
  (24 ships, everyone treated by month 6) is the fallback if operations prefer
  it. Details in `docs/pilot_design.md`.

## What we need
* Historian extracts for the generator sets and steering-gear power units of
  the pilot ships, with maintenance work orders for labels.
* An engineering owner per ship class to confirm the FMECA mapping
  (`configs/*.yaml` `safety:` blocks) - severities and redundancy are theirs to set.
* Chief engineers' feedback channel: every alert can be acknowledged and
  labelled, which is how we measure adoption and precision.

## Risks and how we handle them
| Risk | Mitigation |
|---|---|
| Alarm fatigue | False-alarm budgets fixed in advance; precision and ack rate monitored weekly; recalibrate if precision CI falls below 50% |
| Model drift after overhauls or sensor swaps | Drift monitor on normal-operation data; retrain trigger |
| Over-reliance on the copilot | Answers cite investigation reports and are framed as decision support; the engineer decides |
| A ship class performs worse | Slice parity by ship class in the monitoring view |
