# Experiment Log

## Durable setup

- Approved design: `docs/superpowers/specs/2026-09-26-coupled-sequence-learners-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-26-coupled-sequence-learners-v0.md`
- Current scientific status: NOT RUN

## Preregistration

Not frozen yet. No full frozen-seed results may be interpreted before `predictions.json` is committed.

## Exploratory / not preregistered

No pilots yet.

## Exploratory / not preregistered — Task 8 pilot

Two reduced debug runs were executed before `predictions.json` existed. They are **not results** and are not used to move thresholds.

- Debug config: hidden=8, message=2, 300 training steps, batch=64, message noise=0.02.
- Seed 0: training loss 3.3986 -> 1.3522; unperturbed branch accuracy 0.7109. Under the original `query XOR 1` intervention, live and yoked were exactly identical (0.6797 vs 0.6797). Inspection showed why: the trained A message saturated on the other query coordinate, so XOR-1 changed A state but not A's emitted message at all.
- Seed 1: training loss 3.4374 -> 3.3930 and accuracy remained near chance; useful warning that the small debug model does not always start.
- Shared-only branch probes stayed near chance (0.2422, 0.2578), so no obvious shared-prefix future leak appeared.

**Ruling before preregistration:** change the controlled query intervention from `q XOR 1` to the antipodal `q XOR 3`. The four A queries are encoded by two signed temporal features; XOR-3 flips both encoded features for every episode, avoiding a dimension-specific no-op intervention. This is an assay-correctness fix, not a threshold change. The preregistered thresholds remain those already written in the implementation plan.

### Corrected antipodal debug rerun

With the fixed `q XOR 3` intervention and the same exploratory config:

| seed | loss initial→final | unperturbed acc | live acc | yoked acc | live−yoked | A→B only | B→A only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.3986→1.3522 | 0.7109 | 0.7031 | 0.6406 | +0.0625 | 0.6641 | 0.6836 |
| 1 | 3.4374→3.3930 | 0.2695 | 0.3047 | 0.3047 | 0.0000 | 0.3047 | 0.2227 |

The corrected assay is no longer a guaranteed no-op, but the pilot does **not** establish the preregistered effect. Seed 0 shows only a modest live advantage and a one-way arm is competitive; seed 1 fails to train. Those observations are left as warnings, not used to alter the planned frozen thresholds.
