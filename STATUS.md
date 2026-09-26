# Execution status

Last updated: 2026-09-26, before preregistration.

Branch: `work/coupled-v0`

## Completed locally and verified

- Tasks 1-7 from `docs/superpowers/plans/2026-09-26-coupled-sequence-learners-v0.md`.
- Full local suite: 26 tests passing.
- No-future-leak correction: post-junction observations are neutral; branch continuation exists only as a target.
- Stronger late-query world used for the pilot: A holds one of four queries; B holds a 4-bit context; the query selects which B bit determines the branch. The channel is closed until the junction.
- Counterfactual live/yoked machinery begins from the same saved pair state and validates replay provenance.
- Listener-first pretraining conditions are implemented and pass their intended toy tasks.

## Exploratory pilot only — NOT preregistered results

The first counterfactual used `query XOR 1`. It was discovered to be a defective intervention for this assay: in one trained seed it changed A's hidden state but not A's emitted message because the sender published only the other query coordinate. This made live and yoked exactly identical for a trivial reason.

Ruling before preregistration: use the fixed antipodal intervention `query XOR 3`, which flips both encoded query features for every episode. Thresholds are not changed.

Reduced debug config: hidden=8, message=2, 300 steps, batch=64, message noise=0.02.

- seed 0: loss 3.3986 -> 1.3522; unperturbed acc 0.7109; corrected live 0.7031, yoked 0.6406, delta +0.0625; A->B only 0.6641; B->A only 0.6836.
- seed 1: loss 3.4374 -> 3.3930; unperturbed acc 0.2695; corrected live 0.3047, yoked 0.3047; training effectively failed.
- shared-only branch probe remained near chance in both debug seeds (0.2422, 0.2578).

These observations are warnings, not evidence. They will not be used to move the frozen thresholds already specified in the implementation plan.

## Next exact step

Finish Task 8 GitHub sync, then Task 9: write `predictions.json` with the frozen seed set `[11,23,37,53,71,89,107,131]` and the already-planned thresholds **before** any full frozen-seed run.
