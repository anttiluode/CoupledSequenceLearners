# Experiment Log

## Durable setup

- Approved design: `docs/superpowers/specs/2026-09-26-coupled-sequence-learners-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-26-coupled-sequence-learners-v0.md`
- Current scientific status: FROZEN V0 COMPLETE — PRIMARY GATE FAILED

## Preregistration

Frozen before any full seed run.

- code base SHA: `dfa9e6522c7a2db2f4d088eb470cad43bb371e36`
- seeds: `[11, 23, 37, 53, 71, 89, 107, 131]`
- counterfactual: antipodal query `q XOR 3`
- preregistration digest: `sha256:e67e4533c01659d9d5ca42bcfcedb4225ccf1ce2401f49de451822ffe043a8cb`
- thresholds: exactly those in `predictions.json`; no full frozen-seed run occurred before this file was written.

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

## Frozen v0 run

The frozen run was executed only after `predictions.json` was written and committed. Because repeated PyTorch trainings in one long Python process did not terminate reliably in this container, execution was changed to a resumable **one-process-per-seed/condition harness**. This changed only execution isolation and receipt timing; frozen seeds, task generation, architecture, 1200 training steps, thresholds, counterfactual mask, and evaluation definitions were unchanged.

Environment used for the recorded run:

- Python 3.13.5
- NumPy 2.3.5
- PyTorch 2.10.0+cpu
- pytest 9.0.2
- Linux x86_64, CPU execution

The condition receipt file timestamps span about 44.5 minutes from the earliest preserved fresh seed receipt to the final worker receipt. Earlier failed monolithic execution attempts are not counted as frozen outcomes; only completed worker receipts assembled under the frozen config enter `results/receipt.json`.

### Primary result

The preregistered primary gate **FAILED**.

Across the eight frozen seeds:

- live reciprocal branch accuracy: **0.7954 ± 0.0270 SE**
- yoked replay branch accuracy: **0.7292 ± 0.0160 SE**
- paired live − yoked improvement: **+0.0662 ± 0.0296 SE**
- positive live − yoked seeds: **6/8**

The preregistered requirement was mean live − yoked improvement **≥ +0.10** and positive in at least 6/8 seeds. The seed-count component passed; the effect-size component did not. Therefore `primary_pass=false`.

Per-seed differences:

| seed | live | yoked | live−yoked |
| ---: | ---: | ---: | ---: |
| 11 | 0.7344 | 0.7188 | +0.0156 |
| 23 | 0.8672 | 0.7012 | +0.1660 |
| 37 | 0.8887 | 0.6992 | +0.1895 |
| 53 | 0.7871 | 0.6699 | +0.1172 |
| 71 | 0.8965 | 0.8203 | +0.0762 |
| 89 | 0.7344 | 0.7305 | +0.0039 |
| 107 | 0.7324 | 0.7480 | −0.0156 |
| 131 | 0.7227 | 0.7461 | −0.0234 |

Other preregistered gates passed:

- centralized ceiling mean accuracy: 1.0000 ≥ 0.90
- live held-out accuracy: 0.7954 ≥ 0.70
- live/yoked communication scalar count: exactly equal
- diagnostic gauge logit max difference: 0.0 ≤ 1e−6
- held-out recombination live accuracy: 0.7954 ≥ 0.60

### Secondary results

**Partner specificity passed strongly.** Cyclic partner swap without adaptation produced a mean immediate branch-accuracy drop of **0.2388 ± 0.0634 SE**, above the preregistered secondary threshold of 0.05.

**Listener-first pretraining failed.** Startup counts at the frozen 0.70 accuracy criterion were:

- fresh: **8/8**
- state-dependent pretraining: **6/8**
- state-independent pretraining: **3/8**

The preregistered expected order was state-dependent > state-independent > fresh. The observed order was the reverse at the top: fresh > state-dependent > state-independent. Mean unperturbed accuracies were 0.8047, 0.7888, and 0.6067 respectively.

### Attacker structure

Mean frozen branch accuracies:

- live reciprocal: 0.7954
- A→B only: 0.7703
- frozen partner: 0.7639
- time-shuffled stale trace: 0.7485
- yoked replay: 0.7292
- B→A only: 0.6921
- isolated A/B: 0.6782

Live reciprocal interaction had the best mean accuracy and the lowest future-prediction JS divergence (0.0323), but the one-way A→B arm was close enough that v0 does not justify a strong claim that reciprocity itself is uniquely necessary.

### Frozen interpretation

The preregistered strong statement did not survive v0. A weaker empirical pattern remains: learned live coupling often helped after the counterfactual fork, successful pairs were partner-specific, and their future predictions aligned more closely during live interaction. The effect was too small and too seed-variable to clear the frozen primary threshold, and listener-first pretraining did not behave as predicted.

The correct next move, if this line is continued, is not to relabel v0 as a success. It is to explain why seeds 23/37/53 show large live-replay separation while 107/131 reverse it, and why A→B-only communication captures much of the live advantage.

### Receipt compaction for repository storage

The worker-level `results/work/*.json` files are verbose and duplicate several fields across fresh/counterfactual sections. Before committing the durable `results/seeds/*.json` receipts, those seed receipts were compacted by removing duplicated diagnostic fields that are not consumed by `scripts/summarize_results.py`. The following required scientific fields remain: frozen/git/config digests, model/data seed, centralized ceiling, all counterfactual arm metrics and provenance, fresh unperturbed metrics, listener-pretraining unperturbed metrics, gauge/leakage diagnostics, state-digest equality, and message-budget equality. Regenerating `results/receipt.json` after compaction produced the identical SHA-256 `b3c7814308e7398ab0463ec0d5fcd4427b73d6b1c0ce94789058718c9b22c053`.
