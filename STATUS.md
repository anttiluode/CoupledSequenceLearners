# Execution status

Last updated: 2026-09-26, after frozen v0 completion.

Branch: `work/coupled-v0`

## Completed

- Approved design and implementation plan are committed under `docs/superpowers/`.
- Frozen preregistration is committed in `predictions.json` before the eight-seed result run.
- Frozen seeds: `[11, 23, 37, 53, 71, 89, 107, 131]`.
- Durable per-seed receipts, partner-swap attacker receipt, and mechanical aggregate receipt are under `results/`.
- Local verification: **32 tests passing**.
- Regenerating `results/receipt.json` from the durable receipts is byte-identical, SHA-256 `b3c7814308e7398ab0463ec0d5fcd4427b73d6b1c0ce94789058718c9b22c053`.

## Frozen scientific outcome

The primary preregistered gate **FAILED**.

- live reciprocal accuracy: **0.7954 ± 0.0270 SE**
- yoked replay accuracy: **0.7292 ± 0.0160 SE**
- mean live − yoked advantage: **+0.0662 ± 0.0296 SE**
- positive seeds: **6/8**
- preregistered effect-size threshold: **≥ +0.10**

The seed-count requirement passed, but the mean effect-size requirement did not; therefore `primary_pass=false`.

Secondary results:

- partner-swap immediate accuracy drop: **0.2388 ± 0.0634 SE** — secondary gate passed;
- listener-first startup prediction failed: fresh **8/8**, state-dependent pretraining **6/8**, state-independent pretraining **3/8**;
- live reciprocal had the best mean attacker-arm accuracy, but A→B-only was close at 0.7703, so v0 does not establish that reciprocity itself is uniquely necessary.

## Claim boundary

The defensible v0 conclusion is narrow: two independently stateful sequence learners learned pair-specific communication and live interaction often helped after a counterfactual fork, but the preregistered live-vs-yoked effect was smaller than required. The strong interaction-state hypothesis is therefore **not supported by v0**.

See `README.md`, `docs/experiment_log.md`, and `results/receipt.json` for the durable record.
