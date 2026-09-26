# CoupledSequenceLearners

Status: **FROZEN V0 COMPLETE — PRIMARY GATE FAILED**.

This repository tests one narrow causal question: after two independent sequence learners share the same prefix state, does a **live reciprocal partner** recover from a post-prefix counterfactual perturbation better than a causally stale **yoked replay** with matched communication budget?

The decisive comparison restores the same junction snapshot twice. In the live arm, the partner may react to the intervention. In the yoked arm, the partner is forced to continue the previously recorded unperturbed messages.

The experiment was preregistered before the eight-seed frozen run. The machine-readable preregistration is `predictions.json`; the frozen outcome is `results/receipt.json`.

## Frozen result

The primary preregistered result **did not pass**.

Across the eight frozen seeds, live reciprocal recovery reached **0.7954 ± 0.0270 SE** branch accuracy versus **0.7292 ± 0.0160** for yoked replay. The paired live-minus-yoked advantage was **+0.0662 ± 0.0296**, positive in **6/8 seeds**. The preregistered primary threshold required a mean advantage of **at least +0.10**, so v0 does **not** support the strong claim that reciprocal causal feedback produced the preregistered recovery advantage.

That null is not equivalent to “nothing happened.” Several narrower measurements survived, but they remain secondary because the primary causal gate failed.

## What passed

| preregistered gate | value | threshold | result |
| --- | ---: | ---: | :---: |
| centralized solvability ceiling | 1.0000 | ≥ 0.90 | PASS |
| live held-out branch accuracy | 0.7954 | ≥ 0.70 | PASS |
| live−yoked positive seeds | 6/8 | ≥ 6/8 | PASS |
| live/yoked message scalar count | equal | equal | PASS |
| diagnostic gauge logit difference | 0.0 | ≤ 1e-6 | PASS |
| held-out recombination live accuracy | 0.7954 | ≥ 0.60 | PASS |

The strongest secondary survivor was **partner specificity**. Swapping trained partners without adaptation caused a mean immediate branch-accuracy drop of **0.2388 ± 0.0634 SE**, above the preregistered secondary threshold of 0.05. This says the trained interfaces were pair-specific in this synthetic task; it does not rescue the failed primary live-vs-yoked criterion.

The live arm also had lower mean disagreement between the two learners' future predictions than the stale controls: mean final Jensen–Shannon divergence was **0.0323** for live versus **0.1935** for yoked replay. This is diagnostic, not a separate preregistered proof of reciprocal interaction state.

## What failed

The decisive primary gate failed:

| primary quantity | observed | preregistered threshold |
| --- | ---: | ---: |
| mean live−yoked branch-accuracy improvement | **+0.0662** | **≥ +0.10** |

Per-seed live−yoked differences were:

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

The preregistered **listener-first** secondary prediction also failed. Using 0.70 held-out accuracy as the startup criterion, fresh listeners started in **8/8** seeds, state-dependent-pretrained listeners in **6/8**, and state-independent-pretrained listeners in **3/8**. The expected ordering was state-dependent > state-independent > fresh; the observed ordering was fresh > state-dependent > state-independent.

This is especially useful because it cuts against the motivating Ping/Listener intuition rather than being explainable as a weak positive result. In this v0 architecture, state-dependent listener pretraining was not required for communication startup and did not beat fresh initialization.

## Attacker arms

All values are frozen eight-seed means from `results/receipt.json`.

| arm | branch accuracy | NLL | JS divergence |
| --- | ---: | ---: | ---: |
| live | **0.7954** | **0.3320** | **0.0323** |
| A→B only | 0.7703 | 0.3975 | 0.1331 |
| frozen partner | 0.7639 | 0.5028 | 0.0986 |
| time-shuffled stale trace | 0.7485 | 0.5477 | 0.1602 |
| yoked replay | 0.7292 | 0.6161 | 0.1935 |
| B→A only | 0.6921 | 0.8355 | 0.3214 |
| isolated A | 0.6782 | 0.6586 | 0.3452 |
| isolated B | 0.6782 | 0.6586 | 0.3452 |

The ordering is suggestive: live reciprocal interaction is best on mean accuracy, with A→B-only communication fairly close behind. Because the preregistered live−yoked effect-size gate failed, this table should be read as structure to investigate, not as a confirmed general mechanism.

## What this does not show

This synthetic experiment does **not** establish that:

- human brains form a literal shared dynamical system,
- interpersonal neural synchrony causes understanding,
- mirror neurons implement this mechanism,
- biological spike waveforms carry social interaction state,
- consciousness is distributed between people,
- information theory is the wrong language for communication,
- reciprocal coupling is generally superior to one-way communication,
- or the listener-first hypothesis from `The_Ping_And_The_Listener` is correct.

The narrow defensible v0 conclusion is:

> Two independently stateful sequence learners learned pair-specific communication and live reciprocal continuation was better on average than matched stale replay, but the preregistered live-vs-yoked effect was smaller than the required threshold. Therefore the strong interaction-state hypothesis was **not supported by v0**.

## Reproducibility

The scientific order is preserved in Git:

1. approved design,
2. implementation plan,
3. exploratory debugging and assay correction,
4. frozen `predictions.json`,
5. frozen seed receipts,
6. mechanical summary in `results/receipt.json`,
7. this interpretation.

The counterfactual intervention was fixed **before preregistration** from `query XOR 1` to the antipodal `query XOR 3` after the exploratory pilot showed that XOR-1 could change hidden state without changing the learned outbound message. The frozen thresholds were not changed.

Run:

```bash
python -m pip install -e .
pytest -q
python scripts/summarize_results.py
```

The approved design and implementation plan live under `docs/superpowers/`. The chronological scientific notebook is `docs/experiment_log.md`.
