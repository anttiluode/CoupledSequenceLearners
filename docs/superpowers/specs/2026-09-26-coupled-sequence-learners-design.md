# CoupledSequenceLearners Design

**Date:** 2026-09-26  
**Status:** design contract, no experimental results yet

## Purpose

Build the smallest falsifiable system that tests whether reciprocal interaction between two sequence learners can acquire predictive state that is not reducible to passive exposure to the same messages.

The project is a computational experiment, not a model of a whole brain and not evidence for collective consciousness, mirror-neuron coding, waveform communication, or literal neural synchrony.

It is a direct descendant of the predictive-susceptibility / ping-listener line. The question is narrower:

> When two independently stateful sequence learners repeatedly affect one another, does a closed reciprocal loop provide predictive capability that an otherwise matched non-reactive partner does not?

The strongest intended survivor, if supported, is:

```text
interaction itself can carry predictive state
```

Here "interaction state" means causal state distributed across the current states and recent reciprocal history of both learners. It does not imply a third physical agent.

## Why this experiment is needed

A naive live-partner-versus-exact-replay comparison is insufficient. If learner A begins from the same state and receives the exact same complete input trace, a deterministic A should produce the same behavior. Replaying the complete trace preserves the downstream causes that affected A.

The discriminator therefore has to be counterfactual.

Both live and control conditions share an identical prefix. At a branching junction, perturb A after the shared prefix. In the live condition B can observe A's changed behavior and react. In the yoked/open-loop condition B continues a previously recorded continuation that was generated without that perturbation. The post-perturbation difference isolates the value of reciprocal reactivity rather than the information contained in the shared prefix.

## Scientific question

Can two independently trained sequence learners, with different internal coordinate systems and only a narrow reciprocal channel, form a pair-specific predictive coupling that improves recovery from unseen junction perturbations?

A positive result requires more than correlated hidden states or better average communication. It must survive controls that preserve message statistics while breaking reciprocal contingency.

## Core world

Use a small synthetic branching sequence world so every causal variable is known.

A world episode consists of:

1. a private prefix observed by A,
2. a different private prefix observed by B,
3. a shared observable segment,
4. an ambiguous junction with multiple legal continuations,
5. a joint continuation whose correct branch depends on information distributed across A's and B's private histories.

Neither learner alone receives enough information to choose correctly at the junction.

Training examples include many partial sequence motifs and overlapping junctions. Held-out evaluation contains recombinations that were not presented as complete training sequences.

## Learners

Use two small recurrent learners, A and B.

Each learner contains:

- a private-history encoder,
- a recurrent state,
- a next-event predictor,
- a narrow outbound message head,
- an inbound message pathway whose effect can depend on current receiver state.

The learners do not share parameters.

Their latent coordinate systems must be deliberately made non-comparable by construction. At minimum, use independent initialization. A stronger control applies a fixed invertible or orthogonal reparameterization at one learner's exposed diagnostic state so successful coupling cannot be interpreted as raw coordinate matching.

The v0 implementation must remain non-resonant. Do not use oscillator banks, neural spike waveforms, theta/gamma labels, or biological cell-type assignments. If the effect is real at the abstraction level claimed, it should appear in ordinary stateful sequence learners.

## Interaction loop

At interaction step t:

```math
m_A^t = G_A(x_A^t)
```

```math
x_B^{t+1} = F_B(x_B^t, u_B^t, m_A^t)
```

```math
m_B^t = G_B(x_B^t)
```

```math
x_A^{t+1} = F_A(x_A^t, u_A^t, m_B^t)
```

where `u_A` and `u_B` are each learner's local observations.

The exact implementation may update both messages synchronously from pre-step states or in an explicitly ordered A→B→A substep. The choice must be fixed before results and used identically across experimental arms.

## Primary experiment: shared prefix, counterfactual fork

For each held-out episode:

1. Run a live reciprocal pair through a prefix up to a designated junction.
2. Snapshot both complete learner states at the junction.
3. Record the unperturbed continuation of B from that snapshot.
4. Restore the exact junction snapshot.
5. Apply a controlled perturbation to A only. The perturbation changes information relevant to the correct branch but preserves the externally shared prefix.
6. Evaluate two continuations from the same restored snapshot:
   - **live reciprocal:** B receives A's post-perturbation messages and may react; A then receives B's changed messages.
   - **yoked open-loop:** B is forced to emit the previously recorded unperturbed messages, so B cannot react to A's changed state.
7. Compare branch accuracy, recovery time, predictive loss, and prediction-distribution alignment.

This is the decisive comparison. The open-loop trace is yoked to the same pair and same prefix, but is causally stale after the intervention.

## Pre-registered gates

### G1 — Reciprocal recovery

After the counterfactual fork, live reciprocal pairs must outperform yoked open-loop continuation on held-out branch accuracy or predictive loss.

A positive result must replicate across a frozen seed set, not rely on one lucky pair.

### G2 — Reciprocity rather than bandwidth

The live advantage must remain when live and control arms are matched for:

- message dimensionality,
- number of message exchanges,
- message timing,
- noise level,
- and, where feasible, message norm/energy distribution.

If simply giving more or larger messages explains the effect, the interaction-state claim fails.

### G3 — One-way and frozen-partner attackers

Compare reciprocal interaction with:

- A→B only,
- B→A only,
- a partner whose message head is frozen after the prefix,
- and a partner whose post-junction messages are time-shuffled within the matched trace family.

The interpretation should track reciprocal contingency, not merely access to another learner's representation.

### G4 — Predictive alignment, not raw synchrony

Do not use hidden-state correlation as the primary success metric.

Measure each learner's predicted distribution over the next joint event. Successful pairs should reduce a symmetric divergence between their future predictions during coordination, for example Jensen-Shannon divergence:

```math
JS(P_A(z_{t+1}) || P_B(z_{t+1})).
```

A successful result may coexist with low raw latent correlation.

### G5 — Coordinate scrambling

Apply a fixed invertible/orthogonal diagnostic reparameterization to one learner's latent coordinates, or use an equivalent construction that makes raw coordinates incomparable without changing its input-output computation.

Task performance and predictive alignment should be invariant to this representational gauge.

If the claimed effect depends on coordinate equality, the deeper interpretation is rejected.

### G6 — Partner specificity and swap

After pairs have learned to coordinate, swap partners while preserving each learner's own parameters and local task experience.

Measure:

- immediate task degradation,
- prediction divergence,
- message-response mismatch,
- and recovery across subsequent interaction.

A transient partner-swap cost supports pair-specific coupling. No swap cost would suggest the protocol is universal rather than dyad-specific, which is still useful but weakens the interaction-history interpretation.

### G7 — Listener-first condition

Test a receiver-pretraining manipulation inherited from the Ping/Listener hypothesis.

Compare at least:

- **fresh listener:** ordinary random initialization,
- **state-dependent pretraining:** a separate task that requires combining an inbound signal with the listener's own recurrent state,
- **state-independent pretraining:** a task where the inbound cue is useful without reference to receiver state.

Then train the same reciprocal junction task.

Measure startup rate, steps to criterion, and final reciprocal-recovery performance.

The specific prediction is that state-dependent pretraining improves startup more than state-independent pretraining. This is a hypothesis, not an assumption required for the main reciprocal-coupling experiment.

### G8 — Novel branch recombination

Hold out complete joint trajectories while retaining their component motifs in training.

The pair should solve at least some junction continuations that neither learner nor pair saw as complete sequences during training.

This prevents the strongest result from reducing to replay of memorized pair trajectories.

### G9 — Centralized ceiling

Train or evaluate a matched centralized model that receives both private histories directly.

Its role is a solvability ceiling, not a competitor that reciprocal learners are expected to beat.

If the centralized model cannot solve the task, failures of the paired system are uninterpretable.

## Measurements

Primary metrics:

- held-out joint branch accuracy,
- next-event negative log likelihood,
- recovery steps after intervention,
- Jensen-Shannon divergence between A and B next-joint-event predictions,
- listener startup success across seeds,
- partner-swap performance drop and recovery.

Diagnostic metrics:

- message norm and variance,
- effective message dimensionality,
- raw hidden-state correlation (reported only as a diagnostic),
- decodability of private history from each message,
- decodability of the final branch from pre-junction state,
- performance of each attacker arm.

No single diagnostic is allowed to substitute for task behavior.

## Critical causal controls

### Exact-state restoration

Live and yoked arms must start from byte-equivalent or numerically verified snapshots of A and B at the junction.

### No future leakage

The perturbation and future branch label must not be available to either learner before the designated junction except through the intended private histories.

Probe decodability before the junction to detect accidental leakage.

### Matched stochasticity

Where stochastic noise exists, paired live/yoked comparisons must reuse matched random seeds/noise draws whenever this does not itself destroy the causal intervention.

### Replay provenance

Every replay trace must record which pair, seed, episode, prefix state, and unperturbed continuation produced it. A replay from a different pair is a separate attacker, not the primary yoked control.

### No post-hoc gate movement

Pass/fail thresholds and the seed set must be frozen in a machine-readable predictions file before the first full experimental run. Exploratory debugging runs must be labeled separately and must not silently redefine the preregistered gates.

## Data and reproducibility

The repository will keep the scientific state durable in Git.

Required durable artifacts:

```text
README.md                         scientific ledger and current conclusion
predictions.json                  frozen preregistration and seed set
docs/experiment_log.md            chronological experiment log
results/receipt.json              machine-readable primary receipt
results/*.json                    attacker and diagnostic receipts
src/                              focused implementation modules
tests/                            deterministic and invariant tests
```

The design and implementation plan live under `docs/superpowers/`.

Every meaningful stage gets a Git commit. Failed gates and abandoned implementations remain documented rather than overwritten by the final narrative.

## Implementation boundaries

Keep v0 inexpensive:

- CPU-friendly NumPy or small PyTorch models,
- no external environment simulator,
- no audio,
- no browser demo,
- no large language model,
- no GPU requirement,
- modest frozen seed count for the preregistered run.

Prefer a synthetic task that runs in minutes, not hours.

The experiment should be simple enough that causal controls can be understood by reading the code.

## Claim boundaries

A positive v0 may support only something like:

> In this synthetic sequence task, a learned reciprocal pair used post-intervention causal feedback to recover a distributed branch better than matched non-reactive replay, and successful coordination was better captured by alignment of future predictions than by equality of latent coordinates.

It would not establish that:

- human brains form a literal shared dynamical system,
- interpersonal neural synchrony causes understanding,
- mirror neurons implement the mechanism,
- biological spike waveform carries this interaction state,
- consciousness is distributed between people,
- social meaning cannot be described by ordinary information theory,
- or reciprocal coupling is generally superior to centralized computation.

A null result is scientifically useful. In particular, if live reciprocal interaction cannot beat the counterfactual yoked control once bandwidth and state are matched, the proposed deeper layer should be weakened rather than rescued with a more decorative model.

## Success criterion for v0

The project earns a follow-up only if all of these are true:

1. the centralized ceiling solves the held-out task;
2. at least one reciprocal architecture learns reliably across the frozen seed set;
3. live reciprocal recovery beats the primary counterfactual yoked control after a shared-prefix intervention;
4. the effect survives matched bandwidth/timing controls;
5. raw latent similarity is unnecessary;
6. the receipt and attackers reproduce from a clean checkout.

Listener-first pretraining, partner specificity, and novel recombination are important secondary gates. Their failure limits the interpretation but does not retroactively redefine the primary reciprocal-recovery result.
