# CoupledSequenceLearners

Status: **NOT RUN**.

This repository tests one narrow causal question: after two independent sequence learners share the same prefix state, does a **live reciprocal partner** recover from a post-prefix counterfactual perturbation better than a causally stale **yoked replay** with matched communication budget?

The decisive comparison restores the same junction snapshot twice. In the live arm, the partner may react to the intervention. In the yoked arm, the partner is forced to continue the previously recorded unperturbed messages.

This is a synthetic computation experiment. It does **not** claim shared consciousness, literal interpersonal neural synchrony, mirror-neuron implementation, or biological spike-waveform communication.

The approved design and execution plan live under `docs/superpowers/`.
