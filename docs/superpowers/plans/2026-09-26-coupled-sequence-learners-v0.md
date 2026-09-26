# CoupledSequenceLearners v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and preregister a small CPU-friendly experiment that tests whether live reciprocal reactivity after a shared-prefix intervention improves distributed sequence recovery compared with causally stale yoked replay.

**Architecture:** Two independent recurrent learners receive different private histories and a shared sequence segment, exchange narrow messages, and predict a joint branch at an ambiguous junction. The primary causal assay snapshots both learner states at the junction, perturbs A, and compares a live reciprocal continuation with a yoked open-loop continuation from the exact same snapshot. Auxiliary attackers test one-way communication, frozen or shuffled partners, representational gauge changes, partner swap, listener-first pretraining, and held-out branch recombination.

**Tech Stack:** Python 3.11+, NumPy, PyTorch CPU, pytest, JSON receipts; no GPU or external simulator required.

**Spec:** `docs/superpowers/specs/2026-09-26-coupled-sequence-learners-design.md`

## Global Constraints

- Keep v0 non-resonant: no oscillator bank, spike waveform, theta/gamma, or cell-type mechanism.
- CPU only; target a preregistered run that completes in minutes, not hours.
- Freeze seed set and pass/fail thresholds in `predictions.json` before the first full result run.
- All live/yoked primary comparisons begin from verified identical learner snapshots at the junction.
- No future branch information may leak before the designated intervention except through intended private histories.
- Message dimensionality, exchange count, timing, and noise must be matched across live and primary control arms.
- Raw hidden-state similarity is diagnostic only; predictive-distribution alignment is the primary alignment measure.
- Record nulls and failed gates in Git rather than overwriting them.
- Main claim boundary: synthetic reciprocal causal feedback only; no claim about human shared minds, mirror neurons, spike-waveform coding, or biological synchrony.

## Review Focus

1. **Hidden future leakage:** a learner must not decode the post-junction branch from shared observations alone before the intended private-history/message route; Task 2 adds an explicit leakage test and diagnostic probe.
2. **Snapshot contamination:** live and yoked arms must not mutate a shared state object; Task 4 tests deep-copy state restoration and numerical equality before continuation.
3. **Replay mismatch:** the primary replay must come from the same pair, episode, and junction snapshot; Task 4 tests provenance identity and rejects cross-pair traces as the primary control.
4. **Unequal communication budget:** attacker arms must preserve message width and exchange count unless the attacker definition explicitly changes directionality; Task 5 tests budget accounting.
5. **Gauge diagnostic accidentally affecting behavior:** coordinate scrambling is allowed only on exposed diagnostic coordinates, or via an exact function-preserving reparameterization; Task 6 tests identical logits/performance before and after the gauge transform.

---

## File Structure

```text
README.md                         scientific ledger and current conclusion
pyproject.toml                    package/test configuration
predictions.json                  frozen gates, seeds, thresholds, run config
src/coupled_sequence_learners/
  __init__.py
  world.py                        synthetic branching task generation
  model.py                        recurrent learner and message interface
  pair.py                         reciprocal step, snapshots, replay traces
  metrics.py                      branch/NLL/JS/recovery diagnostics
  pretraining.py                  listener-first curriculum conditions
  experiment.py                   train/evaluate one seed and attacker arms
  run.py                          CLI for debug/preregistered runs
scripts/
  freeze_predictions.py           one-time preregistration writer/validator
  run_preregistered.py            frozen batch runner
  summarize_results.py            receipt aggregation
results/
  .gitkeep
  receipt.json                    final primary machine-readable summary
  seeds/                          per-seed result JSON files
  attackers/                      auxiliary control receipts
docs/
  experiment_log.md               chronological scientific notebook
  superpowers/specs/...           approved design
  superpowers/plans/...           this plan
tests/
  test_world.py
  test_model.py
  test_pair.py
  test_metrics.py
  test_pretraining.py
  test_experiment.py
  test_preregistration.py
```

### Task 1: Package skeleton, deterministic config, and scientific ledger

**Files:**
- Create: `pyproject.toml`
- Create: `src/coupled_sequence_learners/__init__.py`
- Create: `README.md`
- Create: `docs/experiment_log.md`
- Create: `results/.gitkeep`
- Test: `tests/test_preregistration.py`

**Interfaces:**
- Produces package import path `coupled_sequence_learners`.
- Establishes a single frozen configuration schema later written to `predictions.json`.

- [ ] **Step 1: Write the failing import/config smoke test**

```python
def test_package_exports_version_and_default_config():
    import coupled_sequence_learners as csl
    cfg = csl.default_config()
    assert csl.__version__ == "0.1.0"
    assert cfg["message_dim"] == 2
    assert cfg["hidden_dim"] == 16
    assert cfg["junction_step"] == cfg["private_prefix_len"] + cfg["shared_prefix_len"]
```

- [ ] **Step 2: Run the smoke test and verify failure**

Run: `pytest tests/test_preregistration.py::test_package_exports_version_and_default_config -v`

Expected: FAIL because package/config does not exist.

- [ ] **Step 3: Implement package config and minimal project metadata**

Create `default_config() -> dict[str, object]` in `src/coupled_sequence_learners/__init__.py`. Pin initial v0 values: hidden_dim 16, message_dim 2, observation_dim 8, branch_count 4, private_prefix_len 6, shared_prefix_len 6, junction_step 12, post_junction_len 4, message_noise_std 0.05, training_steps 1200, batch_size 128. Keep thresholds out of this function; they belong in `predictions.json`.

- [ ] **Step 4: Add README and experiment log headers**

README must state the causal question, the corrected counterfactual replay design, current status `NOT RUN`, and claim boundaries. `docs/experiment_log.md` starts with design/spec commit references and a section reserved for preregistration.

- [ ] **Step 5: Run smoke test**

Run: `pytest tests/test_preregistration.py::test_package_exports_version_and_default_config -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `chore: scaffold coupled sequence experiment`

---

### Task 2: Synthetic branching world with distributed private information and no leakage

**Files:**
- Create: `src/coupled_sequence_learners/world.py`
- Test: `tests/test_world.py`

**Interfaces:**
- Produces `EpisodeBatch` containing `obs_a`, `obs_b`, `branch`, `junction_step`, `episode_id`, and private latent factors `factor_a`, `factor_b` for diagnostics only.
- Produces `generate_batch(seed: int, batch_size: int, split: str, config: dict) -> EpisodeBatch`.
- Produces `heldout_signature(batch) -> tuple` used to verify complete trajectory recombination is unseen.

- [ ] **Step 1: Write deterministic/distributed-information tests**

Tests assert: same seed gives byte-identical tensors; A private history alone and B private history alone each leave at least two legal branches; the pair of private factors determines one of four branches; shared segment is identical across branch labels; train and held-out splits exclude a frozen set of complete factor/motif combinations while retaining component motifs.

- [ ] **Step 2: Run world tests and verify failure**

Run: `pytest tests/test_world.py -v`

Expected: FAIL because world module is missing.

- [ ] **Step 3: Implement `EpisodeBatch` and generator**

Use two balanced binary private factors, one available only in A's private prefix and one only in B's. Define branch index as the two-bit pair `(factor_a, factor_b)` rather than XOR so neither side alone determines the branch and the centralized ceiling has four classes. Encode private factors through order/motif differences with matched event counts, not direct one-hot labels. Shared prefix must be branch-neutral.

- [ ] **Step 4: Add a leakage diagnostic baseline**

Implement `shared_only_branch_counts(batch) -> np.ndarray` and a test asserting exact branch balance conditional on the shared segment construction. Do not train a learned probe yet; the experiment task adds empirical decodability checks.

- [ ] **Step 5: Run world tests**

Run: `pytest tests/test_world.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add distributed branching sequence world`

---

### Task 3: Independent recurrent learner with state-dependent message reception

**Files:**
- Create: `src/coupled_sequence_learners/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- `LearnerConfig(hidden_dim: int, obs_dim: int, message_dim: int, branch_count: int)`
- `CoupledLearner.forward_step(obs: Tensor, inbound: Tensor, state: Tensor) -> StepOutput`
- `StepOutput.state`, `.message`, `.branch_logits`
- `CoupledLearner.initial_state(batch_size: int, device=None) -> Tensor`
- `CoupledLearner.clone_state(state: Tensor) -> Tensor`

- [ ] **Step 1: Write model shape and independence tests**

Tests assert expected tensor shapes, independent parameter storage for separately constructed A/B models, deterministic forward pass in eval mode, and message reception that depends on recurrent state rather than only adding a fixed message projection.

- [ ] **Step 2: Run model tests and verify failure**

Run: `pytest tests/test_model.py -v`

Expected: FAIL because model is missing.

- [ ] **Step 3: Implement the learner**

Use a compact GRUCell-style recurrent core. Compute an inbound modulation term from `concat(state, inbound)` through a small MLP before recurrent update, so the receiver can represent state-dependent susceptibility without hard-coding a multiplicative task solution. Message head is linear+tanh from state; branch head is linear from state. No shared weights between A and B.

- [ ] **Step 4: Add exact snapshot cloning test**

Verify `clone_state` returns detached independent storage with exact equality before mutation.

- [ ] **Step 5: Run model tests**

Run: `pytest tests/test_model.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add independent stateful sequence learner`

---

### Task 4: Reciprocal pair engine, exact junction snapshots, and causal yoked replay

**Files:**
- Create: `src/coupled_sequence_learners/pair.py`
- Test: `tests/test_pair.py`

**Interfaces:**
- `PairState(state_a, state_b, last_message_a, last_message_b, step)`
- `ReplayTrace(pair_id, episode_id, junction_step, messages_a, messages_b, source_state_digest)`
- `run_prefix(pair, batch) -> PairState`
- `record_unperturbed_continuation(pair, batch, snapshot) -> ReplayTrace`
- `continue_live(pair, batch, snapshot, perturbation) -> Trajectory`
- `continue_yoked(pair, batch, snapshot, perturbation, replay) -> Trajectory`
- `state_digest(PairState) -> str`

- [ ] **Step 1: Write exact-state restoration and replay-provenance tests**

Tests assert live and yoked continuations begin with identical state digests, replay trace pair/episode IDs match the snapshot, the perturbation affects A only at the junction, and mutating one restored state cannot alter another.

- [ ] **Step 2: Write the causal-staleness unit test**

Construct a tiny deterministic mock pair where B's live post-perturbation response changes when A changes. Assert live trajectory differs after the intervention while yoked B messages remain byte-identical to the recorded unperturbed trace.

- [ ] **Step 3: Run pair tests and verify failure**

Run: `pytest tests/test_pair.py -v`

Expected: FAIL because pair engine is missing.

- [ ] **Step 4: Implement synchronous interaction convention**

Freeze v0 convention: at each step both learners emit messages from pre-step states; both then update from local observation plus the other learner's emitted message. Use the same convention in every arm.

- [ ] **Step 5: Implement snapshot/replay functions and provenance guards**

Primary yoked replay must reject any trace whose `pair_id`, `episode_id`, `junction_step`, or source-state digest does not match.

- [ ] **Step 6: Run pair tests**

Run: `pytest tests/test_pair.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: add counterfactual reciprocal replay engine`

---

### Task 5: Metrics and communication-budget accounting

**Files:**
- Create: `src/coupled_sequence_learners/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- `branch_accuracy(logits, labels) -> float`
- `mean_nll(logits, labels) -> float`
- `js_divergence(logits_a, logits_b) -> float`
- `recovery_steps(predictions, labels, junction_index) -> float`
- `message_budget(messages) -> dict[str, float]`
- `effective_rank(messages, eps=1e-8) -> float`

- [ ] **Step 1: Write analytic metric tests**

Use known distributions to pin JS=0 for identical predictions, finite symmetric JS for opposites, exact branch accuracy/NLL on toy logits, and effective rank on rank-1 versus rank-2 message matrices.

- [ ] **Step 2: Add budget-match tests**

Verify equal-width/equal-step message traces report equal scalar count; norm/variance summaries are explicit diagnostics and never silently normalized away.

- [ ] **Step 3: Run metric tests and verify failure**

Run: `pytest tests/test_metrics.py -v`

Expected: FAIL because metrics module is missing.

- [ ] **Step 4: Implement metrics**

Use stable log-softmax for probability metrics. Recovery is first post-junction step after which the correct branch remains top-1 for the rest of the short continuation; return `inf` if never recovered.

- [ ] **Step 5: Run metric tests**

Run: `pytest tests/test_metrics.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add predictive alignment and recovery metrics`

---

### Task 6: Function-preserving coordinate gauge and partner/control attackers

**Files:**
- Modify: `src/coupled_sequence_learners/model.py`
- Modify: `src/coupled_sequence_learners/pair.py`
- Create: `src/coupled_sequence_learners/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- `evaluate_arm(models, batch, arm: str, config, seed) -> dict`
- Supported arms: `live`, `yoked`, `one_way_a_to_b`, `one_way_b_to_a`, `frozen_partner`, `time_shuffled`, `partner_swap`, `centralized_ceiling`, `isolated_a`, `isolated_b`.
- `diagnostic_gauge(state: Tensor, matrix: Tensor) -> Tensor` must not feed back into the model.

- [ ] **Step 1: Write attacker enumeration and budget tests**

Assert every arm returns branch accuracy, NLL, JS, recovery, message budget, and provenance; live/yoked have equal communication scalar counts; time-shuffled preserves the multiset of recorded messages; frozen-partner does not change message width.

- [ ] **Step 2: Write gauge invariance test**

Apply a seeded random orthogonal transform to diagnostic hidden states and assert branch logits, messages, and task outputs are unchanged while raw coordinate correlation changes.

- [ ] **Step 3: Run experiment tests and verify failure**

Run: `pytest tests/test_experiment.py -v`

Expected: FAIL because experiment evaluator is missing.

- [ ] **Step 4: Implement attacker arms**

Keep primary `live` and `yoked` logic in `pair.py`; implement other arms as explicit policy switches, not separate architectures. Partner swap evaluates A_i with B_j after training without further updates for the immediate-cost measurement.

- [ ] **Step 5: Implement centralized ceiling**

Use a small MLP or GRU that receives both complete private histories directly and predicts the four-way branch. It is a solvability ceiling only.

- [ ] **Step 6: Run experiment tests**

Run: `pytest tests/test_experiment.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: add causal attackers and coordinate-gauge diagnostics`

---

### Task 7: Listener-first pretraining conditions

**Files:**
- Create: `src/coupled_sequence_learners/pretraining.py`
- Test: `tests/test_pretraining.py`

**Interfaces:**
- `pretrain_listener(model, condition: str, seed: int, steps: int) -> PretrainReceipt`
- Conditions: `fresh`, `state_dependent`, `state_independent`.
- Receipt includes pretraining accuracy and frozen parameter digest.

- [ ] **Step 1: Write curriculum distinction tests**

State-dependent task must require inbound cue × receiver-state interaction; state-independent task must be solvable from inbound cue alone. Tests verify labels and balanced marginals enforce those properties.

- [ ] **Step 2: Run pretraining tests and verify failure**

Run: `pytest tests/test_pretraining.py -v`

Expected: FAIL because pretraining module is missing.

- [ ] **Step 3: Implement the two small curricula**

Use the same learner architecture and inbound message width as the main experiment. Pretraining changes only listener parameters; the main task data are not used.

- [ ] **Step 4: Verify pretraining learns its intended task**

Tests require >90% held-out accuracy on each pretraining task with a fixed smoke-test seed and confirm fresh condition leaves parameters untouched.

- [ ] **Step 5: Run pretraining tests**

Run: `pytest tests/test_pretraining.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add listener-first curriculum controls`

---

### Task 8: Train/evaluate one seed and run cheap pilot without touching preregistered gates

**Files:**
- Modify: `src/coupled_sequence_learners/experiment.py`
- Create: `src/coupled_sequence_learners/run.py`
- Test: `tests/test_experiment.py`
- Modify: `docs/experiment_log.md`

**Interfaces:**
- `train_pair(seed: int, config: dict, listener_condition: str) -> TrainedPair`
- `run_seed(seed: int, config: dict, mode: str) -> dict`
- CLI: `python -m coupled_sequence_learners.run --mode debug --seed 0`

- [ ] **Step 1: Add trainability smoke tests**

On a tiny config (hidden_dim 8, 200 steps, small batch), centralized ceiling must exceed chance and pair training must reduce training loss. Do not encode preregistered success thresholds in this smoke test.

- [ ] **Step 2: Implement training loop**

Joint loss is sum of A/B branch cross-entropy plus a small identical coefficient on post-junction prediction loss; do not add explicit message-information, synchrony, or communication bonuses in v0.

- [ ] **Step 3: Implement empirical leakage probes**

Fit post-hoc linear probes on shared-only pre-junction state and on each private learner state. Log branch decodability. Shared-only route should remain near chance; private states may decode their own factor.

- [ ] **Step 4: Run debug pilots only**

Run 2 seeds with reduced steps. Record observed bugs, convergence behavior, and any design corrections in `docs/experiment_log.md` under `Exploratory / not preregistered`. Do not call these results.

- [ ] **Step 5: Make only bug-level corrections**

If the world is unsolvable, leaks the answer, or training is numerically broken, fix it before preregistration and document the change. Do not tune pass thresholds from pilot outcomes.

- [ ] **Step 6: Run full test suite**

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: add trainable coupled sequence pilot`

---

### Task 9: Freeze machine-readable preregistration before full runs

**Files:**
- Create: `scripts/freeze_predictions.py`
- Create: `predictions.json`
- Modify: `tests/test_preregistration.py`
- Modify: `docs/experiment_log.md`

**Interfaces:**
- `predictions.json` includes schema_version, git_base_sha, seeds, full config, gate thresholds, primary metric, attacker list, and timestamp/date.
- Script refuses overwrite without explicit `--force` and records content digest.

- [ ] **Step 1: Add preregistration schema tests**

Require frozen seeds `[11, 23, 37, 53, 71, 89, 107, 131]` and these v0 gates:
  - centralized held-out branch accuracy >= 0.90 mean;
  - live pair held-out branch accuracy >= 0.70 mean;
  - primary live-vs-yoked paired branch-accuracy improvement >= 0.10 mean and positive in at least 6/8 seeds;
  - live-vs-yoked message scalar count exactly equal;
  - coordinate-gauge task logits max absolute difference <= 1e-6;
  - held-out recombination live accuracy >= 0.60 mean.
Secondary/non-fatal hypotheses: partner-swap immediate accuracy drop >= 0.05 mean; state-dependent listener pretraining starts more seeds than state-independent and fresh conditions.

- [ ] **Step 2: Run schema tests and verify failure**

Run: `pytest tests/test_preregistration.py -v`

Expected: FAIL because predictions file/script is missing.

- [ ] **Step 3: Implement freeze script and write `predictions.json`**

The file must clearly mark G1/G2/G5/G8 plus ceiling as primary/follow-up criteria according to the spec, and listener-first/partner-swap as secondary.

- [ ] **Step 4: Record preregistration commit/digest in experiment log**

Do this before any full frozen-seed run.

- [ ] **Step 5: Run full tests**

Run: `pytest -q`

Expected: all PASS.

- [ ] **Step 6: Commit**

Commit message: `science: freeze coupled learner v0 predictions`

---

### Task 10: Run frozen seeds, attackers, and durable receipts

**Files:**
- Create: `scripts/run_preregistered.py`
- Create: `scripts/summarize_results.py`
- Create: `results/seeds/*.json`
- Create: `results/attackers/*.json`
- Create: `results/receipt.json`
- Modify: `docs/experiment_log.md`

**Interfaces:**
- Runner reads `predictions.json`; CLI config cannot silently override frozen scientific values.
- Every per-seed receipt stores git SHA, config digest, model seed, data seed, arm metrics, replay provenance, and state-digest equality check.
- Summary computes paired means, standard errors/descriptive intervals, seed wins, and gate booleans without changing thresholds.

- [ ] **Step 1: Write runner immutability tests**

Attempted override of frozen message_dim, seed list, thresholds, or world split must raise an error in preregistered mode.

- [ ] **Step 2: Implement per-seed runner**

For each seed: train ceiling; train fresh reciprocal pair; evaluate isolated, live, yoked, one-way, frozen, shuffled, gauge diagnostics, recombination; then run partner-swap matrix and listener-pretraining conditions as secondary experiments.

- [ ] **Step 3: Run the frozen seed set**

Command: `python scripts/run_preregistered.py`

Expected: eight seed receipts plus attacker receipts. Runtime target: CPU minutes. If runtime explodes, stop and document; do not silently shrink seed count.

- [ ] **Step 4: Summarize without interpretation changes**

Command: `python scripts/summarize_results.py`

Expected: `results/receipt.json` with every preregistered gate marked PASS/FAIL from frozen thresholds.

- [ ] **Step 5: Append experiment log**

Record exact commands, environment versions, commit SHA, elapsed time, all primary outcomes, secondary outcomes, anomalies, and failures.

- [ ] **Step 6: Commit receipts**

Commit message: `results: record coupled learner v0 frozen run`

---

### Task 11: Scientific interpretation and README update

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment_log.md`
- Test: `tests/test_preregistration.py`

**Interfaces:**
- README conclusion must be generated from the actual gate ledger, not from intended narrative.

- [ ] **Step 1: Add README-boundary test**

Test that README contains sections `What passed`, `What failed`, `What this does not show`, and references the frozen receipt path.

- [ ] **Step 2: Update README from receipts**

If primary live-vs-yoked gate passes, state only the synthetic causal-feedback claim allowed by the spec. If it fails, say the interaction-state hypothesis was not supported in v0. Preserve secondary nulls exactly.

- [ ] **Step 3: Add compact table of all arms and secondary conditions**

Include mean branch accuracy/NLL/recovery/JS and seed wins where defined. Do not hide attacks that outperform live.

- [ ] **Step 4: Run full verification**

Run: `pytest -q`

Expected: all PASS.

Run the preregistered summary once more and verify `results/receipt.json` is unchanged except deterministic metadata that was explicitly designed to vary; preferably byte-identical.

- [ ] **Step 5: Commit**

Commit message: `docs: report coupled learner v0 outcome`

---

### Task 12: Final branch verification and integration handoff

**Files:**
- No new required files unless verification finds a documented defect.

**Interfaces:**
- Branch must be reproducible from clean checkout using README commands.

- [ ] **Step 1: Run complete verification**

Commands:

```bash
pytest -q
python scripts/summarize_results.py
```

Expected: tests PASS and summary agrees with committed receipt.

- [ ] **Step 2: Check Git scientific ledger**

Verify commit history has separate design, plan, preregistration, results, and interpretation checkpoints. Confirm no result commit predates `predictions.json` freeze commit.

- [ ] **Step 3: Review diff against approved spec**

Check every primary spec requirement has implementation/receipt coverage and no biological claims were introduced.

- [ ] **Step 4: Prepare integration**

Present branch `work/coupled-v0`, primary outcome ledger, verification evidence, and merge recommendation. Do not merge until final verification has been run on the exact branch tip.
