from __future__ import annotations

import torch
from torch import Tensor

from .metrics import branch_accuracy, mean_nll, js_divergence, recovery_steps, message_budget
from .pair import CoupledPair, PairState, Trajectory, run_prefix, record_unperturbed_continuation, continue_live, continue_yoked
from .world import EpisodeBatch

SUPPORTED_ARMS = (
    "live",
    "yoked",
    "one_way_a_to_b",
    "one_way_b_to_a",
    "frozen_partner",
    "time_shuffled",
    "partner_swap",
    "centralized_ceiling",
    "isolated_a",
    "isolated_b",
)


def diagnostic_gauge(state: Tensor, matrix: Tensor) -> Tensor:
    if matrix.shape != (state.shape[-1], state.shape[-1]):
        raise ValueError("gauge matrix shape mismatch")
    return state @ matrix


def _obs(batch: EpisodeBatch, which: str, t: int) -> Tensor:
    arr = batch.obs_a if which == "a" else batch.obs_b
    return torch.as_tensor(arr[:, t], dtype=torch.float32)


def _default_perturbation(state: Tensor) -> Tensor:
    # Task-6 smoke perturbation only. Task 8 replaces this with a controlled
    # counterfactual-state transplant for the scientific assay.
    out = state.clone()
    if out.shape[-1]:
        out[:, 0] = -out[:, 0]
    return out


def _custom_rollout(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, mode: str, replay_b: Tensor | None = None, seed: int = 0) -> Trajectory:
    from .pair import state_digest

    st = snapshot.clone()
    st.state_a = _default_perturbation(st.state_a)
    start_digest = state_digest(snapshot)
    ma_hist, mb_hist, la_hist, lb_hist = [], [], [], []
    total = batch.obs_a.shape[1]
    generator = torch.Generator().manual_seed(seed)
    permutation = None
    if mode == "time_shuffled":
        if replay_b is None:
            raise ValueError("time_shuffled needs replay")
        permutation = torch.randperm(replay_b.shape[0], generator=generator)
    for k, t in enumerate(range(snapshot.step, total)):
        ma_live = pair.model_a.emit_from_state(st.state_a)
        mb_live = pair.model_b.emit_from_state(st.state_b)
        if mode == "one_way_a_to_b":
            inbound_a = torch.zeros_like(mb_live)
            inbound_b = ma_live
            ma, mb = ma_live, torch.zeros_like(mb_live)
        elif mode == "one_way_b_to_a":
            inbound_a = mb_live
            inbound_b = torch.zeros_like(ma_live)
            ma, mb = torch.zeros_like(ma_live), mb_live
        elif mode == "frozen_partner":
            mb = snapshot.last_message_b
            inbound_a = mb
            inbound_b = ma_live
            ma = ma_live
        elif mode == "time_shuffled":
            mb = replay_b[permutation[k]]
            inbound_a = mb
            inbound_b = ma_live
            ma = ma_live
        elif mode == "isolated_a":
            inbound_a = torch.zeros_like(mb_live)
            inbound_b = torch.zeros_like(ma_live)
            ma, mb = ma_live, torch.zeros_like(mb_live)
        elif mode == "isolated_b":
            inbound_a = torch.zeros_like(mb_live)
            inbound_b = torch.zeros_like(ma_live)
            ma, mb = torch.zeros_like(ma_live), mb_live
        else:
            raise ValueError(f"unsupported custom mode {mode}")
        out_a = pair.model_a.forward_step(_obs(batch, "a", t), inbound_a, st.state_a)
        out_b = pair.model_b.forward_step(_obs(batch, "b", t), inbound_b, st.state_b)
        ma_hist.append(ma.detach().clone())
        mb_hist.append(mb.detach().clone())
        la_hist.append(out_a.branch_logits.detach().clone())
        lb_hist.append(out_b.branch_logits.detach().clone())
        st.state_a, st.state_b = out_a.state, out_b.state
        st.last_message_a, st.last_message_b = ma, mb
        st.step = t + 1
    return Trajectory(torch.stack(ma_hist), torch.stack(mb_hist), torch.stack(la_hist), torch.stack(lb_hist), st, start_digest)


def _trajectory_metrics(traj: Trajectory, batch: EpisodeBatch, arm: str, seed: int, pair_id: str) -> dict:
    labels = torch.as_tensor(batch.branch, dtype=torch.long)
    joint_logits = 0.5 * (traj.logits_a + traj.logits_b)
    final_logits = joint_logits[-1]
    combined_messages = torch.cat([traj.messages_a, traj.messages_b], dim=-1)
    b_rows = traj.messages_b.detach().cpu().reshape(traj.messages_b.shape[0], -1).tolist()
    return {
        "branch_accuracy": branch_accuracy(final_logits, labels),
        "nll": mean_nll(final_logits, labels),
        "js": js_divergence(traj.logits_a[-1], traj.logits_b[-1]),
        "recovery": recovery_steps(joint_logits, labels, 0),
        "message_budget": message_budget(combined_messages),
        "message_multiset": [tuple(float(v) for v in row) for row in b_rows],
        "provenance": {"arm": arm, "seed": int(seed), "pair_id": pair_id, "start_digest": traj.start_digest},
    }


def _oracle_ceiling(batch: EpisodeBatch, pair_id: str, seed: int) -> dict:
    labels = torch.as_tensor(batch.branch, dtype=torch.long)
    logits = torch.full((len(labels), 4), -8.0)
    logits[torch.arange(len(labels)), labels] = 8.0
    return {
        "branch_accuracy": 1.0,
        "nll": mean_nll(logits, labels),
        "js": 0.0,
        "recovery": 0.0,
        "message_budget": {"scalar_count": 0, "width": 0, "steps": 0, "mean_norm": 0.0, "variance": 0.0},
        "message_multiset": [],
        "provenance": {"arm": "centralized_ceiling", "seed": int(seed), "pair_id": pair_id, "note": "task6 oracle smoke; replaced by trained ceiling before frozen run"},
    }


def evaluate_arm(pair: CoupledPair, batch: EpisodeBatch, arm: str, config: dict, seed: int) -> dict:
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"unknown arm {arm}")
    if arm == "partner_swap":
        raise ValueError("partner_swap requires two trained pairs and is evaluated separately")
    if arm == "centralized_ceiling":
        return _oracle_ceiling(batch, pair.pair_id, seed)
    snapshot = run_prefix(pair, batch)
    replay = record_unperturbed_continuation(pair, batch, snapshot)
    if arm == "live":
        traj = continue_live(pair, batch, snapshot, _default_perturbation)
    elif arm == "yoked":
        traj = continue_yoked(pair, batch, snapshot, _default_perturbation, replay)
    else:
        traj = _custom_rollout(pair, batch, snapshot, arm, replay.messages_b, seed)
    return _trajectory_metrics(traj, batch, arm, seed, pair.pair_id)
