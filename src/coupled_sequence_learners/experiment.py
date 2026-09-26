from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metrics import branch_accuracy, mean_nll, js_divergence, recovery_steps, message_budget
from .model import CoupledLearner, LearnerConfig
from .pair import (
    CoupledPair, PairState, Trajectory, run_prefix,
    record_unperturbed_continuation, continue_live, continue_yoked, state_digest,
)
from .pretraining import pretrain_listener, PretrainReceipt
from .world import EpisodeBatch, generate_batch, with_counterfactual_queries

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


@dataclass
class TrainedPair:
    pair: CoupledPair
    initial_loss: float
    final_loss: float
    listener_condition: str
    pretrain_receipt: PretrainReceipt


def diagnostic_gauge(state: Tensor, matrix: Tensor) -> Tensor:
    if matrix.shape != (state.shape[-1], state.shape[-1]):
        raise ValueError("gauge matrix shape mismatch")
    return state @ matrix


def _obs(batch: EpisodeBatch, which: str, t: int) -> Tensor:
    arr = batch.obs_a if which == "a" else batch.obs_b
    return torch.as_tensor(arr[:, t], dtype=torch.float32)


def _slice_batch(batch: EpisodeBatch, idx: np.ndarray) -> EpisodeBatch:
    return EpisodeBatch(
        obs_a=batch.obs_a[idx], obs_b=batch.obs_b[idx], branch=batch.branch[idx],
        junction_step=batch.junction_step, episode_id=batch.episode_id[idx],
        factor_a=batch.factor_a[idx], factor_b=batch.factor_b[idx],
        motif=batch.motif[idx], variant=batch.variant[idx],
        future_tokens=batch.future_tokens[idx], b_bits=batch.b_bits[idx],
    )


def _default_perturbation(state: Tensor) -> Tensor:
    out = state.clone()
    if out.shape[-1]:
        out[:, 0] = -out[:, 0]
    return out


def _training_rollout(pair: CoupledPair, batch: EpisodeBatch, noise_std: float, generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
    n = len(batch.branch)
    sa = pair.model_a.initial_state(n)
    sb = pair.model_b.initial_state(n)
    za = torch.zeros(n, pair.model_a.config.message_dim)
    zb = torch.zeros(n, pair.model_b.config.message_dim)

    # Private history and shared-prefix phase: no communication.
    for t in range(batch.junction_step):
        oa = pair.model_a.forward_step(_obs(batch, "a", t), zb, sa)
        ob = pair.model_b.forward_step(_obs(batch, "b", t), za, sb)
        sa, sb = oa.state, ob.state

    logits_a, logits_b = [], []
    for t in range(batch.junction_step, batch.obs_a.shape[1]):
        ma = pair.model_a.emit_from_state(sa)
        mb = pair.model_b.emit_from_state(sb)
        if noise_std > 0:
            na = torch.randn(ma.shape, generator=generator, dtype=ma.dtype, device=ma.device) * noise_std
            nb = torch.randn(mb.shape, generator=generator, dtype=mb.dtype, device=mb.device) * noise_std
            ma_in, mb_in = ma + na, mb + nb
        else:
            ma_in, mb_in = ma, mb
        oa = pair.model_a.forward_step(_obs(batch, "a", t), mb_in, sa)
        ob = pair.model_b.forward_step(_obs(batch, "b", t), ma_in, sb)
        sa, sb = oa.state, ob.state
        logits_a.append(oa.branch_logits)
        logits_b.append(ob.branch_logits)
    return torch.stack(logits_a), torch.stack(logits_b)


def _pair_loss(pair: CoupledPair, batch: EpisodeBatch, noise_std: float = 0.0, generator: torch.Generator | None = None) -> Tensor:
    la, lb = _training_rollout(pair, batch, noise_std, generator)
    labels = torch.as_tensor(batch.branch, dtype=torch.long)
    final = F.cross_entropy(la[-1], labels) + F.cross_entropy(lb[-1], labels)
    temporal = torch.stack([
        F.cross_entropy(la[t], labels) + F.cross_entropy(lb[t], labels)
        for t in range(la.shape[0])
    ]).mean()
    return final + 0.20 * temporal


def train_pair(seed: int, config: dict, listener_condition: str = "fresh") -> TrainedPair:
    torch.manual_seed(seed)
    lc = LearnerConfig(
        hidden_dim=int(config["hidden_dim"]), obs_dim=int(config["observation_dim"]),
        message_dim=int(config["message_dim"]), branch_count=int(config["branch_count"]),
    )
    a = CoupledLearner(lc)
    b = CoupledLearner(lc)
    pre_steps = int(config.get("pretraining_steps", 250))
    receipt = pretrain_listener(b, listener_condition, seed=seed + 5000, steps=pre_steps)
    pair = CoupledPair(a, b, f"pair-{seed}-{listener_condition}")

    pool_size = max(1024, int(config["batch_size"]) * 8)
    pool = generate_batch(seed + 100, pool_size, "train", config)
    eval_batch = generate_batch(seed + 101, min(512, pool_size), "train", config)
    with torch.no_grad():
        initial = float(_pair_loss(pair, eval_batch, noise_std=0.0).item())

    opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=float(config.get("learning_rate", 0.01)))
    rng = np.random.default_rng(seed + 200)
    noise_gen = torch.Generator().manual_seed(seed + 300)
    pair.model_a.train(); pair.model_b.train()
    steps = int(config["training_steps"])
    batch_size = int(config["batch_size"])
    for _ in range(steps):
        idx = rng.integers(0, pool_size, size=batch_size)
        batch = _slice_batch(pool, idx)
        opt.zero_grad(set_to_none=True)
        loss = _pair_loss(pair, batch, float(config.get("message_noise_std", 0.0)), noise_gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(a.parameters()) + list(b.parameters()), 5.0)
        opt.step()
    pair.model_a.eval(); pair.model_b.eval()
    with torch.no_grad():
        final = float(_pair_loss(pair, eval_batch, noise_std=0.0).item())
    return TrainedPair(pair, initial, final, listener_condition, receipt)


def _custom_rollout(
    pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, mode: str,
    replay_b: Tensor | None = None, seed: int = 0, perturbation=None,
) -> Trajectory:
    st = snapshot.clone()
    perturbation = perturbation or _default_perturbation
    st.state_a = perturbation(st.state_a.detach().clone()).detach().clone()
    start_digest = state_digest(snapshot)
    ma_hist, mb_hist, la_hist, lb_hist = [], [], [], []
    total = batch.obs_a.shape[1]
    generator = torch.Generator().manual_seed(seed)
    permutation = None
    if mode == "time_shuffled":
        if replay_b is None:
            raise ValueError("time_shuffled needs replay")
        permutation = torch.randperm(replay_b.shape[0], generator=generator)
    with torch.no_grad():
        for k, t in enumerate(range(snapshot.step, total)):
            ma_live = pair.model_a.emit_from_state(st.state_a)
            mb_live = pair.model_b.emit_from_state(st.state_b)
            if mode == "one_way_a_to_b":
                inbound_a = torch.zeros_like(mb_live); inbound_b = ma_live
                ma, mb = ma_live, torch.zeros_like(mb_live)
            elif mode == "one_way_b_to_a":
                inbound_a = mb_live; inbound_b = torch.zeros_like(ma_live)
                ma, mb = torch.zeros_like(ma_live), mb_live
            elif mode == "frozen_partner":
                mb = snapshot.last_message_b
                inbound_a = mb; inbound_b = ma_live; ma = ma_live
            elif mode == "time_shuffled":
                mb = replay_b[permutation[k]]
                inbound_a = mb; inbound_b = ma_live; ma = ma_live
            elif mode == "isolated_a":
                inbound_a = torch.zeros_like(mb_live); inbound_b = torch.zeros_like(ma_live)
                ma, mb = ma_live, torch.zeros_like(mb_live)
            elif mode == "isolated_b":
                inbound_a = torch.zeros_like(mb_live); inbound_b = torch.zeros_like(ma_live)
                ma, mb = torch.zeros_like(ma_live), mb_live
            else:
                raise ValueError(f"unsupported custom mode {mode}")
            oa = pair.model_a.forward_step(_obs(batch, "a", t), inbound_a, st.state_a)
            ob = pair.model_b.forward_step(_obs(batch, "b", t), inbound_b, st.state_b)
            ma_hist.append(ma.detach().clone()); mb_hist.append(mb.detach().clone())
            la_hist.append(oa.branch_logits.detach().clone()); lb_hist.append(ob.branch_logits.detach().clone())
            st.state_a, st.state_b = oa.state, ob.state
            st.last_message_a, st.last_message_b = ma, mb
            st.step = t + 1
    return Trajectory(torch.stack(ma_hist), torch.stack(mb_hist), torch.stack(la_hist), torch.stack(lb_hist), st, start_digest)


def _trajectory_metrics(
    traj: Trajectory, batch: EpisodeBatch, arm: str, seed: int, pair_id: str,
    labels: np.ndarray | None = None,
) -> dict:
    y = torch.as_tensor(batch.branch if labels is None else labels, dtype=torch.long)
    joint_logits = 0.5 * (traj.logits_a + traj.logits_b)
    final_logits = joint_logits[-1]
    pred_a = traj.logits_a[-1].argmax(dim=-1)
    pred_b = traj.logits_b[-1].argmax(dim=-1)
    combined_messages = torch.cat([traj.messages_a, traj.messages_b], dim=-1)
    b_rows = traj.messages_b.detach().cpu().reshape(traj.messages_b.shape[0], -1).tolist()
    return {
        "branch_accuracy": branch_accuracy(final_logits, y),
        "branch_accuracy_a": branch_accuracy(traj.logits_a[-1], y),
        "branch_accuracy_b": branch_accuracy(traj.logits_b[-1], y),
        "both_correct": float(((pred_a == y) & (pred_b == y)).float().mean().item()),
        "nll": mean_nll(final_logits, y),
        "js": js_divergence(traj.logits_a[-1], traj.logits_b[-1]),
        "recovery": recovery_steps(joint_logits, y, 0),
        "message_budget": message_budget(combined_messages),
        "message_multiset": [tuple(float(v) for v in row) for row in b_rows],
        "provenance": {"arm": arm, "seed": int(seed), "pair_id": pair_id, "start_digest": traj.start_digest},
    }


def evaluate_unperturbed(pair: CoupledPair, batch: EpisodeBatch, config: dict, seed: int) -> dict:
    snapshot = run_prefix(pair, batch)
    traj = continue_live(pair, batch, snapshot, lambda x: x)
    return _trajectory_metrics(traj, batch, "unperturbed", seed, pair.pair_id)


def evaluate_counterfactual(pair: CoupledPair, batch: EpisodeBatch, config: dict, seed: int) -> dict:
    new_queries = np.bitwise_xor(batch.factor_a, 3)
    counter = with_counterfactual_queries(batch, config, new_queries)
    original_snapshot = run_prefix(pair, batch)
    counter_snapshot = run_prefix(pair, counter)
    # Because the prefix channel is closed and B's private history is unchanged,
    # only A's resident state should differ across these two snapshots.
    b_state_equal = bool(torch.equal(original_snapshot.state_b, counter_snapshot.state_b))
    replay = record_unperturbed_continuation(pair, batch, original_snapshot)
    perturb = lambda _: counter_snapshot.state_a.detach().clone()
    live = continue_live(pair, batch, original_snapshot, perturb)
    yoked = continue_yoked(pair, batch, original_snapshot, perturb, replay)
    out = {
        "live": _trajectory_metrics(live, batch, "live", seed, pair.pair_id, labels=counter.branch),
        "yoked": _trajectory_metrics(yoked, batch, "yoked", seed, pair.pair_id, labels=counter.branch),
        "counterfactual_changed_fraction": float(np.mean(new_queries != batch.factor_a)),
        "target_branch_changed_fraction": float(np.mean(counter.branch != batch.branch)),
        "b_prefix_state_equal": b_state_equal,
    }
    for arm in ("one_way_a_to_b", "one_way_b_to_a", "frozen_partner", "time_shuffled", "isolated_a", "isolated_b"):
        traj = _custom_rollout(pair, batch, original_snapshot, arm, replay.messages_b, seed, perturbation=perturb)
        out[arm] = _trajectory_metrics(traj, batch, arm, seed, pair.pair_id, labels=counter.branch)
    return out


def _oracle_ceiling(batch: EpisodeBatch, pair_id: str, seed: int) -> dict:
    y = torch.as_tensor(batch.branch, dtype=torch.long)
    logits = torch.full((len(y), 4), -8.0)
    logits[torch.arange(len(y)), y] = 8.0
    return {
        "branch_accuracy": 1.0, "nll": mean_nll(logits, y), "js": 0.0, "recovery": 0.0,
        "message_budget": {"scalar_count": 0, "width": 0, "steps": 0, "mean_norm": 0.0, "variance": 0.0},
        "message_multiset": [],
        "provenance": {"arm": "centralized_ceiling", "seed": int(seed), "pair_id": pair_id, "note": "task6 oracle smoke only"},
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


class _Ceiling(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 4))
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _ceiling_features(batch: EpisodeBatch, config: dict) -> Tensor:
    p = int(config["private_prefix_len"])
    a = torch.as_tensor(batch.obs_a[:, :p].reshape(len(batch.branch), -1), dtype=torch.float32)
    b = torch.as_tensor(batch.obs_b[:, :p].reshape(len(batch.branch), -1), dtype=torch.float32)
    return torch.cat([a, b], dim=-1)


def train_centralized_ceiling(seed: int, config: dict, steps: int | None = None) -> dict:
    torch.manual_seed(seed + 7000)
    train = generate_batch(seed + 7001, 1024, "train", config)
    held = generate_batch(seed + 7002, 512, "heldout", config)
    x = _ceiling_features(train, config); y = torch.as_tensor(train.branch, dtype=torch.long)
    model = _Ceiling(x.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    steps = int(steps if steps is not None else min(500, int(config["training_steps"])))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        hx = _ceiling_features(held, config); hy = torch.as_tensor(held.branch, dtype=torch.long)
        logits = model(hx)
        acc = branch_accuracy(logits, hy)
        nll = mean_nll(logits, hy)
    return {"heldout_accuracy": acc, "heldout_nll": nll, "model": model}


def _linear_probe_accuracy(x: Tensor, y: Tensor, classes: int, seed: int) -> float:
    torch.manual_seed(seed)
    n = x.shape[0]
    split = n // 2
    model = nn.Linear(x.shape[1], classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[:split]), y[:split])
        loss.backward(); opt.step()
    with torch.no_grad():
        return branch_accuracy(model(x[split:]), y[split:])


def leakage_diagnostics(pair: CoupledPair, batch: EpisodeBatch, config: dict, seed: int) -> dict:
    snap = run_prefix(pair, batch)
    start = int(config["shared_prefix_start"]); stop = int(config["junction_step"])
    shared = torch.as_tensor(batch.obs_a[:, start:stop].reshape(len(batch.branch), -1), dtype=torch.float32)
    branch = torch.as_tensor(batch.branch, dtype=torch.long)
    return {
        "shared_branch_probe": _linear_probe_accuracy(shared, branch, 4, seed + 1),
        "a_branch_probe": _linear_probe_accuracy(snap.state_a.detach(), branch, 4, seed + 2),
        "b_branch_probe": _linear_probe_accuracy(snap.state_b.detach(), branch, 4, seed + 3),
    }


def run_seed(seed: int, config: dict, mode: str = "debug", listener_condition: str = "fresh") -> dict:
    trained = train_pair(seed, config, listener_condition)
    held = generate_batch(seed + 9000, int(config.get("eval_batch_size", 512)), "heldout", config)
    return {
        "seed": seed, "mode": mode, "initial_loss": trained.initial_loss, "final_loss": trained.final_loss,
        "unperturbed": evaluate_unperturbed(trained.pair, held, config, seed),
        "counterfactual": evaluate_counterfactual(trained.pair, held, config, seed),
        "leakage": leakage_diagnostics(trained.pair, held, config, seed),
    }
