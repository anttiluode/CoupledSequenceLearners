from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable
import torch
from torch import Tensor

from .model import CoupledLearner
from .world import EpisodeBatch


@dataclass
class CoupledPair:
    model_a: CoupledLearner
    model_b: CoupledLearner
    pair_id: str


@dataclass
class PairState:
    state_a: Tensor
    state_b: Tensor
    last_message_a: Tensor
    last_message_b: Tensor
    step: int
    pair_id: str
    episode_ids: tuple[int, ...]

    def clone(self) -> "PairState":
        return PairState(
            self.state_a.detach().clone(), self.state_b.detach().clone(),
            self.last_message_a.detach().clone(), self.last_message_b.detach().clone(),
            self.step, self.pair_id, tuple(self.episode_ids),
        )


@dataclass(frozen=True)
class ReplayTrace:
    pair_id: str
    episode_ids: tuple[int, ...]
    junction_step: int
    messages_a: Tensor
    messages_b: Tensor
    source_state_digest: str


@dataclass
class Trajectory:
    messages_a: Tensor
    messages_b: Tensor
    logits_a: Tensor
    logits_b: Tensor
    final_state: PairState
    start_digest: str


def _digest_tensor(h, x: Tensor) -> None:
    arr = x.detach().cpu().contiguous().numpy()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())


def state_digest(state: PairState) -> str:
    h = hashlib.sha256()
    _digest_tensor(h, state.state_a); _digest_tensor(h, state.state_b)
    _digest_tensor(h, state.last_message_a); _digest_tensor(h, state.last_message_b)
    h.update(str(state.step).encode()); h.update(state.pair_id.encode()); h.update(repr(state.episode_ids).encode())
    return h.hexdigest()


def _obs(batch: EpisodeBatch, which: str, t: int) -> Tensor:
    arr = batch.obs_a if which == "a" else batch.obs_b
    return torch.as_tensor(arr[:, t], dtype=torch.float32)


def run_prefix(pair: CoupledPair, batch: EpisodeBatch) -> PairState:
    n = len(batch.branch)
    state_a = pair.model_a.initial_state(n); state_b = pair.model_b.initial_state(n)
    msg_a = torch.zeros(n, pair.model_a.config.message_dim); msg_b = torch.zeros(n, pair.model_b.config.message_dim)
    with torch.no_grad():
        for t in range(batch.junction_step):
            msg_a = pair.model_a.emit_from_state(state_a); msg_b = pair.model_b.emit_from_state(state_b)
            out_a = pair.model_a.forward_step(_obs(batch, "a", t), msg_b, state_a)
            out_b = pair.model_b.forward_step(_obs(batch, "b", t), msg_a, state_b)
            state_a, state_b = out_a.state, out_b.state
    return PairState(state_a.detach().clone(), state_b.detach().clone(), msg_a.detach().clone(), msg_b.detach().clone(), batch.junction_step, pair.pair_id, tuple(int(x) for x in batch.episode_id))


def _run_continuation(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, perturbation: Callable[[Tensor], Tensor] | None = None, forced_b: Tensor | None = None) -> Trajectory:
    start = state_digest(snapshot)
    st = snapshot.clone()
    if perturbation is not None:
        st.state_a = perturbation(st.state_a.detach().clone()).detach().clone()
    msgs_a, msgs_b, logits_a, logits_b = [], [], [], []
    total = batch.obs_a.shape[1]
    with torch.no_grad():
        for k, t in enumerate(range(snapshot.step, total)):
            ma = pair.model_a.emit_from_state(st.state_a)
            mb_live = pair.model_b.emit_from_state(st.state_b)
            mb = forced_b[k].detach().clone() if forced_b is not None else mb_live
            out_a = pair.model_a.forward_step(_obs(batch, "a", t), mb, st.state_a)
            out_b = pair.model_b.forward_step(_obs(batch, "b", t), ma, st.state_b)
            msgs_a.append(ma.detach().clone()); msgs_b.append(mb.detach().clone())
            logits_a.append(out_a.branch_logits.detach().clone()); logits_b.append(out_b.branch_logits.detach().clone())
            st.state_a, st.state_b = out_a.state, out_b.state
            st.last_message_a, st.last_message_b = ma, mb; st.step = t + 1
    return Trajectory(torch.stack(msgs_a), torch.stack(msgs_b), torch.stack(logits_a), torch.stack(logits_b), st, start)


def record_unperturbed_continuation(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState) -> ReplayTrace:
    traj = _run_continuation(pair, batch, snapshot)
    return ReplayTrace(pair.pair_id, tuple(int(x) for x in batch.episode_id), snapshot.step, traj.messages_a.detach().clone(), traj.messages_b.detach().clone(), state_digest(snapshot))


def _validate_replay(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, replay: ReplayTrace) -> None:
    if replay.pair_id != pair.pair_id: raise ValueError("replay pair provenance mismatch")
    if replay.episode_ids != tuple(int(x) for x in batch.episode_id): raise ValueError("replay episode provenance mismatch")
    if replay.junction_step != snapshot.step: raise ValueError("replay junction mismatch")
    if replay.source_state_digest != state_digest(snapshot): raise ValueError("replay source-state digest mismatch")


def continue_live(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, perturbation) -> Trajectory:
    return _run_continuation(pair, batch, snapshot, perturbation=perturbation)


def continue_yoked(pair: CoupledPair, batch: EpisodeBatch, snapshot: PairState, perturbation, replay: ReplayTrace) -> Trajectory:
    _validate_replay(pair, batch, snapshot, replay)
    return _run_continuation(pair, batch, snapshot, perturbation=perturbation, forced_b=replay.messages_b)
