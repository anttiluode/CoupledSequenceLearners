from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import CoupledLearner


@dataclass(frozen=True)
class PretrainingExamples:
    state_factor: Tensor
    message_factor: Tensor
    labels: Tensor


@dataclass(frozen=True)
class PretrainReceipt:
    condition: str
    heldout_accuracy: float
    parameter_digest: str
    steps: int
    seed: int


def _parameter_digest(model: CoupledLearner) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_pretraining_examples(condition: str, repeats: int = 32) -> PretrainingExamples:
    if condition not in {"state_dependent", "state_independent"}:
        raise ValueError("condition must be state_dependent or state_independent")
    states, messages, labels = [], [], []
    for _ in range(repeats):
        for state_factor, message_factor in itertools.product((-1.0, 1.0), repeat=2):
            states.append(state_factor)
            messages.append(message_factor)
            if condition == "state_dependent":
                labels.append(int(state_factor * message_factor > 0))
            else:
                labels.append(int(message_factor > 0))
    return PretrainingExamples(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(messages, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _forward_examples(model: CoupledLearner, examples: PretrainingExamples, *, noise_std: float = 0.02, generator: torch.Generator | None = None) -> Tensor:
    n = examples.labels.numel()
    state = model.initial_state(n)
    obs_state = torch.zeros(n, model.config.obs_dim)
    obs_state[:, 0] = examples.state_factor
    zero_msg = torch.zeros(n, model.config.message_dim)
    first = model.forward_step(obs_state, zero_msg, state)

    obs_query = torch.zeros(n, model.config.obs_dim)
    inbound = torch.zeros(n, model.config.message_dim)
    inbound[:, 0] = examples.message_factor
    if noise_std:
        noise = torch.randn(inbound.shape, generator=generator, device=inbound.device, dtype=inbound.dtype)
        inbound = inbound + noise_std * noise
    second = model.forward_step(obs_query, inbound, first.state)
    return second.branch_logits[:, :2]


def pretrain_listener(model: CoupledLearner, condition: str, seed: int, steps: int) -> PretrainReceipt:
    if condition == "fresh":
        return PretrainReceipt(condition, 0.5, _parameter_digest(model), 0, seed)
    if condition not in {"state_dependent", "state_independent"}:
        raise ValueError("unknown pretraining condition")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    train = make_pretraining_examples(condition, repeats=16)
    held = make_pretraining_examples(condition, repeats=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = _forward_examples(model, train, noise_std=0.03, generator=generator)
        loss = F.cross_entropy(logits, train.labels)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = _forward_examples(model, held, noise_std=0.0)
        acc = float((logits.argmax(dim=-1) == held.labels).float().mean().item())
    return PretrainReceipt(condition, acc, _parameter_digest(model), steps, seed)
