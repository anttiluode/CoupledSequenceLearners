from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LearnerConfig:
    hidden_dim: int
    obs_dim: int
    message_dim: int
    branch_count: int


@dataclass
class StepOutput:
    state: Tensor
    message: Tensor
    branch_logits: Tensor
    inbound_modulation: Tensor


class CoupledLearner(nn.Module):
    def __init__(self, config: LearnerConfig):
        super().__init__()
        self.config = config
        self.inbound = nn.Sequential(
            nn.Linear(config.hidden_dim + config.message_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.core = nn.GRUCell(config.obs_dim + config.hidden_dim, config.hidden_dim)
        self.message_head = nn.Linear(config.hidden_dim, config.message_dim)
        self.branch_head = nn.Linear(config.hidden_dim, config.branch_count)

    def initial_state(self, batch_size: int, device=None) -> Tensor:
        return torch.zeros(batch_size, self.config.hidden_dim, device=device)

    def clone_state(self, state: Tensor) -> Tensor:
        return state.detach().clone()

    def emit_from_state(self, state: Tensor) -> Tensor:
        return torch.tanh(self.message_head(state))

    def logits_from_state(self, state: Tensor) -> Tensor:
        return self.branch_head(state)

    def forward_step(self, obs: Tensor, inbound: Tensor, state: Tensor) -> StepOutput:
        modulation = self.inbound(torch.cat([state, inbound], dim=-1))
        next_state = self.core(torch.cat([obs, modulation], dim=-1), state)
        message = self.emit_from_state(next_state)
        logits = self.logits_from_state(next_state)
        return StepOutput(next_state, message, logits, modulation)
