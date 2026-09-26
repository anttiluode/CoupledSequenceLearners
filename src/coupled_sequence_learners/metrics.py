from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn.functional as F


def branch_accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def mean_nll(logits: Tensor, labels: Tensor) -> float:
    return float(F.cross_entropy(logits, labels).item())


def js_divergence(logits_a: Tensor, logits_b: Tensor) -> float:
    log_pa = F.log_softmax(logits_a, dim=-1); log_pb = F.log_softmax(logits_b, dim=-1)
    pa, pb = log_pa.exp(), log_pb.exp(); m = 0.5 * (pa + pb); log_m = torch.log(m.clamp_min(1e-12))
    kl_a = (pa * (log_pa - log_m)).sum(dim=-1); kl_b = (pb * (log_pb - log_m)).sum(dim=-1)
    return float((0.5 * (kl_a + kl_b)).mean().item())


def recovery_steps(predictions: Tensor, labels: Tensor, junction_index: int) -> float:
    if predictions.ndim != 3: raise ValueError("predictions must have shape [time,batch,classes]")
    correct = predictions.argmax(dim=-1).eq(labels.unsqueeze(0)); recoveries = []
    for b in range(correct.shape[1]):
        found = None
        for t in range(max(0, junction_index), correct.shape[0]):
            if bool(correct[t:, b].all()): found = t - junction_index; break
        if found is None: return math.inf
        recoveries.append(float(found))
    return float(sum(recoveries) / len(recoveries)) if recoveries else math.inf


def message_budget(messages: Tensor) -> dict[str, float]:
    if messages.ndim < 2: raise ValueError("messages need at least time and width dimensions")
    width = int(messages.shape[-1]); steps = int(messages.shape[0]); flat = messages.reshape(-1, width).float()
    return {"scalar_count": int(messages.numel()), "width": width, "steps": steps, "mean_norm": float(torch.linalg.vector_norm(flat, dim=-1).mean().item()), "variance": float(flat.var(unbiased=False).item())}


def effective_rank(messages: Tensor, eps: float = 1e-8) -> float:
    x = messages.reshape(-1, messages.shape[-1]).float(); s = torch.linalg.svdvals(x); s = s[s > eps]
    if s.numel() == 0: return 0.0
    p = s / s.sum(); entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    return float(torch.exp(entropy).item())
