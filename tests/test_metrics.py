import math
import torch
from coupled_sequence_learners.metrics import branch_accuracy, mean_nll, js_divergence, recovery_steps, message_budget, effective_rank


def test_branch_accuracy_and_nll_are_exact_on_confident_logits():
    logits = torch.tensor([[10., 0.], [0., 10.]])
    labels = torch.tensor([0, 1])
    assert branch_accuracy(logits, labels) == 1.0
    assert mean_nll(logits, labels) < 1e-3


def test_js_is_zero_for_identical_and_symmetric_for_different_predictions():
    a = torch.tensor([[4., 0.], [0., 4.]])
    b = torch.tensor([[0., 4.], [4., 0.]])
    assert js_divergence(a, a) < 1e-9
    assert abs(js_divergence(a, b) - js_divergence(b, a)) < 1e-9
    assert js_divergence(a, b) > 0.5


def test_recovery_steps_requires_stable_correct_prediction():
    logits = torch.tensor([[[0., 2.]], [[2., 0.]], [[0., 2.]], [[0., 3.]]])
    labels = torch.tensor([1])
    assert recovery_steps(logits, labels, 0) == 2.0
    never = torch.tensor([[[2., 0.]], [[0., 2.]], [[2., 0.]]])
    assert math.isinf(recovery_steps(never, labels, 0))


def test_message_budget_and_effective_rank():
    rank1 = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]]])
    rank2 = torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]])
    budget = message_budget(rank1)
    assert budget["scalar_count"] == 6
    assert budget["width"] == 2
    assert budget["steps"] == 3
    assert 0.99 <= effective_rank(rank1) <= 1.01
    assert effective_rank(rank2) > 1.9
