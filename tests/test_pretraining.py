import torch

from coupled_sequence_learners.model import CoupledLearner, LearnerConfig
from coupled_sequence_learners.pretraining import make_pretraining_examples, pretrain_listener


def make_model():
    torch.manual_seed(0)
    return CoupledLearner(LearnerConfig(hidden_dim=8, obs_dim=8, message_dim=2, branch_count=4))


def test_curricula_have_distinct_dependency_structure():
    dep = make_pretraining_examples("state_dependent", repeats=16)
    indep = make_pretraining_examples("state_independent", repeats=16)
    for value in (-1.0, 1.0):
        labels_by_state = dep.labels[dep.state_factor == value]
        labels_by_msg = dep.labels[dep.message_factor == value]
        assert labels_by_state.float().mean().item() == 0.5
        assert labels_by_msg.float().mean().item() == 0.5
    assert torch.equal(indep.labels, (indep.message_factor > 0).long())


def test_pretraining_learns_intended_task_and_fresh_does_not_touch_weights():
    fresh = make_model()
    before = {k: v.detach().clone() for k, v in fresh.state_dict().items()}
    fresh_receipt = pretrain_listener(fresh, "fresh", seed=5, steps=100)
    assert all(torch.equal(before[k], v) for k, v in fresh.state_dict().items())
    assert fresh_receipt.condition == "fresh"

    dep = make_model()
    dep_receipt = pretrain_listener(dep, "state_dependent", seed=7, steps=300)
    assert dep_receipt.heldout_accuracy > 0.90

    indep = make_model()
    indep_receipt = pretrain_listener(indep, "state_independent", seed=7, steps=300)
    assert indep_receipt.heldout_accuracy > 0.90
    assert dep_receipt.parameter_digest != fresh_receipt.parameter_digest
