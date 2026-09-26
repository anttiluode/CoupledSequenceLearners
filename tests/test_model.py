import torch
from coupled_sequence_learners.model import CoupledLearner, LearnerConfig


def make_model(seed=0):
    torch.manual_seed(seed)
    return CoupledLearner(LearnerConfig(hidden_dim=16, obs_dim=8, message_dim=2, branch_count=4))


def test_forward_step_shapes_and_determinism():
    model = make_model(1).eval()
    state = model.initial_state(5)
    obs = torch.zeros(5, 8)
    inbound = torch.ones(5, 2)
    a = model.forward_step(obs, inbound, state)
    b = model.forward_step(obs, inbound, state)
    assert a.state.shape == (5, 16)
    assert a.message.shape == (5, 2)
    assert a.branch_logits.shape == (5, 4)
    assert torch.equal(a.state, b.state)
    assert torch.equal(a.message, b.message)


def test_separate_models_do_not_share_parameters():
    a = make_model(2)
    b = make_model(3)
    assert all(pa.data_ptr() != pb.data_ptr() for pa, pb in zip(a.parameters(), b.parameters()))


def test_receiver_effect_depends_on_current_state():
    model = make_model(4).eval()
    obs = torch.zeros(2, 8)
    inbound = torch.tensor([[0.7, -0.4], [0.7, -0.4]])
    state = torch.stack([torch.zeros(16), torch.ones(16) * 0.5])
    out = model.forward_step(obs, inbound, state)
    delta = out.inbound_modulation
    assert delta.shape == (2, 16)
    assert not torch.allclose(delta[0], delta[1])


def test_clone_state_is_equal_but_independent():
    model = make_model(5)
    state = torch.randn(3, 16)
    clone = model.clone_state(state)
    assert torch.equal(state, clone)
    assert state.data_ptr() != clone.data_ptr()
    clone[0, 0] += 1
    assert not torch.equal(state, clone)
