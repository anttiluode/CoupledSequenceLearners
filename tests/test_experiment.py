import torch
from coupled_sequence_learners import default_config
from coupled_sequence_learners.model import CoupledLearner, LearnerConfig
from coupled_sequence_learners.pair import CoupledPair
from coupled_sequence_learners.world import generate_batch
from coupled_sequence_learners.experiment import evaluate_arm, diagnostic_gauge, SUPPORTED_ARMS


def make_pair():
    torch.manual_seed(2)
    cfg = LearnerConfig(8, 8, 2, 4)
    return CoupledPair(CoupledLearner(cfg), CoupledLearner(cfg), "eval-pair")


def test_every_attacker_returns_common_metrics_and_live_yoked_budget_matches():
    cfg = default_config(); cfg["hidden_dim"] = 8
    batch = generate_batch(20, 128, "heldout", cfg)
    pair = make_pair()
    results = {arm: evaluate_arm(pair, batch, arm, cfg, seed=1) for arm in SUPPORTED_ARMS if arm != "partner_swap"}
    for arm, out in results.items():
        assert {"branch_accuracy", "nll", "js", "recovery", "message_budget", "provenance"} <= set(out)
    assert results["live"]["message_budget"]["scalar_count"] == results["yoked"]["message_budget"]["scalar_count"]
    assert results["live"]["message_budget"]["width"] == results["yoked"]["message_budget"]["width"]
    assert sorted(results["time_shuffled"]["message_multiset"]) == sorted(results["yoked"]["message_multiset"])


def test_diagnostic_gauge_changes_coordinates_not_model_outputs():
    pair = make_pair()
    state = torch.randn(32, 8)
    before_msg = pair.model_a.emit_from_state(state).detach().clone()
    before_logits = pair.model_a.logits_from_state(state).detach().clone()
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    gauged = diagnostic_gauge(state, q)
    assert not torch.allclose(state, gauged)
    assert torch.equal(before_msg, pair.model_a.emit_from_state(state))
    assert torch.equal(before_logits, pair.model_a.logits_from_state(state))
