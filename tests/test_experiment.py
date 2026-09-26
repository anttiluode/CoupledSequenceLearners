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


def test_tiny_config_is_trainable_and_ceiling_solves_above_chance():
    from coupled_sequence_learners.experiment import train_pair, train_centralized_ceiling, evaluate_unperturbed
    cfg = default_config()
    cfg.update({"hidden_dim": 8, "training_steps": 220, "batch_size": 64, "message_noise_std": 0.02})
    trained = train_pair(seed=3, config=cfg, listener_condition="fresh")
    assert trained.final_loss < trained.initial_loss
    held = generate_batch(303, 256, "heldout", cfg)
    pair_metrics = evaluate_unperturbed(trained.pair, held, cfg, seed=3)
    assert 0.0 <= pair_metrics["branch_accuracy"] <= 1.0
    ceiling = train_centralized_ceiling(seed=3, config=cfg, steps=180)
    assert ceiling["heldout_accuracy"] > 0.50


def test_counterfactual_assay_changes_query_after_identical_prefix_snapshot():
    from coupled_sequence_learners.experiment import train_pair, evaluate_counterfactual
    cfg = default_config()
    cfg.update({"hidden_dim": 8, "training_steps": 80, "batch_size": 32, "message_noise_std": 0.0})
    trained = train_pair(seed=4, config=cfg, listener_condition="fresh")
    held = generate_batch(404, 64, "heldout", cfg)
    out = evaluate_counterfactual(trained.pair, held, cfg, seed=4)
    assert out["live"]["provenance"]["start_digest"] == out["yoked"]["provenance"]["start_digest"]
    assert out["live"]["message_budget"]["scalar_count"] == out["yoked"]["message_budget"]["scalar_count"]
    assert out["counterfactual_changed_fraction"] == 1.0
    assert out["b_prefix_state_equal"] is True
