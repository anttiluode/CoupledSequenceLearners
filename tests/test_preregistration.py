import json
from pathlib import Path


def test_package_exports_version_and_default_config():
    import coupled_sequence_learners as csl
    cfg = csl.default_config()
    assert csl.__version__ == "0.1.0"
    assert cfg["message_dim"] == 2
    assert cfg["hidden_dim"] == 16
    assert cfg["junction_step"] > cfg["shared_prefix_start"]


def test_frozen_predictions_schema_and_thresholds():
    path = Path("predictions.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["seeds"] == [11, 23, 37, 53, 71, 89, 107, 131]
    gates = data["gates"]
    assert gates["centralized_accuracy_mean_min"] == 0.90
    assert gates["live_accuracy_mean_min"] == 0.70
    assert gates["live_minus_yoked_mean_min"] == 0.10
    assert gates["live_minus_yoked_positive_seeds_min"] == 6
    assert gates["live_yoked_message_scalar_count_equal"] is True
    assert gates["gauge_logits_max_abs_diff_max"] == 1e-6
    assert gates["heldout_recombination_live_accuracy_mean_min"] == 0.60
    secondary = data["secondary_hypotheses"]
    assert secondary["partner_swap_accuracy_drop_mean_min"] == 0.05
    assert secondary["listener_first_order"] == ["state_dependent", "state_independent", "fresh"]
    assert data["counterfactual_query_xor_mask"] == 3
    assert data["status"] == "FROZEN"
