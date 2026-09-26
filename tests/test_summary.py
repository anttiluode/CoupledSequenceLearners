from scripts.summarize_results import summarize_receipts


def _metric(acc, scalar_count=16):
    return {
        "branch_accuracy": acc,
        "nll": 1.0,
        "js": 0.1,
        "recovery": 1.0,
        "message_budget": {"scalar_count": scalar_count},
    }


def _seed(seed, live, yoked, ceiling=0.95, gauge=0.0, dep=0.8, indep=0.6, fresh=0.7):
    return {
        "model_seed": seed,
        "ceiling": {"heldout_accuracy": ceiling},
        "counterfactual": {
            "live": _metric(live),
            "yoked": _metric(yoked),
            "one_way_a_to_b": _metric(0.6),
            "one_way_b_to_a": _metric(0.6),
            "frozen_partner": _metric(0.55),
            "time_shuffled": _metric(0.50),
            "isolated_a": _metric(0.40),
            "isolated_b": _metric(0.40),
        },
        "fresh": {"unperturbed": _metric(fresh)},
        "gauge": {"logits_max_abs_diff": gauge},
        "message_scalar_count_equal_live_yoked": True,
        "secondary_conditions": {
            "state_dependent": {"unperturbed": _metric(dep)},
            "state_independent": {"unperturbed": _metric(indep)},
        },
    }


def test_summary_applies_frozen_gates_without_rewriting_them():
    pred = {
        "seeds": [1, 2],
        "gates": {
            "centralized_accuracy_mean_min": 0.90,
            "live_accuracy_mean_min": 0.70,
            "live_minus_yoked_mean_min": 0.10,
            "live_minus_yoked_positive_seeds_min": 2,
            "live_yoked_message_scalar_count_equal": True,
            "gauge_logits_max_abs_diff_max": 1e-6,
            "heldout_recombination_live_accuracy_mean_min": 0.60,
        },
        "secondary_hypotheses": {
            "partner_swap_accuracy_drop_mean_min": 0.05,
            "listener_first_order": ["state_dependent", "state_independent", "fresh"],
        },
        "content_digest": "sha256:test",
        "git_base_sha": "abc",
    }
    seeds = [_seed(1, 0.82, 0.65), _seed(2, 0.80, 0.64)]
    swaps = [{"accuracy_drop": 0.07}, {"accuracy_drop": 0.05}]
    out = summarize_receipts(pred, seeds, swaps)
    assert out["metrics"]["live_minus_yoked"]["mean"] == 0.165
    assert out["gates"]["live_minus_yoked_mean"]["passed"] is True
    assert out["gates"]["live_minus_yoked_positive_seeds"]["passed"] is True
    assert out["primary_pass"] is True
    assert out["secondary"]["partner_swap"]["passed"] is True
    assert out["secondary"]["listener_first"]["startup_counts"]["state_dependent"] == 2
