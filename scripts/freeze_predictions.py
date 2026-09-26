from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coupled_sequence_learners import default_config

SEEDS = [11, 23, 37, 53, 71, 89, 107, 131]
GIT_BASE_SHA = "dfa9e6522c7a2db2f4d088eb470cad43bb371e36"
FROZEN_AT = "2026-09-26T13:44:00+03:00"


def build_predictions() -> dict:
    cfg = default_config()
    cfg.update({
        "learning_rate": 0.01,
        "pretraining_steps": 250,
        "eval_batch_size": 512,
        "centralized_steps": 500,
    })
    data = {
        "schema_version": 1,
        "status": "FROZEN",
        "frozen_at": FROZEN_AT,
        "git_base_sha": GIT_BASE_SHA,
        "seeds": SEEDS,
        "config": cfg,
        "counterfactual_query_xor_mask": 3,
        "primary_metric": "heldout counterfactual branch_accuracy: live - yoked",
        "gates": {
            "centralized_accuracy_mean_min": 0.90,
            "live_accuracy_mean_min": 0.70,
            "live_minus_yoked_mean_min": 0.10,
            "live_minus_yoked_positive_seeds_min": 6,
            "live_yoked_message_scalar_count_equal": True,
            "gauge_logits_max_abs_diff_max": 1e-6,
            "heldout_recombination_live_accuracy_mean_min": 0.60,
        },
        "secondary_hypotheses": {
            "partner_swap_accuracy_drop_mean_min": 0.05,
            "listener_first_order": ["state_dependent", "state_independent", "fresh"],
        },
        "attackers": [
            "yoked", "one_way_a_to_b", "one_way_b_to_a", "frozen_partner",
            "time_shuffled", "isolated_a", "isolated_b", "partner_swap",
            "coordinate_gauge", "centralized_ceiling",
        ],
        "claim_boundary": "synthetic reciprocal causal feedback only",
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["content_digest"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing preregistration: {path}")
    data = build_predictions()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(data["content_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
