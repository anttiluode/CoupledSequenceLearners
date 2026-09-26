from __future__ import annotations

import argparse
import json
import os

import torch

from coupled_sequence_learners.experiment import (
    evaluate_counterfactual,
    evaluate_unperturbed,
    leakage_diagnostics,
    train_centralized_ceiling,
    train_pair,
)
from coupled_sequence_learners.world import generate_batch
from scripts.run_preregistered import (
    MODEL_DIR,
    WORK_DIR,
    config_digest,
    gauge_receipt,
    load_predictions,
    safe_json,
    validate_worker_request,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args(argv)
    pred = load_predictions()
    validate_worker_request(pred, args.seed, args.condition)
    cfg = dict(pred["config"])
    torch.set_num_threads(2)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trained = train_pair(args.seed, cfg, listener_condition=args.condition)
    held = generate_batch(args.seed + 9000, int(cfg["eval_batch_size"]), "heldout", cfg)
    unpert = evaluate_unperturbed(trained.pair, held, cfg, args.seed)
    counter = evaluate_counterfactual(trained.pair, held, cfg, args.seed)
    condition_receipt = {
        "condition": args.condition,
        "initial_loss": trained.initial_loss,
        "final_loss": trained.final_loss,
        "pretraining": {
            "condition": trained.pretrain_receipt.condition,
            "heldout_accuracy": trained.pretrain_receipt.heldout_accuracy,
            "parameter_digest": trained.pretrain_receipt.parameter_digest,
            "steps": trained.pretrain_receipt.steps,
            "seed": trained.pretrain_receipt.seed,
        },
        "unperturbed": safe_json(unpert),
        "counterfactual_live": safe_json(counter["live"]),
        "counterfactual_yoked": safe_json(counter["yoked"]),
    }
    partial = {
        "schema_version": 1,
        "preregistration_digest": pred["content_digest"],
        "config_digest": config_digest(cfg),
        "seed": args.seed,
        "condition": args.condition,
        "condition_receipt": condition_receipt,
    }
    if args.condition == "fresh":
        ceiling = train_centralized_ceiling(args.seed, cfg, steps=int(cfg["centralized_steps"]))
        primary = {
            "schema_version": 1,
            "git_sha": pred["git_base_sha"],
            "preregistration_digest": pred["content_digest"],
            "config_digest": config_digest(cfg),
            "model_seed": args.seed,
            "data_seed": args.seed + 9000,
            "ceiling": {"heldout_accuracy": ceiling["heldout_accuracy"], "heldout_nll": ceiling["heldout_nll"]},
            "fresh": condition_receipt,
            "counterfactual": safe_json(counter),
            "leakage": leakage_diagnostics(trained.pair, held, cfg, args.seed),
            "gauge": gauge_receipt(trained.pair, held, args.seed),
            "state_digest_equal_live_yoked": counter["live"]["provenance"]["start_digest"] == counter["yoked"]["provenance"]["start_digest"],
            "message_scalar_count_equal_live_yoked": counter["live"]["message_budget"]["scalar_count"] == counter["yoked"]["message_budget"]["scalar_count"],
            "secondary_conditions": {},
        }
        partial["primary_receipt"] = primary
        torch.save(
            {"model_a": trained.pair.model_a.state_dict(), "model_b": trained.pair.model_b.state_dict()},
            MODEL_DIR / f"seed_{args.seed}_fresh.pt",
        )
    path = WORK_DIR / f"seed_{args.seed}_{args.condition}.json"
    path.write_text(json.dumps(safe_json(partial), indent=2, sort_keys=True) + "\n")
    print(
        f"worker seed {args.seed} {args.condition}: unpert={unpert['branch_accuracy']:.3f} "
        f"live={counter['live']['branch_accuracy']:.3f} yoked={counter['yoked']['branch_accuracy']:.3f}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
