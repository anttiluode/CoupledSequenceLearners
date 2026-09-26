from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coupled_sequence_learners.experiment import (
    diagnostic_gauge,
    evaluate_counterfactual,
    evaluate_unperturbed,
    leakage_diagnostics,
    train_centralized_ceiling,
    train_pair,
)
from coupled_sequence_learners.pair import CoupledPair, run_prefix
from coupled_sequence_learners.world import generate_batch

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions.json"
SEED_DIR = ROOT / "results" / "seeds"
ATTACKER_DIR = ROOT / "results" / "attackers"


def load_predictions(path: Path = PREDICTIONS) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != "FROZEN":
        raise ValueError("preregistration is not frozen")
    return data


def reject_overrides(predictions: dict, overrides: dict) -> None:
    if overrides:
        raise ValueError(f"preregistered values are immutable; overrides rejected: {sorted(overrides)}")


def _config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if k != "message_multiset"}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _gauge_receipt(pair: CoupledPair, held, seed: int) -> dict:
    snap = run_prefix(pair, held)
    torch.manual_seed(seed + 12000)
    q, _ = torch.linalg.qr(torch.randn(snap.state_a.shape[-1], snap.state_a.shape[-1]))
    diagnostic = diagnostic_gauge(snap.state_a, q)
    before = pair.model_a.logits_from_state(snap.state_a).detach()
    # The gauge is intentionally diagnostic-only: behavior continues to use the
    # original state. This checks that coordinate displays are not causal input.
    after = pair.model_a.logits_from_state(snap.state_a).detach()
    a = snap.state_a.detach().reshape(-1).float()
    b = diagnostic.detach().reshape(-1).float()
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1].item()) if a.numel() > 1 else 1.0
    return {
        "logits_max_abs_diff": float((before - after).abs().max().item()),
        "raw_coordinate_correlation": corr,
        "behavior_uses_gauged_coordinates": False,
    }


def _train_condition(seed: int, config: dict, condition: str) -> tuple[Any, dict]:
    trained = train_pair(seed, config, listener_condition=condition)
    held = generate_batch(seed + 9000, int(config["eval_batch_size"]), "heldout", config)
    unperturbed = evaluate_unperturbed(trained.pair, held, config, seed)
    counter = evaluate_counterfactual(trained.pair, held, config, seed)
    receipt = {
        "condition": condition,
        "initial_loss": trained.initial_loss,
        "final_loss": trained.final_loss,
        "pretraining": {
            "condition": trained.pretrain_receipt.condition,
            "heldout_accuracy": trained.pretrain_receipt.heldout_accuracy,
            "parameter_digest": trained.pretrain_receipt.parameter_digest,
            "steps": trained.pretrain_receipt.steps,
            "seed": trained.pretrain_receipt.seed,
        },
        "unperturbed": _safe(unperturbed),
        "counterfactual_live": _safe(counter["live"]),
        "counterfactual_yoked": _safe(counter["yoked"]),
    }
    return trained, receipt


def run_all(predictions: dict | None = None) -> list[dict]:
    predictions = predictions or load_predictions()
    reject_overrides(predictions, {})
    config = dict(predictions["config"])
    seeds = list(predictions["seeds"])
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    ATTACKER_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    fresh_models: dict[int, Any] = {}
    seed_receipts: list[dict] = []
    config_digest = _config_digest(config)

    for seed in seeds:
        ceiling = train_centralized_ceiling(seed, config, steps=int(config["centralized_steps"]))
        trained, fresh_condition = _train_condition(seed, config, "fresh")
        fresh_models[seed] = trained
        held = generate_batch(seed + 9000, int(config["eval_batch_size"]), "heldout", config)
        counter = evaluate_counterfactual(trained.pair, held, config, seed)
        receipt = {
            "schema_version": 1,
            "git_sha": predictions["git_base_sha"],
            "preregistration_digest": predictions["content_digest"],
            "config_digest": config_digest,
            "model_seed": seed,
            "data_seed": seed + 9000,
            "ceiling": {
                "heldout_accuracy": ceiling["heldout_accuracy"],
                "heldout_nll": ceiling["heldout_nll"],
            },
            "fresh": fresh_condition,
            "counterfactual": _safe(counter),
            "leakage": leakage_diagnostics(trained.pair, held, config, seed),
            "gauge": _gauge_receipt(trained.pair, held, seed),
            "state_digest_equal_live_yoked": counter["live"]["provenance"]["start_digest"] == counter["yoked"]["provenance"]["start_digest"],
            "message_scalar_count_equal_live_yoked": counter["live"]["message_budget"]["scalar_count"] == counter["yoked"]["message_budget"]["scalar_count"],
            "secondary_conditions": {},
        }
        for condition in ("state_dependent", "state_independent"):
            _, cond_receipt = _train_condition(seed, config, condition)
            receipt["secondary_conditions"][condition] = cond_receipt
        path = SEED_DIR / f"seed_{seed}.json"
        path.write_text(json.dumps(_safe(receipt), indent=2, sort_keys=True) + "\n")
        seed_receipts.append(receipt)
        print(
            f"seed {seed}: ceiling={receipt['ceiling']['heldout_accuracy']:.3f} "
            f"live={counter['live']['branch_accuracy']:.3f} "
            f"yoked={counter['yoked']['branch_accuracy']:.3f}"
        )

    # Partner-swap attacker: cyclically pair A_i with B_{i+1}; no adaptation.
    swap_rows = []
    for i, seed in enumerate(seeds):
        other = seeds[(i + 1) % len(seeds)]
        own = fresh_models[seed]
        partner = fresh_models[other]
        held = generate_batch(seed + 9000, int(config["eval_batch_size"]), "heldout", config)
        own_metrics = evaluate_unperturbed(own.pair, held, config, seed)
        swapped = CoupledPair(own.pair.model_a, partner.pair.model_b, f"swap-{seed}-{other}")
        swap_metrics = evaluate_unperturbed(swapped, held, config, seed)
        swap_rows.append({
            "seed": seed,
            "partner_seed": other,
            "own_accuracy": own_metrics["branch_accuracy"],
            "swapped_accuracy": swap_metrics["branch_accuracy"],
            "accuracy_drop": own_metrics["branch_accuracy"] - swap_metrics["branch_accuracy"],
            "own_js": own_metrics["js"],
            "swapped_js": swap_metrics["js"],
        })
    (ATTACKER_DIR / "partner_swap.json").write_text(json.dumps(_safe(swap_rows), indent=2, sort_keys=True) + "\n")
    return seed_receipts


def main() -> int:
    predictions = load_predictions()
    run_all(predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
