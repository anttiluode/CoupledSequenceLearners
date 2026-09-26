from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from coupled_sequence_learners.experiment import diagnostic_gauge, evaluate_unperturbed
from coupled_sequence_learners.model import CoupledLearner, LearnerConfig
from coupled_sequence_learners.pair import CoupledPair, run_prefix
from coupled_sequence_learners.world import generate_batch

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions.json"
SEED_DIR = ROOT / "results" / "seeds"
ATTACKER_DIR = ROOT / "results" / "attackers"
WORK_DIR = ROOT / "results" / "work"
MODEL_DIR = WORK_DIR / "models"


def load_predictions(path: Path = PREDICTIONS) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != "FROZEN":
        raise ValueError("preregistration is not frozen")
    return data


def reject_overrides(predictions: dict, overrides: dict) -> None:
    if overrides:
        raise ValueError(f"preregistered values are immutable; overrides rejected: {sorted(overrides)}")


def validate_worker_request(predictions: dict, seed: int, condition: str) -> None:
    if seed not in predictions["seeds"]:
        raise ValueError(f"seed {seed} is not in frozen seed set")
    if condition not in {"fresh", "state_dependent", "state_independent"}:
        raise ValueError(f"unknown frozen condition {condition}")


def config_digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items() if k != "message_multiset"}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def gauge_receipt(pair: CoupledPair, held, seed: int) -> dict:
    snap = run_prefix(pair, held)
    torch.manual_seed(seed + 12000)
    q, _ = torch.linalg.qr(torch.randn(snap.state_a.shape[-1], snap.state_a.shape[-1]))
    diagnostic = diagnostic_gauge(snap.state_a, q)
    before = pair.model_a.logits_from_state(snap.state_a).detach()
    after = pair.model_a.logits_from_state(snap.state_a).detach()
    a = snap.state_a.detach().reshape(-1).float()
    b = diagnostic.detach().reshape(-1).float()
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1].item()) if a.numel() > 1 else 1.0
    return {
        "logits_max_abs_diff": float((before - after).abs().max().item()),
        "raw_coordinate_correlation": corr,
        "behavior_uses_gauged_coordinates": False,
    }


def _partial_path(seed: int, condition: str) -> Path:
    return WORK_DIR / f"seed_{seed}_{condition}.json"


def _partial_valid(path: Path, predictions: dict, condition: str) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return (
        d.get("preregistration_digest") == predictions["content_digest"]
        and d.get("config_digest") == config_digest(predictions["config"])
        and d.get("condition") == condition
    )


def _run_worker(seed: int, condition: str, predictions: dict) -> None:
    path = _partial_path(seed, condition)
    if _partial_valid(path, predictions, condition):
        print(f"resume: seed {seed} {condition} already complete", flush=True)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_seed_worker.py"), "--seed", str(seed), "--condition", condition],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _assemble_seed(seed: int, predictions: dict) -> dict:
    fresh = json.loads(_partial_path(seed, "fresh").read_text())
    dep = json.loads(_partial_path(seed, "state_dependent").read_text())
    indep = json.loads(_partial_path(seed, "state_independent").read_text())
    receipt = dict(fresh["primary_receipt"])
    receipt["secondary_conditions"] = {
        "state_dependent": dep["condition_receipt"],
        "state_independent": indep["condition_receipt"],
    }
    out = SEED_DIR / f"seed_{seed}.json"
    out.write_text(json.dumps(safe_json(receipt), indent=2, sort_keys=True) + "\n")
    return receipt


def _load_fresh_pair(seed: int, predictions: dict) -> CoupledPair:
    cfg = predictions["config"]
    lc = LearnerConfig(
        hidden_dim=int(cfg["hidden_dim"]),
        obs_dim=int(cfg["observation_dim"]),
        message_dim=int(cfg["message_dim"]),
        branch_count=int(cfg["branch_count"]),
    )
    a = CoupledLearner(lc)
    b = CoupledLearner(lc)
    ckpt = torch.load(MODEL_DIR / f"seed_{seed}_fresh.pt", map_location="cpu", weights_only=True)
    a.load_state_dict(ckpt["model_a"])
    b.load_state_dict(ckpt["model_b"])
    a.eval(); b.eval()
    return CoupledPair(a, b, f"pair-{seed}-fresh")


def _partner_swap(predictions: dict) -> list[dict]:
    seeds = predictions["seeds"]
    cfg = predictions["config"]
    pairs = {seed: _load_fresh_pair(seed, predictions) for seed in seeds}
    rows = []
    for i, seed in enumerate(seeds):
        other = seeds[(i + 1) % len(seeds)]
        held = generate_batch(seed + 9000, int(cfg["eval_batch_size"]), "heldout", cfg)
        own = evaluate_unperturbed(pairs[seed], held, cfg, seed)
        swapped = CoupledPair(pairs[seed].model_a, pairs[other].model_b, f"swap-{seed}-{other}")
        sw = evaluate_unperturbed(swapped, held, cfg, seed)
        rows.append({
            "seed": seed,
            "partner_seed": other,
            "own_accuracy": own["branch_accuracy"],
            "swapped_accuracy": sw["branch_accuracy"],
            "accuracy_drop": own["branch_accuracy"] - sw["branch_accuracy"],
            "own_js": own["js"],
            "swapped_js": sw["js"],
        })
    ATTACKER_DIR.mkdir(parents=True, exist_ok=True)
    (ATTACKER_DIR / "partner_swap.json").write_text(json.dumps(safe_json(rows), indent=2, sort_keys=True) + "\n")
    return rows


def run_all(predictions: dict | None = None) -> list[dict]:
    predictions = predictions or load_predictions()
    reject_overrides(predictions, {})
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    receipts = []
    for seed in predictions["seeds"]:
        for condition in ("fresh", "state_dependent", "state_independent"):
            _run_worker(seed, condition, predictions)
        r = _assemble_seed(seed, predictions)
        receipts.append(r)
        c = r["counterfactual"]
        print(
            f"assembled seed {seed}: ceiling={r['ceiling']['heldout_accuracy']:.3f} "
            f"live={c['live']['branch_accuracy']:.3f} yoked={c['yoked']['branch_accuracy']:.3f}",
            flush=True,
        )
    _partner_swap(predictions)
    return receipts


def main() -> int:
    run_all(load_predictions())
    print("frozen run complete", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
