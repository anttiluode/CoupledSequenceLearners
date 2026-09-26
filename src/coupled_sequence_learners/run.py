from __future__ import annotations

import argparse
import json

from . import default_config
from .experiment import run_seed


def _debug_config() -> dict:
    cfg = default_config()
    cfg.update({
        "hidden_dim": 8,
        "training_steps": 300,
        "batch_size": 64,
        "eval_batch_size": 256,
        "message_noise_std": 0.02,
        "pretraining_steps": 200,
    })
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CoupledSequenceLearners experiments")
    parser.add_argument("--mode", choices=("debug",), default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--listener-condition",
        choices=("fresh", "state_dependent", "state_independent"),
        default="fresh",
    )
    args = parser.parse_args(argv)
    cfg = _debug_config()
    result = run_seed(args.seed, cfg, mode=args.mode, listener_condition=args.listener_condition)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
