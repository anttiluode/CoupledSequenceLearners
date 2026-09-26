import json
from pathlib import Path
import pytest


def test_preregistered_runner_rejects_frozen_overrides():
    from scripts.run_preregistered import load_predictions, reject_overrides
    pred = load_predictions(Path("predictions.json"))
    for overrides in (
        {"message_dim": 4},
        {"seeds": [1]},
        {"live_minus_yoked_mean_min": 0.0},
        {"world_split": "train"},
    ):
        with pytest.raises(ValueError):
            reject_overrides(pred, overrides)


def test_preregistered_runner_accepts_no_overrides():
    from scripts.run_preregistered import load_predictions, reject_overrides
    pred = load_predictions(Path("predictions.json"))
    assert reject_overrides(pred, {}) is None


def test_worker_request_must_use_frozen_seed_and_known_condition():
    from scripts.run_preregistered import load_predictions, validate_worker_request
    pred = load_predictions(Path("predictions.json"))
    assert validate_worker_request(pred, 11, "fresh") is None
    with pytest.raises(ValueError):
        validate_worker_request(pred, 999, "fresh")
    with pytest.raises(ValueError):
        validate_worker_request(pred, 11, "mystery")
