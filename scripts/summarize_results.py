from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions.json"
SEED_DIR = ROOT / "results" / "seeds"
SWAP_FILE = ROOT / "results" / "attackers" / "partner_swap.json"
OUT = ROOT / "results" / "receipt.json"


def _summary(values: list[float]) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": None, "se": None, "n": 0}
    m = mean(vals)
    se = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return {"mean": round(m, 12), "se": round(se, 12), "n": len(vals)}


def _gate(value, threshold, op: str) -> dict:
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == "==":
        passed = value == threshold
    else:
        raise ValueError(op)
    return {"value": value, "threshold": threshold, "op": op, "passed": bool(passed)}


def summarize_receipts(predictions: dict, seeds: list[dict], swaps: list[dict]) -> dict:
    gates = predictions["gates"]
    live = [r["counterfactual"]["live"]["branch_accuracy"] for r in seeds]
    yoked = [r["counterfactual"]["yoked"]["branch_accuracy"] for r in seeds]
    deltas = [a - b for a, b in zip(live, yoked)]
    ceiling = [r["ceiling"]["heldout_accuracy"] for r in seeds]
    gauge = [r["gauge"]["logits_max_abs_diff"] for r in seeds]
    msg_equal = all(bool(r["message_scalar_count_equal_live_yoked"]) for r in seeds)
    positive = sum(d > 0 for d in deltas)

    arm_names = [
        "live", "yoked", "one_way_a_to_b", "one_way_b_to_a",
        "frozen_partner", "time_shuffled", "isolated_a", "isolated_b",
    ]
    arms = {}
    for arm in arm_names:
        rows = [r["counterfactual"][arm] for r in seeds]
        arms[arm] = {
            "branch_accuracy": _summary([x["branch_accuracy"] for x in rows]),
            "nll": _summary([x["nll"] for x in rows]),
            "js": _summary([x["js"] for x in rows]),
            "recovery": _summary([x["recovery"] for x in rows if x.get("recovery") is not None]),
        }

    gate_rows = {
        "centralized_accuracy_mean": _gate(mean(ceiling), gates["centralized_accuracy_mean_min"], ">="),
        "live_accuracy_mean": _gate(mean(live), gates["live_accuracy_mean_min"], ">="),
        "live_minus_yoked_mean": _gate(mean(deltas), gates["live_minus_yoked_mean_min"], ">="),
        "live_minus_yoked_positive_seeds": _gate(positive, gates["live_minus_yoked_positive_seeds_min"], ">="),
        "live_yoked_message_scalar_count_equal": _gate(msg_equal, gates["live_yoked_message_scalar_count_equal"], "=="),
        "gauge_logits_max_abs_diff": _gate(max(gauge), gates["gauge_logits_max_abs_diff_max"], "<="),
        "heldout_recombination_live_accuracy_mean": _gate(mean(live), gates["heldout_recombination_live_accuracy_mean_min"], ">="),
    }

    startup_threshold = gates["live_accuracy_mean_min"]
    condition_names = ["state_dependent", "state_independent", "fresh"]
    condition_accs = {name: [] for name in condition_names}
    for r in seeds:
        condition_accs["fresh"].append(r["fresh"]["unperturbed"]["branch_accuracy"])
        for name in ("state_dependent", "state_independent"):
            condition_accs[name].append(r["secondary_conditions"][name]["unperturbed"]["branch_accuracy"])
    startup_counts = {name: sum(v >= startup_threshold for v in vals) for name, vals in condition_accs.items()}
    listener_order = predictions["secondary_hypotheses"]["listener_first_order"]
    listener_pass = all(startup_counts[a] > startup_counts[b] for a, b in zip(listener_order, listener_order[1:]))

    swap_drops = [float(r["accuracy_drop"]) for r in swaps]
    swap_mean = mean(swap_drops) if swap_drops else 0.0
    secondary = {
        "partner_swap": {
            **_summary(swap_drops),
            "threshold": predictions["secondary_hypotheses"]["partner_swap_accuracy_drop_mean_min"],
            "passed": bool(swap_mean >= predictions["secondary_hypotheses"]["partner_swap_accuracy_drop_mean_min"]),
        },
        "listener_first": {
            "startup_threshold": startup_threshold,
            "startup_counts": startup_counts,
            "accuracy": {name: _summary(vals) for name, vals in condition_accs.items()},
            "expected_order": listener_order,
            "passed": listener_pass,
        },
    }

    return {
        "schema_version": 1,
        "preregistration_digest": predictions["content_digest"],
        "git_base_sha": predictions["git_base_sha"],
        "seeds": [r["model_seed"] for r in seeds],
        "metrics": {
            "ceiling_accuracy": _summary(ceiling),
            "live_accuracy": _summary(live),
            "yoked_accuracy": _summary(yoked),
            "live_minus_yoked": _summary(deltas),
            "live_minus_yoked_positive_seeds": positive,
            "arms": arms,
            "gauge_logits_max_abs_diff": max(gauge),
            "all_live_yoked_message_budgets_equal": msg_equal,
        },
        "gates": gate_rows,
        "primary_pass": all(row["passed"] for row in gate_rows.values()),
        "secondary": secondary,
    }


def load_all() -> tuple[dict, list[dict], list[dict]]:
    pred = json.loads(PREDICTIONS.read_text())
    seeds = []
    for seed in pred["seeds"]:
        seeds.append(json.loads((SEED_DIR / f"seed_{seed}.json").read_text()))
    swaps = json.loads(SWAP_FILE.read_text()) if SWAP_FILE.exists() else []
    return pred, seeds, swaps


def main() -> int:
    pred, seeds, swaps = load_all()
    out = summarize_receipts(pred, seeds, swaps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"primary_pass": out["primary_pass"], "gates": out["gates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
