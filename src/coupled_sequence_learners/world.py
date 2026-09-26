from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EpisodeBatch:
    obs_a: np.ndarray
    obs_b: np.ndarray
    branch: np.ndarray
    junction_step: int
    episode_id: np.ndarray
    factor_a: np.ndarray
    factor_b: np.ndarray
    motif: np.ndarray
    variant: np.ndarray
    future_tokens: np.ndarray


def _private_pattern(length: int, first: int, second: int, factor: int, dim: int) -> np.ndarray:
    x = np.zeros((length, dim), dtype=np.float32)
    early, late = 1, length - 2
    if factor == 0:
        x[early, first] = 1.0
        x[late, second] = 1.0
    else:
        x[early, second] = 1.0
        x[late, first] = 1.0
    return x


def _shared_pattern(length: int, motif: int, dim: int) -> np.ndarray:
    x = np.zeros((length, dim), dtype=np.float32)
    c0 = 4 + (motif % 4)
    c1 = 4 + ((motif + 1) % 4)
    for t in range(length):
        x[t, c0 if t % 2 == 0 else c1] = 1.0
    return x


def _is_heldout(fa: int, fb: int, motif: int, variant: int) -> bool:
    return variant == ((fa + 2 * fb + motif) & 1)


def generate_batch(seed: int, batch_size: int, split: str, config: dict) -> EpisodeBatch:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be 'train' or 'heldout'")
    rng = np.random.default_rng(seed)
    dim = int(config["observation_dim"])
    private_len = int(config["private_prefix_len"])
    shared_len = int(config["shared_prefix_len"])
    post_len = int(config["post_junction_len"])
    junction = private_len + shared_len
    total_len = junction + post_len

    obs_a = np.zeros((batch_size, total_len, dim), dtype=np.float32)
    obs_b = np.zeros_like(obs_a)
    fa_arr = np.empty(batch_size, dtype=np.int64)
    fb_arr = np.empty(batch_size, dtype=np.int64)
    motif_arr = np.empty(batch_size, dtype=np.int64)
    variant_arr = np.empty(batch_size, dtype=np.int64)
    branch_arr = np.empty(batch_size, dtype=np.int64)
    future_tokens = np.empty((batch_size, post_len), dtype=np.int64)

    n = 0
    while n < batch_size:
        fa = int(rng.integers(0, 2))
        fb = int(rng.integers(0, 2))
        motif = int(rng.integers(0, 4))
        variant = int(rng.integers(0, 2))
        held = _is_heldout(fa, fb, motif, variant)
        if (split == "heldout") != held:
            continue
        branch = 2 * fa + fb
        a = _private_pattern(private_len, 0, 1, fa, dim)
        b = _private_pattern(private_len, 2, 3, fb, dim)
        shared = _shared_pattern(shared_len, motif, dim)
        future = np.zeros((post_len, dim), dtype=np.float32)
        obs_a[n] = np.concatenate([a, shared, future], axis=0)
        obs_b[n] = np.concatenate([b, shared, future], axis=0)
        fa_arr[n], fb_arr[n] = fa, fb
        motif_arr[n], variant_arr[n] = motif, variant
        branch_arr[n] = branch
        future_tokens[n] = np.array([(branch + (variant if t % 2 else 0)) % 4 for t in range(post_len)], dtype=np.int64)
        n += 1

    ids = np.arange(batch_size, dtype=np.int64) + np.int64(seed) * 1_000_000
    return EpisodeBatch(
        obs_a=obs_a,
        obs_b=obs_b,
        branch=branch_arr,
        junction_step=junction,
        episode_id=ids,
        factor_a=fa_arr,
        factor_b=fb_arr,
        motif=motif_arr,
        variant=variant_arr,
        future_tokens=future_tokens,
    )


def heldout_signature(batch: EpisodeBatch) -> list[tuple[int, int, int, int]]:
    return [(int(a), int(b), int(m), int(v)) for a, b, m, v in zip(batch.factor_a, batch.factor_b, batch.motif, batch.variant)]


def shared_only_branch_counts(batch: EpisodeBatch) -> np.ndarray:
    counts = np.zeros((4, 4), dtype=np.int64)
    for motif, branch in zip(batch.motif, batch.branch):
        counts[int(motif), int(branch)] += 1
    return counts
