from __future__ import annotations

from dataclasses import dataclass, replace
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
    b_bits: np.ndarray


def _a_private_pattern(length: int, query: int, dim: int) -> np.ndarray:
    x = np.zeros((length, dim), dtype=np.float32)
    # Four query states encoded by two signed temporal features. The query is
    # private to A until the communication phase starts at the junction.
    bit0 = (query >> 0) & 1
    bit1 = (query >> 1) & 1
    x[1, 0] = 1.0 if bit0 else -1.0
    x[length - 2, 1] = 1.0 if bit1 else -1.0
    return x


def _b_private_pattern(length: int, context: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((length, dim), dtype=np.float32)
    bits = np.array([(context >> i) & 1 for i in range(4)], dtype=np.int64)
    # B holds four candidate answers. A's query chooses which one matters.
    # Two binary channel symbols cannot losslessly publish all 16 contexts at
    # once; B must hear the query to know which answer to return.
    for i, bit in enumerate(bits):
        x[i, i] = 1.0 if bit else -1.0
    return x, bits


def _shared_pattern(length: int, motif: int, dim: int) -> np.ndarray:
    x = np.zeros((length, dim), dtype=np.float32)
    c0 = 4 + (motif % 4)
    c1 = 4 + ((motif + 1) % 4)
    for t in range(length):
        x[t, c0 if t % 2 == 0 else c1] = 1.0
    return x


def _is_heldout(query: int, context: int, motif: int, variant: int) -> bool:
    return variant == ((query + context + motif) & 1)


def _branch_for(query: int, bits: np.ndarray) -> int:
    # Query selects one of B's four hidden bits. That bit toggles the query's
    # low branch bit, leaving four balanced branch classes overall.
    return int(query ^ int(bits[query]))


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
    qa = np.empty(batch_size, dtype=np.int64)
    context_arr = np.empty(batch_size, dtype=np.int64)
    motif_arr = np.empty(batch_size, dtype=np.int64)
    variant_arr = np.empty(batch_size, dtype=np.int64)
    branch_arr = np.empty(batch_size, dtype=np.int64)
    future_tokens = np.empty((batch_size, post_len), dtype=np.int64)
    b_bits = np.empty((batch_size, 4), dtype=np.int64)

    n = 0
    while n < batch_size:
        query = int(rng.integers(0, 4))
        context = int(rng.integers(0, 16))
        motif = int(rng.integers(0, 4))
        variant = int(rng.integers(0, 2))
        held = _is_heldout(query, context, motif, variant)
        if (split == "heldout") != held:
            continue
        a = _a_private_pattern(private_len, query, dim)
        b, bits = _b_private_pattern(private_len, context, dim)
        shared = _shared_pattern(shared_len, motif, dim)
        future_obs = np.zeros((post_len, dim), dtype=np.float32)
        branch = _branch_for(query, bits)
        obs_a[n] = np.concatenate([a, shared, future_obs], axis=0)
        obs_b[n] = np.concatenate([b, shared, future_obs], axis=0)
        qa[n] = query
        context_arr[n] = context
        motif_arr[n] = motif
        variant_arr[n] = variant
        branch_arr[n] = branch
        b_bits[n] = bits
        future_tokens[n] = np.array(
            [(branch + (variant if t % 2 else 0)) % 4 for t in range(post_len)],
            dtype=np.int64,
        )
        n += 1

    ids = np.arange(batch_size, dtype=np.int64) + np.int64(seed) * 1_000_000
    return EpisodeBatch(
        obs_a=obs_a,
        obs_b=obs_b,
        branch=branch_arr,
        junction_step=junction,
        episode_id=ids,
        factor_a=qa,
        factor_b=context_arr,
        motif=motif_arr,
        variant=variant_arr,
        future_tokens=future_tokens,
        b_bits=b_bits,
    )


def heldout_signature(batch: EpisodeBatch) -> list[tuple[int, int, int, int]]:
    return [
        (int(a), int(b), int(m), int(v))
        for a, b, m, v in zip(batch.factor_a, batch.factor_b, batch.motif, batch.variant)
    ]


def shared_only_branch_counts(batch: EpisodeBatch) -> np.ndarray:
    counts = np.zeros((4, 4), dtype=np.int64)
    for motif, branch in zip(batch.motif, batch.branch):
        counts[int(motif), int(branch)] += 1
    return counts


def with_counterfactual_queries(batch: EpisodeBatch, config: dict, queries: np.ndarray) -> EpisodeBatch:
    queries = np.asarray(queries, dtype=np.int64)
    if queries.shape != batch.factor_a.shape:
        raise ValueError("counterfactual query shape mismatch")
    if np.any((queries < 0) | (queries > 3)):
        raise ValueError("queries must be in 0..3")
    obs_a = batch.obs_a.copy()
    private_len = int(config["private_prefix_len"])
    for i, query in enumerate(queries):
        obs_a[i, :private_len] = _a_private_pattern(private_len, int(query), obs_a.shape[-1])
    branches = np.array([_branch_for(int(q), bits) for q, bits in zip(queries, batch.b_bits)], dtype=np.int64)
    future_tokens = np.empty_like(batch.future_tokens)
    for i, (branch, variant) in enumerate(zip(branches, batch.variant)):
        future_tokens[i] = np.array(
            [(int(branch) + (int(variant) if t % 2 else 0)) % 4 for t in range(future_tokens.shape[1])],
            dtype=np.int64,
        )
    return replace(batch, obs_a=obs_a, branch=branches, factor_a=queries.copy(), future_tokens=future_tokens)
