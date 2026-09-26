import numpy as np
from coupled_sequence_learners import default_config
from coupled_sequence_learners.world import generate_batch, heldout_signature, shared_only_branch_counts


def test_generate_batch_is_deterministic():
    cfg = default_config()
    a = generate_batch(7, 64, "train", cfg)
    b = generate_batch(7, 64, "train", cfg)
    assert np.array_equal(a.obs_a, b.obs_a)
    assert np.array_equal(a.obs_b, b.obs_b)
    assert np.array_equal(a.branch, b.branch)


def test_private_information_is_distributed():
    cfg = default_config()
    batch = generate_batch(8, 512, "train", cfg)
    for fa in (0, 1):
        assert len(set(batch.branch[batch.factor_a == fa].tolist())) >= 2
    for fb in (0, 1):
        assert len(set(batch.branch[batch.factor_b == fb].tolist())) >= 2
    expected = np.array([int(q) ^ int(bits[int(q)]) for q, bits in zip(batch.factor_a, batch.b_bits)])
    assert np.array_equal(batch.branch, expected)


def test_shared_segment_is_branch_neutral_within_motif():
    cfg = default_config()
    batch = generate_batch(9, 1024, "train", cfg)
    start = cfg["shared_prefix_start"]
    stop = cfg["junction_step"]
    for motif in np.unique(batch.motif):
        rows = np.where(batch.motif == motif)[0]
        ref = batch.obs_a[rows[0], start:stop]
        assert all(np.array_equal(batch.obs_a[i, start:stop], ref) for i in rows)
        assert all(np.array_equal(batch.obs_b[i, start:stop], ref) for i in rows)
    counts = shared_only_branch_counts(batch)
    assert counts.shape == (4, 4)
    assert np.all(counts > 0)


def test_train_and_holdout_are_complete_trajectory_recombinations():
    cfg = default_config()
    train = generate_batch(10, 4096, "train", cfg)
    held = generate_batch(11, 2048, "heldout", cfg)
    train_sigs = set(heldout_signature(train))
    held_sigs = set(heldout_signature(held))
    assert train_sigs.isdisjoint(held_sigs)
    assert set(train.motif.tolist()) == set(held.motif.tolist()) == {0, 1, 2, 3}
    assert set(train.factor_a.tolist()) == set(held.factor_a.tolist()) == {0, 1, 2, 3}
    assert set(train.factor_b.tolist()) == set(held.factor_b.tolist()) == set(range(16))


def test_post_junction_observations_do_not_reveal_branch():
    cfg = default_config()
    batch = generate_batch(12, 512, "train", cfg)
    j = cfg["junction_step"]
    assert np.all(batch.obs_a[:, j:] == 0)
    assert np.all(batch.obs_b[:, j:] == 0)
    assert batch.future_tokens.shape == (512, cfg["post_junction_len"])
    # Targets still encode the continuation; observations do not.
    assert len({tuple(row) for row in batch.future_tokens.tolist()}) >= 4


def test_world_uses_late_query_against_larger_listener_context():
    cfg = default_config()
    batch = generate_batch(44, 4096, "train", cfg)
    assert set(batch.factor_a.tolist()) == {0, 1, 2, 3}
    assert len(set(batch.factor_b.tolist())) >= 12
    # A query alone leaves two possible answers; B context alone leaves multiple
    # possible query-conditioned branches.
    for q in range(4):
        assert len(set(batch.branch[batch.factor_a == q].tolist())) == 2
    for ctx in np.unique(batch.factor_b):
        rows = batch.branch[batch.factor_b == ctx]
        if len(rows) >= 8:
            assert len(set(rows.tolist())) >= 2
