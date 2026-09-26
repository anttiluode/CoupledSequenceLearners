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
    expected = batch.factor_a * 2 + batch.factor_b
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
    assert set(train.factor_a.tolist()) == set(held.factor_a.tolist()) == {0, 1}
    assert set(train.factor_b.tolist()) == set(held.factor_b.tolist()) == {0, 1}
