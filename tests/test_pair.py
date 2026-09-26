import torch
import pytest
from dataclasses import replace
from coupled_sequence_learners import default_config
from coupled_sequence_learners.model import CoupledLearner, LearnerConfig
from coupled_sequence_learners.pair import CoupledPair, run_prefix, record_unperturbed_continuation, continue_live, continue_yoked, state_digest
from coupled_sequence_learners.world import generate_batch


def make_pair(pair_id="p0"):
    torch.manual_seed(0)
    cfg = LearnerConfig(8, 8, 2, 4)
    return CoupledPair(CoupledLearner(cfg), CoupledLearner(cfg), pair_id)


def test_snapshot_clones_are_independent_and_digest_stable():
    cfg = default_config(); cfg["hidden_dim"] = 8
    batch = generate_batch(3, 4, "train", cfg)
    pair = make_pair()
    snap = run_prefix(pair, batch)
    d = state_digest(snap)
    clone = snap.clone()
    assert state_digest(clone) == d
    clone.state_a[0, 0] += 1
    assert state_digest(snap) == d
    assert state_digest(clone) != d


def test_replay_provenance_must_match_pair_episode_and_snapshot():
    cfg = default_config(); cfg["hidden_dim"] = 8
    batch = generate_batch(4, 3, "train", cfg)
    pair = make_pair("pair-a")
    snap = run_prefix(pair, batch)
    replay = record_unperturbed_continuation(pair, batch, snap)
    with pytest.raises(ValueError):
        continue_yoked(make_pair("pair-b"), batch, snap, lambda x: x, replay)
    bad = replace(replay, source_state_digest="wrong")
    with pytest.raises(ValueError):
        continue_yoked(pair, batch, snap, lambda x: x, bad)


def test_live_partner_reacts_but_yoked_partner_emits_stale_trace():
    cfg = default_config(); cfg["hidden_dim"] = 8
    batch = generate_batch(5, 2, "train", cfg)
    pair = make_pair("pair-c")
    snap = run_prefix(pair, batch)
    replay = record_unperturbed_continuation(pair, batch, snap)
    perturb = lambda s: s + 3.0
    live = continue_live(pair, batch, snap, perturb)
    yoked = continue_yoked(pair, batch, snap, perturb, replay)
    assert torch.equal(yoked.messages_b, replay.messages_b)
    assert not torch.allclose(live.messages_b, replay.messages_b)
    assert live.start_digest == yoked.start_digest == state_digest(snap)
