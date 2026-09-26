"""Coupled sequence learner experiments."""

__version__ = "0.1.0"


def default_config() -> dict[str, object]:
    private_prefix_len = 6
    shared_prefix_len = 6
    return {
        "hidden_dim": 16,
        "message_dim": 2,
        "observation_dim": 8,
        "branch_count": 4,
        "private_prefix_len": private_prefix_len,
        "shared_prefix_len": shared_prefix_len,
        "post_junction_len": 4,
        "shared_prefix_start": private_prefix_len,
        "junction_step": private_prefix_len + shared_prefix_len,
        "message_noise_std": 0.05,
        "training_steps": 1200,
        "batch_size": 128,
    }
