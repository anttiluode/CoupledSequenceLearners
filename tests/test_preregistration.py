def test_package_exports_version_and_default_config():
    import coupled_sequence_learners as csl
    cfg = csl.default_config()
    assert csl.__version__ == "0.1.0"
    assert cfg["message_dim"] == 2
    assert cfg["hidden_dim"] == 16
    assert cfg["junction_step"] > cfg["shared_prefix_start"]
