"""Tests for the Test-locked Practice 2.2 training runner."""

from configs.experiment_config import EXPERIMENTS
from practice_2_2.train_practice_2_2 import RUN_SPECS, validate_run_specs


def test_run_specs_contain_only_e1_e2_and_one_controlled_difference():
    assert [name for name, _ in RUN_SPECS] == [
        "E1_head_only",
        "E2_partial_finetune",
    ]
    assert validate_run_specs()
    e1 = EXPERIMENTS[RUN_SPECS[0][1]]
    e2 = EXPERIMENTS[RUN_SPECS[1][1]]
    differing = {
        key
        for key in set(e1).union(e2)
        if e1.get(key) != e2.get(key)
    }
    assert differing == {"training_mode"}


def test_controlled_configs_lock_test_and_require_multi_epoch_training():
    for _, config_id in RUN_SPECS:
        config = EXPERIMENTS[config_id]
        assert config["epochs"] == 15
        assert config["selection_source"] == "validation_only"
        assert config["test_data_used"] is False
        assert not any(
            key.startswith("test_") and key != "test_data_used"
            for key in config
        )
