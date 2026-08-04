"""Tests for the canonical, Test-locked Practice 2.2 training lineage."""

from processing_own_phase.canonical_train_practice_2_2 import (
    CANONICAL_BASE_CONFIG,
    EXPERIMENTS,
    EXPECTED_MODEL_COUNTS,
    RUN_ID,
    validate_controlled_configs,
)


def test_canonical_run_id_and_counts_are_frozen():
    assert RUN_ID == "canonical_26dc4625_52aaf974_s42_v1"
    assert EXPECTED_MODEL_COUNTS == {
        "Train": 2016,
        "Validation": 438,
        "Test": 440,
    }


def test_canonical_e1_e2_differ_only_by_training_mode():
    assert validate_controlled_configs()
    e1 = EXPERIMENTS["E1_head_only"]
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {
        key for key in set(e1).union(e2) if e1.get(key) != e2.get(key)
    }
    assert differing == {"training_mode"}
    assert e1["training_mode"] == "head_only"
    assert e2["training_mode"] == "partial_finetune"


def test_canonical_config_is_validation_only_and_test_locked():
    assert CANONICAL_BASE_CONFIG["selection_source"] == "validation_only"
    assert CANONICAL_BASE_CONFIG["test_data_used"] is False
    assert CANONICAL_BASE_CONFIG["test_loader_constructed"] is False
    assert CANONICAL_BASE_CONFIG["test_evaluated"] is False
    assert CANONICAL_BASE_CONFIG["offline_generated_files_used"] is False
    assert CANONICAL_BASE_CONFIG["resume_from_legacy"] is False
    assert CANONICAL_BASE_CONFIG["augment_strength"] == "base"
    assert "mixup" not in CANONICAL_BASE_CONFIG
    assert "cutmix" not in CANONICAL_BASE_CONFIG
    assert "ema" not in CANONICAL_BASE_CONFIG
