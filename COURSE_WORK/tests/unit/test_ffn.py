import pytest
import torch

from course_work.sweeps.ffn import (
    CONDITIONS,
    build_phase_36_preflight,
    inspect_ffn_geometry,
    prepare_phase_36_condition,
    verify_phase_36_f256_best,
    verify_phase_36_f64_best,
)
from course_work.utils.artifacts import get_project_root, read_json
from course_work.models.transformer_regressor import TransformerRegressor


def model_config() -> dict:
    return {
        "input_size": 33,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.1,
        "pooling": "LAST_STEP",
        "activation": "GELU",
    }


def test_ffn_registry_is_exact() -> None:
    assert [(item.condition_id, item.ffn_dim, item.execution_mode) for item in CONDITIONS] == [
        ("F64", 64, "TRAIN_NEW"),
        ("F128", 128, "REUSE_REFERENCE"),
        ("F256", 256, "TRAIN_NEW"),
    ]


def test_ffn_geometry_changes_only_ffn_width() -> None:
    comparison = inspect_ffn_geometry(model_config())
    assert comparison["status"] == "PASS"
    assert comparison["config_delta"] == ["ffn_dim"]
    assert comparison["state_dict_key_sets_equal"] is True
    assert comparison["ffn_only_shape_delta"] is True
    assert comparison["mha_invariance"] == "PASS"
    assert comparison["non_ffn_invariance"] == "PASS"
    assert comparison["parameter_monotonicity"] == "PASS"
    assert comparison["parameter_delta_audit"] == "PASS"
    assert comparison["optimizer_coverage"] == "PASS"
    assert comparison["parameter_counts"] == {
        "F64": 52673,
        "F128": 69185,
        "F256": 102209,
    }
    assert comparison["parameter_deltas"] == comparison["expected_parameter_deltas"]
    assert comparison["sanity"] == {"F64": "PASS", "F256": "PASS"}
    changed = [row for row in comparison["state_dict_shape_rows"] if row["shape_changes"]]
    assert changed
    assert all(row["ffn_width_dependent"] and row["status"] == "PASS" for row in changed)
    for candidate in comparison["candidates"]:
        assert candidate["optimizer_coverage"]["status"] == "PASS"
        assert candidate["sanity"]["optimizer_step_executed"] is False
        assert candidate["sanity"]["training_engine_called"] is False
        assert all(row["status"] == "PASS" for row in candidate["layer_widths"])


def test_wrong_ffn_width_checkpoint_strict_load_fails() -> None:
    source = TransformerRegressor(model_config())
    target_config = {**model_config(), "ffn_dim": 256}
    target = TransformerRegressor(target_config)
    with pytest.raises(RuntimeError):
        target.load_state_dict(source.state_dict(), strict=True)


def test_phase_36_canonical_preflight_reuses_f128_with_warning() -> None:
    preflight = build_phase_36_preflight(get_project_root())
    assert preflight["ready"] is True
    assert preflight["handoff"]["status"] == "PASS_WITH_WARNING"
    assert preflight["handoff"]["issues"] == []
    assert preflight["handoff"]["winner_run_id"] == "RUN_TR_S09_0016_AE0FB819"
    assert preflight["decision"]["inspection"]["conditions"]["missing_conditions"] == []
    verified = {
        item["condition_id"]: item
        for item in preflight["decision"]["inspection"]["conditions"]["verified_conditions"]
    }
    assert verified["F128"]["evidence_status"] == "PASS_WITH_WARNING"
    assert verified["F128"]["run_id"] == "RUN_TR_S09_0016_AE0FB819"
    assert verified["F64"]["run_id"] == "RUN_TR_S14_0022_AA048302"
    assert verified["F256"]["run_id"] == "RUN_TR_S14_0023_A711A9B8"
    assert preflight["test_access"] == "FORBIDDEN"


def test_f64_best_checkpoint_recomputes_full_ordered_validation() -> None:
    verification = verify_phase_36_f64_best(project_root=get_project_root())
    assert verification["status"] == "VERIFIED"
    assert verification["strict_load_verification"] == "PASS"
    assert verification["strict_best_verification"] == "PASS"
    assert verification["population_verification"] == "PASS"
    assert verification["parameter_count"] == 52673
    assert verification["parameter_monotonicity"] == "PASS"
    assert verification["optimizer_coverage"] == "PASS"
    assert verification["test_access"] == "FORBIDDEN"
    assert verification["metric_delta"] <= verification["verification_tolerance"]


def test_f256_best_checkpoint_recomputes_full_ordered_validation() -> None:
    verification = verify_phase_36_f256_best(project_root=get_project_root())
    assert verification["status"] == "VERIFIED"
    assert verification["strict_load_verification"] == "PASS"
    assert verification["strict_best_verification"] == "PASS"
    assert verification["population_verification"] == "PASS"
    assert verification["parameter_count"] == 102209
    assert verification["parameter_monotonicity"] == "PASS"
    assert verification["optimizer_coverage"] == "PASS"
    assert verification["test_access"] == "FORBIDDEN"
    assert verification["metric_delta"] <= verification["verification_tolerance"]


def test_prepare_f128_never_authorizes_training() -> None:
    prepared = prepare_phase_36_condition("F128", get_project_root())
    assert prepared["execution_mode"] == "REUSE_REFERENCE"
    assert prepared["reference_run_id"] == "RUN_TR_S09_0016_AE0FB819"
    assert prepared["ffn_dim"] == 128
    assert prepared["test_access"] == "FORBIDDEN"


def test_completed_f256_cannot_be_retrained() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        prepare_phase_36_condition("F256", get_project_root())


def test_phase_36_final_signoff_and_phase_37_policy() -> None:
    root = get_project_root()
    signoff = read_json(root / "artifacts/sweeps/S14_ffn/phase_36_signoff.json")
    reference = read_json(root / "artifacts/sweeps/S14_ffn/s14_reference_update.json")
    assert signoff["status"] == "PASS_WITH_WARNING"
    assert signoff["winner_ffn_id"] == "F256"
    assert signoff["approved_for_phase37"] is True
    assert reference["phase37_condition_policy"] == {"MSE": "REUSE_REFERENCE", "Huber": "TRAIN_NEW"}
    assert reference["test_status"] == "FORBIDDEN"


def test_unknown_ffn_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        prepare_phase_36_condition("F512", get_project_root())
