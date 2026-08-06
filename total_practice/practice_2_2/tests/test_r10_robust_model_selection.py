import ast
import json
from pathlib import Path

import pytest

from practice_2_2.r10_robust_model_selection import (
    confusion_stability,
    freeze_selected_checkpoint,
    json_sha256,
    load_r10_policy,
    metrics_from_confusion,
    select_robust_finalist,
    summarize_finalist_results,
    wilson_interval,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r10_robust_validation_selection_s42_123_2026_v1"


def cyclic_confusion(correct, support=50):
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for index in range(10):
        matrix[index][index] = correct
        matrix[index][(index + 1) % 10] = support - correct
    return matrix


def configurations():
    return {
        "F0": {"architecture": "resnet18", "protocol": "staged"},
        "F1": {"architecture": "efficientnet_b0", "protocol": "staged"},
    }


def result_fixture(policy, inconsistent=False):
    if inconsistent:
        correct = {
            "F0": {42: 38, 123: 38, 2026: 38},
            "F1": {42: 44, 123: 36, 2026: 36},
        }
    else:
        correct = {
            "F0": {42: 35, 123: 36, 2026: 37},
            "F1": {42: 38, 123: 39, 2026: 40},
        }
    config = configurations()
    return [
        {
            "finalist_id": finalist_id,
            "seed": seed,
            "configuration_sha256": json_sha256(config[finalist_id]),
            "initial_state_sha256": json_sha256(
                {"finalist_id": finalist_id, "kind": "initial"}
            ),
            "model_state_sha256": json_sha256(
                {"finalist_id": finalist_id, "seed": seed, "kind": "model"}
            ),
            "checkpoint_sha256": json_sha256(
                {"finalist_id": finalist_id, "seed": seed, "kind": "checkpoint"}
            ),
            "train_accuracy": (0.88 if finalist_id == "F0" else 0.86),
            "best_epoch": 9,
            "confusion_matrix": cyclic_confusion(correct[finalist_id][seed]),
        }
        for finalist_id in config
        for seed in policy["required_seeds"]
    ]


def test_r10_policy_predeclares_metrics_seeds_and_checkpoint_rule():
    policy = load_r10_policy()
    assert policy["required_seeds"] == [42, 123, 2026]
    assert policy["selection"]["primary"] == "mean_validation_macro_f1"
    assert policy["selection"]["secondary"] == "mean_validation_accuracy"
    assert policy["selection"]["minimum_seed_wins"] == 2
    assert policy["final_checkpoint"]["seed"] == 42
    assert policy["selection_record_test_fields_allowed"] is False
    assert policy["test_evidence_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["test_loader_construction_allowed"] is False


def test_r10_wilson_interval_has_expected_symmetry():
    lower, upper = wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038298286, abs=1e-9)
    assert upper == pytest.approx(0.5961701714, abs=1e-9)
    assert lower < 0.5 < upper


def test_r10_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError, match="invalid"):
        wilson_interval(1, 0)
    with pytest.raises(ValueError, match="invalid"):
        wilson_interval(11, 10)


def test_r10_metrics_are_derived_from_confusion_matrix():
    policy = load_r10_policy()
    metrics = metrics_from_confusion(cyclic_confusion(40), policy["class_names"])
    assert metrics["sample_count"] == 500
    assert metrics["correct_count"] == 400
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx(0.8)
    assert metrics["macro_recall"] == pytest.approx(0.8)
    assert all(
        values["recall"] == pytest.approx(0.8)
        for values in metrics["per_class"].values()
    )


def test_r10_confusion_stability_is_zero_for_identical_matrices():
    matrix = cyclic_confusion(38)
    stability = confusion_stability([matrix, matrix, matrix])
    assert stability["mean_pairwise_normalized_confusion_difference"] == 0.0
    assert stability["maximum_normalized_cell_standard_deviation"] == 0.0


def test_r10_confusion_stability_detects_seed_variation():
    stability = confusion_stability(
        [cyclic_confusion(36), cyclic_confusion(38), cyclic_confusion(40)]
    )
    assert stability["mean_pairwise_normalized_confusion_difference"] > 0
    assert stability["maximum_normalized_cell_standard_deviation"] > 0


def test_r10_summary_reports_all_uncertainty_and_class_fields():
    policy = load_r10_policy()
    summaries = summarize_finalist_results(
        result_fixture(policy), ["F0", "F1"], policy
    )
    for summary in summaries:
        assert summary["seeds"] == [42, 123, 2026]
        assert summary["std_validation_accuracy"] > 0
        assert summary["std_validation_macro_f1"] > 0
        assert len(summary["validation_accuracy_wilson_95_by_seed"]) == 3
        assert len(summary["descriptive_pooled_validation_accuracy_wilson_95"]) == 2
        assert set(summary["per_class_recall"]) == set(policy["class_names"])
        assert set(summary["confusion_stability"]) == set(
            policy["confusion_stability"]["reported_fields"]
        )


def test_r10_consistent_winner_is_selected_across_seeds():
    policy = load_r10_policy()
    results = result_fixture(policy)
    summaries = summarize_finalist_results(results, ["F0", "F1"], policy)
    selection = select_robust_finalist(
        results, summaries, ["F0", "F1"], policy
    )
    assert selection["selected_finalist_id"] == "F1"
    assert selection["seed_wins"]["F1"] == 3
    assert selection["consistent_across_seeds"] is True
    assert selection["selection_source"] == "Validation_only"
    assert selection["test_evidence_used"] is False
    assert len(selection["selection_record_sha256"]) == 64


def test_r10_inconsistent_mean_leader_is_not_selected():
    policy = load_r10_policy()
    results = result_fixture(policy, inconsistent=True)
    summaries = summarize_finalist_results(results, ["F0", "F1"], policy)
    selection = select_robust_finalist(
        results, summaries, ["F0", "F1"], policy
    )
    assert selection["seed_wins"]["F1"] == 1
    assert selection["selected_finalist_id"] is None
    assert selection["consistent_across_seeds"] is False


def test_r10_results_reject_test_fields_and_missing_seed():
    policy = load_r10_policy()
    results = result_fixture(policy)
    unauthorized = [dict(row) for row in results]
    unauthorized[0]["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="unauthorized"):
        summarize_finalist_results(unauthorized, ["F0", "F1"], policy)
    with pytest.raises(ValueError, match="cover"):
        summarize_finalist_results(results[:-1], ["F0", "F1"], policy)


def test_r10_results_reject_mixed_initialization_or_configuration():
    policy = load_r10_policy()
    mixed_initial = [dict(row) for row in result_fixture(policy)]
    mixed_initial[-1]["initial_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="share initialization"):
        summarize_finalist_results(mixed_initial, ["F0", "F1"], policy)
    mixed_config = [dict(row) for row in result_fixture(policy)]
    mixed_config[-1]["configuration_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="one configuration"):
        summarize_finalist_results(mixed_config, ["F0", "F1"], policy)


def test_r10_results_reject_invalid_confusion_matrix():
    policy = load_r10_policy()
    results = [dict(row) for row in result_fixture(policy)]
    results[0]["confusion_matrix"] = [[1, 0], [0, 1]]
    with pytest.raises(ValueError, match="shape"):
        summarize_finalist_results(results, ["F0", "F1"], policy)


def test_r10_freeze_record_contains_all_predeclared_hashes(tmp_path):
    policy = load_r10_policy()
    results = result_fixture(policy)
    summaries = summarize_finalist_results(results, ["F0", "F1"], policy)
    selection = select_robust_finalist(
        results, summaries, ["F0", "F1"], policy
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"R10 checkpoint fixture")
    selected_row = next(
        row
        for row in results
        if row["finalist_id"] == "F1" and row["seed"] == 42
    )
    selected_row["checkpoint_sha256"] = file_sha256(checkpoint)
    record = freeze_selected_checkpoint(
        selection,
        results,
        configurations()["F1"],
        checkpoint,
        policy,
    )
    assert record["final_checkpoint_seed"] == 42
    assert record["frozen_before_test_authorization"] is True
    assert record["selection_source"] == "Validation_only"
    assert record["test_evidence_used"] is False
    for field in policy["final_checkpoint"]["required_frozen_hashes"]:
        assert len(record[field]) == 64
    assert len(record["freeze_record_sha256"]) == 64


def test_r10_freeze_rejects_tampered_selection_config_or_checkpoint(tmp_path):
    policy = load_r10_policy()
    results = result_fixture(policy)
    summaries = summarize_finalist_results(results, ["F0", "F1"], policy)
    selection = select_robust_finalist(
        results, summaries, ["F0", "F1"], policy
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"R10 checkpoint fixture")
    selected_row = next(
        row
        for row in results
        if row["finalist_id"] == "F1" and row["seed"] == 42
    )
    selected_row["checkpoint_sha256"] = file_sha256(checkpoint)
    tampered = dict(selection)
    tampered["mean_macro_f1_margin"] = 1.0
    with pytest.raises(RuntimeError, match="selection record"):
        freeze_selected_checkpoint(
            tampered, results, configurations()["F1"], checkpoint, policy
        )
    with pytest.raises(RuntimeError, match="configuration"):
        freeze_selected_checkpoint(
            selection, results, {"different": True}, checkpoint, policy
        )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint file"):
        freeze_selected_checkpoint(
            selection, results, configurations()["F1"], checkpoint, policy
        )


def test_r10_report_is_protocol_complete_but_selection_gate_blocked():
    report = json.loads((ROOT / "docs/R10_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["protocol_exit_checks_passed"] is True
    assert report["blocked_reasons"] == [
        "R9_GATE_NOT_PASSED",
        "R9_ARCHITECTURE_NOT_AUTHORIZED",
        "REPEATED_SEED_FINALIST_RESULTS_UNAVAILABLE",
        "NO_ROBUST_FINALIST_SELECTED",
        "FINAL_CHECKPOINT_NOT_FROZEN",
    ]
    assert report["selected_finalist_id"] is None
    assert report["final_checkpoint_frozen"] is False
    assert report["test_evidence_used"] is False
    assert report["test_loader_constructed"] is False
    assert report["real_training_performed"] is False
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["successor_phase_executed"] is False


def test_r10_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["synthetic_protocol_verification_only"] is True
    assert manifest["fixture_is_model_selection_evidence"] is False
    assert manifest["selected_finalist_id"] is None
    assert manifest["final_checkpoint_frozen"] is False
    assert manifest["real_training_performed"] is False
    assert manifest["validation_content_access_count"] == 0
    assert manifest["test_content_access_count"] == 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r10_entrypoint_imports_no_data_loader_and_performs_no_real_training():
    paths = [
        ROOT / "src/practice_2_2/r10_robust_model_selection.py",
        ROOT / "scripts/run_r10_robust_model_selection.py",
    ]
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "DataLoader" not in imported_names
        assert "torch.utils.data" not in imported_names
        assert ".backward(" not in source
        assert "optimizer.step(" not in source


def test_r10_did_not_mutate_canonical_notebook():
    report = json.loads((ROOT / "docs/R10_VERIFICATION.json").read_text())
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    assert report["canonical_notebook_mutated"] is False
    assert file_sha256(notebook) == report["canonical_notebook_sha256"]
