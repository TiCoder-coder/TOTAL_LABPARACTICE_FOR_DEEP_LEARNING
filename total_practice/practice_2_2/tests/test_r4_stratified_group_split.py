import csv
import json
from pathlib import Path

import pytest

from practice_2_2.r4_split import (
    assign_component_splits,
    authorize_test_access,
    build_split_fingerprints,
    load_r4_policy,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/v3_product_visual_group_s42_v1"


def test_r4_policy_is_group_safe_test_safe_and_deterministic():
    policy = load_r4_policy()
    assert policy["seed"] == 42
    assert sum(policy["splits"].values()) == 1.0
    assert policy["generated_assets_excluded"] is True
    assert policy["component_splitting_allowed"] is False
    assert policy["predecessor_gate_required"] is True
    assert policy["blocked_manifest_use_for_model"] is False
    assert policy["automatic_training_allowed"] is False
    assert policy["automatic_test_evaluation_allowed"] is False


def test_r4_assignment_is_identical_for_reversed_component_input():
    groups = json.loads(
        (
            ROOT
            / "artifacts/new_work/r3_product_visual_groups_v1/group_manifest.json"
        ).read_text()
    )["components"]
    first = assign_component_splits(groups)
    second = assign_component_splits(list(reversed(groups)))
    assert first == second


def test_r4_manifest_assigns_every_original_once_and_excludes_generated_assets():
    rows = json.loads((ARTIFACT_ROOT / "candidate_split_manifest.json").read_text())
    assert len(rows) == 2896
    assert len({row["asset_id"] for row in rows}) == 2896
    assert all(row["is_generated"] is False for row in rows)
    assert all(row["use_for_model"] is False for row in rows)
    assert all(
        row["eligibility"] == "provisional_pending_predecessor_gates"
        for row in rows
    )


def test_r4_json_and_csv_manifests_have_the_same_assignment():
    json_rows = json.loads(
        (ARTIFACT_ROOT / "candidate_split_manifest.json").read_text()
    )
    with (ARTIFACT_ROOT / "candidate_split_manifest.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_assignment = {
        row["asset_id"]: (row["component_id"], row["split"], row["relative_path"])
        for row in json_rows
    }
    csv_assignment = {
        row["asset_id"]: (row["component_id"], row["split"], row["relative_path"])
        for row in csv_rows
    }
    assert csv_assignment == json_assignment


def test_r4_components_never_cross_split_boundaries():
    rows = json.loads((ARTIFACT_ROOT / "candidate_split_manifest.json").read_text())
    component_splits = {}
    for row in rows:
        component_splits.setdefault(row["component_id"], set()).add(row["split"])
    assert len(component_splits) == 2462
    assert all(len(splits) == 1 for splits in component_splits.values())


def test_r4_ratios_and_class_coverage_meet_declared_limits():
    policy = load_r4_policy()
    summary = json.loads((ARTIFACT_ROOT / "split_summary.json").read_text())
    assert summary["every_class_in_every_split"] is True
    assert summary["maximum_overall_image_ratio_deviation"] <= policy[
        "maximum_overall_image_ratio_deviation"
    ]
    assert summary["maximum_per_class_image_ratio_deviation"] <= policy[
        "maximum_per_class_image_ratio_deviation"
    ]
    assert summary["maximum_per_class_component_ratio_deviation"] <= policy[
        "maximum_per_class_group_ratio_deviation"
    ]
    assert len(summary["class_inventory"]) == 10


def test_r4_leakage_report_has_zero_confirmed_group_violations():
    leakage = json.loads((ARTIFACT_ROOT / "leakage_report.json").read_text())
    assert leakage["component_cross_split_violation_count"] == 0
    assert leakage["confirmed_visual_cross_split_violation_count"] == 0
    assert leakage["strong_leakage_violation_count"] == 0
    assert all(
        count == 0
        for count in leakage["identity_cross_split_violation_counts"].values()
    )
    pairs = json.loads(
        (ARTIFACT_ROOT / "review_only_cross_split_pairs.json").read_text()
    )
    assert leakage["review_only_cross_split_pair_count"] == len(pairs)
    assert leakage["all_review_only_cross_split_pairs_documented"] is True
    assert len({pair["edge_id"] for pair in pairs}) == len(pairs)


def test_r4_fingerprints_match_current_manifest_and_components():
    rows = json.loads((ARTIFACT_ROOT / "candidate_split_manifest.json").read_text())
    groups = json.loads(
        (
            ROOT
            / "artifacts/new_work/r3_product_visual_groups_v1/group_manifest.json"
        ).read_text()
    )["components"]
    expected = build_split_fingerprints(rows, groups)
    actual = json.loads((ARTIFACT_ROOT / "split_fingerprints.json").read_text())
    for key, value in expected.items():
        assert actual[key] == value
    assert actual["manifest_json_sha256"] == file_sha256(
        ARTIFACT_ROOT / "candidate_split_manifest.json"
    )
    assert actual["manifest_csv_sha256"] == file_sha256(
        ARTIFACT_ROOT / "candidate_split_manifest.csv"
    )


def test_r4_test_guard_is_sealed_and_rejects_access():
    guard_path = ARTIFACT_ROOT / "test_guard.json"
    guard = json.loads(guard_path.read_text())
    assert guard["guard_state"] == "sealed_candidate_blocked"
    assert guard["lineage_authorized"] is False
    assert guard["model_training_authorized"] is False
    assert guard["test_partition_cryptographically_sealed"] is True
    assert guard["test_access_allowed"] is False
    assert guard["test_access_count"] == 0
    assert guard["test_evaluation_count"] == 0
    with pytest.raises(RuntimeError, match="lineage is not authorized"):
        authorize_test_access(
            guard_path, guard["split_fingerprint_sha256"], authorization_token=None
        )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        authorize_test_access(guard_path, "0" * 64, authorization_token=None)


def test_r4_report_stays_blocked_without_claiming_metric_comparability():
    report = json.loads((ROOT / "docs/R4_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["manifest_kind"] == "candidate"
    assert report["strong_leakage_violation_count"] == 0
    assert report["model_training_authorized"] is False
    assert report["test_access_count"] == 0
    assert report["test_evaluation_count"] == 0
    assert report["historical_metric_comparison_allowed"] is False
    assert report["canonical_notebook_mutated"] is False
    assert report["successor_phase_executed"] is False


def test_r4_artifact_manifest_verifies_every_persisted_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["source_images_mutated"] is False
    assert manifest["canonical_notebook_mutated"] is False
    assert manifest["test_access_count"] == 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r4_implementation_has_no_training_loader_evaluation_or_delete_operations():
    paths = [
        ROOT / "src/practice_2_2/r4_split.py",
        ROOT / "scripts/run_r4_split.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "optimizer" not in source
    assert "DataLoader" not in source
    assert "model(" not in source
    assert "evaluate(" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
