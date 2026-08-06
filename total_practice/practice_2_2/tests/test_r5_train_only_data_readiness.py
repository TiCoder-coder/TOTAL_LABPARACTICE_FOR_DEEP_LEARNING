import json
from pathlib import Path

import numpy as np

from practice_2_2.r5_data_readiness import (
    deterministic_kmeans,
    load_r5_policy,
    select_train_rows,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r5_train_only_data_readiness_v1"
R4_ROOT = ROOT / "artifacts/new_work/v3_product_visual_group_s42_v1"


def test_r5_policy_is_train_only_non_destructive_and_non_training():
    policy = load_r5_policy()
    assert policy["seed"] == 42
    assert policy["authorized_content_split"] == "Train"
    assert policy["predecessor_gate_required"] is True
    assert policy["automatic_semantic_decisions_allowed"] is False
    assert policy["automatic_quarantine_allowed"] is False
    assert policy["validation_content_access_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["model_training_allowed"] is False


def test_r5_train_selection_is_exact_and_disjoint_from_heldout():
    manifest = json.loads((R4_ROOT / "candidate_split_manifest.json").read_text())
    train_rows, heldout_ids = select_train_rows(manifest)
    train_ids = {row["asset_id"] for row in train_rows}
    expected = {row["asset_id"] for row in manifest if row["split"] == "Train"}
    assert train_ids == expected
    assert len(train_ids) == 2034
    assert len(heldout_ids) == 862
    assert not train_ids & heldout_ids


def test_r5_kmeans_is_deterministic():
    generator = np.random.default_rng(42)
    values = generator.normal(size=(60, 8)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    asset_ids = [f"asset_{index:03d}" for index in range(len(values))]
    first_assignments, first_centers, first_iterations = deterministic_kmeans(
        values, asset_ids, 6, 42
    )
    second_assignments, second_centers, second_iterations = deterministic_kmeans(
        values, asset_ids, 6, 42
    )
    assert np.array_equal(first_assignments, second_assignments)
    assert np.array_equal(first_centers, second_centers)
    assert first_iterations == second_iterations


def test_r5_content_access_manifest_contains_train_only():
    access = json.loads((ARTIFACT_ROOT / "content_access_manifest.json").read_text())
    manifest = json.loads((R4_ROOT / "candidate_split_manifest.json").read_text())
    train_ids = {row["asset_id"] for row in manifest if row["split"] == "Train"}
    assert access["authorized_split"] == "Train"
    assert set(access["accessed_asset_ids"]) == train_ids
    assert access["accessed_asset_count"] == 2034
    assert access["validation_content_access_count"] == 0
    assert access["test_content_access_count"] == 0


def test_r5_pixel_statistics_are_derived_from_train_only():
    statistics = json.loads((ARTIFACT_ROOT / "train_pixel_statistics.json").read_text())
    assert statistics["source_split"] == "Train"
    assert statistics["image_count"] == 2034
    assert statistics["pixel_count"] > 0
    assert len(statistics["rgb_mean"]) == 3
    assert len(statistics["rgb_std"]) == 3
    assert statistics["validation_images_used"] == 0
    assert statistics["test_images_used"] == 0


def test_r5_review_queue_covers_every_train_asset_without_auto_decisions():
    queue = json.loads((ARTIFACT_ROOT / "train_review_queue.json").read_text())
    assert len(queue) == 2034
    assert len({item["asset_id"] for item in queue}) == 2034
    assert all(item["review_status"] == "pending" for item in queue)
    assert all(item["semantic_decision"] is None for item in queue)
    assert all(item["reviewer_id"] is None for item in queue)


def test_r5_embedding_audits_use_only_train_embeddings():
    clusters = json.loads((ARTIFACT_ROOT / "embedding_cluster_report.json").read_text())
    neighbors = json.loads((ARTIFACT_ROOT / "embedding_neighbor_report.json").read_text())
    assert clusters["source_split"] == "Train"
    assert clusters["asset_count"] == 2034
    assert clusters["cluster_count"] == 20
    assert len(clusters["assignments"]) == 2034
    assert clusters["validation_embeddings_used"] == 0
    assert clusters["test_embeddings_used"] == 0
    assert neighbors["source_split"] == "Train"
    assert neighbors["asset_count"] == 2034
    assert len(neighbors["neighborhoods"]) == 2034
    assert neighbors["validation_embeddings_used"] == 0
    assert neighbors["test_embeddings_used"] == 0


def test_r5_contact_sheets_are_complete_and_train_only():
    index = json.loads((ARTIFACT_ROOT / "contact_sheet_index.json").read_text())
    assert len(index["class_sheets"]) == 10
    assert len(index["cluster_sheets"]) == 20
    assert len(index["overlap_sheet"]["pairs"]) == 12
    assert len(list((ARTIFACT_ROOT / "class_contact_sheets").glob("*.png"))) == 10
    assert len(list((ARTIFACT_ROOT / "cluster_contact_sheets").glob("*.png"))) == 20
    assert (ARTIFACT_ROOT / "cross_label_neighbor_pairs.png").is_file()


def test_r5_report_is_fail_closed_and_does_not_claim_data_readiness():
    report = json.loads((ROOT / "docs/R5_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["data_ready"] is False
    assert report["train_manifest_assets"] == 2034
    assert report["train_content_assets_accessed"] == 2034
    assert report["heldout_manifest_assets"] == 862
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["train_only_content_isolation_passed"] is True
    assert report["automatic_semantic_decisions_performed"] is False
    assert report["automatic_quarantine_performed"] is False
    assert report["model_training_performed"] is False
    assert report["validation_evaluated"] is False
    assert report["test_evaluated"] is False
    assert report["successor_phase_executed"] is False


def test_r5_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["content_authority"] == "Train"
    assert manifest["validation_content_access_count"] == 0
    assert manifest["test_content_access_count"] == 0
    assert manifest["source_images_mutated"] is False
    assert manifest["canonical_notebook_mutated"] is False
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r5_implementation_has_no_training_validation_test_or_delete_operations():
    paths = [
        ROOT / "src/practice_2_2/r5_data_readiness.py",
        ROOT / "scripts/run_r5_data_readiness.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "optimizer" not in source
    assert "DataLoader" not in source
    assert "backward(" not in source
    assert "evaluate(" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
