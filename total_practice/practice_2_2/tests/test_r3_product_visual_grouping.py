import json
from pathlib import Path

import numpy as np

from practice_2_2.r3_grouping import (
    DeterministicUnionFind,
    load_r3_policy,
    stable_component_id,
    stable_edge_id,
    structural_similarity,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r3_product_visual_groups_v1"


def test_r3_policy_is_deterministic_non_destructive_and_split_safe():
    policy = load_r3_policy()
    assert policy["seed"] == 42
    assert policy["generated_assets_excluded"] is True
    assert policy["split_creation_allowed"] is False
    assert policy["automatic_deletion_allowed"] is False
    assert policy["automatic_quarantine_allowed"] is False
    assert policy["cross_label_union_allowed"] is False


def test_union_find_and_component_ids_ignore_input_order():
    first = DeterministicUnionFind(["c", "a", "b"])
    second = DeterministicUnionFind(["b", "c", "a"])
    first.union("c", "b")
    first.union("a", "b")
    second.union("a", "c")
    second.union("b", "a")
    assert first.components() == second.components() == [["a", "b", "c"]]
    assert stable_component_id(["c", "a", "b"]) == stable_component_id(
        ["b", "c", "a"]
    )
    assert stable_edge_id("b", "a") == stable_edge_id("a", "b")


def test_structural_similarity_has_expected_identity_and_difference_behavior():
    left = np.arange(96 * 96, dtype=np.float32).reshape(96, 96) % 255
    right = np.flipud(left).copy()
    assert structural_similarity(left, left) > 0.9999
    assert structural_similarity(left, right) < 0.95


def test_r3_report_assigns_every_original_once_and_remains_blocked():
    report = json.loads((ROOT / "docs/R3_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["seed"] == 42
    assert report["eligible_original_assets"] == 2896
    assert report["component_assignment_count"] == 2896
    assert report["unique_component_assignment_count"] == 2896
    assert report["generated_assets_excluded"] is True
    assert report["cross_label_component_count"] == 0
    assert report["split_manifest_created"] is False
    assert report["automatic_quarantine_performed"] is False
    assert report["automatic_deletion_performed"] is False
    assert report["test_evaluated"] is False


def test_r3_manifest_component_ids_and_members_are_unique():
    manifest = json.loads((ARTIFACT_ROOT / "group_manifest.json").read_text())
    components = manifest["components"]
    component_ids = [component["component_id"] for component in components]
    asset_ids = [asset_id for component in components for asset_id in component["member_asset_ids"]]
    assert len(component_ids) == len(set(component_ids))
    assert len(asset_ids) == len(set(asset_ids)) == 2896
    for component in components:
        assert component["component_id"] == stable_component_id(
            component["member_asset_ids"]
        )
        assert len(component["labels"]) == 1


def test_r3_all_cross_label_and_uncertain_edges_are_review_only():
    policy = load_r3_policy()
    thresholds = policy["visual_evidence"]["confirmed_same_label"]
    queue = json.loads((ARTIFACT_ROOT / "review_queue.json").read_text())
    accepted = json.loads((ARTIFACT_ROOT / "accepted_edges.json").read_text())
    assert len({edge["edge_id"] for edge in queue}) == len(queue)
    assert all(edge["review_status"] == "pending" for edge in queue)
    assert all(not edge["auto_confirmed"] for edge in queue)
    assert all(edge["left_label"] == edge["right_label"] for edge in accepted)
    assert all(edge["auto_confirmed"] for edge in accepted)
    for edge in accepted:
        if edge.get("edge_type") == "identity":
            continue
        assert edge["phash_distance"] <= thresholds["maximum_phash_distance"]
        assert edge["dhash_distance"] <= thresholds["maximum_dhash_distance"]
        assert edge["ssim"] >= thresholds["minimum_ssim"]
        assert edge["embedding_cosine"] >= thresholds["minimum_embedding_cosine"]
    assert all(
        edge["review_status"] == "pending"
        for edge in queue
        if edge["left_label"] != edge["right_label"]
    )


def test_r3_embedding_uses_verified_official_pretrained_weights():
    metadata = json.loads((ARTIFACT_ROOT / "embedding_metadata.json").read_text())
    assert metadata["model"] == "resnet18"
    assert metadata["weights"] == "IMAGENET1K_V1"
    assert metadata["checkpoint_sha256"] == (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )
    assert metadata["embedding_dimension"] == 512
    assert metadata["asset_count"] == 2896
    assert metadata["normalized"] is True
    stored = np.load(ARTIFACT_ROOT / "pretrained_embeddings.npz", allow_pickle=False)
    assert stored["embeddings"].shape == (2896, 512)
    assert np.allclose(np.linalg.norm(stored["embeddings"], axis=1), 1.0, atol=1e-5)


def test_r3_artifact_manifest_verifies_every_persisted_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["source_images_mutated"] is False
    assert manifest["split_manifest_created"] is False
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r3_implementation_has_no_training_split_test_or_delete_operations():
    paths = [
        ROOT / "src/practice_2_2/r3_grouping.py",
        ROOT / "scripts/run_r3_grouping.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "optimizer" not in source
    assert "train_test_split" not in source
    assert "random_split" not in source
    assert "DataLoader(test" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
