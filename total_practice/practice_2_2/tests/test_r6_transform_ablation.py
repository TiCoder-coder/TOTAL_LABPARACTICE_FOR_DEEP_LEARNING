import json
from pathlib import Path

import numpy as np
from PIL import Image

from practice_2_2.r6_transform_ablation import (
    RECIPE_FIELDS,
    apply_recipe,
    load_r6_policy,
    normalize_image,
    select_train_rows,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r6_transform_ablation_s42_v1"
R4_ROOT = ROOT / "artifacts/new_work/v3_product_visual_group_s42_v1"


def test_r6_policy_is_train_only_non_training_and_fail_closed():
    policy = load_r6_policy()
    assert policy["seed"] == 42
    assert policy["authorized_content_split"] == "Train"
    assert policy["predecessor_gate_required"] is True
    assert policy["automatic_recipe_selection_allowed"] is False
    assert policy["validation_content_access_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["model_training_allowed"] is False
    assert tuple(policy["recipes"]) == ("T0", "T1", "T2", "T3", "T4")


def test_r6_every_child_recipe_changes_exactly_one_parent_field():
    recipes = load_r6_policy()["recipes"]
    for recipe_id in ("T1", "T2", "T3", "T4"):
        recipe = recipes[recipe_id]
        parent = recipes[recipe["parent"]]
        changed = [field for field in RECIPE_FIELDS if recipe[field] != parent[field]]
        assert changed == [recipe["single_change"]]


def test_r6_train_selection_is_exact_and_disjoint_from_heldout():
    manifest = json.loads((R4_ROOT / "candidate_split_manifest.json").read_text())
    train_rows, heldout_ids = select_train_rows(manifest)
    train_ids = {row["asset_id"] for row in train_rows}
    assert len(train_ids) == 2034
    assert len(heldout_ids) == 862
    assert not train_ids & heldout_ids


def test_r6_recipe_application_is_deterministic_and_normalization_is_finite():
    values = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    image = Image.fromarray((values % 256).astype(np.uint8))
    first_train, first_evaluation, first_metrics = apply_recipe(
        image, "synthetic_asset", "T2"
    )
    second_train, second_evaluation, second_metrics = apply_recipe(
        image, "synthetic_asset", "T2"
    )
    assert first_train.tobytes() == second_train.tobytes()
    assert first_evaluation.tobytes() == second_evaluation.tobytes()
    assert first_metrics == second_metrics
    normalized = normalize_image(first_train)
    assert normalized.shape == (3, 224, 224)
    assert np.isfinite(normalized.numpy()).all()


def test_r6_t0_t1_share_train_transform_and_change_only_evaluation():
    image = Image.new("RGB", (224, 224), "white")
    values = np.asarray(image).copy()
    values[:, :30] = [20, 40, 60]
    image = Image.fromarray(values)
    t0_train, t0_evaluation, _ = apply_recipe(image, "edge_asset", "T0")
    t1_train, t1_evaluation, _ = apply_recipe(image, "edge_asset", "T1")
    assert t0_train.tobytes() == t1_train.tobytes()
    assert t0_evaluation.tobytes() != t1_evaluation.tobytes()


def test_r6_transform_audit_and_summary_cover_all_recipes():
    records = json.loads((ARTIFACT_ROOT / "transform_audit.json").read_text())
    summary = json.loads((ARTIFACT_ROOT / "transform_summary.json").read_text())
    assert len(records) == 2034 * 5
    assert len({(item["asset_id"], item["recipe_id"]) for item in records}) == len(
        records
    )
    assert [item["recipe_id"] for item in summary] == ["T0", "T1", "T2", "T3", "T4"]
    by_id = {item["recipe_id"]: item for item in summary}
    assert by_id["T0"]["evaluation_area_retention"] == (224 / 256) ** 2
    assert all(by_id[item]["evaluation_area_retention"] == 1.0 for item in ("T1", "T2", "T3", "T4"))
    assert by_id["T2"]["mean_train_crop_area_retention"] > by_id["T1"]["mean_train_crop_area_retention"]
    assert by_id["T3"]["horizontal_flip_count"] == 0
    assert by_id["T4"]["random_erasing_count"] == 0
    assert all(item["normalization_nonfinite_count"] == 0 for item in summary)


def test_r6_content_access_manifest_contains_train_only():
    access = json.loads((ARTIFACT_ROOT / "content_access_manifest.json").read_text())
    manifest = json.loads((R4_ROOT / "candidate_split_manifest.json").read_text())
    train_ids = {row["asset_id"] for row in manifest if row["split"] == "Train"}
    assert set(access["accessed_asset_ids"]) == train_ids
    assert access["accessed_asset_count"] == 2034
    assert access["validation_content_access_count"] == 0
    assert access["test_content_access_count"] == 0


def test_r6_contact_sheets_and_review_template_are_complete():
    index = json.loads((ARTIFACT_ROOT / "contact_sheet_index.json").read_text())
    review = json.loads((ARTIFACT_ROOT / "transform_review_template.json").read_text())
    assert index["columns"] == ["original", "train_transform", "evaluation_transform"]
    assert len(index["sheets"]) == 50
    assert all(len(item["asset_ids"]) == 12 for item in index["sheets"])
    assert len(list((ARTIFACT_ROOT / "transformed_contact_sheets").rglob("*.png"))) == 50
    assert len(review) == 50
    assert all(item["review_status"] == "pending" for item in review)
    assert all(item["reviewer_id"] is None for item in review)


def test_r6_report_does_not_select_or_advance_a_recipe():
    report = json.loads((ROOT / "docs/R6_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["train_manifest_assets"] == 2034
    assert report["train_content_assets_accessed"] == 2034
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["train_only_content_isolation_passed"] is True
    assert report["selected_recipe_id"] is None
    assert report["advanced_recipe_count"] == 0
    assert report["automatic_recipe_selection_performed"] is False
    assert report["model_training_performed"] is False
    assert report["validation_evaluated"] is False
    assert report["test_evaluated"] is False
    assert report["successor_phase_executed"] is False


def test_r6_artifact_manifest_verifies_every_output():
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


def test_r6_implementation_has_no_training_loader_evaluation_or_delete_operations():
    paths = [
        ROOT / "src/practice_2_2/r6_transform_ablation.py",
        ROOT / "scripts/run_r6_transform_ablation.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "optimizer" not in source
    assert "DataLoader" not in source
    assert "backward(" not in source
    assert "evaluate(" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
