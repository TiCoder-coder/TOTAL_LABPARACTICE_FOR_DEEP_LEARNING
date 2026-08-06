import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from practice_2_2.r2_provenance import (
    build_provenance_record,
    evaluate_product_candidate,
    load_r2_policy,
    select_packshot_candidates,
    write_immutable_asset,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r2_provenance_semantic_audit_v1"


def test_r2_policy_requires_provenance_and_prohibits_destructive_collection():
    policy = load_r2_policy()
    assert policy["raw_assets_immutable"] is True
    assert policy["automatic_deletion_allowed"] is False
    assert policy["detail_gallery_allowed"] is False
    assert policy["minimum_unique_products_per_class"] == 100
    assert len(policy["required_provenance_fields"]) == 13
    assert len(policy["classes"]) == 10


def test_toner_product_filter_rejects_non_cosmetic_query_matches():
    policy = load_r2_policy()
    rejected = evaluate_product_candidate(
        "toner",
        "Gói dưỡng hoa tươi lâu",
        "Phụ kiện cắm hoa",
        policy,
    )
    accepted = evaluate_product_candidate(
        "toner",
        "Toner nước cân bằng da mặt",
        "Chăm sóc da mặt",
        policy,
    )
    assert rejected["accepted"] is False
    assert rejected["negative_terms"]
    assert accepted["accepted"] is True


def test_packshot_selection_is_bounded_and_does_not_accept_detail_payload():
    policy = load_r2_policy()
    product = {
        "thumbnail_url": "https://example.com/thumbnail.jpg",
        "images": [
            {"base_url": "https://example.com/primary.jpg"},
            {"base_url": "https://example.com/second.jpg"},
        ],
    }
    selected = select_packshot_candidates(product, policy)
    assert len(selected) == 2
    assert [item["image_role"] for item in selected] == [
        "primary_thumbnail",
        "primary_gallery",
    ]
    assert all("detail" not in item for item in selected)


def test_immutable_asset_and_sidecar_are_idempotent(tmp_path):
    policy = load_r2_policy()
    content = b"immutable-r2-image-bytes"
    digest = hashlib.sha256(content).hexdigest()
    record = build_provenance_record(
        class_name="toner",
        product_id="123",
        listing_url="https://tiki.vn/product-p123.html",
        image_url="https://example.com/image.jpg",
        query=policy["classes"]["toner"]["query"],
        category_id="456",
        category_name="Chăm sóc da mặt",
        image_index=0,
        image_role="primary_thumbnail",
        crawl_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        raw_sha256=digest,
        policy=policy,
    )
    first = write_immutable_asset(
        content, ".jpg", record, tmp_path / "raw", tmp_path / "metadata"
    )
    second = write_immutable_asset(
        content, ".jpg", record, tmp_path / "raw", tmp_path / "metadata"
    )
    assert first == second
    assert first["raw_path"].read_bytes() == content
    assert json.loads(first["metadata_path"].read_text()) == record
    with pytest.raises(RuntimeError, match="raw content hash"):
        write_immutable_asset(
            b"changed", ".jpg", record, tmp_path / "raw", tmp_path / "metadata"
        )


def test_r2_generated_audit_matches_canonical_dataset_and_stays_blocked():
    report = json.loads((ROOT / "docs/R2_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["files"] == 3202
    assert report["original_files"] == 2896
    assert report["generated_files_excluded"] == 306
    assert report["invalid_images"] == 0
    assert report["dataset"]["directory_sha256"] == (
        "fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5"
    )
    assert report["provenance_coverage"] == 0.0
    assert report["semantic_review_queue_count"] == 2896
    assert report["cross_label_visual_pair_count"] == 325
    assert report["source_images_mutated"] is False
    assert report["automatic_deletion_performed"] is False
    assert report["test_loader_constructed"] is False
    assert report["test_evaluated"] is False


def test_r2_review_queue_covers_every_original_without_generated_files():
    queue = json.loads((ARTIFACT_ROOT / "semantic_review_queue.json").read_text())
    assert len(queue) == 2896
    assert len({item["asset_id"] for item in queue}) == 2896
    assert all("_aug" not in Path(item["relative_path"]).stem for item in queue)
    assert all(item["review_status"] == "pending" for item in queue)
    assert all(item["decision"] is None for item in queue)


def test_r2_cross_label_pairs_are_cross_class_and_pending_review():
    policy = load_r2_policy()
    pairs = json.loads((ARTIFACT_ROOT / "cross_label_visual_pairs.json").read_text())
    assert len(pairs) == 325
    assert all(item["left_class_name"] != item["right_class_name"] for item in pairs)
    assert all(
        item["decoded_pixel_exact"]
        or item["dhash_distance"]
        <= policy["quality_thresholds"]["cross_label_dhash_distance"]
        for item in pairs
    )
    assert all(item["review_status"] == "pending" for item in pairs)


def test_r2_artifact_manifest_verifies_every_persisted_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["source_images_mutated"] is False
    assert manifest["dataset_directory_sha256"] == (
        "fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5"
    )
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r2_implementation_has_no_training_test_or_delete_operations():
    paths = [
        ROOT / "src/practice_2_2/r2_provenance.py",
        ROOT / "src/practice_2_2/r2_semantic_audit.py",
        ROOT / "scripts/run_r2_audit.py",
        ROOT / "scripts/collect_tiki_r2.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    assert "torch" not in source
    assert "DataLoader" not in source
    assert "optimizer" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
    assert "PRODUCT_DETAIL_URL" not in source
