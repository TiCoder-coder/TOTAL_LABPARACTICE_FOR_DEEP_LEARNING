from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r5_data_readiness import (
    audit_train_images,
    build_class_readiness,
    build_embedding_audits,
    build_r5_report,
    build_train_review_queue,
    create_contact_sheets,
    load_r5_policy,
    select_train_rows,
    write_json,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r2-inventory", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r4-verification", type=Path, required=True)
    parser.add_argument("--r3-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r5_policy()
    inventory = json.loads(args.r2_inventory.read_text())
    manifest = json.loads(args.r4_manifest.read_text())
    r4_report = json.loads(args.r4_verification.read_text())
    train_rows, heldout_ids = select_train_rows(manifest, policy)
    audited, pixel_statistics, accessed = audit_train_images(
        train_rows, inventory, args.dataset_root, policy
    )
    stored = np.load(args.r3_embeddings, allow_pickle=False)
    cluster_report, neighbor_report, _, _ = build_embedding_audits(
        train_rows,
        stored["asset_ids"].tolist(),
        stored["embeddings"],
        policy,
    )
    class_readiness = build_class_readiness(
        train_rows, audited, cluster_report, policy
    )
    review_queue = build_train_review_queue(
        train_rows, audited, neighbor_report, policy
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths, contact_sheet_index = create_contact_sheets(
        train_rows,
        cluster_report,
        neighbor_report,
        args.dataset_root,
        args.output_dir,
        accessed,
        policy,
    )
    report = build_r5_report(
        train_rows,
        heldout_ids,
        accessed,
        class_readiness,
        review_queue,
        r4_report,
        policy,
    )
    content_access_manifest = {
        "schema_version": 1,
        "authorized_split": "Train",
        "accessed_asset_ids": sorted(accessed),
        "accessed_asset_count": len(accessed),
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
    }
    artifact_paths = [
        write_json(audited, args.output_dir / "train_image_audit.json"),
        write_json(pixel_statistics, args.output_dir / "train_pixel_statistics.json"),
        write_json(class_readiness, args.output_dir / "class_readiness.json"),
        write_json(cluster_report, args.output_dir / "embedding_cluster_report.json"),
        write_json(neighbor_report, args.output_dir / "embedding_neighbor_report.json"),
        write_json(review_queue, args.output_dir / "train_review_queue.json"),
        write_json(contact_sheet_index, args.output_dir / "contact_sheet_index.json"),
        write_json(
            content_access_manifest,
            args.output_dir / "content_access_manifest.json",
        ),
        write_json(report, args.output_dir / "data_readiness_report.json"),
        write_json(report, args.report_output),
        *figure_paths,
    ]
    root = get_practice_2_2_root().resolve()
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "content_authority": "Train",
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "artifacts": [
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(artifact_manifest, args.output_dir / "artifact_manifest.json")
    output = {
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "train_assets": report["train_manifest_assets"],
        "train_content_assets_accessed": report["train_content_assets_accessed"],
        "validation_content_access_count": report["validation_content_access_count"],
        "test_content_access_count": report["test_content_access_count"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
