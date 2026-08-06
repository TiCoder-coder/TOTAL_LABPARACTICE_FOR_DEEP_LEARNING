from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r3_grouping import (
    _identity_edges,
    build_group_report,
    build_groups,
    build_visual_edges,
    extract_pretrained_embeddings,
    load_r3_policy,
    write_json,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r2-inventory", type=Path, required=True)
    parser.add_argument("--r2-verification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r3_policy()
    inventory = json.loads(args.r2_inventory.read_text())
    records = sorted(
        (
            record
            for record in inventory
            if not record["is_generated"] and record["image_error"] is None
        ),
        key=lambda item: item["asset_id"],
    )
    r2_report = json.loads(args.r2_verification.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "pretrained_embeddings.npz"
    embedding_metadata_path = args.output_dir / "embedding_metadata.json"
    expected_asset_ids = [record["asset_id"] for record in records]
    reusable = embedding_path.is_file() and embedding_metadata_path.is_file()
    if reusable:
        stored = np.load(embedding_path, allow_pickle=False)
        asset_ids = stored["asset_ids"].tolist()
        embeddings = stored["embeddings"]
        embedding_metadata = json.loads(embedding_metadata_path.read_text())
        reusable = (
            asset_ids == expected_asset_ids
            and embeddings.shape == (len(records), policy["visual_evidence"]["embedding_dimension"])
            and embedding_metadata["checkpoint_sha256"] == file_sha256(args.checkpoint)
        )
    if not reusable:
        asset_ids, embeddings, embedding_metadata = extract_pretrained_embeddings(
            records, args.dataset_root, args.checkpoint
        )
        np.savez_compressed(
            embedding_path,
            asset_ids=np.asarray(asset_ids),
            embeddings=embeddings,
        )
    visual_edges, phashes = build_visual_edges(
        records, args.dataset_root, asset_ids, embeddings, policy
    )
    identity_edges = _identity_edges(records, policy["identity_keys"])
    components, accepted_edges, review_edges = build_groups(
        records, identity_edges, visual_edges, policy
    )
    manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "seed": policy["seed"],
        "eligibility": "provisional_original_pending_r2_acceptance",
        "components": components,
    }
    report = build_group_report(
        records,
        components,
        accepted_edges,
        review_edges,
        embedding_metadata,
        r2_report,
        r2_report["dataset"]["directory_sha256"],
    )
    artifact_paths = [
        embedding_path,
        write_json(embedding_metadata, embedding_metadata_path),
        write_json(phashes, args.output_dir / "phashes.json"),
        write_json(visual_edges, args.output_dir / "visual_edges.json"),
        write_json(accepted_edges, args.output_dir / "accepted_edges.json"),
        write_json(review_edges, args.output_dir / "review_queue.json"),
        write_json(manifest, args.output_dir / "group_manifest.json"),
        write_json(report, args.output_dir / "group_report.json"),
        write_json(report, args.report_output),
    ]
    root = get_practice_2_2_root().resolve()
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "dataset_directory_sha256": report["dataset_directory_sha256"],
        "source_images_mutated": False,
        "split_manifest_created": False,
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
    summary = {
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "eligible_original_assets": report["eligible_original_assets"],
        "component_count": report["component_count"],
        "pending_review_edge_count": report["pending_review_edge_count"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
