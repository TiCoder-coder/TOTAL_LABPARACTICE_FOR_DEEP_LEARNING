from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r2_provenance import load_r2_policy
from practice_2_2.r2_semantic_audit import (
    build_r2_report,
    build_semantic_review_queue,
    cross_label_visual_pairs,
    inventory_legacy_dataset,
    write_json,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--r1-pilot", type=Path, required=True)
    parser.add_argument("--r0-verification", type=Path, required=True)
    parser.add_argument("--r1-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r2_policy()
    records = inventory_legacy_dataset(
        args.dataset_root, args.metadata_root, policy
    )
    pairs = cross_label_visual_pairs(
        records, policy["quality_thresholds"]["cross_label_dhash_distance"]
    )
    historical = get_practice_2_2_root() / policy[
        "historical_review_candidates_path"
    ]
    queue = build_semantic_review_queue(
        records,
        pairs,
        args.r1_pilot,
        historical,
        policy,
    )
    report = build_r2_report(
        args.dataset_root,
        records,
        pairs,
        queue,
        args.r0_verification,
        args.r1_verification,
        historical,
        policy=policy,
    )
    artifact_paths = [
        write_json(records, args.output_dir / "legacy_inventory.json"),
        write_json(pairs, args.output_dir / "cross_label_visual_pairs.json"),
        write_json(queue, args.output_dir / "semantic_review_queue.json"),
        write_json(report, args.output_dir / "audit_report.json"),
        write_json(report, args.report_output),
    ]
    root = get_practice_2_2_root().resolve()
    artifact_manifest = {
        "schema_version": 1,
        "audit_lineage": "r2_provenance_semantic_audit_v1",
        "policy_version": policy["policy_version"],
        "dataset_directory_sha256": report["dataset"]["directory_sha256"],
        "source_images_mutated": False,
        "artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
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
        "files": report["files"],
        "original_files": report["original_files"],
        "generated_files_excluded": report["generated_files_excluded"],
        "semantic_review_queue_count": report["semantic_review_queue_count"],
        "cross_label_visual_pair_count": report[
            "cross_label_visual_pair_count"
        ],
        "provenance_coverage": report["provenance_coverage"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(summary, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
