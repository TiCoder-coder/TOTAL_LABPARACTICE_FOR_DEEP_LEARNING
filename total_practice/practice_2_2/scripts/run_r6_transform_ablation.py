from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r6_transform_ablation import (
    audit_recipes,
    build_r6_report,
    build_review_template,
    create_transformed_contact_sheets,
    load_r6_policy,
    select_train_rows,
    write_json,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r4-manifest", type=Path, required=True)
    parser.add_argument("--r5-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r6_policy()
    manifest = json.loads(args.r4_manifest.read_text())
    r5_report = json.loads(args.r5_verification.read_text())
    train_rows, heldout_ids = select_train_rows(manifest, policy)
    records, summaries, accessed = audit_recipes(
        train_rows, args.dataset_root, policy
    )
    figure_paths, contact_sheet_index = create_transformed_contact_sheets(
        train_rows, args.dataset_root, args.output_dir, accessed, policy
    )
    review_template = build_review_template(policy)
    report = build_r6_report(
        train_rows,
        heldout_ids,
        accessed,
        summaries,
        review_template,
        r5_report,
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
        write_json(policy["recipes"], args.output_dir / "recipe_specs.json"),
        write_json(records, args.output_dir / "transform_audit.json"),
        write_json(summaries, args.output_dir / "transform_summary.json"),
        write_json(
            contact_sheet_index, args.output_dir / "contact_sheet_index.json"
        ),
        write_json(
            review_template, args.output_dir / "transform_review_template.json"
        ),
        write_json(
            content_access_manifest,
            args.output_dir / "content_access_manifest.json",
        ),
        write_json(report, args.output_dir / "transform_ablation_report.json"),
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
        "validation_content_access_count": report["validation_content_access_count"],
        "test_content_access_count": report["test_content_access_count"],
        "selected_recipe_id": report["selected_recipe_id"],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
