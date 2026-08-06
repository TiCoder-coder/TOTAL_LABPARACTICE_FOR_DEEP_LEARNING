from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r4_split import (
    assign_component_splits,
    build_asset_manifest,
    build_leakage_report,
    build_r4_report,
    build_split_fingerprints,
    build_split_summary,
    build_test_guard,
    load_r4_policy,
    write_json,
    write_manifest_csv,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-inventory", type=Path, required=True)
    parser.add_argument("--r3-groups", type=Path, required=True)
    parser.add_argument("--r3-accepted-edges", type=Path, required=True)
    parser.add_argument("--r3-review-queue", type=Path, required=True)
    parser.add_argument("--r3-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r4_policy()
    inventory = json.loads(args.r2_inventory.read_text())
    component_document = json.loads(args.r3_groups.read_text())
    components = component_document["components"]
    accepted_edges = json.loads(args.r3_accepted_edges.read_text())
    review_edges = json.loads(args.r3_review_queue.read_text())
    r3_report = json.loads(args.r3_verification.read_text())
    assignments = assign_component_splits(components, policy)
    repeated_assignments = assign_component_splits(list(reversed(components)), policy)
    if assignments != repeated_assignments:
        raise RuntimeError("R4 component split is not deterministic")
    rows = build_asset_manifest(
        components, inventory, assignments, lineage_authorized=False
    )
    summary = build_split_summary(rows, policy)
    leakage, cross_split_review = build_leakage_report(
        rows, accepted_edges, review_edges
    )
    report = build_r4_report(rows, summary, leakage, r3_report, policy)
    if report["gate_passed"]:
        rows = build_asset_manifest(
            components, inventory, assignments, lineage_authorized=True
        )
        summary = build_split_summary(rows, policy)
    fingerprints = build_split_fingerprints(rows, components)
    guard = build_test_guard(report, fingerprints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json_path = write_json(
        rows, args.output_dir / "candidate_split_manifest.json"
    )
    manifest_csv_path = write_manifest_csv(
        rows, args.output_dir / "candidate_split_manifest.csv"
    )
    fingerprints["manifest_json_sha256"] = file_sha256(manifest_json_path)
    fingerprints["manifest_csv_sha256"] = file_sha256(manifest_csv_path)
    artifact_paths = [
        manifest_json_path,
        manifest_csv_path,
        write_json(summary, args.output_dir / "split_summary.json"),
        write_json(fingerprints, args.output_dir / "split_fingerprints.json"),
        write_json(leakage, args.output_dir / "leakage_report.json"),
        write_json(
            cross_split_review,
            args.output_dir / "review_only_cross_split_pairs.json",
        ),
        write_json(guard, args.output_dir / "test_guard.json"),
        write_json(report, args.output_dir / "split_report.json"),
        write_json(report, args.report_output),
    ]
    root = get_practice_2_2_root().resolve()
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "test_access_count": 0,
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
    output = {
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "assets": report["assets"],
        "components": report["components"],
        "review_only_cross_split_pair_count": report[
            "review_only_cross_split_pair_count"
        ],
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
