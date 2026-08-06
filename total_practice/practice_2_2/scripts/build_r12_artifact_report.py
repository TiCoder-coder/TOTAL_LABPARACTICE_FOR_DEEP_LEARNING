from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r12_artifact_report import (
    audit_notebook_execution,
    audit_notebook_links,
    audit_notebook_safety,
    build_artifact_manifest,
    build_notebook,
    build_r12_report,
    build_source_registry,
    execute_notebook,
    export_notebook_html,
    load_r12_policy,
    verify_source_registry,
    write_json,
)
from practice_2_2.resources import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    root = get_practice_2_2_root().resolve()
    policy = load_r12_policy(args.policy)
    canonical_path = root / policy["canonical_notebook"]["path"]
    canonical_hash_before = file_sha256(canonical_path)
    expected_hash = policy["canonical_notebook"]["expected_sha256"]
    if canonical_hash_before != expected_hash:
        raise RuntimeError("R12 canonical notebook hash does not match policy")
    artifact_root = root / policy["outputs"]["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    policy_snapshot_path = write_json(policy, artifact_root / "policy_snapshot.json")
    registry = build_source_registry(root, policy)
    registry_path = write_json(
        registry, artifact_root / "source_artifact_registry.json"
    )
    registry_audit = verify_source_registry(root, registry)
    registry_audit_path = write_json(
        registry_audit, artifact_root / "registry_verification.json"
    )
    notebook_path = root / policy["outputs"]["notebook"]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(policy)
    executed = execute_notebook(
        notebook,
        root,
        policy["execution"]["kernel_name"],
        policy["execution"]["timeout_seconds"],
    )
    nbformat.write(executed, notebook_path)
    execution_audit = audit_notebook_execution(executed)
    safety_audit = audit_notebook_safety(executed)
    link_audit = audit_notebook_links(executed)
    execution_path = write_json(
        execution_audit, artifact_root / "notebook_execution_verification.json"
    )
    safety_path = write_json(
        safety_audit, artifact_root / "notebook_safety_verification.json"
    )
    links_path = write_json(
        link_audit, artifact_root / "local_link_verification.json"
    )
    html_path = export_notebook_html(
        executed, root / policy["outputs"]["html"]
    )
    report = build_r12_report(
        root,
        policy,
        registry,
        registry_audit,
        execution_audit,
        safety_audit,
        link_audit,
        notebook_path,
        html_path,
        canonical_hash_before,
    )
    report_path = write_json(report, root / policy["outputs"]["verification_report"])
    hash_path = write_json(
        {
            "schema_version": 1,
            "notebook_path": report["notebook_path"],
            "notebook_sha256": report["notebook_sha256"],
            "html_path": report["html_path"],
            "html_sha256": report["html_sha256"],
            "canonical_notebook_sha256": report["canonical_notebook_sha256"],
        },
        artifact_root / "notebook_export_hashes.json",
    )
    manifest_paths = [
        policy_snapshot_path,
        registry_path,
        registry_audit_path,
        execution_path,
        safety_path,
        links_path,
        hash_path,
        report_path,
        notebook_path,
        html_path,
    ]
    manifest = build_artifact_manifest(root, manifest_paths, report)
    write_json(manifest, artifact_root / "artifact_manifest.json")
    if file_sha256(canonical_path) != canonical_hash_before:
        raise RuntimeError("R12 mutated the canonical notebook")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
