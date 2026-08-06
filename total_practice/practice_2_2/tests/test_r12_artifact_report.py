import ast
import json
from pathlib import Path

import nbformat

from practice_2_2.r12_artifact_report import (
    audit_notebook_execution,
    audit_notebook_links,
    audit_notebook_safety,
    build_source_registry,
    load_r12_policy,
    verify_source_registry,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r12_artifact_report_v1"


def test_r12_policy_is_artifact_only_and_fail_closed():
    policy = load_r12_policy()
    assert policy["predecessor_phase"] == "R11"
    assert policy["predecessor_gate_required"] is True
    assert policy["r11_completion_required"] is True
    assert policy["canonical_notebook"]["mutation_allowed"] is False
    assert policy["execution"]["training_allowed"] is False
    assert policy["execution"]["test_evaluation_allowed"] is False
    assert policy["execution"]["test_loader_construction_allowed"] is False
    assert policy["execution"]["network_access_allowed"] is False
    assert policy["execution"]["persisted_artifact_writes_allowed"] is False
    assert policy["successor_phase_authorized"] is False


def test_r12_registry_covers_r0_to_r10_and_verifies_hashes():
    policy = load_r12_policy()
    registry = build_source_registry(ROOT, policy)
    assert [
        next(
            entry["phase"]
            for entry in registry["source_artifacts"]
            if entry["artifact_id"] == artifact_id
        )
        for artifact_id in registry["phase_report_ids"]
    ] == [f"R{index}" for index in range(11)]
    audit = verify_source_registry(ROOT, registry)
    assert audit["passed"] is True
    assert audit["verified_path_count"] == audit["registry_path_count"]


def test_r12_notebook_executes_without_errors_and_has_safe_code():
    notebook = nbformat.read(ROOT / "notebooks/05_product_visualsafe_report.ipynb", as_version=4)
    execution = audit_notebook_execution(notebook)
    safety = audit_notebook_safety(notebook)
    assert execution["passed"] is True
    assert execution["error_output_count"] == 0
    assert execution["executed_code_cell_count"] == execution["code_cell_count"]
    assert safety["passed"] is True
    assert safety["forbidden_imports"] == []
    assert safety["forbidden_calls"] == []
    assert safety["training_operations_detected"] is False
    assert safety["test_evaluation_operations_detected"] is False


def test_r12_notebook_links_resolve_to_declared_anchors():
    notebook = nbformat.read(ROOT / "notebooks/05_product_visualsafe_report.ipynb", as_version=4)
    audit = audit_notebook_links(notebook)
    assert audit["passed"] is True
    assert audit["local_link_count"] >= 7
    assert audit["missing_anchor_targets"] == []


def test_r12_notebook_reports_missing_r11_without_test_metrics():
    notebook = nbformat.read(ROOT / "notebooks/05_product_visualsafe_report.ipynb", as_version=4)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "R11 was not executed" in source
    assert "no new-lineage Test Accuracy" in source
    assert "New-lineage Test evaluations performed by this notebook: 0" in source
    assert "synthetic protocol fixture" in source
    assert "must not be presented as measured Validation performance" in source


def test_r12_report_build_passes_but_lineage_gate_is_blocked():
    report = json.loads((ROOT / "docs/R12_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["report_build_checks_passed"] is True
    assert report["failed_report_build_checks"] == []
    assert report["blocked_reasons"] == [
        "R10_GATE_NOT_PASSED",
        "REPEATED_SEED_RESULTS_UNAVAILABLE",
        "FINAL_CHECKPOINT_NOT_FROZEN",
        "R11_NOT_EXECUTED",
    ]
    assert report["r11_executed"] is False
    assert report["one_time_test_evaluation_available"] is False
    assert report["test_evaluation_count"] == 0
    assert report["real_training_performed"] is False
    assert report["test_content_access_count"] == 0
    assert report["successor_phase_executed"] is False


def test_r12_notebook_and_html_hashes_match_report():
    report = json.loads((ROOT / "docs/R12_VERIFICATION.json").read_text())
    notebook = ROOT / report["notebook_path"]
    html = ROOT / report["html_path"]
    assert file_sha256(notebook) == report["notebook_sha256"]
    assert file_sha256(html) == report["html_sha256"]


def test_r12_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["status"] == "blocked"
    assert manifest["gate_passed"] is False
    assert manifest["report_build_checks_passed"] is True
    assert manifest["r11_executed"] is False
    assert manifest["real_training_performed"] is False
    assert manifest["test_evaluation_count"] == 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r12_entrypoint_contains_no_training_or_test_loader_imports():
    paths = [
        ROOT / "src/practice_2_2/r12_artifact_report.py",
        ROOT / "scripts/build_r12_artifact_report.py",
    ]
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "DataLoader" not in imported_names
        assert "torch.utils.data" not in imported_names
        assert ".backward(" not in source
        assert "optimizer.step(" not in source


def test_r12_did_not_mutate_canonical_notebook():
    report = json.loads((ROOT / "docs/R12_VERIFICATION.json").read_text())
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    assert report["canonical_notebook_mutated"] is False
    assert file_sha256(notebook) == report["canonical_notebook_sha256"]
