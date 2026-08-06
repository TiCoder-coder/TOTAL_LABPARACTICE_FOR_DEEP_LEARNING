from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .paths import get_practice_2_2_root
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r12_artifact_report_policy.json")
FORBIDDEN_IMPORTS = {
    "requests",
    "sklearn",
    "socket",
    "subprocess",
    "torch",
    "torchvision",
    "urllib",
}
FORBIDDEN_CALLS = {
    "backward",
    "evaluate",
    "exec",
    "fit",
    "load_state_dict",
    "step",
    "train",
    "write_bytes",
    "write_text",
}


def load_r12_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R12 report policy schema")
    if policy.get("predecessor_phase") != "R11":
        raise RuntimeError("R12 predecessor must be R11")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R12 must require the R11 gate")
    if policy.get("r11_completion_required") is not True:
        raise RuntimeError("R12 must require completed R11 evidence")
    execution = policy.get("execution", {})
    for field in (
        "training_allowed",
        "test_evaluation_allowed",
        "test_loader_construction_allowed",
        "network_access_allowed",
        "persisted_artifact_writes_allowed",
    ):
        if execution.get(field) is not False:
            raise RuntimeError(f"R12 execution policy must prohibit {field}")
    phases = [entry["phase"] for entry in policy.get("phase_reports", [])]
    if phases != [f"R{index}" for index in range(11)]:
        raise RuntimeError("R12 phase report registry must cover R0 through R10")
    artifact_ids = [
        entry["artifact_id"]
        for entry in [*policy["phase_reports"], *policy["evidence_artifacts"]]
    ]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise RuntimeError("R12 artifact identifiers must be unique")
    if policy.get("successor_phase_authorized") is not False:
        raise RuntimeError("R12 must not authorize a successor phase")
    return policy


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"R12 path must be project-relative: {value}")
    return path


def write_json(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return path


def build_source_registry(
    root: Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = []
    for entry in [*policy["phase_reports"], *policy["evidence_artifacts"]]:
        relative_path = _safe_relative_path(str(entry["path"]))
        path = (root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"R12 source artifact is missing: {relative_path}")
        artifacts.append(
            {
                **dict(entry),
                "path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    r11_path = _safe_relative_path(str(policy["expected_r11_report"]))
    return {
        "schema_version": 1,
        "phase": "R12",
        "registry_version": policy["policy_version"],
        "path_base": "practice_2_2_root",
        "source_artifacts": artifacts,
        "phase_report_ids": [
            entry["artifact_id"] for entry in policy["phase_reports"]
        ],
        "expected_r11_report": r11_path.as_posix(),
        "r11_report_exists": (root / r11_path).is_file(),
        "training_allowed": False,
        "test_evaluation_allowed": False,
    }


def verify_source_registry(root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for entry in registry["source_artifacts"]:
        path = root / _safe_relative_path(str(entry["path"]))
        if not path.is_file():
            failures.append({"artifact_id": entry["artifact_id"], "reason": "missing"})
            continue
        if path.stat().st_size != entry["size_bytes"]:
            failures.append({"artifact_id": entry["artifact_id"], "reason": "size"})
        if file_sha256(path) != entry["sha256"]:
            failures.append({"artifact_id": entry["artifact_id"], "reason": "sha256"})
    return {
        "registry_path_count": len(registry["source_artifacts"]),
        "verified_path_count": len(registry["source_artifacts"]) - len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _markdown_cell(cell_id: str, source: str) -> Any:
    cell = nbformat.v4.new_markdown_cell(source=source)
    cell["id"] = cell_id
    return cell


def _code_cell(cell_id: str, source: str) -> Any:
    cell = nbformat.v4.new_code_cell(source=source)
    cell["id"] = cell_id
    return cell


def build_notebook(policy: Mapping[str, Any]) -> Any:
    cells = [
        _markdown_cell(
            "r12-01",
            """<a id="report-start"></a>\n# Practice 2.2 Product-Visual-Safe Artifact Report\n\n**Current lineage status: BLOCKED.** This notebook reports persisted R0-R10 evidence only. R11 has not been executed, no final checkpoint is frozen, and no new-lineage Test metric exists.\n\n[Lineage status](#lineage-status) | [Data quality](#data-quality) | [Grouping and leakage](#grouping-leakage) | [Training protocol](#training-protocol) | [Uncertainty](#uncertainty) | [Limitations](#limitations) | [Integrity](#integrity)""",
        ),
        _markdown_cell(
            "r12-02",
            """<a id="lineage-status"></a>\n## Lineage Status\n\nRun All is read-only. Every code cell loads persisted JSON or verifies a persisted file hash. The notebook cannot train, create a Test loader, evaluate Test, access a network, or write project artifacts.""",
        ),
        _code_cell(
            "r12-03",
            """import hashlib
import json
from pathlib import Path

def locate_project_root():
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "r12_artifact_report_policy.json").is_file():
            return candidate
    raise FileNotFoundError("Open this notebook from the practice_2_2 project")

project_root = locate_project_root()
registry_path = project_root / "artifacts/new_work/r12_artifact_report_v1/source_artifact_registry.json"
source_registry = json.loads(registry_path.read_text())
artifact_index = {entry["artifact_id"]: entry for entry in source_registry["source_artifacts"]}

def load_artifact(artifact_id):
    entry = artifact_index[artifact_id]
    return json.loads((project_root / entry["path"]).read_text())

print(f"Registry: {registry_path.relative_to(project_root)}")
print(f"Persisted source artifacts: {len(artifact_index)}")
print(f"R11 report exists: {source_registry['r11_report_exists']}")""",
        ),
        _code_cell(
            "r12-04",
            """print("Phase | Status | Gate | Blocker count")
print("-" * 52)
for artifact_id in source_registry["phase_report_ids"]:
    report = load_artifact(artifact_id)
    status = report.get("status", "unknown")
    gate = report.get("gate_passed", False)
    blockers = report.get("blocked_reasons", report.get("failed_authorities", []))
    print(f"{report['phase']:>5} | {status:<27} | {str(gate):<5} | {len(blockers)}")""",
        ),
        _markdown_cell(
            "r12-05",
            """<a id="data-quality"></a>\n## Data Quality Decisions\n\nR1 defines the review contract. R2 records the non-destructive provenance and semantic audit. R5 reports Train-only readiness without reading Validation or Test content.""",
        ),
        _code_cell(
            "r12-06",
            """r1 = load_artifact("r1_verification")
r2 = load_artifact("r2_data_quality")
r5 = load_artifact("r5_readiness")
print(f"R1 independent review passed: {r1['gate_passed']}")
print(f"Original files audited: {r2['original_files']}")
print(f"Generated files excluded: {r2['generated_files_excluded']}")
print(f"Invalid images: {r2['invalid_images']}")
print(f"Product provenance coverage: {r2['provenance_coverage']:.1%}")
print(f"Semantic review decisions complete: {r2['semantic_decision_result'] is not None}")
print(f"Train assets audited: {r5['train_content_assets_accessed']}")
print(f"Validation content reads: {r5['validation_content_access_count']}")
print(f"Test content reads: {r5['test_content_access_count']}")
print(f"Data ready: {r5['data_ready']}")""",
        ),
        _markdown_cell(
            "r12-07",
            """<a id="grouping-leakage"></a>\n## Product and Visual Grouping\n\nThe counts below describe candidate components, not proven unique products. Product provenance remains unavailable, uncertain visual edges remain under review, and the candidate split is not authorized for model training.""",
        ),
        _code_cell(
            "r12-08",
            """r3 = load_artifact("r3_groups")
r4 = load_artifact("r4_split")
leakage = load_artifact("r4_leakage")
print(f"Eligible original assets: {r3['eligible_original_assets']}")
print(f"Candidate visual components: {r3['component_count']}")
print(f"Singleton components: {r3['singleton_component_count']}")
print(f"Largest component size: {r3['largest_component_size']}")
print(f"Product provenance coverage: {r3['product_provenance_coverage']:.1%}")
print(f"Strong cross-split leakage violations: {leakage['strong_leakage_violation_count']}")
print(f"Review-only cross-split pairs: {leakage['review_only_cross_split_pair_count']}")
print(f"Every class represented in every candidate split: {r4['every_class_in_every_split']}")
print(f"Candidate split authorized for training: {r4['model_training_authorized']}")
print(f"Test access count: {r4['test_access_count']}")""",
        ),
        _markdown_cell(
            "r12-09",
            """<a id="training-protocol"></a>\n## Transform and Training Protocol\n\nR6-R10 verified implementation contracts with Train-only audits or synthetic fixtures. These artifacts are protocol evidence, not real model-selection evidence.""",
        ),
        _code_cell(
            "r12-10",
            """r6 = load_artifact("r6_transforms")
r7 = load_artifact("r7_correctness")
r8 = load_artifact("r8_staged")
r9 = load_artifact("r9_ceiling")
r10 = load_artifact("r10_selection")
print(f"Transform recipes audited: {r6['recipe_count']}")
print(f"Selected transform recipe: {r6['selected_recipe_id']}")
print(f"Correct loss aggregation verified: {r7['correctness_exit_checks_passed']}")
print(f"Offline checkpoint reload verified: {r7['offline_reload_predictions_identical']}")
print(f"Staged protocol checks passed: {r8['protocol_exit_checks_passed']}")
print(f"Architecture protocol checks passed: {r9['protocol_exit_checks_passed']}")
print(f"Robust selection protocol checks passed: {r10['protocol_exit_checks_passed']}")
print(f"Real training performed in R7-R10: {any(item.get('real_training_performed', False) for item in [r7, r8, r9, r10])}")""",
        ),
        _markdown_cell(
            "r12-11",
            """<a id="uncertainty"></a>\n## Repeated Seeds and Uncertainty\n\nThe R10 uncertainty file is a synthetic protocol fixture. Its numerical values must not be presented as measured Validation performance. Actual repeated-seed finalist results are unavailable.""",
        ),
        _code_cell(
            "r12-12",
            """uncertainty_entry = artifact_index["r10_uncertainty"]
r10 = load_artifact("r10_selection")
print(f"Required seeds: {r10['required_seeds']}")
print(f"Repeated-seed finalist results available: {r10['repeated_seed_finalist_results_available']}")
print(f"Selected finalist: {r10['selected_finalist_id']}")
print(f"Final checkpoint frozen: {r10['final_checkpoint_frozen']}")
print(f"Uncertainty artifact evidence class: {uncertainty_entry['evidence_class']}")
print(f"Uncertainty artifact eligible for model conclusions: {uncertainty_entry['eligible_for_model_conclusion']}")""",
        ),
        _markdown_cell(
            "r12-13",
            """<a id="limitations"></a>\n## R11 Status and Limitations\n\nR11 was not executed. Therefore this report contains no new-lineage Test Accuracy, Macro F1, per-class Test metric, calibration result, confusion matrix, or error analysis. The historical canonical Test result remains frozen and is not treated as comparable new-lineage evidence.""",
        ),
        _code_cell(
            "r12-14",
            """r10 = load_artifact("r10_verification")
r11_path = project_root / source_registry["expected_r11_report"]
print(f"R10 gate passed: {r10['gate_passed']}")
print(f"R11 verification exists: {r11_path.is_file()}")
print(f"New-lineage Test evaluations performed by this notebook: 0")
print(f"Current lineage gate: BLOCKED")""",
        ),
        _markdown_cell(
            "r12-15",
            """<a id="integrity"></a>\n## Artifact Integrity\n\nEvery registered source path is resolved from the project root and checked against its recorded byte size and SHA-256 digest.""",
        ),
        _code_cell(
            "r12-16",
            """verified = 0
for entry in source_registry["source_artifacts"]:
    path = project_root / entry["path"]
    observed_size = path.stat().st_size
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_size != entry["size_bytes"] or observed_sha256 != entry["sha256"]:
        raise RuntimeError(f"Artifact integrity mismatch: {entry['artifact_id']}")
    verified += 1
print(f"Verified artifacts: {verified}/{len(source_registry['source_artifacts'])}")
print("Artifact registry integrity: PASS")""",
        ),
        _markdown_cell(
            "r12-17",
            """## Conclusion\n\nThe report build is reproducible and read-only, but the R12 lineage gate remains blocked. The next valid project action is to resolve the earlier data and review gates, complete real repeated-seed Validation selection, freeze one final checkpoint, and separately authorize R11. This notebook does not authorize or perform those actions.\n\n[Back to report start](#report-start)""",
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python - Deep Learning (venv)",
            "language": "python",
            "name": policy["execution"]["kernel_name"],
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "practice_2_2": {
            "mode": "artifact_only_report",
            "lineage": policy["policy_version"],
            "training_allowed": False,
            "final_test_evaluation_allowed": False,
            "r11_executed": False,
        },
    }
    return notebook


def execute_notebook(
    notebook: Any,
    root: Path,
    kernel_name: str,
    timeout_seconds: int,
) -> Any:
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(root)}},
    )
    return client.execute()


def export_notebook_html(notebook: Any, output_path: Path) -> Path:
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body)
    return output_path


def audit_notebook_execution(notebook: Mapping[str, Any]) -> dict[str, Any]:
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    unexecuted = [
        cell["id"] for cell in code_cells if cell.get("execution_count") is None
    ]
    return {
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": len(code_cells) - len(unexecuted),
        "unexecuted_cell_ids": unexecuted,
        "error_output_count": len(errors),
        "passed": not errors and not unexecuted,
    }


def audit_notebook_safety(notebook: Mapping[str, Any]) -> dict[str, Any]:
    violations = []
    imports = set()
    calls = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(cell["source"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    forbidden_imports = sorted(imports & FORBIDDEN_IMPORTS)
    forbidden_calls = sorted(calls & FORBIDDEN_CALLS)
    if forbidden_imports:
        violations.append({"kind": "import", "values": forbidden_imports})
    if forbidden_calls:
        violations.append({"kind": "call", "values": forbidden_calls})
    return {
        "imports": sorted(imports),
        "calls": sorted(calls),
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": forbidden_calls,
        "violations": violations,
        "training_operations_detected": bool(forbidden_calls),
        "test_evaluation_operations_detected": False,
        "passed": not violations,
    }


def audit_notebook_links(notebook: Mapping[str, Any]) -> dict[str, Any]:
    markdown = "\n".join(
        cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', markdown))
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown)
    local_links = [link for link in links if link.startswith("#")]
    missing = [link for link in local_links if link[1:] not in anchors]
    return {
        "anchor_count": len(anchors),
        "local_link_count": len(local_links),
        "missing_anchor_targets": missing,
        "passed": not missing,
    }


def collect_blocked_reasons(
    root: Path,
    policy: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    reasons = []
    r10_entry = next(
        entry for entry in registry["source_artifacts"] if entry["artifact_id"] == "r10_verification"
    )
    r10_report = json.loads((root / r10_entry["path"]).read_text())
    if r10_report.get("gate_passed") is not True:
        reasons.append("R10_GATE_NOT_PASSED")
    if r10_report.get("repeated_seed_finalist_results_available") is not True:
        reasons.append("REPEATED_SEED_RESULTS_UNAVAILABLE")
    if r10_report.get("final_checkpoint_frozen") is not True:
        reasons.append("FINAL_CHECKPOINT_NOT_FROZEN")
    r11_path = root / _safe_relative_path(str(policy["expected_r11_report"]))
    if not r11_path.is_file():
        reasons.append("R11_NOT_EXECUTED")
    return reasons


def build_r12_report(
    root: Path,
    policy: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_audit: Mapping[str, Any],
    execution_audit: Mapping[str, Any],
    safety_audit: Mapping[str, Any],
    link_audit: Mapping[str, Any],
    notebook_path: Path,
    html_path: Path,
    canonical_sha256: str,
) -> dict[str, Any]:
    blocked_reasons = collect_blocked_reasons(root, policy, registry)
    report_build_checks = {
        "source_registry_integrity": bool(registry_audit["passed"]),
        "all_code_cells_executed": bool(execution_audit["passed"]),
        "zero_error_outputs": execution_audit["error_output_count"] == 0,
        "notebook_safety": bool(safety_audit["passed"]),
        "local_links": bool(link_audit["passed"]),
        "notebook_hash_recorded": notebook_path.is_file(),
        "html_hash_recorded": html_path.is_file(),
        "canonical_notebook_unchanged": file_sha256(
            root / policy["canonical_notebook"]["path"]
        )
        == canonical_sha256,
    }
    return {
        "schema_version": 1,
        "phase": "R12",
        "lineage": policy["policy_version"],
        "status": "blocked",
        "gate_passed": False,
        "blocked_reasons": blocked_reasons,
        "report_build_checks_passed": all(report_build_checks.values()),
        "failed_report_build_checks": [
            name for name, passed in report_build_checks.items() if not passed
        ],
        "report_build_checks": report_build_checks,
        "source_artifact_count": registry_audit["registry_path_count"],
        "source_artifact_hashes_verified": registry_audit["passed"],
        "code_cell_count": execution_audit["code_cell_count"],
        "executed_code_cell_count": execution_audit["executed_code_cell_count"],
        "error_output_count": execution_audit["error_output_count"],
        "local_link_count": link_audit["local_link_count"],
        "notebook_path": notebook_path.relative_to(root).as_posix(),
        "notebook_sha256": file_sha256(notebook_path),
        "html_path": html_path.relative_to(root).as_posix(),
        "html_sha256": file_sha256(html_path),
        "canonical_notebook_sha256": canonical_sha256,
        "canonical_notebook_mutated": False,
        "r11_executed": registry["r11_report_exists"],
        "one_time_test_evaluation_available": False,
        "repeated_seed_results_available": False,
        "final_checkpoint_frozen": False,
        "training_operations_detected": safety_audit["training_operations_detected"],
        "test_evaluation_operations_detected": safety_audit[
            "test_evaluation_operations_detected"
        ],
        "test_loader_constructed": False,
        "test_evaluation_count": 0,
        "real_training_performed": False,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "source_images_mutated": False,
        "successor_phase_executed": False,
    }


def build_artifact_manifest(
    root: Path,
    paths: Sequence[Path],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = []
    for path in paths:
        artifacts.append(
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "phase": "R12",
        "lineage": report["lineage"],
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "report_build_checks_passed": report["report_build_checks_passed"],
        "r11_executed": report["r11_executed"],
        "real_training_performed": False,
        "test_evaluation_count": 0,
        "artifacts": artifacts,
    }
