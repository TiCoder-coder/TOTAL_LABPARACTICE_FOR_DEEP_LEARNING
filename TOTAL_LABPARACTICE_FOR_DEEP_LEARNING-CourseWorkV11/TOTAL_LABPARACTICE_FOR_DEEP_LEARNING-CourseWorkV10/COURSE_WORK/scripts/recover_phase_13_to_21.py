from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent.parent.resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PRESERVATION_MANIFEST = Path(
    "docs/save_log_in_processing/phase_13_to_21_recovery_preservation_manifest.json"
)
ACTIVE_ROOTS = (
    "artifacts/experiments",
    "artifacts/runs",
    "artifacts/baselines/persistence",
    "artifacts/models/lstm",
    "artifacts/models/transformer",
    "artifacts/attention_verification",
    "artifacts/forward_sanity",
    "artifacts/training_engine",
    "artifacts/lstm_baseline",
    "artifacts/transformer_b0",
)
PHASE_OWNERS = {
    13: (
        "course_work.experiments.registry",
        "materialize_phase_13",
        "verify_existing_signoff",
        "artifacts/experiments/phase_13_signoff.json",
        "artifacts/experiments",
    ),
    14: (
        "course_work.baselines.persistence",
        "materialize_phase_14",
        "verify_existing_signoff",
        "artifacts/baselines/persistence/phase_14_signoff.json",
        "artifacts/baselines/persistence",
    ),
    15: (
        "course_work.models.lstm_regressor",
        "materialize_phase_15",
        "verify_existing_signoff",
        "artifacts/models/lstm/phase_15_signoff.json",
        "artifacts/models/lstm",
    ),
    16: (
        "course_work.models.transformer_regressor",
        "materialize_phase_16",
        "verify_existing_signoff",
        "artifacts/models/transformer/phase_16_signoff.json",
        "artifacts/models/transformer",
    ),
    17: (
        "course_work.attention.verification",
        "materialize_phase_17",
        "verify_existing_signoff",
        "artifacts/attention_verification/phase_17_signoff.json",
        "artifacts/attention_verification",
    ),
    18: (
        "course_work.sanity.forward_sanity",
        "materialize_phase_18",
        "verify_existing_signoff",
        "artifacts/forward_sanity/phase_18_signoff.json",
        "artifacts/forward_sanity",
    ),
    19: (
        "course_work.training.engine_materialize",
        "materialize_phase_19",
        "verify_existing_signoff",
        "artifacts/training_engine/phase_19_signoff.json",
        "artifacts/training_engine",
    ),
    20: (
        "course_work.baselines.lstm_baseline",
        "materialize_phase_20",
        "verify_existing_signoff",
        "artifacts/lstm_baseline/phase_20_signoff.json",
        "artifacts/lstm_baseline",
    ),
    21: (
        "course_work.baselines.transformer_b0",
        "materialize_phase_21",
        "verify_existing_signoff",
        "artifacts/transformer_b0/phase_21_signoff.json",
        "artifacts/transformer_b0",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_state(actual_root: Path, logical_root: str) -> dict[str, Any]:
    files = sorted(
        (path for path in actual_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(actual_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        logical_path = (Path(logical_root) / path.relative_to(actual_root)).as_posix()
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return {
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "tree_sha256": digest.hexdigest(),
    }


def _load_manifest(project_root: Path) -> dict[str, Any]:
    path = project_root / PRESERVATION_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("Recovery preservation manifest is not PASS")
    if manifest.get("artifact_version") != "PHASE-13-21-RECOVERY-PRESERVATION-v1":
        raise RuntimeError("Recovery preservation manifest version mismatch")
    return manifest


def _history_root(project_root: Path, manifest: dict[str, Any]) -> Path:
    return project_root / "artifacts" / "_history" / manifest["recovery_revision_id"]


def _history_path(history_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if not path.parts or path.parts[0] != "artifacts":
        raise ValueError(f"Invalid recovery root: {relative_path}")
    return history_root.joinpath(*path.parts[1:])


def _assert_tree(actual_root: Path, expected: dict[str, Any]) -> None:
    if not actual_root.is_dir():
        raise RuntimeError(f"Recovery root missing: {actual_root}")
    actual = _tree_state(actual_root, expected["path"])
    for key in ("file_count", "total_size_bytes", "tree_sha256"):
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"Recovery preservation mismatch: {expected['path']} {key} "
                f"expected={expected[key]} actual={actual[key]}"
            )


def _verify_notebook(project_root: Path, manifest: dict[str, Any]) -> None:
    notebook = manifest["notebook"]
    path = project_root / notebook["path"]
    if not path.is_file():
        raise RuntimeError("Recovery notebook is missing")
    if path.stat().st_size != notebook["size_bytes"] or _sha256(path) != notebook["sha256"]:
        raise RuntimeError("Recovery notebook preservation mismatch")


def _verify_environment(project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = next(item for item in manifest["roots"] if item["path"] == "artifacts/environment")
    _assert_tree(project_root / expected["path"], expected)
    report_path = project_root / "artifacts/environment/environment_report.json"
    signoff_path = project_root / "artifacts/environment/phase_1_signoff.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    ready = (
        signoff.get("status") == "PASS"
        and report.get("selected_device") in {"mps", "cuda"}
        and bool(report.get("mps_available") or report.get("cuda_available"))
    )
    if not ready:
        raise RuntimeError("Current environment is not ready for recovery training")
    return {
        "environment_revision_id": report.get("environment_revision_id"),
        "python_version": report.get("python_version"),
        "selected_device": report.get("selected_device"),
        "status": "PASS",
    }


def _verify_history(project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    history_root = _history_root(project_root, manifest)
    relocation_path = history_root / "recovery_relocation_manifest.json"
    if not relocation_path.is_file():
        raise RuntimeError("Recovery history exists without relocation manifest")
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    if relocation.get("status") != "PASS":
        raise RuntimeError("Recovery relocation manifest is not PASS")
    expected_by_path = {item["path"]: item for item in manifest["roots"]}
    for relative_path in ACTIVE_ROOTS:
        _assert_tree(_history_path(history_root, relative_path), expected_by_path[relative_path])
    return relocation


def _phase_state(project_root: Path, phase_id: int) -> dict[str, Any]:
    module_name, _, verifier_name, signoff_relative, _ = PHASE_OWNERS[phase_id]
    signoff_path = project_root / signoff_relative
    if not signoff_path.is_file():
        return {"phase_id": phase_id, "state": "MISSING"}
    module = importlib.import_module(module_name)
    verifier = getattr(module, verifier_name)
    try:
        signoff = verifier(project_root, signoff_path)
    except Exception as error:
        return {
            "phase_id": phase_id,
            "state": "INVALID",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "phase_id": phase_id,
        "state": "VALID_REUSABLE",
        "status": signoff.get("status"),
        "run_id": signoff.get("run_id"),
    }


def audit_recovery(project_root: Path = ROOT) -> dict[str, Any]:
    root = project_root.resolve()
    manifest = _load_manifest(root)
    _verify_notebook(root, manifest)
    environment = _verify_environment(root, manifest)
    history_root = _history_root(root, manifest)
    if history_root.exists():
        relocation = _verify_history(root, manifest)
        preservation_state = "PRESERVED"
    else:
        expected_by_path = {item["path"]: item for item in manifest["roots"]}
        for relative_path in ACTIVE_ROOTS:
            _assert_tree(root / relative_path, expected_by_path[relative_path])
        relocation = None
        preservation_state = "READY_TO_PRESERVE"
    return {
        "plan_id": "CW-PHASE-13-21-REVISIONED-CANONICAL-RECOVERY-v1",
        "recovery_revision_id": manifest["recovery_revision_id"],
        "preservation_state": preservation_state,
        "history_root": str(history_root.relative_to(root)),
        "environment": environment,
        "phases": [_phase_state(root, phase_id) for phase_id in PHASE_OWNERS],
        "relocation": relocation,
        "notebook_preserved": True,
        "scientific_artifacts_written": False,
        "status": "PASS",
    }


def preserve_evidence(project_root: Path = ROOT) -> dict[str, Any]:
    root = project_root.resolve()
    manifest = _load_manifest(root)
    _verify_notebook(root, manifest)
    _verify_environment(root, manifest)
    history_root = _history_root(root, manifest)
    if history_root.exists():
        return _verify_history(root, manifest)
    expected_by_path = {item["path"]: item for item in manifest["roots"]}
    for relative_path in ACTIVE_ROOTS:
        source = root / relative_path
        destination = _history_path(history_root, relative_path)
        _assert_tree(source, expected_by_path[relative_path])
        if destination.exists():
            raise FileExistsError(destination)
    moved: list[tuple[Path, Path]] = []
    try:
        history_root.mkdir(parents=True, exist_ok=False)
        for relative_path in ACTIVE_ROOTS:
            source = root / relative_path
            destination = _history_path(history_root, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        for relative_path in ACTIVE_ROOTS:
            _assert_tree(
                _history_path(history_root, relative_path),
                expected_by_path[relative_path],
            )
        relocation = {
            "artifact_version": "PHASE-13-21-RECOVERY-RELOCATION-v1",
            "recovery_revision_id": manifest["recovery_revision_id"],
            "mappings": [
                {
                    "canonical_path": relative_path,
                    "historical_path": str(
                        _history_path(history_root, relative_path).relative_to(root)
                    ),
                    "tree_sha256": expected_by_path[relative_path]["tree_sha256"],
                }
                for relative_path in ACTIVE_ROOTS
            ],
            "notebook_sha256": manifest["notebook"]["sha256"],
            "status": "PASS",
        }
        path = history_root / "recovery_relocation_manifest.json"
        path.write_text(
            json.dumps(relocation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return _verify_history(root, manifest)
    except BaseException:
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise


def _verify_test_firewall(project_root: Path) -> None:
    path = project_root / "artifacts/experiments/experiment_registry.jsonl"
    if not path.is_file():
        raise RuntimeError("Active experiment registry is missing")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        if record.get("test_access_authorized"):
            raise PermissionError(f"Test access authorized in {record.get('run_id')}")
        if any(metric.get("split_id") == "TEST" for metric in record.get("metrics", [])):
            raise PermissionError(f"Test metric detected in {record.get('run_id')}")


def execute_through(target_phase: int, project_root: Path = ROOT) -> dict[str, Any]:
    if target_phase not in PHASE_OWNERS:
        raise ValueError(f"Unsupported recovery target Phase {target_phase}")
    root = project_root.resolve()
    before = audit_recovery(root)
    relocation = preserve_evidence(root)
    completed = []
    for phase_id in range(13, target_phase + 1):
        module_name, materializer_name, verifier_name, signoff_relative, artifact_relative = PHASE_OWNERS[phase_id]
        module = importlib.import_module(module_name)
        materializer = getattr(module, materializer_name)
        verifier = getattr(module, verifier_name)
        signoff_path = root / signoff_relative
        artifact_root = root / artifact_relative
        if signoff_path.is_file():
            signoff = verifier(root, signoff_path)
            action = "REUSED"
        else:
            if artifact_root.exists() and any(artifact_root.iterdir()):
                raise RuntimeError(f"Phase {phase_id} has partial active artifacts")
            signoff = materializer(root)
            signoff = verifier(root, signoff_path)
            action = "MATERIALIZED"
        if signoff.get("status") != "PASS":
            raise RuntimeError(f"Phase {phase_id} sign-off is not PASS")
        _verify_test_firewall(root)
        completed.append(
            {
                "phase_id": phase_id,
                "action": action,
                "status": signoff.get("status"),
                "run_id": signoff.get("run_id"),
            }
        )
    _verify_notebook(root, _load_manifest(root))
    return {
        "plan_id": before["plan_id"],
        "recovery_revision_id": before["recovery_revision_id"],
        "target_phase": target_phase,
        "relocation": relocation,
        "completed_phases": completed,
        "notebook_preserved": True,
        "test_firewall": "PASS",
        "status": "PASS",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--preserve-only", action="store_true")
    mode.add_argument("--execute-through", type=int, choices=tuple(PHASE_OWNERS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.audit_only:
            result = audit_recovery(ROOT)
        elif args.preserve_only:
            result = preserve_evidence(ROOT)
        else:
            result = execute_through(args.execute_through, ROOT)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "status": "FAIL",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
