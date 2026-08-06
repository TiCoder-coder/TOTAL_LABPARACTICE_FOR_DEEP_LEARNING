from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .paths import get_practice_2_2_root
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r0_authority_policy.json")


def load_r0_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R0 authority policy schema")
    if policy.get("implementation_authorized") is not False:
        raise RuntimeError("R0 policy must not authorize implementation automatically")
    if policy.get("canonical_notebook_mutation_authorized") is not False:
        raise RuntimeError("Canonical notebook must remain read-only")
    if policy.get("final_test_re_evaluation_authorized") is not False:
        raise RuntimeError("Final Test re-evaluation must remain disabled")
    return policy


def resolve_policy_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe R0 authority path: {relative_path}")
    return (get_practice_2_2_root() / relative).resolve()


def directory_digest(root: Path) -> dict[str, Any]:
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_sha256(path)} {size} {relative}\n".encode())
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "directory_sha256": digest.hexdigest(),
    }


def _pandas_manifest_hashes(path: Path) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_csv(path)

    def frame_hash(columns: list[str] | None = None) -> str:
        selected = frame if columns is None else frame[columns]
        values = pd.util.hash_pandas_object(selected, index=True).values
        return hashlib.sha256(values).hexdigest()

    return {
        "dataframe_sha256": frame_hash(),
        "dataset_sha256": frame_hash(
            ["image_path", "label", "split", "visual_group_id", "is_generated"]
        ),
        "split_sha256": frame_hash(["image_path", "split"]),
    }


def _matches_expected(actual: Mapping[str, Any], authority: Mapping[str, Any]) -> bool:
    checks = {
        "sha256": "expected_sha256",
        "file_count": "expected_file_count",
        "total_bytes": "expected_total_bytes",
        "directory_sha256": "expected_directory_sha256",
        "dataframe_sha256": "expected_dataframe_sha256",
        "dataset_sha256": "expected_dataset_sha256",
        "split_sha256": "expected_split_sha256",
    }
    expected_keys = [expected for expected in checks.values() if expected in authority]
    if not expected_keys:
        return False
    return all(actual[key] == authority[expected] for key, expected in checks.items() if expected in authority)


def inspect_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_policy_path(str(authority["path"]))
    kind = str(authority["kind"])
    record = {
        "name": authority["name"],
        "classification": authority["classification"],
        "path": authority["path"],
        "kind": kind,
        "required": bool(authority["required"]),
        "exists": path.exists(),
    }
    if not path.exists():
        record["status"] = "missing"
        return record
    if kind == "directory":
        if not path.is_dir():
            record["status"] = "wrong_type"
            return record
        actual = directory_digest(path)
    elif kind in {"file", "pandas_manifest"}:
        if not path.is_file():
            record["status"] = "wrong_type"
            return record
        actual = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if kind == "pandas_manifest":
            actual.update(_pandas_manifest_hashes(path))
    else:
        raise RuntimeError(f"Unsupported R0 authority kind: {kind}")
    record["actual"] = actual
    record["status"] = "verified" if _matches_expected(actual, authority) else "mismatch"
    return record


def build_r0_inventory(policy_path: Path | None = None) -> dict[str, Any]:
    policy = load_r0_policy(policy_path)
    records = [inspect_authority(authority) for authority in policy["authorities"]]
    failures = [
        record["name"]
        for record in records
        if record["required"] and record["status"] != "verified"
    ]
    return {
        "schema_version": 1,
        "phase": "R0",
        "canonical_run_id": policy["canonical_run_id"],
        "status": "ready_for_manual_authorization" if not failures else "blocked",
        "implementation_authorized": False,
        "canonical_notebook_mutation_authorized": False,
        "final_test_re_evaluation_authorized": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "failed_authorities": failures,
        "authorities": records,
        "classifications": policy["classifications"],
    }


def assert_r0_ready(inventory: Mapping[str, Any] | None = None) -> None:
    result = dict(inventory or build_r0_inventory())
    if result.get("status") != "ready_for_manual_authorization":
        failed = ", ".join(result.get("failed_authorities", []))
        raise RuntimeError(f"R0 authority gate is blocked: {failed}")


def write_r0_inventory(inventory: Mapping[str, Any], output_path: Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if path != root and root not in path.parents:
        raise RuntimeError("R0 inventory output must remain inside Practice 2.2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(inventory), indent=2) + "\n")
    return path
