"""Read-only canonical registry, resolver and immutable-artifact adapter.

This module never regenerates resources and never writes persisted lineage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .paths import get_practice_2_2_root


REGISTRY_RELATIVE_PATH = Path("configs/canonical_registry.json")
LEGACY_REGISTRY_RELATIVE_PATH = Path("configs/canonical_registry_legacy_compat.json")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_registry(registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path or (get_practice_2_2_root() / REGISTRY_RELATIVE_PATH)
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical registry is missing: {path}")
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1:
        raise RuntimeError("Unsupported canonical registry schema")
    if registry.get("path_base") != "practice_2_2_root":
        raise RuntimeError("Canonical registry must use practice_2_2-relative paths")
    return registry


def load_legacy_compat_registry() -> dict[str, Any]:
    """Load the old layout only when compatibility is explicitly requested."""

    registry = load_canonical_registry(
        get_practice_2_2_root() / LEGACY_REGISTRY_RELATIVE_PATH
    )
    if registry.get("authority_status") != "legacy_compatibility_read_only":
        raise RuntimeError("Legacy registry is not marked read-only compatibility")
    return registry


def resolve_registry_resource(
    registry: Mapping[str, Any],
    name: str,
    *,
    require_file: bool | None = None,
) -> Path:
    try:
        relative = Path(registry["resources"][name])
    except KeyError as error:
        raise KeyError(f"Unknown canonical resource: {name}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Registry resource must be a safe relative path: {name}")
    path = (get_practice_2_2_root() / relative).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Canonical resource is missing: {name} -> {path}")
    if require_file is True and not path.is_file():
        raise FileNotFoundError(f"Canonical resource must be a file: {name} -> {path}")
    if require_file is False and not path.is_dir():
        raise FileNotFoundError(f"Canonical resource must be a directory: {name} -> {path}")
    return path


def resolve_resource(name: str, *, require_file: bool | None = None) -> Path:
    return resolve_registry_resource(
        load_canonical_registry(), name, require_file=require_file
    )


def resolve_legacy_resource(
    name: str, *, require_file: bool | None = None
) -> Path:
    """Resolve a pre-migration path through the explicit read-only registry."""

    return resolve_registry_resource(
        load_legacy_compat_registry(), name, require_file=require_file
    )


def resolve_dataset_root() -> Path:
    return resolve_resource("dataset_root", require_file=False)


def resolve_split_manifest() -> Path:
    return resolve_resource("split_manifest", require_file=True)


def resolve_split_summary() -> Path:
    return resolve_resource("split_summary", require_file=True)


def resolve_best_checkpoint() -> Path:
    return resolve_resource("e2_best_checkpoint", require_file=True)


def resolve_final_test_dir() -> Path:
    return resolve_resource("final_test_dir", require_file=False)


def load_immutable_artifact(name: str) -> dict[str, Any]:
    """Read an immutable JSON through the registry without writing it back."""

    registry = load_canonical_registry()
    try:
        resource_name = registry["immutable_artifacts"][name]
    except KeyError as error:
        raise KeyError(f"Unknown immutable artifact: {name}") from error
    path = resolve_resource(resource_name, require_file=True)
    return json.loads(path.read_text())


def resolve_persisted_path(
    artifact: Mapping[str, Any],
    field: str,
    fallback_resource: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Resolve a historical absolute path, falling back read-only to registry.

    The immutable ``artifact`` mapping is never mutated. When a persisted path
    no longer exists, the current registry location is used and optionally
    hash-verified before it is returned.
    """

    stored = artifact.get(field)
    candidate = Path(stored).expanduser() if stored else None
    if candidate is None or not candidate.exists():
        candidate = resolve_resource(fallback_resource)
    candidate = candidate.resolve()
    if expected_sha256 and (
        not candidate.is_file() or file_sha256(candidate) != expected_sha256
    ):
        raise RuntimeError(f"Resolved resource SHA-256 mismatch: {candidate}")
    return candidate


def verify_canonical_resources(
    *,
    verify_dataset_contents: bool = True,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed on missing/mismatched resources without loading Final Test."""

    registry = load_canonical_registry(registry_path)
    expected = registry["expected"]
    resolve = lambda name, require_file=None: resolve_registry_resource(
        registry, name, require_file=require_file
    )
    required_files = [
        "split_manifest",
        "split_summary",
        "quarantine_csv",
        "e2_best_checkpoint",
        "e2_latest_checkpoint",
        "final_selection",
        "final_test_summary",
        "final_test_guard",
        "canonical_report_notebook",
        "html_report",
    ]
    for name in required_files:
        resolve(name, require_file=True)
    resolve("canonical_output_dir", require_file=False)
    resolve("final_test_dir", require_file=False)

    checkpoint_path = resolve("e2_best_checkpoint", require_file=True)
    manifest_path = resolve("split_manifest", require_file=True)
    summary_path = resolve("split_summary", require_file=True)
    dataset_path = resolve("dataset_root", require_file=False)
    checkpoint_hash = file_sha256(checkpoint_path)
    manifest_hash = file_sha256(manifest_path)
    guard_path = resolve("final_test_guard", require_file=True)
    guard_hash = file_sha256(guard_path)
    if checkpoint_hash != expected["checkpoint_sha256"]:
        raise RuntimeError("Canonical checkpoint SHA-256 mismatch")
    if manifest_hash != expected["split_manifest_sha256"]:
        raise RuntimeError("Canonical split manifest SHA-256 mismatch")
    if guard_hash != expected["final_test_guard_sha256"]:
        raise RuntimeError("Final Test guard SHA-256 mismatch")

    summary = json.loads(summary_path.read_text())
    if summary.get("dataset_fingerprint_sha256") != expected["dataset_fingerprint_sha256"]:
        raise RuntimeError("Dataset fingerprint mismatch in canonical summary")
    if summary.get("split_fingerprint_sha256") != expected["split_fingerprint_sha256"]:
        raise RuntimeError("Split fingerprint mismatch in canonical summary")
    guard = json.loads(guard_path.read_text())
    if not (
        guard.get("FINAL_TEST_COMPLETED") is True
        and guard.get("final_test_evaluation_count") == 1
        and guard.get("repeat_evaluation_allowed") is False
        and guard.get("maintenance_override_used") is False
    ):
        raise RuntimeError("Final Test guard contract mismatch")

    live = None
    if verify_dataset_contents:
        from .canonical_train_practice_2_2 import verify_canonical_input

        live = verify_canonical_input(
            dataset_path, manifest_path, summary_path
        )

    return {
        "canonical_run_id": registry["canonical_run_id"],
        "checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": manifest_hash,
        "dataset_fingerprint_sha256": expected["dataset_fingerprint_sha256"],
        "split_fingerprint_sha256": expected["split_fingerprint_sha256"],
        "final_test_guard_sha256": guard_hash,
        "authority_status": registry.get("authority_status", "active"),
        "live_dataset_verification": live,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
