"""Initialize real NOT_STARTED metadata for Practice 3 v2 without training."""

from __future__ import annotations

from typing import Any

from .dataset_protocol_v2 import (
    create_split_manifest,
    load_development_pool,
    save_split_manifest,
)
from .experiment_protocol_v2 import (
    PART_2_1_PLAN,
    PROTOCOL_MANIFEST_PATH,
    V2_RESULT_DIR,
    atomic_write_json,
    build_protocol_manifest,
    optimizer_compatibility_preflight,
    sha256_file,
    validate_protocol_manifest,
)
from .experiment_registry_v2 import initialize_registry
from .holdout_guard_v2 import initialize_holdout_state


def initialize_protocol_v2() -> dict[str, Any]:
    """Create real protocol metadata only; no model, Trainer, Test, or evaluation."""
    if not PART_2_1_PLAN.is_file():
        raise FileNotFoundError("Frozen Part 2.1 plan is missing")
    records, upstream_fingerprints = load_development_pool()
    split_manifest = create_split_manifest(records, upstream_fingerprints)
    optimizer = optimizer_compatibility_preflight()
    if optimizer["status"] != "SUPPORTED":
        raise RuntimeError(f"Frozen optimizer is incompatible: {optimizer}")
    protocol = build_protocol_manifest(
        optimizer,
        sha256_file(PART_2_1_PLAN),
        split_manifest["split_manifest_hash"],
    )
    validate_protocol_manifest(protocol)

    V2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_split_manifest(V2_RESULT_DIR / "dataset_split_manifest.json", split_manifest)
    atomic_write_json(PROTOCOL_MANIFEST_PATH, protocol)
    holdout_state = initialize_holdout_state(
        V2_RESULT_DIR / "holdout_access_state.json"
    )
    configs, registry = initialize_registry(
        V2_RESULT_DIR,
        split_manifest["development_pool_hash"],
        split_manifest["split_manifest_hash"],
    )
    return {
        "protocol": protocol,
        "dataset_split": split_manifest,
        "holdout_state": holdout_state,
        "run_configs": configs,
        "registry": registry,
        "training_performed": False,
        "trainer_created": False,
        "v1_test_accessed": False,
        "v2_holdout_evaluated": False,
        "winner_created": False,
    }


if __name__ == "__main__":
    report = initialize_protocol_v2()
    print({
        "protocol": report["protocol"]["status"],
        "dataset_split": report["dataset_split"]["status"],
        "holdout": report["protocol"]["holdout_status"],
        "registry": report["registry"]["status"],
        "optimizer": report["protocol"]["optimizer_preflight"]["status"],
    })

