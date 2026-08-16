"""Winner-lock validation and one-time Holdout guard for Practice 3 v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .experiment_protocol_v2 import (
    PROTOCOL_VERSION,
    RANKING_TOLERANCE,
    atomic_write_json,
    rank_experiments,
    sha256_file,
    sha256_payload,
)


def initial_holdout_state() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "winner_locked": False,
        "holdout_access_allowed": False,
        "holdout_evaluation_count": 0,
        "attempt_claimed": False,
        "holdout_evaluated": False,
        "v1_test_evaluation_count": 1,
    }


def validate_initial_holdout_state(state: dict[str, Any]) -> None:
    checks = (
        state.get("protocol_version") == PROTOCOL_VERSION,
        state.get("winner_locked") is False,
        state.get("holdout_access_allowed") is False,
        state.get("holdout_evaluation_count") == 0,
        state.get("attempt_claimed") is False,
        state.get("holdout_evaluated") is False,
        state.get("v1_test_evaluation_count") == 1,
    )
    if not all(checks):
        raise ValueError("Initial Holdout state must remain sealed with count zero")


def initialize_holdout_state(path: Path) -> dict[str, Any]:
    state = initial_holdout_state()
    validate_initial_holdout_state(state)
    atomic_write_json(path, state)
    return state


def build_winner_manifest(
    registry: dict[str, Any],
    run_configs: dict[str, dict[str, Any]],
    checkpoint_paths: dict[str, Path],
    dataset_fingerprint: str,
    split_fingerprint: str,
    *,
    locked_at: str | None = None,
) -> dict[str, Any]:
    """Build a winner only from three completed Validation-only records."""
    records = registry.get("runs", [])
    if len(records) != 3 or any(record.get("status") != "COMPLETED" for record in records):
        raise ValueError("All three registry runs must be COMPLETED before winner lock")
    ranked = rank_experiments(records)
    winner = ranked[0]
    run_id = winner["run_id"]
    config = run_configs.get(run_id)
    checkpoint = checkpoint_paths.get(run_id)
    if config is None or checkpoint is None or not checkpoint.is_file():
        raise ValueError("Winner config/checkpoint evidence is missing")
    if config.get("config_hash") != winner.get("config_hash"):
        raise ValueError("Winner config hash mismatch")
    if winner.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Winner dataset fingerprint mismatch")
    if winner.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Winner split fingerprint mismatch")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "learning_rate": winner["learning_rate"],
        "best_epoch": winner["best_epoch"],
        "checkpoint": str(checkpoint),
        "validation_metrics": {
            "loss": winner["best_val_loss"],
            "accuracy": winner["best_val_accuracy"],
            "precision": winner["best_val_precision"],
            "recall": winner["best_val_recall"],
            "f1": winner["best_val_f1"],
        },
        "config_hash": config["config_hash"],
        "checkpoint_hash": sha256_file(checkpoint),
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "selection_rule": [
            "lowest_validation_loss",
            "higher_validation_f1_within_tolerance",
            "higher_validation_accuracy_within_tolerance",
            "earlier_best_epoch_within_tolerance",
            "lexical_run_id",
        ],
        "ranking_tolerance": RANKING_TOLERANCE,
        "locked_at": locked_at or datetime.now(timezone.utc).isoformat(),
    }
    manifest["winner_manifest_hash"] = sha256_payload(manifest)
    return manifest


def validate_winner_manifest(
    winner: dict[str, Any],
    registry: dict[str, Any],
    run_configs: dict[str, dict[str, Any]],
    dataset_fingerprint: str,
    split_fingerprint: str,
) -> None:
    supplied_hash = winner.get("winner_manifest_hash")
    body = {key: value for key, value in winner.items() if key != "winner_manifest_hash"}
    if supplied_hash != sha256_payload(body):
        raise ValueError("Winner manifest hash mismatch")
    if winner.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Winner protocol version mismatch")
    records = registry.get("runs", [])
    ranked = rank_experiments(records)
    if winner.get("run_id") != ranked[0]["run_id"]:
        raise ValueError("Winner does not match deterministic registry ranking")
    config = run_configs.get(winner["run_id"])
    if config is None or winner.get("config_hash") != config.get("config_hash"):
        raise ValueError("Winner config hash mismatch")
    checkpoint = Path(winner.get("checkpoint", ""))
    if not checkpoint.is_file() or winner.get("checkpoint_hash") != sha256_file(checkpoint):
        raise ValueError("Winner checkpoint hash mismatch")
    if winner.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Winner dataset fingerprint mismatch")
    if winner.get("split_fingerprint") != split_fingerprint:
        raise ValueError("Winner split fingerprint mismatch")
    if float(winner.get("ranking_tolerance")) != RANKING_TOLERANCE:
        raise ValueError("Winner ranking tolerance mismatch")


def unlock_state_after_valid_winner(
    state: dict[str, Any],
    winner: dict[str, Any],
    registry: dict[str, Any],
    run_configs: dict[str, dict[str, Any]],
    dataset_fingerprint: str,
    split_fingerprint: str,
) -> dict[str, Any]:
    validate_initial_holdout_state(state)
    validate_winner_manifest(
        winner, registry, run_configs, dataset_fingerprint, split_fingerprint
    )
    unlocked = json.loads(json.dumps(state))
    unlocked["winner_locked"] = True
    unlocked["holdout_access_allowed"] = True
    unlocked["winner_manifest_hash"] = winner["winner_manifest_hash"]
    return unlocked


def claim_holdout_attempt(path: Path, expected_winner_hash: str) -> dict[str, Any]:
    """Atomically consume the one permitted attempt before Holdout loading."""
    if not path.is_file():
        raise FileNotFoundError("Holdout state does not exist")
    state = json.loads(path.read_text(encoding="utf-8"))
    checks = (
        state.get("protocol_version") == PROTOCOL_VERSION,
        state.get("winner_locked") is True,
        state.get("holdout_access_allowed") is True,
        state.get("winner_manifest_hash") == expected_winner_hash,
        state.get("holdout_evaluation_count") == 0,
        state.get("attempt_claimed") is False,
    )
    if not all(checks):
        raise PermissionError("Holdout access denied by one-time guard")
    claimed = json.loads(json.dumps(state))
    claimed["holdout_evaluation_count"] = 1
    claimed["attempt_claimed"] = True
    claimed["holdout_access_allowed"] = False
    claimed["attempt_claimed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, claimed, overwrite=True)
    return claimed


def guarded_holdout_request(
    state_path: Path,
    expected_winner_hash: str,
    provider: Callable[[], Any],
) -> Any:
    """Reject before invoking a Holdout provider unless the attempt is valid."""
    claim_holdout_attempt(state_path, expected_winner_hash)
    return provider()
