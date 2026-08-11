"""Atomic recovery checkpoints for epoch-based PyTorch training."""

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from typing import Any, Optional, Union

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
PathLike = Union[str, Path]


class TrainingCheckpointError(RuntimeError):
    """Base error for unreadable or invalid recovery checkpoints."""


class CheckpointCompatibilityError(TrainingCheckpointError):
    """Raised when a checkpoint belongs to a different training run."""


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Checkpoint signature values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Checkpoint signature keys must be strings")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_canonicalize(item) for item in value]
    raise TypeError(
        f"Unsupported checkpoint signature value: {type(value).__name__}"
    )


def build_checkpoint_signature(metadata: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 signature for config and data metadata."""
    canonical_metadata = _canonicalize(metadata)
    encoded_metadata = json.dumps(
        canonical_metadata,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded_metadata).hexdigest()


def fingerprint_indices(
    indices: Union[Sequence[int], torch.Tensor],
) -> str:
    """Hash an ordered integer index sequence without serializing it to JSON."""
    digest = hashlib.sha256()
    digest.update(len(indices).to_bytes(8, byteorder="little", signed=False))
    for index in indices:
        integer_index = int(index)
        digest.update(
            integer_index.to_bytes(8, byteorder="little", signed=True)
        )
    return digest.hexdigest()


def clone_to_cpu(value: Any) -> Any:
    """Recursively clone tensors to CPU for portable serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    return value


def capture_rng_state(
    data_loader_generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    """Capture process and optional DataLoader random-number states."""
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [],
        "torch_mps": None,
        "data_loader_generator": (
            data_loader_generator.get_state().cpu().clone()
            if data_loader_generator is not None
            else None
        ),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = [
            rng_state.cpu().clone()
            for rng_state in torch.cuda.get_rng_state_all()
        ]
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state().cpu().clone()

    return state


def restore_rng_state(
    state: Mapping[str, Any],
    data_loader_generator: Optional[torch.Generator] = None,
) -> None:
    """Restore states captured by :func:`capture_rng_state`."""
    required_fields = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "torch_mps",
        "data_loader_generator",
    }
    missing_fields = sorted(required_fields.difference(state))
    if missing_fields:
        raise TrainingCheckpointError(
            "Missing RNG state fields: " + ", ".join(missing_fields)
        )

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise TrainingCheckpointError("NumPy RNG state must be a mapping")

    try:
        random.setstate(state["python"])
        np.random.set_state((
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
        torch.set_rng_state(state["torch_cpu"].cpu())

        cuda_states = state["torch_cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointCompatibilityError(
                    "Checkpoint contains CUDA RNG state, but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in cuda_states]
            )

        mps_state = state["torch_mps"]
        if mps_state is not None:
            if not torch.backends.mps.is_available():
                raise CheckpointCompatibilityError(
                    "Checkpoint contains MPS RNG state, but MPS is unavailable"
                )
            torch.mps.set_rng_state(mps_state.cpu())

        loader_state = state["data_loader_generator"]
        if loader_state is not None:
            if data_loader_generator is None:
                raise CheckpointCompatibilityError(
                    "Checkpoint requires a DataLoader generator"
                )
            data_loader_generator.set_state(loader_state.cpu())
    except CheckpointCompatibilityError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingCheckpointError(
            f"Invalid RNG state: {error}"
        ) from error


def atomic_torch_save(payload: Mapping[str, Any], destination: PathLike) -> None:
    """Atomically replace a checkpoint after a complete durable write."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination_path)

        directory_descriptor = os.open(
            destination_path.parent,
            os.O_RDONLY,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class TrainingCheckpointManager:
    """Validate, save and load one stable recovery checkpoint path."""

    def __init__(
        self,
        path: PathLike,
        run_kind: str,
        run_id: str,
        signature: str,
        enabled: bool = True,
    ) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path")
        if not isinstance(run_kind, str) or not run_kind.strip():
            raise ValueError("run_kind must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(signature, str) or len(signature) != 64:
            raise ValueError("signature must be a SHA-256 hexadecimal string")
        try:
            int(signature, 16)
        except ValueError as error:
            raise ValueError(
                "signature must be a SHA-256 hexadecimal string"
            ) from error
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        self.path = Path(path)
        self.run_kind = run_kind.strip()
        self.run_id = run_id.strip()
        self.signature = signature
        self.enabled = enabled

    @property
    def exists(self) -> bool:
        """Report whether the enabled manager has a recovery file."""
        return self.enabled and self.path.is_file()

    def save(self, state: Mapping[str, Any]) -> None:
        """Validate and atomically persist the supplied training state."""
        if not self.enabled:
            return
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")

        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_kind": self.run_kind,
            "run_id": self.run_id,
            "signature": self.signature,
            "saved_at_unix": time.time(),
            **clone_to_cpu(state),
        }
        self._validate_payload(payload, check_compatibility=True)
        atomic_torch_save(payload, self.path)

    def load(
        self,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Optional[dict[str, Any]]:
        """Load a compatible checkpoint, or return ``None`` when absent."""
        if not self.enabled or not self.path.exists():
            return None
        if not self.path.is_file():
            raise TrainingCheckpointError(
                f"Recovery checkpoint is not a file: {self.path}"
            )

        try:
            payload = torch.load(
                self.path,
                map_location=map_location,
                weights_only=True,
            )
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            raise TrainingCheckpointError(
                f"Cannot load recovery checkpoint {self.path}: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise TrainingCheckpointError(
                "Recovery checkpoint root must be a dictionary"
            )
        self._validate_payload(payload, check_compatibility=True)
        return payload

    def _validate_payload(
        self,
        payload: Mapping[str, Any],
        check_compatibility: bool,
    ) -> None:
        required_fields = {
            "schema_version",
            "run_kind",
            "run_id",
            "signature",
            "saved_at_unix",
            "status",
            "completed_epoch",
            "total_epochs",
            "model_state_dict",
            "optimizer_state_dict",
            "history",
            "best_state_dict",
            "best_epoch",
            "best_validation_accuracy",
            "best_validation_loss",
            "rng_state",
            "elapsed_seconds",
            "log_directory",
        }
        missing_fields = sorted(required_fields.difference(payload))
        if missing_fields:
            raise TrainingCheckpointError(
                "Missing recovery checkpoint fields: "
                + ", ".join(missing_fields)
            )

        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                "Unsupported recovery checkpoint schema version: "
                f"{payload['schema_version']}"
            )
        if check_compatibility:
            compatibility_fields = {
                "run_kind": self.run_kind,
                "run_id": self.run_id,
                "signature": self.signature,
            }
            for field, expected_value in compatibility_fields.items():
                if payload[field] != expected_value:
                    raise CheckpointCompatibilityError(
                        f"Recovery checkpoint {field} mismatch: expected "
                        f"{expected_value!r}, received {payload[field]!r}"
                    )

        status = payload["status"]
        if status not in {"in_progress", "completed"}:
            raise TrainingCheckpointError(
                "Recovery checkpoint status must be in_progress or completed"
            )

        completed_epoch = payload["completed_epoch"]
        total_epochs = payload["total_epochs"]
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or isinstance(total_epochs, bool)
            or not isinstance(total_epochs, int)
            or total_epochs <= 0
            or not 0 <= completed_epoch <= total_epochs
        ):
            raise TrainingCheckpointError(
                "Recovery checkpoint has an invalid epoch range"
            )
        if status == "completed" and completed_epoch != total_epochs:
            raise TrainingCheckpointError(
                "A completed checkpoint must contain the final epoch"
            )
        if status == "in_progress" and completed_epoch >= total_epochs:
            raise TrainingCheckpointError(
                "An in-progress checkpoint cannot contain the final epoch"
            )

        history = payload["history"]
        if not isinstance(history, list) or len(history) != completed_epoch:
            raise TrainingCheckpointError(
                "Recovery history length must match completed_epoch"
            )
        if not isinstance(payload["model_state_dict"], Mapping):
            raise TrainingCheckpointError("model_state_dict must be a mapping")
        if not isinstance(payload["optimizer_state_dict"], Mapping):
            raise TrainingCheckpointError(
                "optimizer_state_dict must be a mapping"
            )
        if not isinstance(payload["rng_state"], Mapping):
            raise TrainingCheckpointError("rng_state must be a mapping")
        if not isinstance(payload["log_directory"], str):
            raise TrainingCheckpointError("log_directory must be a string")

        elapsed_seconds = payload["elapsed_seconds"]
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise TrainingCheckpointError(
                "elapsed_seconds must be a finite non-negative number"
            )
