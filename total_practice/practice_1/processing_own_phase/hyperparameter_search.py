"""Staged hyperparameter-search utilities for FashionMNIST experiments."""

from collections.abc import Mapping, Sequence
import csv
import io
import json
import math
import os
from pathlib import Path
from statistics import fmean, median, pstdev
import tempfile
from typing import Any, Union


PathLike = Union[str, Path]
REQUIRED_BASE_FIELDS = {
    "num_classes",
    "input_dim",
    "weight_decay",
    "batch_size",
    "dropout",
    "optimizer",
    "use_augmentation",
}


def _validate_learning_rate(learning_rate: float) -> float:
    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate,
        (int, float),
    ):
        raise TypeError("learning_rate must be a real number")
    resolved_learning_rate = float(learning_rate)
    if not math.isfinite(resolved_learning_rate):
        raise ValueError("learning_rate must be finite")
    if not 0.0 < resolved_learning_rate <= 1.0:
        raise ValueError("learning_rate must be in the interval (0, 1]")
    return resolved_learning_rate


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(hidden_dims, Sequence) or isinstance(
        hidden_dims,
        (str, bytes, bytearray),
    ):
        raise TypeError("hidden_dims must be a sequence of integers")
    resolved_hidden_dims = tuple(hidden_dims)
    if not resolved_hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    if any(
        isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
        for hidden_dim in resolved_hidden_dims
    ):
        raise ValueError("hidden_dims must contain positive integers")
    return resolved_hidden_dims


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _learning_rate_slug(learning_rate: float) -> str:
    return (
        f"{learning_rate:.3e}"
        .replace(".", "p")
        .replace("+", "")
    )


def candidate_key(config: Mapping[str, Any]) -> str:
    """Return a stable identity that excludes stage, seed and epoch budget."""
    hidden_dims = _validate_hidden_dims(config["hidden_dims"])
    learning_rate = _validate_learning_rate(config["learning_rate"])
    optimizer = str(config["optimizer"]).lower()
    dropout = float(config.get("dropout", 0.0))
    augmentation = int(bool(config.get("use_augmentation", False)))
    hidden_slug = "x".join(str(hidden_dim) for hidden_dim in hidden_dims)
    dropout_slug = f"{dropout:.2f}".replace(".", "p")
    return (
        f"h{hidden_slug}_lr{_learning_rate_slug(learning_rate)}_"
        f"do{dropout_slug}_{optimizer}_aug{augmentation}"
    )


def build_trial_config(
    base_config: Mapping[str, Any],
    *,
    stage: str,
    hidden_dims: Sequence[int],
    learning_rate: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Create one validated config with a deterministic experiment ID."""
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    missing_fields = sorted(REQUIRED_BASE_FIELDS.difference(base_config))
    if missing_fields:
        raise KeyError(
            "Missing base config fields: " + ", ".join(missing_fields)
        )
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")

    resolved_hidden_dims = _validate_hidden_dims(hidden_dims)
    resolved_learning_rate = _validate_learning_rate(learning_rate)
    resolved_epochs = _validate_positive_integer("epochs", epochs)
    resolved_seed = _validate_positive_integer("seed", seed)
    stage_slug = stage.strip().lower().replace("_", "-")
    hidden_slug = "x".join(str(hidden_dim) for hidden_dim in resolved_hidden_dims)
    experiment_id = (
        f"H_{stage_slug}_h{hidden_slug}_"
        f"lr{_learning_rate_slug(resolved_learning_rate)}_"
        f"e{resolved_epochs}_s{resolved_seed}"
    )
    config = {
        **dict(base_config),
        "experiment_id": experiment_id,
        "hidden_dims": resolved_hidden_dims,
        "learning_rate": resolved_learning_rate,
        "epochs": resolved_epochs,
        "seed": resolved_seed,
        "search_stage": stage_slug,
    }
    config["candidate_key"] = candidate_key(config)
    return config


def _ensure_unique_trials(
    trial_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_trials = [dict(config) for config in trial_configs]
    experiment_ids = [config["experiment_id"] for config in normalized_trials]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Generated hyperparameter trial IDs must be unique")
    return normalized_trials


def generate_learning_rate_trials(
    base_config: Mapping[str, Any],
    learning_rates: Sequence[float],
    *,
    hidden_dims: Sequence[int],
    epochs: int,
    seed: int,
    stage: str = "learning-rate",
) -> list[dict[str, Any]]:
    """Generate one controlled trial per learning rate."""
    if not learning_rates:
        raise ValueError("learning_rates must not be empty")
    trials = [
        build_trial_config(
            base_config,
            stage=stage,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed,
        )
        for learning_rate in learning_rates
    ]
    return _ensure_unique_trials(trials)


def generate_architecture_trials(
    base_config: Mapping[str, Any],
    hidden_dim_candidates: Sequence[Sequence[int]],
    *,
    learning_rate: float,
    epochs: int,
    seed: int,
    stage: str = "architecture",
) -> list[dict[str, Any]]:
    """Generate controlled depth/capacity trials at one learning rate."""
    if not hidden_dim_candidates:
        raise ValueError("hidden_dim_candidates must not be empty")
    trials = [
        build_trial_config(
            base_config,
            stage=stage,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            epochs=epochs,
            seed=seed,
        )
        for hidden_dims in hidden_dim_candidates
    ]
    return _ensure_unique_trials(trials)


def refinement_learning_rates(
    best_learning_rate: float,
    *,
    factor: float = math.sqrt(3.0),
    minimum: float = 1e-5,
    maximum: float = 1e-2,
) -> tuple[float, ...]:
    """Return bounded geometric neighbors around a coarse best rate."""
    best_rate = _validate_learning_rate(best_learning_rate)
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("factor must be finite and greater than one")
    minimum_rate = _validate_learning_rate(minimum)
    maximum_rate = _validate_learning_rate(maximum)
    if minimum_rate >= maximum_rate:
        raise ValueError("minimum must be smaller than maximum")

    candidates = {
        max(minimum_rate, min(maximum_rate, best_rate / factor)),
        max(minimum_rate, min(maximum_rate, best_rate)),
        max(minimum_rate, min(maximum_rate, best_rate * factor)),
    }
    return tuple(sorted(candidates))


def generate_confirmation_trials(
    base_config: Mapping[str, Any],
    candidate_configs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    epochs: int,
) -> list[dict[str, Any]]:
    """Repeat top candidates over independent seeds and a longer budget."""
    if not candidate_configs:
        raise ValueError("candidate_configs must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    trials = []
    for candidate_config in candidate_configs:
        for seed in seeds:
            trials.append(build_trial_config(
                base_config,
                stage="confirmation",
                hidden_dims=candidate_config["hidden_dims"],
                learning_rate=candidate_config["learning_rate"],
                epochs=epochs,
                seed=seed,
            ))
    return _ensure_unique_trials(trials)


def _validated_result_metrics(
    result: Mapping[str, Any],
) -> tuple[float, float, int]:
    validation_accuracy = float(result["best_validation_accuracy"])
    validation_loss = float(result["best_validation_loss"])
    parameter_count = int(result["parameter_count"])
    if not 0.0 <= validation_accuracy <= 1.0:
        raise ValueError("validation accuracy must be in the [0, 1] range")
    if not math.isfinite(validation_loss) or validation_loss < 0.0:
        raise ValueError("validation loss must be finite and non-negative")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return validation_accuracy, validation_loss, parameter_count


def rank_results(
    results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Rank trials by accuracy, loss, model size and deterministic ID."""
    if not results:
        raise ValueError("results must not be empty")
    for result in results:
        _validated_result_metrics(result)
    return sorted(
        results,
        key=lambda result: (
            -float(result["best_validation_accuracy"]),
            float(result["best_validation_loss"]),
            int(result["parameter_count"]),
            str(result["experiment_id"]),
        ),
    )


def aggregate_confirmation_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate confirmation trials by seed-independent candidate key."""
    if not results:
        raise ValueError("results must not be empty")
    grouped_results: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        _validated_result_metrics(result)
        config = result["config"]
        key = str(config.get("candidate_key") or candidate_key(config))
        grouped_results.setdefault(key, []).append(result)

    summaries = []
    for key, candidate_results in grouped_results.items():
        accuracies = [
            float(result["best_validation_accuracy"])
            for result in candidate_results
        ]
        losses = [
            float(result["best_validation_loss"])
            for result in candidate_results
        ]
        best_epochs = [int(result["best_epoch"]) for result in candidate_results]
        runtimes = [float(result["total_seconds"]) for result in candidate_results]
        first_config = dict(candidate_results[0]["config"])
        summaries.append({
            "candidate_key": key,
            "hidden_dims": tuple(first_config["hidden_dims"]),
            "learning_rate": float(first_config["learning_rate"]),
            "optimizer": str(first_config["optimizer"]),
            "parameter_count": int(candidate_results[0]["parameter_count"]),
            "trial_count": len(candidate_results),
            "mean_validation_accuracy": fmean(accuracies),
            "std_validation_accuracy": (
                pstdev(accuracies) if len(accuracies) > 1 else 0.0
            ),
            "mean_validation_loss": fmean(losses),
            "median_best_epoch": int(round(median(best_epochs))),
            "mean_total_seconds": fmean(runtimes),
            "trial_ids": [
                str(result["experiment_id"])
                for result in candidate_results
            ],
            "config": first_config,
        })

    return sorted(
        summaries,
        key=lambda summary: (
            -summary["mean_validation_accuracy"],
            summary["mean_validation_loss"],
            summary["std_validation_accuracy"],
            summary["parameter_count"],
            summary["candidate_key"],
        ),
    )


def result_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project an in-memory result to JSON/CSV-safe search metadata."""
    config = result["config"]
    return {
        "experiment_id": str(result["experiment_id"]),
        "search_stage": str(config["search_stage"]),
        "candidate_key": str(config["candidate_key"]),
        "hidden_dims": list(config["hidden_dims"]),
        "learning_rate": float(config["learning_rate"]),
        "epochs": int(config["epochs"]),
        "seed": int(config["seed"]),
        "best_epoch": int(result["best_epoch"]),
        "validation_accuracy": float(result["best_validation_accuracy"]),
        "validation_loss": float(result["best_validation_loss"]),
        "parameter_count": int(result["parameter_count"]),
        "total_seconds": float(result["total_seconds"]),
        "resumed_from_epoch": int(result["resumed_from_epoch"]),
        "recovery_checkpoint_path": result["recovery_checkpoint_path"],
    }


def _atomic_write_text(text: str, destination: PathLike) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def export_search_outputs(
    results: Sequence[Mapping[str, Any]],
    confirmation_summaries: Sequence[Mapping[str, Any]],
    selected_summary: Mapping[str, Any],
    output_directory: PathLike,
) -> None:
    """Atomically export trial records, confirmation ranking and best config."""
    output_path = Path(output_directory)
    records = [result_record(result) for result in results]
    serializable_summaries = []
    for summary in confirmation_summaries:
        normalized_summary = dict(summary)
        normalized_summary["hidden_dims"] = list(summary["hidden_dims"])
        normalized_summary["config"] = {
            **dict(summary["config"]),
            "hidden_dims": list(summary["config"]["hidden_dims"]),
        }
        serializable_summaries.append(normalized_summary)

    manifest = {
        "trials": records,
        "confirmation_ranking": serializable_summaries,
    }
    _atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        output_path / "search_summary.json",
    )

    csv_buffer = io.StringIO()
    fieldnames = list(records[0]) if records else []
    if fieldnames:
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = dict(record)
            csv_record["hidden_dims"] = "x".join(
                str(hidden_dim) for hidden_dim in record["hidden_dims"]
            )
            writer.writerow(csv_record)
    _atomic_write_text(csv_buffer.getvalue(), output_path / "search_summary.csv")

    best_payload = {
        "candidate_key": selected_summary["candidate_key"],
        "hidden_dims": list(selected_summary["hidden_dims"]),
        "learning_rate": selected_summary["learning_rate"],
        "optimizer": selected_summary["optimizer"],
        "median_best_epoch": selected_summary["median_best_epoch"],
        "mean_validation_accuracy": (
            selected_summary["mean_validation_accuracy"]
        ),
        "std_validation_accuracy": (
            selected_summary["std_validation_accuracy"]
        ),
        "mean_validation_loss": selected_summary["mean_validation_loss"],
        "parameter_count": selected_summary["parameter_count"],
        "trial_ids": list(selected_summary["trial_ids"]),
    }
    _atomic_write_text(
        json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
        output_path / "best_hyperparameters.json",
    )
