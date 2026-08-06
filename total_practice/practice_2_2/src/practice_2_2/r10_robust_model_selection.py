from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import get_practice_2_2_root
from .resources import file_sha256


POLICY_RELATIVE_PATH = Path("configs/r10_robust_model_selection_policy.json")
RESULT_FIELDS = {
    "finalist_id",
    "seed",
    "configuration_sha256",
    "initial_state_sha256",
    "model_state_sha256",
    "checkpoint_sha256",
    "train_accuracy",
    "best_epoch",
    "confusion_matrix",
}


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_r10_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R10 robust-selection policy schema")
    if policy.get("predecessor_phase") != "R9":
        raise RuntimeError("R10 predecessor must be R9")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R10 must require the R9 gate")
    if policy.get("required_seeds") != [42, 123, 2026]:
        raise RuntimeError("R10 required seeds are invalid")
    if len(policy.get("class_names", [])) != 10:
        raise RuntimeError("R10 must define ten ordered classes")
    selection = policy.get("selection", {})
    if selection.get("primary") != "mean_validation_macro_f1":
        raise RuntimeError("R10 primary metric must be mean Validation Macro F1")
    if selection.get("secondary") != "mean_validation_accuracy":
        raise RuntimeError("R10 secondary metric must be mean Validation Accuracy")
    if selection.get("minimum_seed_wins", 0) < 2:
        raise RuntimeError("R10 winner must be consistent across seeds")
    checkpoint = policy.get("final_checkpoint", {})
    if checkpoint.get("seed") != 42:
        raise RuntimeError("R10 final checkpoint seed must be predeclared as 42")
    if checkpoint.get("strict_file_sha256_verification") is not True:
        raise RuntimeError("R10 checkpoint file hash verification must be strict")
    if policy.get("minimum_finalists", 0) < 2:
        raise RuntimeError("R10 requires at least two finalists")
    if policy.get("selection_record_test_fields_allowed") is not False:
        raise RuntimeError("R10 selection record must prohibit Test fields")
    if policy.get("test_evidence_allowed") is not False:
        raise RuntimeError("R10 must prohibit Test evidence")
    if policy.get("test_content_access_allowed") is not False:
        raise RuntimeError("R10 must prohibit Test content access")
    if policy.get("test_loader_construction_allowed") is not False:
        raise RuntimeError("R10 must prohibit Test loader construction")
    return policy


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval counts are invalid")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    class_count: int,
) -> list[list[int]]:
    matrix = [list(row) for row in confusion_matrix]
    if len(matrix) != class_count or any(len(row) != class_count for row in matrix):
        raise ValueError("R10 confusion matrix shape is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for row in matrix
        for value in row
    ):
        raise ValueError("R10 confusion matrix counts must be non-negative integers")
    if any(sum(row) == 0 for row in matrix):
        raise ValueError("R10 every Validation class must have positive support")
    return matrix


def metrics_from_confusion(
    confusion_matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
) -> dict[str, Any]:
    matrix = validate_confusion_matrix(confusion_matrix, len(class_names))
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    recalls = []
    precisions = []
    f1_scores = []
    per_class = {}
    for index, class_name in enumerate(class_names):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        recall = true_positive / support
        precision = true_positive / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    lower, upper = wilson_interval(correct, total)
    return {
        "sample_count": total,
        "correct_count": correct,
        "accuracy": correct / total,
        "macro_f1": statistics.mean(f1_scores),
        "macro_recall": statistics.mean(recalls),
        "macro_precision": statistics.mean(precisions),
        "accuracy_wilson_95": [lower, upper],
        "per_class": per_class,
    }


def normalized_confusion(
    confusion_matrix: Sequence[Sequence[int]],
) -> list[list[float]]:
    matrix = [list(row) for row in confusion_matrix]
    return [
        [value / sum(row) for value in row]
        for row in matrix
    ]


def confusion_stability(
    confusion_matrices: Sequence[Sequence[Sequence[int]]],
) -> dict[str, float]:
    if len(confusion_matrices) < 2:
        raise ValueError("R10 confusion stability requires multiple seeds")
    normalized = [normalized_confusion(matrix) for matrix in confusion_matrices]
    pairwise = []
    for left, right in itertools.combinations(normalized, 2):
        differences = [
            abs(left[row][column] - right[row][column])
            for row in range(len(left))
            for column in range(len(left[row]))
        ]
        pairwise.append(statistics.mean(differences))
    cell_standard_deviations = [
        statistics.pstdev(
            matrix[row][column] for matrix in normalized
        )
        for row in range(len(normalized[0]))
        for column in range(len(normalized[0][row]))
    ]
    return {
        "mean_pairwise_normalized_confusion_difference": statistics.mean(pairwise),
        "maximum_normalized_cell_standard_deviation": max(
            cell_standard_deviations
        ),
    }


def _validate_result_rows(
    results: Sequence[Mapping[str, Any]],
    finalist_ids: Sequence[str],
    policy: Mapping[str, Any],
) -> None:
    if len(set(finalist_ids)) < policy["minimum_finalists"]:
        raise ValueError("R10 requires at least two unique finalists")
    expected_pairs = {
        (finalist_id, seed)
        for finalist_id in finalist_ids
        for seed in policy["required_seeds"]
    }
    observed_pairs = set()
    initial_hashes = {finalist_id: set() for finalist_id in finalist_ids}
    configuration_hashes = {finalist_id: set() for finalist_id in finalist_ids}
    for row in results:
        if set(row) != RESULT_FIELDS:
            raise ValueError("R10 result contains missing or unauthorized fields")
        finalist_id = str(row["finalist_id"])
        if finalist_id not in initial_hashes:
            raise ValueError("R10 result contains an undeclared finalist")
        pair = (finalist_id, int(row["seed"]))
        if pair in observed_pairs:
            raise ValueError("R10 result contains a duplicate finalist seed")
        observed_pairs.add(pair)
        for field in (
            "configuration_sha256",
            "initial_state_sha256",
            "model_state_sha256",
            "checkpoint_sha256",
        ):
            if not _is_sha256(row[field]):
                raise ValueError(f"R10 result has an invalid hash: {field}")
        initial_hashes[finalist_id].add(str(row["initial_state_sha256"]))
        configuration_hashes[finalist_id].add(str(row["configuration_sha256"]))
        if not 0.0 <= float(row["train_accuracy"]) <= 1.0:
            raise ValueError("R10 Train Accuracy is invalid")
        if int(row["best_epoch"]) <= 0:
            raise ValueError("R10 best epoch is invalid")
        validate_confusion_matrix(row["confusion_matrix"], len(policy["class_names"]))
    if observed_pairs != expected_pairs:
        raise ValueError("R10 results do not cover every finalist seed")
    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError("R10 finalist seeds do not share initialization")
    if any(len(values) != 1 for values in configuration_hashes.values()):
        raise ValueError("R10 finalist seeds do not share one configuration")


def summarize_finalist_results(
    results: Sequence[Mapping[str, Any]],
    finalist_ids: Sequence[str],
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r10_policy())
    _validate_result_rows(results, finalist_ids, policy)
    class_names = policy["class_names"]
    summaries = []
    for finalist_id in finalist_ids:
        rows = sorted(
            (row for row in results if row["finalist_id"] == finalist_id),
            key=lambda row: int(row["seed"]),
        )
        seed_metrics = [
            {
                "seed": int(row["seed"]),
                **metrics_from_confusion(row["confusion_matrix"], class_names),
            }
            for row in rows
        ]
        accuracies = [item["accuracy"] for item in seed_metrics]
        macro_f1 = [item["macro_f1"] for item in seed_metrics]
        train_accuracies = [float(row["train_accuracy"]) for row in rows]
        gaps = [
            train_accuracy - validation_accuracy
            for train_accuracy, validation_accuracy in zip(
                train_accuracies, accuracies
            )
        ]
        per_class_recall = {}
        for class_name in class_names:
            values = [
                item["per_class"][class_name]["recall"]
                for item in seed_metrics
            ]
            per_class_recall[class_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        pooled_correct = sum(item["correct_count"] for item in seed_metrics)
        pooled_total = sum(item["sample_count"] for item in seed_metrics)
        pooled_wilson = wilson_interval(
            pooled_correct,
            pooled_total,
            z=float(policy["uncertainty"]["wilson_z"]),
        )
        summaries.append(
            {
                "finalist_id": finalist_id,
                "seeds": [item["seed"] for item in seed_metrics],
                "seed_count": len(seed_metrics),
                "configuration_sha256": str(rows[0]["configuration_sha256"]),
                "initial_state_sha256": str(rows[0]["initial_state_sha256"]),
                "mean_train_accuracy": statistics.mean(train_accuracies),
                "mean_validation_accuracy": statistics.mean(accuracies),
                "std_validation_accuracy": statistics.stdev(accuracies),
                "mean_validation_macro_f1": statistics.mean(macro_f1),
                "std_validation_macro_f1": statistics.stdev(macro_f1),
                "mean_generalization_gap": statistics.mean(gaps),
                "validation_accuracy_wilson_95_by_seed": [
                    {
                        "seed": item["seed"],
                        "interval": item["accuracy_wilson_95"],
                    }
                    for item in seed_metrics
                ],
                "descriptive_pooled_validation_accuracy_wilson_95": list(
                    pooled_wilson
                ),
                "per_class_recall": per_class_recall,
                "confusion_stability": confusion_stability(
                    [row["confusion_matrix"] for row in rows]
                ),
                "test_evidence_used": False,
            }
        )
    return summaries


def _seed_winners(
    results: Sequence[Mapping[str, Any]],
    finalist_ids: Sequence[str],
    policy: Mapping[str, Any],
) -> dict[str, int]:
    wins = {finalist_id: 0 for finalist_id in finalist_ids}
    for seed in policy["required_seeds"]:
        candidates = []
        for finalist_id in finalist_ids:
            row = next(
                row
                for row in results
                if row["finalist_id"] == finalist_id and int(row["seed"]) == seed
            )
            metrics = metrics_from_confusion(
                row["confusion_matrix"], policy["class_names"]
            )
            candidates.append(
                (
                    metrics["macro_f1"],
                    metrics["accuracy"],
                    -(
                        float(row["train_accuracy"])
                        - metrics["accuracy"]
                    ),
                    finalist_id,
                )
            )
        winner = max(candidates)[3]
        wins[winner] += 1
    return wins


def select_robust_finalist(
    results: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    finalist_ids: Sequence[str],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r10_policy())
    _validate_result_rows(results, finalist_ids, policy)
    by_id = {str(summary["finalist_id"]): summary for summary in summaries}
    if set(by_id) != set(finalist_ids):
        raise ValueError("R10 finalist summary coverage is incomplete")
    if any(summary.get("test_evidence_used") is not False for summary in summaries):
        raise ValueError("R10 selection cannot use Test evidence")
    ranking = sorted(
        summaries,
        key=lambda summary: (
            float(summary["mean_validation_macro_f1"]),
            float(summary["mean_validation_accuracy"]),
            -float(summary["mean_generalization_gap"]),
            str(summary["finalist_id"]),
        ),
        reverse=True,
    )
    leader = ranking[0]
    runner_up = ranking[1]
    margin = (
        float(leader["mean_validation_macro_f1"])
        - float(runner_up["mean_validation_macro_f1"])
    )
    wins = _seed_winners(results, finalist_ids, policy)
    consistent = (
        wins[str(leader["finalist_id"])]
        >= policy["selection"]["minimum_seed_wins"]
        and margin >= policy["selection"]["minimum_mean_macro_f1_margin"]
    )
    record = {
        "selected_finalist_id": str(leader["finalist_id"]) if consistent else None,
        "primary_metric": policy["selection"]["primary"],
        "secondary_metric": policy["selection"]["secondary"],
        "leader_mean_validation_macro_f1": leader[
            "mean_validation_macro_f1"
        ],
        "runner_up_mean_validation_macro_f1": runner_up[
            "mean_validation_macro_f1"
        ],
        "mean_macro_f1_margin": margin,
        "seed_wins": wins,
        "minimum_seed_wins": policy["selection"]["minimum_seed_wins"],
        "consistent_across_seeds": consistent,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
    }
    record["selection_record_sha256"] = json_sha256(record)
    return record


def freeze_selected_checkpoint(
    selection_record: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    configuration_snapshot: Mapping[str, Any],
    checkpoint_path: Path,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r10_policy())
    selected = selection_record.get("selected_finalist_id")
    if selected is None or selection_record.get("test_evidence_used") is not False:
        raise ValueError("R10 cannot freeze an unselected or Test-derived finalist")
    selection_payload = dict(selection_record)
    stored_selection_hash = selection_payload.pop("selection_record_sha256", None)
    if stored_selection_hash != json_sha256(selection_payload):
        raise RuntimeError("R10 selection record hash mismatch")
    seed = int(policy["final_checkpoint"]["seed"])
    row = next(
        (
            row
            for row in results
            if row["finalist_id"] == selected and int(row["seed"]) == seed
        ),
        None,
    )
    if row is None:
        raise ValueError("R10 final checkpoint seed result is unavailable")
    configuration_hash = json_sha256(configuration_snapshot)
    if configuration_hash != row["configuration_sha256"]:
        raise RuntimeError("R10 configuration hash mismatch")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != row["checkpoint_sha256"]:
        raise RuntimeError("R10 checkpoint file hash mismatch")
    freeze_record = {
        "schema_version": 1,
        "phase": "R10",
        "selected_finalist_id": selected,
        "final_checkpoint_rule": policy["final_checkpoint"]["rule"],
        "final_checkpoint_seed": seed,
        "selection_record_sha256": selection_record[
            "selection_record_sha256"
        ],
        "configuration_sha256": configuration_hash,
        "model_state_sha256": row["model_state_sha256"],
        "checkpoint_file_sha256": checkpoint_hash,
        "checkpoint_path": str(checkpoint_path),
        "frozen_before_test_authorization": True,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
    }
    freeze_record["freeze_record_sha256"] = json_sha256(freeze_record)
    return freeze_record


def build_r10_report(
    r9_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    repeated_seed_results_available: bool,
    selected_finalist_id: str | None = None,
    final_checkpoint_frozen: bool = False,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r10_policy())
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    blocked_reasons = []
    if not r9_report.get("gate_passed"):
        blocked_reasons.append("R9_GATE_NOT_PASSED")
    if r9_report.get("authorized_architecture_experiment_id") is None:
        blocked_reasons.append("R9_ARCHITECTURE_NOT_AUTHORIZED")
    if not repeated_seed_results_available:
        blocked_reasons.append("REPEATED_SEED_FINALIST_RESULTS_UNAVAILABLE")
    if failed_checks:
        blocked_reasons.append("R10_PROTOCOL_CHECKS_FAILED")
    if selected_finalist_id is None:
        blocked_reasons.append("NO_ROBUST_FINALIST_SELECTED")
    if not final_checkpoint_frozen:
        blocked_reasons.append("FINAL_CHECKPOINT_NOT_FROZEN")
    return {
        "schema_version": 1,
        "phase": "R10",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "protocol_exit_checks_passed": not failed_checks,
        "failed_protocol_checks": failed_checks,
        "required_seeds": policy["required_seeds"],
        "primary_metric": policy["selection"]["primary"],
        "secondary_metric": policy["selection"]["secondary"],
        "mean_and_standard_deviation_reported": checks.get(
            "mean_and_standard_deviation", False
        ),
        "wilson_intervals_reported": checks.get("wilson_intervals", False),
        "per_class_recall_reported": checks.get("per_class_recall", False),
        "confusion_stability_reported": checks.get(
            "confusion_stability", False
        ),
        "consistent_seed_winner_logic_verified": checks.get(
            "consistent_seed_winner", False
        ),
        "test_derived_field_rejection_verified": checks.get(
            "test_evidence_rejected", False
        ),
        "freeze_hash_contract_verified": checks.get("freeze_hash_contract", False),
        "repeated_seed_finalist_results_available": repeated_seed_results_available,
        "selected_finalist_id": selected_finalist_id,
        "final_checkpoint_frozen": final_checkpoint_frozen,
        "selection_source": "Validation_only",
        "test_evidence_used": False,
        "test_loader_imported": False,
        "test_loader_constructed": False,
        "real_training_performed": False,
        "synthetic_protocol_verification_only": True,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "validation_evaluated": False,
        "test_evaluated": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "successor_phase_executed": False,
    }


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R10 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
