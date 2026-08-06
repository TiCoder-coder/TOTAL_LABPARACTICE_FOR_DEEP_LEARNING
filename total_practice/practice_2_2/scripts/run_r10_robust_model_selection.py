from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r10_robust_model_selection import (
    build_r10_report,
    freeze_selected_checkpoint,
    json_sha256,
    load_r10_policy,
    select_robust_finalist,
    summarize_finalist_results,
    write_json,
)
from practice_2_2.resources import file_sha256


def cyclic_confusion(correct_per_class: int, support_per_class: int = 50) -> list[list[int]]:
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for index in range(10):
        matrix[index][index] = correct_per_class
        matrix[index][(index + 1) % 10] = support_per_class - correct_per_class
    return matrix


def build_result_fixture(
    policy: dict,
    configurations: dict,
    inconsistent: bool = False,
) -> list[dict]:
    finalist_ids = list(configurations)
    if inconsistent:
        correct = {
            "F0": {42: 38, 123: 38, 2026: 38},
            "F1": {42: 44, 123: 36, 2026: 36},
        }
        train = {"F0": 0.88, "F1": 0.89}
    else:
        correct = {
            "F0": {42: 35, 123: 36, 2026: 37},
            "F1": {42: 38, 123: 39, 2026: 40},
        }
        train = {"F0": 0.88, "F1": 0.86}
    rows = []
    for finalist_id in finalist_ids:
        configuration_hash = json_sha256(configurations[finalist_id])
        initial_hash = json_sha256(
            {"finalist_id": finalist_id, "state": "shared_initialization"}
        )
        for seed in policy["required_seeds"]:
            rows.append(
                {
                    "finalist_id": finalist_id,
                    "seed": seed,
                    "configuration_sha256": configuration_hash,
                    "initial_state_sha256": initial_hash,
                    "model_state_sha256": json_sha256(
                        {
                            "finalist_id": finalist_id,
                            "seed": seed,
                            "kind": "model_state",
                        }
                    ),
                    "checkpoint_sha256": json_sha256(
                        {
                            "finalist_id": finalist_id,
                            "seed": seed,
                            "kind": "checkpoint_fixture",
                        }
                    ),
                    "train_accuracy": train[finalist_id] + (seed % 3) / 1000.0,
                    "best_epoch": 9,
                    "confusion_matrix": cyclic_confusion(correct[finalist_id][seed]),
                }
            )
    return rows


def verify_robust_selection(policy: dict) -> tuple[dict, list[dict], dict[str, bool]]:
    configurations = {
        "F0": {
            "architecture": "resnet18",
            "protocol": "staged",
            "dropout": 0.2,
        },
        "F1": {
            "architecture": "efficientnet_b0",
            "protocol": "staged",
            "dropout": 0.2,
        },
    }
    finalist_ids = list(configurations)
    results = build_result_fixture(policy, configurations)
    summaries = summarize_finalist_results(results, finalist_ids, policy)
    selection = select_robust_finalist(results, summaries, finalist_ids, policy)
    inconsistent_results = build_result_fixture(
        policy, configurations, inconsistent=True
    )
    inconsistent_summaries = summarize_finalist_results(
        inconsistent_results, finalist_ids, policy
    )
    inconsistent_selection = select_robust_finalist(
        inconsistent_results,
        inconsistent_summaries,
        finalist_ids,
        policy,
    )
    unauthorized = [dict(row) for row in results]
    unauthorized[0]["test_accuracy"] = 0.99
    test_evidence_rejected = False
    try:
        summarize_finalist_results(unauthorized, finalist_ids, policy)
    except ValueError:
        test_evidence_rejected = True
    summary_fields = all(
        "std_validation_accuracy" in summary
        and "std_validation_macro_f1" in summary
        for summary in summaries
    )
    wilson_fields = all(
        len(summary["validation_accuracy_wilson_95_by_seed"])
        == len(policy["required_seeds"])
        and len(summary["descriptive_pooled_validation_accuracy_wilson_95"]) == 2
        for summary in summaries
    )
    per_class_fields = all(
        set(summary["per_class_recall"]) == set(policy["class_names"])
        for summary in summaries
    )
    confusion_fields = all(
        set(summary["confusion_stability"])
        == set(policy["confusion_stability"]["reported_fields"])
        for summary in summaries
    )
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_robust_selection_logic_only",
        "fixture_is_model_selection_evidence": False,
        "finalist_ids": finalist_ids,
        "configuration_snapshots": configurations,
        "consistent_fixture_selection": selection,
        "inconsistent_fixture_selection": inconsistent_selection,
        "test_field_rejection_verified": test_evidence_rejected,
        "authorized_finalist_id": None,
        "test_evidence_used": False,
    }
    checks = {
        "mean_and_standard_deviation": summary_fields,
        "wilson_intervals": wilson_fields,
        "per_class_recall": per_class_fields,
        "confusion_stability": confusion_fields,
        "consistent_seed_winner": selection["selected_finalist_id"] == "F1"
        and selection["seed_wins"]["F1"] >= 2,
        "inconsistent_seed_leader_rejected": inconsistent_selection[
            "selected_finalist_id"
        ]
        is None,
        "test_evidence_rejected": test_evidence_rejected,
        "no_fixture_finalist_authorized": report["authorized_finalist_id"] is None,
    }
    return report, summaries, checks


def verify_freeze_logic(
    policy: dict,
    selection_report: dict,
    summaries: list[dict],
) -> tuple[dict, dict[str, bool]]:
    configurations = selection_report["configuration_snapshots"]
    finalist_ids = selection_report["finalist_ids"]
    results = build_result_fixture(policy, configurations)
    selection = selection_report["consistent_fixture_selection"]
    selected = selection["selected_finalist_id"]
    with tempfile.TemporaryDirectory(prefix="r10_checkpoint_freeze_") as directory:
        checkpoint_path = Path(directory) / "selected_seed_42.pt"
        checkpoint_path.write_bytes(b"R10 synthetic checkpoint freeze fixture")
        checkpoint_hash = file_sha256(checkpoint_path)
        selected_row = next(
            row
            for row in results
            if row["finalist_id"] == selected and row["seed"] == 42
        )
        selected_row["checkpoint_sha256"] = checkpoint_hash
        freeze_record = freeze_selected_checkpoint(
            selection,
            results,
            configurations[selected],
            checkpoint_path,
            policy,
        )
        tampered_selection = dict(selection)
        tampered_selection["mean_macro_f1_margin"] = 1.0
        tampered_selection_rejected = False
        try:
            freeze_selected_checkpoint(
                tampered_selection,
                results,
                configurations[selected],
                checkpoint_path,
                policy,
            )
        except RuntimeError:
            tampered_selection_rejected = True
    required_hashes = set(policy["final_checkpoint"]["required_frozen_hashes"])
    frozen_hashes = {
        name: freeze_record[name]
        for name in required_hashes
    }
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_checkpoint_freeze_logic_only",
        "fixture_is_final_checkpoint": False,
        "selected_fixture_finalist_id": selected,
        "final_checkpoint_rule": freeze_record["final_checkpoint_rule"],
        "final_checkpoint_seed": freeze_record["final_checkpoint_seed"],
        "frozen_hashes": frozen_hashes,
        "freeze_record_sha256": freeze_record["freeze_record_sha256"],
        "selection_record_tamper_rejected": tampered_selection_rejected,
        "authorized_final_checkpoint": None,
        "test_evidence_used": False,
    }
    checks = {
        "freeze_hash_contract": set(frozen_hashes) == required_hashes
        and all(len(value) == 64 for value in frozen_hashes.values()),
        "predeclared_checkpoint_seed": freeze_record["final_checkpoint_seed"] == 42,
        "selection_record_tamper_rejected": tampered_selection_rejected,
        "no_fixture_checkpoint_authorized": report[
            "authorized_final_checkpoint"
        ]
        is None,
    }
    return report, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r9-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r10_policy()
    r9_report = json.loads(args.r9_verification.read_text())
    root = get_practice_2_2_root().resolve()
    notebook_path = root / "notebooks/04_canonical_report.ipynb"
    notebook_hash_before = file_sha256(notebook_path)
    selection_report, summaries, selection_checks = verify_robust_selection(policy)
    freeze_report, freeze_checks = verify_freeze_logic(
        policy, selection_report, summaries
    )
    checks = {**selection_checks, **freeze_checks}
    checks["canonical_notebook_unchanged"] = (
        file_sha256(notebook_path) == notebook_hash_before
    )
    report = build_r10_report(
        r9_report,
        checks,
        repeated_seed_results_available=False,
        selected_finalist_id=None,
        final_checkpoint_frozen=False,
        policy=policy,
    )
    report["protocol_checks"] = checks
    report["canonical_notebook_sha256"] = notebook_hash_before
    prerequisite_report = {
        "schema_version": 1,
        "r9_gate_passed": bool(r9_report.get("gate_passed")),
        "r9_authorized_architecture_experiment_id": r9_report.get(
            "authorized_architecture_experiment_id"
        ),
        "repeated_seed_finalist_results_available": False,
        "authorized_finalist_id": None,
        "authorized_final_checkpoint": None,
        "test_evidence_used": False,
    }
    artifact_paths = [
        write_json(policy, args.output_dir / "policy_snapshot.json"),
        write_json(
            prerequisite_report, args.output_dir / "prerequisite_report.json"
        ),
        write_json(
            selection_report, args.output_dir / "selection_logic_verification.json"
        ),
        write_json(summaries, args.output_dir / "uncertainty_report.json"),
        write_json(
            freeze_report, args.output_dir / "freeze_logic_verification.json"
        ),
        write_json(report, args.output_dir / "robust_model_selection_report.json"),
        write_json(report, args.report_output),
    ]
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "synthetic_protocol_verification_only": True,
        "fixture_is_model_selection_evidence": False,
        "selected_finalist_id": None,
        "final_checkpoint_frozen": False,
        "real_training_performed": False,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "test_loader_constructed": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "canonical_notebook_sha256": notebook_hash_before,
        "artifacts": [
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(artifact_manifest, args.output_dir / "artifact_manifest.json")
    output = {
        "status": report["status"],
        "gate_passed": report["gate_passed"],
        "protocol_exit_checks_passed": report["protocol_exit_checks_passed"],
        "selected_finalist_id": None,
        "final_checkpoint_frozen": False,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
