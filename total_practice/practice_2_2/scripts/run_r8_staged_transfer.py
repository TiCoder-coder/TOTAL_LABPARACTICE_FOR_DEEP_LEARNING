from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torchvision import models

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r7_training_correctness import (
    LOSS_CONTRACT,
    optimizer_learning_rates,
    state_dict_sha256,
)
from practice_2_2.r8_staged_transfer import (
    StagedTransferSession,
    build_r8_report,
    load_authorized_pretrained_state,
    load_r8_policy,
    select_candidate_from_validation,
    summarize_repeated_seed_results,
    trainable_parameter_names,
    write_json,
)
from practice_2_2.resources import file_sha256


def build_synthetic_pretrained_fixture() -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8042)
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in models.resnet18(weights=None).state_dict().items()
        }


def warmup_metrics(epoch: int) -> dict:
    losses = [1.4, 1.2, 1.25, 1.3]
    accuracies = [0.42, 0.51, 0.49, 0.48]
    macro_f1 = [0.4, 0.52, 0.5, 0.49]
    return {
        "split": "Validation",
        "contract": LOSS_CONTRACT,
        "validation_loss": losses[epoch - 1],
        "validation_accuracy": accuracies[epoch - 1],
        "validation_macro_f1": macro_f1[epoch - 1],
    }


def verify_staged_protocol(
    pretrained_state: dict[str, torch.Tensor],
    policy: dict,
) -> tuple[dict, dict[str, bool]]:
    sessions = []
    transition_records = []
    warmup_head_only = True
    layer4_only = True
    restore_verified = True
    controls_verified = True
    learning_rates_verified = True
    frozen_batchnorm_verified = True
    loss_configuration_verified = True
    loss_values = {}
    torch.manual_seed(8118)
    fixture_logits = torch.randn(6, policy["num_classes"], dtype=torch.float64)
    fixture_targets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    with tempfile.TemporaryDirectory(prefix="r8_staged_transfer_") as directory:
        root = Path(directory)
        for experiment_id in policy["experiments"]:
            session = StagedTransferSession(
                experiment_id,
                pretrained_state,
                root / f"{experiment_id}_warmup_best.pt",
                policy,
            )
            initial_names = trainable_parameter_names(session.model)
            session.prepare_training_mode()
            batchnorm_states = [
                (
                    name,
                    module.training,
                    any(
                        parameter.requires_grad
                        for parameter in module.parameters(recurse=False)
                    ),
                )
                for name, module in session.model.named_modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            ]
            frozen_batchnorm_verified &= all(
                not training
                for _, training, trainable in batchnorm_states
                if not trainable
            )
            loss_values[experiment_id] = float(
                session.compute_loss_terms(
                    fixture_logits, fixture_targets
                ).optimization_loss
            )
            if session.experiment["warmup_epochs"] == 0:
                layer4_only &= all(
                    name.startswith("layer4.") or name.startswith("fc.")
                    for name in initial_names
                )
                learning_rates_verified &= optimizer_learning_rates(
                    session.optimizer
                ) == {
                    "backbone": policy["optimizer"]["backbone_learning_rate"],
                    "head": policy["optimizer"]["head_learning_rate"],
                }
            else:
                warmup_head_only &= bool(initial_names) and all(
                    name.startswith("fc.") for name in initial_names
                )
                learning_rates_verified &= optimizer_learning_rates(
                    session.optimizer
                ) == {"head": policy["optimizer"]["head_learning_rate"]}
                for epoch in range(1, session.experiment["warmup_epochs"] + 1):
                    with torch.no_grad():
                        session.model.fc[1].bias.add_(epoch / 1000.0)
                    record = session.record_warmup_epoch(warmup_metrics(epoch))
                    controls_verified &= (
                        record["metric_split"] == "Validation"
                        and record["metric_contract"] == LOSS_CONTRACT
                        and record["scheduler_metric"]
                        == warmup_metrics(epoch)["validation_loss"]
                        and record["early_stopping_metric"]
                        == warmup_metrics(epoch)["validation_loss"]
                    )
                pre_transition_hash = state_dict_sha256(session.model.state_dict())
                with patch(
                    "torch.hub.load_state_dict_from_url",
                    side_effect=RuntimeError("network access forbidden"),
                ), patch(
                    "torchvision.models._api.load_state_dict_from_url",
                    side_effect=RuntimeError("network access forbidden"),
                ):
                    transition = session.transition_to_layer4()
                session.prepare_training_mode()
                transition_batchnorm_states = [
                    (
                        module.training,
                        any(
                            parameter.requires_grad
                            for parameter in module.parameters(recurse=False)
                        ),
                    )
                    for module in session.model.modules()
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                ]
                frozen_batchnorm_verified &= all(
                    not training
                    for training, trainable in transition_batchnorm_states
                    if not trainable
                ) and any(
                    training
                    for training, trainable in transition_batchnorm_states
                    if trainable
                )
                transition_names = transition["trainable_parameter_names"]
                layer4_only &= all(
                    name.startswith("layer4.") or name.startswith("fc.")
                    for name in transition_names
                )
                restore_verified &= (
                    session.best_warmup_epoch == 2
                    and pre_transition_hash != transition["restored_state_sha256"]
                    and transition["restored_state_sha256"]
                    == session.best_warmup_state_sha256
                )
                learning_rates_verified &= optimizer_learning_rates(
                    session.optimizer
                ) == {
                    "backbone": policy["optimizer"]["backbone_learning_rate"],
                    "head": policy["optimizer"]["head_learning_rate"],
                }
                transition_records.append(
                    {
                        "experiment_id": experiment_id,
                        "best_warmup_epoch": session.best_warmup_epoch,
                        "pre_transition_state_sha256": pre_transition_hash,
                        "restored_state_sha256": transition[
                            "restored_state_sha256"
                        ],
                        "post_transition_learning_rates": optimizer_learning_rates(
                            session.optimizer
                        ),
                        "test_evidence_used": False,
                    }
                )
            sessions.append(session.snapshot())
    loss_configuration_verified &= (
        loss_values["M0"] == loss_values["M1"]
        and loss_values["M1"] == loss_values["M2"]
        and loss_values["M3"] != loss_values["M1"]
    )
    initial_hashes = {
        session["initial_state_sha256"] for session in sessions
    }
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_protocol_only",
        "fixture_is_performance_evidence": False,
        "experiment_snapshots": sessions,
        "transition_records": transition_records,
        "shared_initial_state_sha256": next(iter(initial_hashes)),
        "shared_initial_state_count": len(initial_hashes),
        "loss_values_by_experiment": loss_values,
        "test_evidence_used": False,
        "real_training_performed": False,
    }
    checks = {
        "shared_initial_state": len(initial_hashes) == 1,
        "warmup_head_only": warmup_head_only,
        "warmup_checkpoint_restore": restore_verified,
        "layer4_only_unfreeze": layer4_only,
        "learning_rate_ranges": learning_rates_verified,
        "corrected_validation_controls": controls_verified,
        "offline_checkpoint_restore": restore_verified,
        "frozen_batchnorm": frozen_batchnorm_verified,
        "loss_configuration": loss_configuration_verified,
    }
    return report, checks


def build_selection_fixture(policy: dict, initial_hash: str) -> list[dict]:
    profiles = {
        "M0": (0.9, 0.7, 0.68),
        "M1": (0.86, 0.72, 0.71),
        "M2": (0.87, 0.725, 0.72),
        "M3": (0.85, 0.69, 0.7),
    }
    offsets = {42: -0.002, 123: 0.0, 2026: 0.002}
    rows = []
    for experiment_id, values in profiles.items():
        train_accuracy, validation_accuracy, macro_f1 = values
        for seed in policy["experiment_seeds"]:
            offset = offsets[seed]
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "seed": seed,
                    "initial_state_sha256": initial_hash,
                    "train_accuracy": train_accuracy + offset,
                    "validation_accuracy": validation_accuracy + offset,
                    "validation_macro_f1": macro_f1 + offset,
                    "best_epoch": 8,
                }
            )
    return rows


def verify_selection_logic(
    policy: dict,
    initial_hash: str,
) -> tuple[dict, dict[str, bool]]:
    fixture = build_selection_fixture(policy, initial_hash)
    summaries = summarize_repeated_seed_results(fixture, policy)
    selection = select_candidate_from_validation(summaries, policy)
    unauthorized = dict(fixture[0])
    unauthorized["test_accuracy"] = 0.99
    rejected_test_evidence = False
    try:
        summarize_repeated_seed_results([unauthorized, *fixture[1:]], policy)
    except ValueError:
        rejected_test_evidence = True
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_selection_logic_only",
        "fixture_is_performance_evidence": False,
        "summaries": summaries,
        "fixture_selection": selection,
        "authorized_candidate_experiment_id": None,
        "test_field_rejection_verified": rejected_test_evidence,
        "test_evidence_used": False,
    }
    checks = {
        "repeated_seed_selection": selection["selected_experiment_id"] == "M2",
        "all_seed_coverage": all(
            summary["seeds"] == policy["experiment_seeds"]
            for summary in summaries
        ),
        "test_evidence_rejected": rejected_test_evidence,
        "no_fixture_candidate_authorized": report[
            "authorized_candidate_experiment_id"
        ]
        is None,
    }
    return report, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r7-verification", type=Path, required=True)
    parser.add_argument("--r6-verification", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r8_policy()
    r7_report = json.loads(args.r7_verification.read_text())
    r6_report = json.loads(args.r6_verification.read_text())
    root = get_practice_2_2_root().resolve()
    notebook_path = root / "notebooks/04_canonical_report.ipynb"
    notebook_hash_before = file_sha256(notebook_path)
    pretrained_available = False
    pretrained_metadata = {
        "authorized_pretrained_state_available": False,
        "expected_weights": policy["pretrained_weights"],
        "expected_checkpoint_sha256": policy["pretrained_checkpoint_sha256"],
        "network_download_attempted": False,
    }
    if args.pretrained_checkpoint is not None:
        pretrained_state, loaded_metadata = load_authorized_pretrained_state(
            args.pretrained_checkpoint,
            policy["pretrained_checkpoint_sha256"],
        )
        pretrained_available = True
        pretrained_metadata.update(loaded_metadata)
        pretrained_metadata["authorized_pretrained_state_available"] = True
        protocol_fixture_type = "authorized_pretrained_state"
    else:
        pretrained_state = build_synthetic_pretrained_fixture()
        protocol_fixture_type = "synthetic_protocol_only"
    protocol_report, protocol_checks = verify_staged_protocol(
        pretrained_state, policy
    )
    protocol_report["initialization_fixture_type"] = protocol_fixture_type
    selection_report, selection_checks = verify_selection_logic(
        policy, protocol_report["shared_initial_state_sha256"]
    )
    checks = {**protocol_checks, **selection_checks}
    checks["canonical_notebook_unchanged"] = (
        file_sha256(notebook_path) == notebook_hash_before
    )
    report = build_r8_report(
        r7_report,
        r6_report,
        checks,
        pretrained_state_available=pretrained_available,
        repeated_seed_evidence_available=False,
        authorized_candidate_id=None,
        policy=policy,
    )
    report["protocol_checks"] = checks
    report["canonical_notebook_sha256"] = notebook_hash_before
    artifact_paths = [
        write_json(policy, args.output_dir / "policy_snapshot.json"),
        write_json(
            policy["experiments"], args.output_dir / "experiment_registry.json"
        ),
        write_json(
            pretrained_metadata, args.output_dir / "pretrained_authority.json"
        ),
        write_json(
            protocol_report, args.output_dir / "staged_protocol_verification.json"
        ),
        write_json(
            selection_report, args.output_dir / "selection_logic_verification.json"
        ),
        write_json(report, args.output_dir / "staged_transfer_report.json"),
        write_json(report, args.report_output),
    ]
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "synthetic_protocol_verification_only": not pretrained_available,
        "fixture_is_performance_evidence": False,
        "authorized_candidate_experiment_id": None,
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
        "authorized_candidate_experiment_id": None,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
