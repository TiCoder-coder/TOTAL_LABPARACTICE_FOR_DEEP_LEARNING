from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from practice_2_2.paths import get_practice_2_2_root
from practice_2_2.r7_training_correctness import state_dict_sha256, tensor_sha256
from practice_2_2.r8_staged_transfer import load_r8_policy
from practice_2_2.r9_feature_extractor_ceiling import (
    build_architecture_optimizer,
    build_classifier_model,
    build_r9_report,
    configure_architecture_stage,
    create_backbone,
    load_architecture_checkpoint,
    load_r9_policy,
    optimizer_learning_rates,
    prepare_training_mode,
    profile_architecture,
    save_architecture_checkpoint,
    select_architecture_from_validation,
    summarize_architecture_results,
    trainable_parameter_names,
    write_json,
)
from practice_2_2.resources import file_sha256


def build_synthetic_backbone_fixture(
    architecture: str,
) -> dict[str, torch.Tensor]:
    seeds = {"resnet18": 9018, "efficientnet_b0": 9000}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds[architecture])
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in create_backbone(architecture).state_dict().items()
        }


def verify_architecture_protocol(
    policy: dict,
    r8_policy: dict,
) -> tuple[dict, list[dict], dict[str, bool]]:
    experiment_records = []
    profiles = []
    checkpoint_reload_verified = True
    stage_compatibility_verified = True
    frozen_batchnorm_verified = True
    learning_rates_verified = True
    output_shapes_verified = True
    r8_protocol_match = (
        policy["experiment_seeds"] == r8_policy["experiment_seeds"]
        and policy["staged_warmup_epochs"] == r8_policy["staged_warmup_epochs"]
        and policy["optimizer"] == r8_policy["optimizer"]
        and policy["scheduler"] == r8_policy["scheduler"]
        and policy["early_stopping"] == r8_policy["early_stopping"]
    )
    profiling = policy["profiling"]
    with tempfile.TemporaryDirectory(prefix="r9_architecture_ceiling_") as directory:
        checkpoint_root = Path(directory)
        for experiment_id, experiment in policy["experiments"].items():
            architecture = experiment["architecture"]
            backbone_state = build_synthetic_backbone_fixture(architecture)
            model = build_classifier_model(
                architecture,
                backbone_state,
                num_classes=policy["num_classes"],
                dropout=0.2,
                head_seed=policy["head_initialization_seed"],
            )
            initial_hash = state_dict_sha256(model.state_dict())
            configure_architecture_stage(model, architecture, "head_warmup")
            head_names = trainable_parameter_names(model)
            stage_compatibility_verified &= bool(head_names) and all(
                name.startswith("fc.")
                if architecture == "resnet18"
                else name.startswith("classifier.")
                for name in head_names
            )
            head_optimizer = build_architecture_optimizer(model, architecture, policy)
            learning_rates_verified &= optimizer_learning_rates(head_optimizer) == {
                "head": policy["optimizer"]["head_learning_rate"]
            }
            prepare_training_mode(model)
            frozen_batchnorm_verified &= all(
                not module.training
                for module in model.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
            configure_architecture_stage(
                model, architecture, "final_block_finetune"
            )
            final_names = trainable_parameter_names(model)
            if architecture == "resnet18":
                allowed = all(
                    name.startswith("layer4.") or name.startswith("fc.")
                    for name in final_names
                )
            else:
                allowed = all(
                    name.startswith("features.7.")
                    or name.startswith("features.8.")
                    or name.startswith("classifier.")
                    for name in final_names
                )
            stage_compatibility_verified &= allowed
            final_optimizer = build_architecture_optimizer(model, architecture, policy)
            learning_rates_verified &= optimizer_learning_rates(final_optimizer) == {
                "backbone": policy["optimizer"]["backbone_learning_rate"],
                "head": policy["optimizer"]["head_learning_rate"],
            }
            prepare_training_mode(model)
            frozen_batchnorm_verified &= all(
                not module.training
                for module in model.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                and not any(
                    parameter.requires_grad
                    for parameter in module.parameters(recurse=False)
                )
            )
            model.eval()
            torch.manual_seed(9099)
            inputs = torch.randn(1, 3, 96, 96)
            with torch.inference_mode():
                expected = model(inputs)
            checkpoint_path = save_architecture_checkpoint(
                checkpoint_root / f"{experiment_id}.pt",
                model,
                architecture,
                num_classes=policy["num_classes"],
                dropout=0.2,
            )
            with patch(
                "torch.hub.load_state_dict_from_url",
                side_effect=RuntimeError("network access forbidden"),
            ), patch(
                "torchvision.models._api.load_state_dict_from_url",
                side_effect=RuntimeError("network access forbidden"),
            ):
                loaded, checkpoint = load_architecture_checkpoint(checkpoint_path)
            with torch.inference_mode():
                actual = loaded(inputs)
            identical = torch.equal(expected, actual)
            checkpoint_reload_verified &= (
                identical
                and checkpoint["constructor_weights"] is None
                and state_dict_sha256(loaded.state_dict())
                == checkpoint["model_state_sha256"]
            )
            output_shapes_verified &= tuple(actual.shape) == (1, policy["num_classes"])
            profiles.append(
                profile_architecture(
                    model,
                    architecture,
                    image_size=policy["image_size"],
                    warmup_iterations=profiling["warmup_iterations"],
                    measured_iterations=profiling["measured_iterations"],
                )
            )
            experiment_records.append(
                {
                    "experiment_id": experiment_id,
                    "architecture": architecture,
                    "initial_state_sha256": initial_hash,
                    "head_warmup_trainable_parameter_count": len(head_names),
                    "final_block_trainable_parameter_count": len(final_names),
                    "head_learning_rates": optimizer_learning_rates(head_optimizer),
                    "final_block_learning_rates": optimizer_learning_rates(
                        final_optimizer
                    ),
                    "checkpoint_output_sha256_before": tensor_sha256(expected),
                    "checkpoint_output_sha256_after": tensor_sha256(actual),
                    "checkpoint_predictions_identical": identical,
                    "test_evidence_used": False,
                }
            )
    by_architecture = {profile["architecture"]: profile for profile in profiles}
    profile_fields_verified = all(
        profile["parameter_count"] > 0
        and profile["parameter_memory_bytes"] > 0
        and profile["forward_activation_bytes_sum"] > 0
        and profile["latency_ms_mean"] > 0
        and profile["output_shape"] == [1, policy["num_classes"]]
        for profile in profiles
    )
    efficient_parameter_advantage = (
        by_architecture["efficientnet_b0"]["parameter_count"]
        < by_architecture["resnet18"]["parameter_count"]
    )
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_architecture_protocol_only",
        "fixture_is_model_quality_evidence": False,
        "experiments": experiment_records,
        "r8_protocol_match": r8_protocol_match,
        "test_evidence_used": False,
        "real_training_performed": False,
    }
    checks = {
        "staged_protocol_compatibility": stage_compatibility_verified,
        "r8_protocol_match": r8_protocol_match,
        "frozen_batchnorm": frozen_batchnorm_verified,
        "named_learning_rates": learning_rates_verified,
        "offline_checkpoint_reload": checkpoint_reload_verified,
        "output_shapes": output_shapes_verified,
        "architecture_profiles": profile_fields_verified,
        "efficient_parameter_advantage": efficient_parameter_advantage,
    }
    return report, profiles, checks


def selection_fixture(policy: dict, train_only: bool = False) -> list[dict]:
    if train_only:
        profiles = {
            "A0": (0.88, 0.72, 0.7),
            "A1": (0.95, 0.72, 0.7),
        }
    else:
        profiles = {
            "A0": (0.88, 0.72, 0.7),
            "A1": (0.86, 0.74, 0.73),
        }
    offsets = {42: -0.002, 123: 0.0, 2026: 0.002}
    hashes = {"A0": "a" * 64, "A1": "b" * 64}
    return [
        {
            "experiment_id": experiment_id,
            "seed": seed,
            "initial_state_sha256": hashes[experiment_id],
            "train_accuracy": values[0] + offsets[seed],
            "validation_accuracy": values[1] + offsets[seed],
            "validation_macro_f1": values[2] + offsets[seed],
            "best_epoch": 9,
        }
        for experiment_id, values in profiles.items()
        for seed in policy["experiment_seeds"]
    ]


def verify_selection_logic(policy: dict) -> tuple[dict, dict[str, bool]]:
    improvement_rows = selection_fixture(policy)
    improvement_summaries = summarize_architecture_results(
        improvement_rows, policy
    )
    improvement_selection = select_architecture_from_validation(
        improvement_summaries, policy
    )
    train_only_rows = selection_fixture(policy, train_only=True)
    train_only_summaries = summarize_architecture_results(train_only_rows, policy)
    train_only_selection = select_architecture_from_validation(
        train_only_summaries, policy
    )
    unauthorized = [dict(row) for row in improvement_rows]
    unauthorized[0]["test_accuracy"] = 0.99
    rejected_test_evidence = False
    try:
        summarize_architecture_results(unauthorized, policy)
    except ValueError:
        rejected_test_evidence = True
    report = {
        "schema_version": 1,
        "fixture_type": "synthetic_selection_logic_only",
        "fixture_is_model_quality_evidence": False,
        "improvement_fixture_summaries": improvement_summaries,
        "improvement_fixture_selection": improvement_selection,
        "train_only_fixture_summaries": train_only_summaries,
        "train_only_fixture_selection": train_only_selection,
        "test_field_rejection_verified": rejected_test_evidence,
        "authorized_architecture_experiment_id": None,
        "test_evidence_used": False,
    }
    checks = {
        "repeated_seed_selection": improvement_selection["comparator_advances"]
        and improvement_selection["selected_experiment_id"] == "A1",
        "train_only_improvement_rejected": not train_only_selection[
            "comparator_advances"
        ]
        and train_only_selection["selected_experiment_id"] == "A0",
        "test_evidence_rejected": rejected_test_evidence,
        "no_fixture_architecture_authorized": report[
            "authorized_architecture_experiment_id"
        ]
        is None,
    }
    return report, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-verification", type=Path, required=True)
    parser.add_argument("--r6-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_r9_policy()
    r8_policy = load_r8_policy()
    r8_report = json.loads(args.r8_verification.read_text())
    r6_report = json.loads(args.r6_verification.read_text())
    root = get_practice_2_2_root().resolve()
    notebook_path = root / "notebooks/04_canonical_report.ipynb"
    notebook_hash_before = file_sha256(notebook_path)
    protocol_report, profiles, protocol_checks = verify_architecture_protocol(
        policy, r8_policy
    )
    selection_report, selection_checks = verify_selection_logic(policy)
    checks = {**protocol_checks, **selection_checks}
    checks["canonical_notebook_unchanged"] = (
        file_sha256(notebook_path) == notebook_hash_before
    )
    baseline_available = False
    comparator_available = False
    comparator_hash_frozen = (
        policy["experiments"]["A1"]["pretrained_checkpoint_sha256"] is not None
    )
    pretrained_authority = {
        "schema_version": 1,
        "network_download_attempted": False,
        "A0": {
            "architecture": "resnet18",
            "authorized_pretrained_state_available": baseline_available,
            "expected_checkpoint_sha256": policy["experiments"]["A0"][
                "pretrained_checkpoint_sha256"
            ],
        },
        "A1": {
            "architecture": "efficientnet_b0",
            "authorized_pretrained_state_available": comparator_available,
            "expected_checkpoint_sha256": policy["experiments"]["A1"][
                "pretrained_checkpoint_sha256"
            ],
            "checkpoint_hash_frozen": comparator_hash_frozen,
        },
    }
    report = build_r9_report(
        r8_report,
        r6_report,
        checks,
        baseline_pretrained_available=baseline_available,
        comparator_pretrained_available=comparator_available,
        comparator_hash_frozen=comparator_hash_frozen,
        repeated_seed_evidence_available=False,
        authorized_architecture_id=None,
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
            pretrained_authority, args.output_dir / "pretrained_authority.json"
        ),
        write_json(
            protocol_report, args.output_dir / "architecture_protocol_verification.json"
        ),
        write_json(profiles, args.output_dir / "architecture_profiles.json"),
        write_json(
            selection_report, args.output_dir / "selection_logic_verification.json"
        ),
        write_json(report, args.output_dir / "feature_ceiling_report.json"),
        write_json(report, args.report_output),
    ]
    artifact_manifest = {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "synthetic_protocol_verification_only": True,
        "architecture_profile_evidence_only": True,
        "fixture_is_model_quality_evidence": False,
        "authorized_architecture_experiment_id": None,
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
        "authorized_architecture_experiment_id": None,
        "validation_content_access_count": 0,
        "test_content_access_count": 0,
        "blocked_reasons": report["blocked_reasons"],
    }
    print(json.dumps(output, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
