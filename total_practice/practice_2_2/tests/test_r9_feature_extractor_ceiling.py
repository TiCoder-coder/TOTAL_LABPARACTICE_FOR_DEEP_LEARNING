import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from practice_2_2.r7_training_correctness import state_dict_sha256
from practice_2_2.r9_feature_extractor_ceiling import (
    build_architecture_optimizer,
    build_classifier_model,
    configure_architecture_stage,
    create_backbone,
    load_architecture_checkpoint,
    load_authorized_backbone_state,
    load_r9_policy,
    optimizer_learning_rates,
    prepare_training_mode,
    profile_architecture,
    save_architecture_checkpoint,
    select_architecture_from_validation,
    summarize_architecture_results,
    trainable_parameter_names,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r9_resnet18_vs_efficientnet_b0_s42_123_2026_v1"


def synthetic_state(architecture):
    seeds = {"resnet18": 9018, "efficientnet_b0": 9000}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seeds[architecture])
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in create_backbone(architecture).state_dict().items()
        }


def build_model(architecture):
    return build_classifier_model(
        architecture,
        synthetic_state(architecture),
        num_classes=10,
        dropout=0.2,
        head_seed=42,
    )


def result_fixture(policy, train_only=False):
    profiles = (
        {"A0": (0.88, 0.72, 0.7), "A1": (0.95, 0.72, 0.7)}
        if train_only
        else {"A0": (0.88, 0.72, 0.7), "A1": (0.86, 0.74, 0.73)}
    )
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


def test_r9_policy_is_controlled_image_only_and_test_locked():
    policy = load_r9_policy()
    assert policy["experiment_seeds"] == [42, 123, 2026]
    assert policy["image_size"] == 224
    assert policy["higher_resolution_experiment_enabled"] is False
    assert policy["ocr_or_image_text_features_enabled"] is False
    assert policy["same_split_required"] is True
    assert policy["same_transform_required"] is True
    assert policy["same_seeds_required"] is True
    assert policy["same_staged_protocol_required"] is True
    assert policy["test_evidence_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["test_loader_construction_allowed"] is False


def test_r9_a1_changes_architecture_only_at_same_resolution():
    policy = load_r9_policy()
    experiments = policy["experiments"]
    parent = experiments[experiments["A1"]["parent"]]
    changed = [
        field
        for field in policy["controlled_experiment_fields"]
        if experiments["A1"][field] != parent[field]
    ]
    assert changed == ["architecture"]
    assert experiments["A0"]["image_size"] == experiments["A1"]["image_size"]


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_models_produce_ten_class_logits_without_network(architecture):
    with patch(
        "torch.hub.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ), patch(
        "torchvision.models._api.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ):
        model = build_model(architecture).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 96, 96))
    assert tuple(output.shape) == (1, 10)


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_stage_adapter_limits_trainable_parameters(architecture):
    model = build_model(architecture)
    configure_architecture_stage(model, architecture, "head_warmup")
    head_names = trainable_parameter_names(model)
    prefix = "fc." if architecture == "resnet18" else "classifier."
    assert head_names
    assert all(name.startswith(prefix) for name in head_names)
    configure_architecture_stage(model, architecture, "final_block_finetune")
    final_names = trainable_parameter_names(model)
    if architecture == "resnet18":
        assert all(
            name.startswith("layer4.") or name.startswith("fc.")
            for name in final_names
        )
    else:
        assert all(
            name.startswith("features.7.")
            or name.startswith("features.8.")
            or name.startswith("classifier.")
            for name in final_names
        )


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_named_learning_rates_match_r8_protocol(architecture):
    policy = load_r9_policy()
    model = build_model(architecture)
    configure_architecture_stage(model, architecture, "head_warmup")
    optimizer = build_architecture_optimizer(model, architecture, policy)
    assert optimizer_learning_rates(optimizer) == {
        "head": policy["optimizer"]["head_learning_rate"]
    }
    configure_architecture_stage(model, architecture, "final_block_finetune")
    optimizer = build_architecture_optimizer(model, architecture, policy)
    assert optimizer_learning_rates(optimizer) == {
        "backbone": policy["optimizer"]["backbone_learning_rate"],
        "head": policy["optimizer"]["head_learning_rate"],
    }


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_frozen_batchnorm_remains_in_eval(architecture):
    model = build_model(architecture)
    configure_architecture_stage(model, architecture, "final_block_finetune")
    result = prepare_training_mode(model)
    assert result["frozen_batchnorm_count"] > 0
    for module in model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        trainable = any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
        if not trainable:
            assert not module.training


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_offline_checkpoint_reload_is_identical(architecture, tmp_path):
    model = build_model(architecture).eval()
    inputs = torch.randn(1, 3, 96, 96)
    with torch.inference_mode():
        expected = model(inputs)
    path = save_architecture_checkpoint(
        tmp_path / f"{architecture}.pt", model, architecture, 10, 0.2
    )
    with patch(
        "torch.hub.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ), patch(
        "torchvision.models._api.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ):
        loaded, checkpoint = load_architecture_checkpoint(path)
    with torch.inference_mode():
        actual = loaded(inputs)
    assert checkpoint["constructor_weights"] is None
    assert checkpoint["model_state_sha256"] == state_dict_sha256(
        loaded.state_dict()
    )
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_r9_profile_reports_parameters_memory_latency_and_shapes(architecture):
    model = build_model(architecture)
    configure_architecture_stage(model, architecture, "final_block_finetune")
    profile = profile_architecture(model, architecture, 96, 0, 1)
    assert profile["parameter_count"] > 0
    assert profile["trainable_parameter_count"] > 0
    assert profile["parameter_memory_bytes"] > 0
    assert profile["buffer_memory_bytes"] > 0
    assert profile["forward_activation_bytes_sum"] > 0
    assert profile["latency_ms_mean"] > 0
    assert profile["output_shape"] == [1, 10]
    assert profile["performance_metric_source"] == "synthetic_input_only"


def test_r9_repeated_seed_selector_advances_validation_improvement():
    policy = load_r9_policy()
    summaries = summarize_architecture_results(result_fixture(policy), policy)
    selection = select_architecture_from_validation(summaries, policy)
    assert all(summary["seed_count"] == 3 for summary in summaries)
    assert selection["selected_experiment_id"] == "A1"
    assert selection["comparator_advances"] is True
    assert selection["selection_source"] == "Validation_only"
    assert selection["test_evidence_used"] is False


def test_r9_rejects_architecture_that_only_increases_train_accuracy():
    policy = load_r9_policy()
    summaries = summarize_architecture_results(
        result_fixture(policy, train_only=True), policy
    )
    selection = select_architecture_from_validation(summaries, policy)
    assert selection["selected_experiment_id"] == "A0"
    assert selection["comparator_advances"] is False
    assert selection["train_only_improvement"] is True


def test_r9_result_contract_rejects_test_fields_and_incomplete_seed_coverage():
    policy = load_r9_policy()
    rows = result_fixture(policy)
    unauthorized = [dict(row) for row in rows]
    unauthorized[0]["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="unauthorized"):
        summarize_architecture_results(unauthorized, policy)
    with pytest.raises(ValueError, match="cover"):
        summarize_architecture_results(rows[:-1], policy)


def test_r9_each_architecture_requires_one_initialization_across_seeds():
    policy = load_r9_policy()
    rows = result_fixture(policy)
    mixed = [dict(row) for row in rows]
    mixed[-1]["initial_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="share initialization"):
        summarize_architecture_results(mixed, policy)


def test_r9_authorized_pretrained_loader_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_authorized_backbone_state(
            "resnet18", tmp_path / "missing.pt", "a" * 64
        )
    path = tmp_path / "state.pt"
    torch.save(synthetic_state("resnet18"), path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_authorized_backbone_state("resnet18", path, "a" * 64)


def test_r9_report_is_protocol_complete_but_architecture_gate_blocked():
    report = json.loads((ROOT / "docs/R9_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["protocol_exit_checks_passed"] is True
    assert report["blocked_reasons"] == [
        "R8_GATE_NOT_PASSED",
        "R8_CANDIDATE_NOT_AUTHORIZED",
        "TRANSFORM_RECIPE_NOT_SELECTED",
        "TRAIN_MANIFEST_NOT_AUTHORIZED",
        "BASELINE_PRETRAINED_STATE_UNAVAILABLE",
        "COMPARATOR_PRETRAINED_STATE_UNAVAILABLE",
        "COMPARATOR_PRETRAINED_HASH_NOT_FROZEN",
        "REPEATED_SEED_VALIDATION_EVIDENCE_UNAVAILABLE",
        "NO_R9_ARCHITECTURE_AUTHORIZED",
    ]
    assert report["architecture_profiles_reported"] is True
    assert report["authorized_architecture_experiment_id"] is None
    assert report["test_evidence_used"] is False
    assert report["test_loader_constructed"] is False
    assert report["real_training_performed"] is False
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["successor_phase_executed"] is False


def test_r9_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["synthetic_protocol_verification_only"] is True
    assert manifest["architecture_profile_evidence_only"] is True
    assert manifest["fixture_is_model_quality_evidence"] is False
    assert manifest["authorized_architecture_experiment_id"] is None
    assert manifest["real_training_performed"] is False
    assert manifest["validation_content_access_count"] == 0
    assert manifest["test_content_access_count"] == 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r9_entrypoint_imports_no_data_loader_and_performs_no_real_training():
    paths = [
        ROOT / "src/practice_2_2/r9_feature_extractor_ceiling.py",
        ROOT / "scripts/run_r9_feature_extractor_ceiling.py",
    ]
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "DataLoader" not in imported_names
        assert "torch.utils.data" not in imported_names
        assert ".backward(" not in source
        assert "optimizer.step(" not in source


def test_r9_did_not_mutate_canonical_notebook():
    report = json.loads((ROOT / "docs/R9_VERIFICATION.json").read_text())
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    assert report["canonical_notebook_mutated"] is False
    assert file_sha256(notebook) == report["canonical_notebook_sha256"]
