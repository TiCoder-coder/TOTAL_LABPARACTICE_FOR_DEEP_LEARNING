import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torchvision import models

from practice_2_2.r7_training_correctness import (
    LOSS_CONTRACT,
    optimizer_learning_rates,
    state_dict_sha256,
)
from practice_2_2.r8_staged_transfer import (
    StagedTransferSession,
    build_experiment_model,
    load_authorized_pretrained_state,
    load_r8_policy,
    select_candidate_from_validation,
    summarize_repeated_seed_results,
    trainable_parameter_names,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r8_staged_transfer_s42_123_2026_v1"


def synthetic_pretrained_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8042)
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in models.resnet18(weights=None).state_dict().items()
        }


def validation_metrics(epoch):
    return {
        "split": "Validation",
        "contract": LOSS_CONTRACT,
        "validation_loss": [1.4, 1.2, 1.25, 1.3][epoch - 1],
        "validation_accuracy": [0.42, 0.51, 0.49, 0.48][epoch - 1],
        "validation_macro_f1": [0.4, 0.52, 0.5, 0.49][epoch - 1],
    }


def selection_fixture(policy, initial_hash):
    profiles = {
        "M0": (0.9, 0.7, 0.68),
        "M1": (0.86, 0.72, 0.71),
        "M2": (0.87, 0.725, 0.72),
        "M3": (0.85, 0.69, 0.7),
    }
    offsets = {42: -0.002, 123: 0.0, 2026: 0.002}
    return [
        {
            "experiment_id": experiment_id,
            "seed": seed,
            "initial_state_sha256": initial_hash,
            "train_accuracy": values[0] + offsets[seed],
            "validation_accuracy": values[1] + offsets[seed],
            "validation_macro_f1": values[2] + offsets[seed],
            "best_epoch": 8,
        }
        for experiment_id, values in profiles.items()
        for seed in policy["experiment_seeds"]
    ]


def test_r8_policy_predeclares_staged_protocol_and_test_lock():
    policy = load_r8_policy()
    assert policy["experiment_seeds"] == [42, 123, 2026]
    assert policy["staged_warmup_epochs"] == 4
    assert policy["warmup_epoch_range"] == [3, 5]
    assert policy["fine_tune_block"] == "layer4"
    assert 3e-4 <= policy["optimizer"]["head_learning_rate"] <= 1e-3
    assert 1e-5 <= policy["optimizer"]["backbone_learning_rate"] <= 3e-5
    assert policy["test_evidence_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["test_loader_construction_allowed"] is False


def test_r8_controlled_experiments_change_one_parent_field():
    experiments = load_r8_policy()["experiments"]
    fields = ("warmup_epochs", "dropout", "label_smoothing")
    for experiment_id in ("M1", "M2", "M3"):
        experiment = experiments[experiment_id]
        parent = experiments[experiment["parent"]]
        changed = [field for field in fields if experiment[field] != parent[field]]
        assert changed == [experiment["single_change"]]


def test_r8_all_experiments_share_one_initial_tensor_hash(tmp_path):
    policy = load_r8_policy()
    pretrained = synthetic_pretrained_state()
    sessions = [
        StagedTransferSession(
            experiment_id,
            pretrained,
            tmp_path / f"{experiment_id}.pt",
            policy,
        )
        for experiment_id in policy["experiments"]
    ]
    assert len({session.initial_state_sha256 for session in sessions}) == 1


def test_r8_initial_freeze_policy_distinguishes_m0_and_staged_models(tmp_path):
    policy = load_r8_policy()
    pretrained = synthetic_pretrained_state()
    m0 = StagedTransferSession("M0", pretrained, tmp_path / "m0.pt", policy)
    m1 = StagedTransferSession("M1", pretrained, tmp_path / "m1.pt", policy)
    assert m0.stage == "layer4_finetune"
    assert all(
        name.startswith("layer4.") or name.startswith("fc.")
        for name in trainable_parameter_names(m0.model)
    )
    assert m1.stage == "head_warmup"
    assert all(name.startswith("fc.") for name in trainable_parameter_names(m1.model))


def test_r8_cannot_unfreeze_layer4_before_warmup_finishes(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    session.record_warmup_epoch(validation_metrics(1))
    with pytest.raises(RuntimeError, match="finish"):
        session.transition_to_layer4()


def test_r8_restores_best_warmup_checkpoint_before_layer4_unfreeze(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    for epoch in range(1, 5):
        with torch.no_grad():
            session.model.fc[1].bias.add_(epoch / 1000.0)
        session.record_warmup_epoch(validation_metrics(epoch))
    final_warmup_hash = state_dict_sha256(session.model.state_dict())
    with patch(
        "torch.hub.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ), patch(
        "torchvision.models._api.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ):
        transition = session.transition_to_layer4()
    assert session.best_warmup_epoch == 2
    assert final_warmup_hash != transition["restored_state_sha256"]
    assert transition["restored_state_sha256"] == session.best_warmup_state_sha256
    assert session.stage == "layer4_finetune"
    assert all(
        name.startswith("layer4.") or name.startswith("fc.")
        for name in trainable_parameter_names(session.model)
    )


def test_r8_uses_predeclared_named_learning_rates_for_each_stage(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    assert optimizer_learning_rates(session.optimizer) == {
        "head": policy["optimizer"]["head_learning_rate"]
    }
    for epoch in range(1, 5):
        session.record_warmup_epoch(validation_metrics(epoch))
    session.transition_to_layer4()
    assert optimizer_learning_rates(session.optimizer) == {
        "backbone": policy["optimizer"]["backbone_learning_rate"],
        "head": policy["optimizer"]["head_learning_rate"],
    }


def test_r8_warmup_controls_receive_corrected_validation_loss(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    record = session.record_warmup_epoch(validation_metrics(1))
    assert record["metric_split"] == "Validation"
    assert record["metric_contract"] == LOSS_CONTRACT
    assert record["scheduler_metric"] == 1.4
    assert record["early_stopping_metric"] == 1.4


def test_r8_training_mode_keeps_frozen_batchnorm_in_eval(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    session.prepare_training_mode()
    assert not session.model.bn1.training
    assert not session.model.layer4[0].bn1.training
    for epoch in range(1, 5):
        session.record_warmup_epoch(validation_metrics(epoch))
    session.transition_to_layer4()
    session.prepare_training_mode()
    assert not session.model.bn1.training
    assert not session.model.layer3[1].bn2.training
    assert session.model.layer4[0].bn1.training


def test_r8_loss_configuration_applies_only_m3_label_smoothing(tmp_path):
    policy = load_r8_policy()
    pretrained = synthetic_pretrained_state()
    torch.manual_seed(88)
    logits = torch.randn(6, 10, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    losses = {}
    for experiment_id in policy["experiments"]:
        session = StagedTransferSession(
            experiment_id,
            pretrained,
            tmp_path / f"{experiment_id}.pt",
            policy,
        )
        losses[experiment_id] = float(
            session.compute_loss_terms(logits, targets).optimization_loss
        )
    assert losses["M0"] == losses["M1"] == losses["M2"]
    assert losses["M3"] != losses["M1"]


def test_r8_warmup_metrics_reject_test_or_legacy_contract(tmp_path):
    policy = load_r8_policy()
    session = StagedTransferSession(
        "M1", synthetic_pretrained_state(), tmp_path / "best.pt", policy
    )
    metrics = validation_metrics(1)
    metrics["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="contract"):
        session.record_warmup_epoch(metrics)
    metrics = validation_metrics(1)
    metrics["contract"] = "legacy_batch_mean"
    with pytest.raises(ValueError, match="corrected"):
        session.record_warmup_epoch(metrics)


def test_r8_repeated_seed_selector_uses_mean_validation_and_gap():
    policy = load_r8_policy()
    rows = selection_fixture(policy, "a" * 64)
    summaries = summarize_repeated_seed_results(rows, policy)
    selection = select_candidate_from_validation(summaries, policy)
    assert all(summary["seed_count"] == 3 for summary in summaries)
    assert all(summary["seeds"] == [42, 123, 2026] for summary in summaries)
    assert selection["selected_experiment_id"] == "M2"
    assert selection["selection_source"] == "Validation_only"
    assert selection["test_evidence_used"] is False


def test_r8_selector_rejects_test_fields_missing_seeds_and_mixed_initial_hashes():
    policy = load_r8_policy()
    rows = selection_fixture(policy, "a" * 64)
    with_test = [dict(row) for row in rows]
    with_test[0]["test_accuracy"] = 0.99
    with pytest.raises(ValueError, match="unauthorized"):
        summarize_repeated_seed_results(with_test, policy)
    with pytest.raises(ValueError, match="cover"):
        summarize_repeated_seed_results(rows[:-1], policy)
    mixed = [dict(row) for row in rows]
    mixed[-1]["initial_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial tensor hash"):
        summarize_repeated_seed_results(mixed, policy)


def test_r8_authorized_pretrained_loader_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_authorized_pretrained_state(tmp_path / "missing.pt", "a" * 64)
    path = tmp_path / "state.pt"
    torch.save(synthetic_pretrained_state(), path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_authorized_pretrained_state(path, "a" * 64)


def test_r8_build_experiment_model_never_downloads_weights():
    pretrained = synthetic_pretrained_state()
    with patch(
        "torch.hub.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ), patch(
        "torchvision.models._api.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ):
        model = build_experiment_model(pretrained, 0.2, 10, 42)
    assert tuple(model.fc[1].weight.shape) == (10, 512)


def test_r8_report_is_protocol_complete_but_experiment_gate_blocked():
    report = json.loads((ROOT / "docs/R8_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["protocol_exit_checks_passed"] is True
    assert report["blocked_reasons"] == [
        "R7_GATE_NOT_PASSED",
        "TRANSFORM_RECIPE_NOT_SELECTED",
        "TRAIN_MANIFEST_NOT_AUTHORIZED",
        "PRETRAINED_INITIAL_STATE_UNAVAILABLE",
        "REPEATED_SEED_VALIDATION_EVIDENCE_UNAVAILABLE",
        "NO_R8_CANDIDATE_AUTHORIZED",
    ]
    assert report["authorized_candidate_experiment_id"] is None
    assert report["test_evidence_used"] is False
    assert report["test_loader_constructed"] is False
    assert report["real_training_performed"] is False
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["successor_phase_executed"] is False


def test_r8_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["synthetic_protocol_verification_only"] is True
    assert manifest["fixture_is_performance_evidence"] is False
    assert manifest["authorized_candidate_experiment_id"] is None
    assert manifest["real_training_performed"] is False
    assert manifest["validation_content_access_count"] == 0
    assert manifest["test_content_access_count"] == 0
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r8_entrypoint_imports_no_data_loader_and_performs_no_real_training():
    paths = [
        ROOT / "src/practice_2_2/r8_staged_transfer.py",
        ROOT / "scripts/run_r8_staged_transfer.py",
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


def test_r8_did_not_mutate_canonical_notebook():
    report = json.loads((ROOT / "docs/R8_VERIFICATION.json").read_text())
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    assert report["canonical_notebook_mutated"] is False
    assert file_sha256(notebook) == report["canonical_notebook_sha256"]
