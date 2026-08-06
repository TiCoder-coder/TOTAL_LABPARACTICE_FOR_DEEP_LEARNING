import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as functional

from practice_2_2.r7_training_correctness import (
    LOSS_CONTRACT,
    EarlyStoppingState,
    ExactCrossEntropyAccumulator,
    build_named_adamw,
    build_resnet18_classifier,
    configure_trainable_parameters,
    cross_entropy_batch_terms,
    load_complete_checkpoint,
    load_r7_policy,
    optimizer_learning_rates,
    route_corrected_validation_loss,
    save_complete_checkpoint,
    set_frozen_batchnorm_eval,
    state_dict_sha256,
)
from practice_2_2.resources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/new_work/r7_training_correctness_v1"


def aggregate(
    logits,
    targets,
    partitions,
    split,
    class_weights=None,
    label_smoothing=0.0,
):
    accumulator = ExactCrossEntropyAccumulator(split)
    offset = 0
    for size in partitions:
        selected = slice(offset, offset + size)
        accumulator.update(
            cross_entropy_batch_terms(
                logits[selected],
                targets[selected],
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
        )
        offset += size
    assert offset == targets.numel()
    return accumulator.metric()


def test_r7_policy_is_unweighted_offline_and_fail_closed():
    policy = load_r7_policy()
    assert policy["loss_contract"] == LOSS_CONTRACT
    assert policy["baseline_loss"] == {
        "name": "cross_entropy",
        "class_weights": None,
        "label_smoothing": 0.0,
    }
    assert policy["class_weighting_requires_separate_experiment"] is True
    assert policy["label_smoothing_requires_separate_experiment"] is True
    assert policy["checkpoint"]["constructor_weights"] is None
    assert policy["checkpoint"]["strict_state_dict_loading"] is True
    assert policy["real_training_allowed"] is False
    assert policy["validation_content_access_allowed"] is False
    assert policy["test_content_access_allowed"] is False
    assert policy["test_loader_construction_allowed"] is False


@pytest.mark.parametrize(
    ("class_weights", "label_smoothing"),
    [
        (None, 0.0),
        (torch.tensor([0.5, 1.2, 2.0, 0.8], dtype=torch.float64), 0.0),
        (None, 0.1),
        (torch.tensor([0.5, 1.2, 2.0, 0.8], dtype=torch.float64), 0.1),
    ],
)
def test_r7_epoch_loss_is_batch_partition_invariant(
    class_weights, label_smoothing
):
    torch.manual_seed(19)
    logits = torch.randn(11, 4, dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 3, 1, 2, 2, 3, 0, 3, 1])
    reference = functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="mean",
    ).item()
    values = [
        aggregate(
            logits,
            targets,
            partitions,
            split,
            class_weights,
            label_smoothing,
        )["value"]
        for split in ("Train", "Validation", "Test")
        for partitions in ([11], [1, 10], [2, 3, 1, 5], [4, 4, 3])
    ]
    assert max(abs(value - reference) for value in values) < 1e-12
    assert max(values) - min(values) < 1e-12


def test_r7_batch_optimization_loss_matches_pytorch_mean():
    torch.manual_seed(23)
    logits = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1, 2, 2, 0, 1, 2])
    weights = torch.tensor([0.6, 1.4, 2.3], dtype=torch.float64)
    terms = cross_entropy_batch_terms(
        logits, targets, class_weights=weights, label_smoothing=0.1
    )
    reference = functional.cross_entropy(
        logits,
        targets,
        weight=weights,
        label_smoothing=0.1,
        reduction="mean",
    )
    assert torch.allclose(terms.optimization_loss, reference, atol=1e-14, rtol=0)
    terms.optimization_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_r7_float32_epoch_metric_is_batch_partition_invariant():
    torch.manual_seed(29)
    logits = torch.randn(101, 10, dtype=torch.float32)
    targets = torch.randint(0, 10, (101,))
    weights = torch.linspace(0.5, 2.0, 10, dtype=torch.float32)
    for class_weights in (None, weights):
        values = [
            aggregate(
                logits,
                targets,
                partitions,
                "Train",
                class_weights,
                0.1,
            )["value"]
            for partitions in (
                [101],
                [1, 100],
                [32, 32, 32, 5],
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3],
            )
        ]
        assert max(values) - min(values) < 1e-12


def test_r7_validation_controls_reject_uncorrected_or_non_validation_metrics():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    early_stopping = EarlyStoppingState(patience=2)
    with pytest.raises(ValueError, match="Validation"):
        route_corrected_validation_loss(
            {"split": "Train", "contract": LOSS_CONTRACT, "value": 1.0},
            scheduler,
            early_stopping,
        )
    with pytest.raises(ValueError, match="corrected"):
        route_corrected_validation_loss(
            {"split": "Validation", "contract": "batch_mean", "value": 1.0},
            scheduler,
            early_stopping,
        )


def test_r7_scheduler_and_early_stopping_receive_corrected_validation_loss():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    early_stopping = EarlyStoppingState(patience=2)
    record = route_corrected_validation_loss(
        {"split": "Validation", "contract": LOSS_CONTRACT, "value": 0.875},
        scheduler,
        early_stopping,
    )
    assert record["scheduler_metric"] == 0.875
    assert record["early_stopping_metric"] == 0.875
    assert scheduler.best == 0.875
    assert early_stopping.best == 0.875


def test_r7_named_learning_rates_and_frozen_batchnorm():
    model = build_resnet18_classifier(num_classes=10, dropout=0.2)
    configure_trainable_parameters(model, "partial_finetune")
    model.train()
    frozen_count = set_frozen_batchnorm_eval(model)
    assert frozen_count > 0
    assert not model.bn1.training
    assert not model.layer3[1].bn2.training
    assert model.layer4[0].bn1.training
    optimizer = build_named_adamw(
        model,
        head_learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        weight_decay=2e-4,
    )
    assert optimizer_learning_rates(optimizer) == {
        "backbone": 1e-4,
        "head": 1e-3,
    }
    grouped = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert grouped == trainable


def test_r7_offline_checkpoint_reload_produces_identical_outputs(tmp_path):
    torch.manual_seed(42)
    model = build_resnet18_classifier(num_classes=10, dropout=0.2)
    model.eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = model(inputs)
    checkpoint_path = save_complete_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        num_classes=10,
        dropout=0.2,
        metadata={"training_performed": False},
    )
    with patch(
        "torch.hub.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ), patch(
        "torchvision.models._api.load_state_dict_from_url",
        side_effect=RuntimeError("network access forbidden"),
    ):
        loaded, checkpoint = load_complete_checkpoint(checkpoint_path)
    with torch.inference_mode():
        actual = loaded(inputs)
    assert checkpoint["constructor_weights"] is None
    assert checkpoint["model_state_sha256"] == state_dict_sha256(
        loaded.state_dict()
    )
    assert torch.equal(expected, actual)


def test_r7_report_is_correctness_complete_but_predecessor_blocked():
    report = json.loads((ROOT / "docs/R7_VERIFICATION.json").read_text())
    assert report["status"] == "blocked"
    assert report["gate_passed"] is False
    assert report["correctness_exit_checks_passed"] is True
    assert report["blocked_reasons"] == [
        "R6_GATE_NOT_PASSED",
        "TRANSFORM_RECIPE_NOT_SELECTED",
        "TRAIN_MANIFEST_NOT_AUTHORIZED",
    ]
    assert report["offline_reload_predictions_identical"] is True
    assert report["scheduler_consumed_corrected_validation_loss"] is True
    assert report["early_stopping_consumed_corrected_validation_loss"] is True
    assert report["test_loader_imported"] is False
    assert report["test_loader_constructed"] is False
    assert report["real_training_performed"] is False
    assert report["validation_content_access_count"] == 0
    assert report["test_content_access_count"] == 0
    assert report["successor_phase_executed"] is False


def test_r7_artifact_manifest_verifies_every_output():
    manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    assert manifest["synthetic_tensor_verification_only"] is True
    assert manifest["real_training_performed"] is False
    assert manifest["validation_content_access_count"] == 0
    assert manifest["test_content_access_count"] == 0
    assert manifest["test_loader_constructed"] is False
    assert manifest["source_images_mutated"] is False
    assert manifest["canonical_notebook_mutated"] is False
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]


def test_r7_entrypoint_imports_no_data_loader_and_performs_no_training_step():
    paths = [
        ROOT / "src/practice_2_2/r7_training_correctness.py",
        ROOT / "scripts/run_r7_training_correctness.py",
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


def test_r7_did_not_mutate_canonical_notebook():
    report = json.loads((ROOT / "docs/R7_VERIFICATION.json").read_text())
    notebook = ROOT / "notebooks/04_canonical_report.ipynb"
    assert report["canonical_notebook_mutated"] is False
    assert file_sha256(notebook) == report["canonical_notebook_sha256"]
