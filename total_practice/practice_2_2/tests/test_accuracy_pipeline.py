from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from practice_2_2.accuracy_pipeline import (
    CLASS_NAMES,
    AccuracyConfig,
    ManifestImageDataset,
    build_evaluation_transform,
    build_loader,
    build_model,
    build_train_transform,
    configure_training_stage,
    inspect_manifest_authorization,
    load_accuracy_checkpoint,
    load_authorized_development_manifest,
    run_epoch,
    save_accuracy_checkpoint,
    select_validation_inference_strategy,
    train_single_seed,
)
from practice_2_2.r7_training_correctness import set_frozen_batchnorm_eval


def build_manifest(tmp_path: Path, authorized: bool = True) -> tuple[Path, Path]:
    dataset_root = tmp_path / "images"
    rows = []
    index = 1
    for class_name in CLASS_NAMES:
        for split, count in (("Train", 2), ("Validation", 1)):
            for _ in range(count):
                relative_path = Path(class_name) / f"image_{index}.png"
                image_path = dataset_root / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (32, 32),
                    color=(index % 255, (index * 3) % 255, (index * 7) % 255),
                ).save(image_path)
                rows.append(
                    {
                        "asset_id": f"asset_{index}",
                        "relative_path": str(relative_path),
                        "class_name": class_name,
                        "component_id": f"component_{index}",
                        "split": split,
                        "raw_sha256": f"{index:064x}",
                        "decoded_pixel_sha256": f"{index + 1000:064x}",
                        "is_generated": False,
                        "use_for_model": authorized,
                    }
                )
                index += 1
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, dataset_root


def test_manifest_authorization_is_fail_closed(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path, authorized=False)
    report = inspect_manifest_authorization(manifest_path, dataset_root)
    assert report["authorized"] is False
    assert "DEVELOPMENT_ROWS_NOT_AUTHORIZED" in report["blocked_reasons"]
    with pytest.raises(RuntimeError, match="not authorized"):
        load_authorized_development_manifest(manifest_path, dataset_root)


def test_authorized_manifest_returns_development_rows_only(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    frame = pd.read_csv(manifest_path)
    test_row = frame.iloc[0].copy()
    test_row["asset_id"] = "test_asset"
    test_row["relative_path"] = "unreadable/test.png"
    test_row["component_id"] = "test_component"
    test_row["split"] = "Test"
    test_row["raw_sha256"] = "f" * 64
    test_row["decoded_pixel_sha256"] = "e" * 64
    frame.loc[len(frame)] = test_row
    frame.to_csv(manifest_path, index=False)
    development = load_authorized_development_manifest(manifest_path, dataset_root)
    assert set(development["split"]) == {"Train", "Validation"}
    assert "test_asset" not in set(development["asset_id"])


def test_direct_training_api_rejects_unauthorized_frame(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    development = load_authorized_development_manifest(manifest_path, dataset_root)
    development.loc[0, "use_for_model"] = False
    config = AccuracyConfig(
        pretrained=False,
        image_size=32,
        batch_size=10,
        warmup_epochs=1,
        finetune_epochs=1,
    )
    with pytest.raises(RuntimeError, match="unauthorized"):
        train_single_seed(
            development,
            dataset_root,
            tmp_path / "blocked_run",
            config,
            42,
            device=torch.device("cpu"),
        )


def test_dataset_and_loader_reject_test_split(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    development = load_authorized_development_manifest(manifest_path, dataset_root)
    transform = build_evaluation_transform(32)
    with pytest.raises(ValueError, match="Unsupported development split"):
        ManifestImageDataset(development, dataset_root, "Test", transform)
    with pytest.raises(ValueError, match="Unsupported development split"):
        build_loader(development, dataset_root, "Test", transform, 4, 42)


def test_manifest_rejects_train_validation_leakage(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    frame = pd.read_csv(manifest_path)
    train_index = frame.index[frame["split"] == "Train"][0]
    validation_index = frame.index[frame["split"] == "Validation"][0]
    frame.loc[validation_index, "component_id"] = frame.loc[train_index, "component_id"]
    frame.to_csv(manifest_path, index=False)
    report = inspect_manifest_authorization(manifest_path, dataset_root)
    assert report["authorized"] is False
    assert "TRAIN_VALIDATION_LEAKAGE" in report["blocked_reasons"]


def test_domain_safe_transforms_preserve_expected_shape():
    image = Image.new("RGB", (41, 53), color=(20, 30, 40))
    train_transform = build_train_transform(64)
    representation = repr(train_transform)
    assert "RandomHorizontalFlip" not in representation
    assert "RandomResizedCrop" not in representation
    assert "RandomErasing" not in representation
    assert train_transform(image).shape == (3, 64, 64)
    for scale in (1.0, 0.95, 0.9):
        assert build_evaluation_transform(64, scale)(image).shape == (3, 64, 64)


def test_resnet_staged_freezing_and_frozen_batchnorm():
    config = AccuracyConfig(
        pretrained=False,
        image_size=64,
        batch_size=2,
        warmup_epochs=1,
        finetune_epochs=1,
    )
    model = build_model(config)
    configure_training_stage(model, config.architecture, "head_warmup")
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    assert not any(parameter.requires_grad for parameter in model.layer4.parameters())
    configure_training_stage(model, config.architecture, "final_block_finetune")
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.layer3.parameters())
    model.train()
    assert set_frozen_batchnorm_eval(model) > 0
    assert model.layer3[-1].bn2.training is False
    assert model.layer4[-1].bn2.training is True


def test_efficientnet_staged_freezing():
    config = AccuracyConfig(
        architecture="efficientnet_b0",
        pretrained=False,
        image_size=64,
        batch_size=2,
        warmup_epochs=1,
        finetune_epochs=1,
    )
    model = build_model(config)
    configure_training_stage(model, config.architecture, "head_warmup")
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    configure_training_stage(model, config.architecture, "final_block_finetune")
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert any(parameter.requires_grad for parameter in model.features[-2:].parameters())
    assert not any(parameter.requires_grad for parameter in model.features[0].parameters())


class IndexedTensorDataset(TensorDataset):
    def __getitem__(self, index):
        inputs, target = super().__getitem__(index)
        return inputs, target, f"asset_{index}"


def test_epoch_metrics_and_test_guard():
    torch.manual_seed(42)
    inputs = torch.randn(20, 3, 8, 8)
    targets = torch.arange(20) % len(CLASS_NAMES)
    loader = DataLoader(IndexedTensorDataset(inputs, targets), batch_size=5)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, len(CLASS_NAMES)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_metrics = run_epoch(
        model,
        loader,
        "Train",
        torch.device("cpu"),
        0.05,
        len(CLASS_NAMES),
        optimizer=optimizer,
    )
    validation_metrics = run_epoch(
        model,
        loader,
        "Validation",
        torch.device("cpu"),
        0.05,
        len(CLASS_NAMES),
    )
    assert train_metrics["sample_count"] == 20
    assert validation_metrics["sample_count"] == 20
    assert train_metrics["loss_denominator"] == 20
    with pytest.raises(ValueError, match="Unsupported development split"):
        run_epoch(
            model,
            loader,
            "Test",
            torch.device("cpu"),
            0.05,
            len(CLASS_NAMES),
        )


def test_checkpoint_reloads_without_pretrained_weights(tmp_path):
    config = AccuracyConfig(
        pretrained=False,
        image_size=64,
        batch_size=2,
        warmup_epochs=1,
        finetune_epochs=1,
    )
    model = build_model(config)
    metrics = {
        "loss": 1.0,
        "accuracy": 0.5,
        "macro_f1": 0.4,
    }
    checkpoint_path = save_accuracy_checkpoint(
        tmp_path / "model.pt",
        model,
        config,
        CLASS_NAMES,
        42,
        "head_warmup",
        1,
        metrics,
        metrics,
    )
    loaded, checkpoint = load_accuracy_checkpoint(checkpoint_path)
    assert checkpoint["constructor_weights"] is None
    assert checkpoint["config"]["pretrained"] is False
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(expected, actual)


def test_validation_only_tta_selection_runs_from_checkpoint(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    config = AccuracyConfig(
        pretrained=False,
        image_size=64,
        batch_size=10,
        warmup_epochs=1,
        finetune_epochs=1,
    )
    model = build_model(config)
    metrics = {
        "loss": 1.0,
        "accuracy": 0.5,
        "macro_f1": 0.4,
    }
    checkpoint_path = save_accuracy_checkpoint(
        tmp_path / "tta_model.pt",
        model,
        config,
        CLASS_NAMES,
        42,
        "final_block_finetune",
        2,
        metrics,
        metrics,
    )
    result = select_validation_inference_strategy(
        [checkpoint_path],
        manifest_path,
        dataset_root,
        tta_scales=(1.0, 0.95, 0.9),
        maximum_ensemble_members=1,
        batch_size=10,
        device=torch.device("cpu"),
    )
    assert result["selection_split"] == "Validation"
    assert result["validation_asset_count"] == len(CLASS_NAMES)
    assert result["selected"]["tta_scales"] in ([1.0], [1.0, 0.95, 0.9])
    assert result["test_loader_constructed"] is False
    assert result["test_evaluated"] is False


def test_tta_selection_rejects_duplicate_views(tmp_path):
    with pytest.raises(ValueError, match="TTA scales must be unique"):
        select_validation_inference_strategy(
            [tmp_path / "unused.pt"],
            tmp_path / "unused.csv",
            tmp_path,
            tta_scales=(1.0, 1.0),
        )


def test_two_stage_training_runs_end_to_end_on_synthetic_data(tmp_path):
    manifest_path, dataset_root = build_manifest(tmp_path)
    development = load_authorized_development_manifest(manifest_path, dataset_root)
    config = AccuracyConfig(
        pretrained=False,
        image_size=32,
        batch_size=10,
        warmup_epochs=1,
        finetune_epochs=1,
        patience=1,
    )
    result = train_single_seed(
        development,
        dataset_root,
        tmp_path / "training_run",
        config,
        42,
        device=torch.device("cpu"),
    )
    assert result["warmup_best"]["stage"] == "head_warmup"
    assert result["finetune_best"]["stage"] == "final_block_finetune"
    assert Path(result["checkpoint_path"]).is_file()
    assert (tmp_path / "training_run/history.json").is_file()
    assert (tmp_path / "training_run/validation_result.json").is_file()
    assert result["test_loader_constructed"] is False
    assert result["test_evaluated"] is False
