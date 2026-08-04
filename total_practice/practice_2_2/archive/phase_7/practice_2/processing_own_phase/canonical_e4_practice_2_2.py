"""Controlled Phase 2.6 E4 augmentation experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from processing_own_phase.canonical_e3_practice_2_2 import E2_BASELINE
from processing_own_phase.canonical_train_practice_2_2 import (
    DATASET_FINGERPRINT,
    EXPECTED_MODEL_COUNTS,
    EXPERIMENTS,
    RUN_ID as BASELINE_RUN_ID,
    SEED,
    _json_hash,
    _state_dict_hash,
    verify_canonical_input,
)
from processing_own_phase.data_practice_2_2 import (
    compute_train_class_weights,
    create_train_validation_datasets,
    file_sha256,
    get_practice_2_2_transforms,
    make_train_validation_loaders,
)
from processing_own_phase.model import build_model
from processing_own_phase.save_load import load_model_from_checkpoint
from processing_own_phase.train import evaluate, get_criterion, train_model
from processing_own_phase.utils import get_device, setup_reproducibility


RUN_ID = "canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1"
EXPERIMENT_ID = "E4_moderate_augmentation"
E4_CONFIG = {
    **EXPERIMENTS["E2_partial_finetune"],
    "augment_strength": "e4_moderate",
}
E3_RESULT = {
    "experiment": "E3_layer4_1_head",
    "strategy": "reduced_capacity",
    "train_accuracy": 90.07936507936508,
    "validation_accuracy": 70.31963470319634,
    "validation_loss": 1.279932244183266,
    "validation_macro_f1": 0.7016717195510864,
    "generalization_gap": 19.759730376168733,
    "trainable_parameters": 4_725_770,
    "best_epoch": 8,
}


def validate_e4_contract():
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {
        key for key in set(e2).union(E4_CONFIG) if e2.get(key) != E4_CONFIG.get(key)
    }
    if differing != {"augment_strength"}:
        raise RuntimeError(
            "E4 must differ from canonical E2 only by augmentation; found "
            + ", ".join(sorted(differing))
        )
    if E4_CONFIG["training_mode"] != "partial_finetune":
        raise RuntimeError("E4 must train full layer4 and classifier")
    if any(
        E4_CONFIG[key]
        for key in ("test_data_used", "test_loader_constructed", "test_evaluated")
    ):
        raise RuntimeError("Test is locked during Phase 2.6")
    _, e2_validation = get_practice_2_2_transforms(224, "base")
    _, e4_validation = get_practice_2_2_transforms(224, "e4_moderate")
    if repr(e2_validation) != repr(e4_validation):
        raise RuntimeError("E4 changed the deterministic Validation transform")
    return True


def choose_provisional_winner(e4):
    accuracy_margin = float(e4["validation_accuracy"] - E2_BASELINE["validation_accuracy"])
    f1_margin = float(e4["validation_macro_f1"] - E2_BASELINE["validation_macro_f1"])
    accuracy_not_materially_lower = accuracy_margin >= -0.5
    e4_wins = accuracy_margin > 0 or (f1_margin > 0 and accuracy_not_materially_lower)
    return {
        "winner": EXPERIMENT_ID if e4_wins else E2_BASELINE["experiment"],
        "policy": (
            "Validation Accuracy, then Macro F1, Validation Loss, and gap; "
            "E4 cannot win on gap alone when Validation performance is lower"
        ),
        "validation_accuracy_margin_points": accuracy_margin,
        "validation_macro_f1_margin": f1_margin,
        "test_data_used": False,
    }


def run_e4(
    dataset_root,
    manifest_path,
    split_summary_path,
    baseline_lineage_path,
    e3_report_path,
    runs_root,
    outputs_root,
    device=None,
):
    validate_e4_contract()
    verification = verify_canonical_input(dataset_root, manifest_path, split_summary_path)
    baseline = json.loads(Path(baseline_lineage_path).read_text())
    if baseline.get("run_id") != BASELINE_RUN_ID:
        raise RuntimeError("Unexpected canonical E2 lineage")
    e3_report = json.loads(Path(e3_report_path).read_text())
    if e3_report.get("test_loader_constructed") is not False:
        raise RuntimeError("E3 lineage does not preserve the Test lock")

    selected_device = device or get_device()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    run_root = Path(runs_root) / RUN_ID / EXPERIMENT_ID
    output_root = Path(outputs_root) / RUN_ID
    if run_root.parent.exists() or output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2.6 lineage: {RUN_ID}")
    run_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    setup_reproducibility(SEED)
    train_dataset, validation_dataset = create_train_validation_datasets(
        dataset_root,
        manifest_path,
        image_size=E4_CONFIG["image_size"],
        augment_strength=E4_CONFIG["augment_strength"],
    )
    if len(train_dataset) != EXPECTED_MODEL_COUNTS["Train"]:
        raise RuntimeError("Unexpected E4 Train count")
    if len(validation_dataset) != EXPECTED_MODEL_COUNTS["Validation"]:
        raise RuntimeError("Unexpected E4 Validation count")
    train_loader, validation_loader = make_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=E4_CONFIG["batch_size"],
        num_workers=0,
        seed=SEED,
    )

    model = build_model(
        E4_CONFIG["model_name"],
        E4_CONFIG["training_mode"],
        num_classes=E4_CONFIG["num_classes"],
        dropout=E4_CONFIG["dropout"],
    ).to(selected_device)
    initial_hash = _state_dict_hash(model)
    if initial_hash != baseline["initial_model_state_sha256"]:
        raise RuntimeError("E4 initial model state differs from canonical E2")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer4_trainable = sum(p.numel() for p in model.network.layer4.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.network.fc.parameters() if p.requires_grad)
    if trainable != E2_BASELINE["trainable_parameters"]:
        raise RuntimeError("E4 trainable parameter count differs from E2")
    if layer4_trainable == 0 or head_trainable == 0:
        raise RuntimeError("E4 did not unfreeze full layer4 and classifier")

    train_transform, validation_transform = get_practice_2_2_transforms(
        E4_CONFIG["image_size"], E4_CONFIG["augment_strength"]
    )
    config = {
        **E4_CONFIG,
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": verification["split_fingerprint_sha256"],
        "initial_model_state_sha256": initial_hash,
        "trainable_parameters": trainable,
        "train_transform": repr(train_transform),
        "validation_transform": repr(validation_transform),
        "output_dir": str(run_root.resolve()),
    }
    config["config_fingerprint_sha256"] = _json_hash(config)
    (output_root / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    class_weights = compute_train_class_weights(train_dataset)
    started = time.perf_counter()
    history = train_model(
        model,
        train_loader,
        validation_loader,
        config,
        selected_device,
        class_weights=class_weights,
    )
    training_seconds = time.perf_counter() - started
    best_path = run_root / "best.pt"
    latest_path = run_root / "latest.pt"
    if not best_path.is_file() or not latest_path.is_file():
        raise FileNotFoundError("E4 latest.pt or best.pt is missing")

    reloaded = load_model_from_checkpoint(str(best_path), device=selected_device)
    criterion = get_criterion(config, class_weights.to(selected_device))
    val_loss, val_acc, val_f1 = evaluate(
        reloaded,
        validation_loader,
        criterion,
        selected_device,
        return_macro_f1=True,
    )
    best_epoch = int(history["best_epoch"])
    index = best_epoch - 1
    recorded_loss = float(history["val_loss"][index])
    recorded_acc = float(history["val_acc"][index])
    if abs(val_loss - recorded_loss) > 1e-6 or abs(val_acc - recorded_acc) > 1e-6:
        raise RuntimeError("E4 checkpoint reload metrics do not match")

    e4 = {
        "experiment": EXPERIMENT_ID,
        "strategy": "moderate_online_augmentation",
        "best_epoch": best_epoch,
        "epochs_trained": len(history["epoch"]),
        "train_accuracy": float(history["train_acc"][index]),
        "validation_accuracy": val_acc,
        "validation_loss": val_loss,
        "validation_macro_f1": val_f1,
        "generalization_gap": float(history["train_acc"][index] - val_acc),
        "trainable_parameters": trainable,
        "training_seconds": training_seconds,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": file_sha256(best_path),
        "reload_status": "PASS",
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    e2 = {**E2_BASELINE, "strategy": "baseline_augmentation"}
    decision = choose_provisional_winner(e4)
    pd.DataFrame([e2, E3_RESULT, e4]).to_csv(
        output_root / "validation_comparison_e2_e3_e4.csv", index=False
    )
    (output_root / "training_history_e4.json").write_text(json.dumps(history, indent=2))
    report = {
        "run_id": RUN_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "dataset_verification": verification,
        "initial_model_state_sha256": initial_hash,
        "exact_train_augmentation": repr(train_transform),
        "validation_transform": repr(validation_transform),
        "e2": e2,
        "e3": E3_RESULT,
        "e4": e4,
        "decision": decision,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_metrics_recorded": False,
        "further_experiments_allowed": False,
    }
    (output_root / "phase_2_6_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--baseline-lineage", required=True, type=Path)
    parser.add_argument("--e3-report", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_e4(
        args.dataset_root,
        args.manifest,
        args.split_summary,
        args.baseline_lineage,
        args.e3_report,
        args.runs_root,
        args.outputs_root,
    ), indent=2))


if __name__ == "__main__":
    main()
