"""Controlled Phase 2.5 E3 fine-tuning-depth experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from processing_own_phase.canonical_train_practice_2_2 import (
    CANONICAL_BASE_CONFIG,
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
    make_train_validation_loaders,
)
from processing_own_phase.model import build_model
from processing_own_phase.save_load import load_model_from_checkpoint
from processing_own_phase.train import evaluate, get_criterion, train_model
from processing_own_phase.utils import get_device, setup_reproducibility


RUN_ID = "canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1"
EXPERIMENT_ID = "E3_layer4_1_head"
E3_CONFIG = {
    **CANONICAL_BASE_CONFIG,
    "training_mode": "last_block_finetune",
}
E2_BASELINE = {
    "experiment": "E2_partial_finetune",
    "train_accuracy": 98.31349206349206,
    "validation_accuracy": 78.31050228310502,
    "validation_loss": 1.021770380947688,
    "validation_macro_f1": 0.7820960879325867,
    "generalization_gap": 20.00298978038704,
    "trainable_parameters": 8_398_858,
    "best_epoch": 14,
}


def validate_e3_contract():
    e2 = EXPERIMENTS["E2_partial_finetune"]
    differing = {
        key for key in set(e2).union(E3_CONFIG) if e2.get(key) != E3_CONFIG.get(key)
    }
    if differing != {"training_mode"}:
        raise RuntimeError(
            "E3 must differ from canonical E2 only by training_mode; found "
            + ", ".join(sorted(differing))
        )
    if E3_CONFIG["training_mode"] != "last_block_finetune":
        raise RuntimeError("E3 must train layer4.1 and the classifier only")
    if any(
        E3_CONFIG[key]
        for key in ("test_data_used", "test_loader_constructed", "test_evaluated")
    ):
        raise RuntimeError("Test is locked during Phase 2.5")
    return True


def choose_provisional_winner(e3):
    accuracy_margin = float(e3["validation_accuracy"] - E2_BASELINE["validation_accuracy"])
    f1_margin = float(e3["validation_macro_f1"] - E2_BASELINE["validation_macro_f1"])
    accuracy_not_materially_lower = accuracy_margin >= -0.5
    e3_wins = accuracy_margin > 0 or (f1_margin > 0 and accuracy_not_materially_lower)
    return {
        "winner": EXPERIMENT_ID if e3_wins else E2_BASELINE["experiment"],
        "policy": (
            "E3 wins on higher Validation Accuracy, or higher Validation Macro F1 "
            "with Validation Accuracy no more than 0.5 percentage points lower; "
            "generalization gap breaks near-ties only"
        ),
        "validation_accuracy_margin_points": accuracy_margin,
        "validation_macro_f1_margin": f1_margin,
        "test_data_used": False,
    }


def run_e3(
    dataset_root,
    manifest_path,
    split_summary_path,
    baseline_lineage_path,
    runs_root,
    outputs_root,
    device=None,
):
    validate_e3_contract()
    dataset_verification = verify_canonical_input(
        dataset_root, manifest_path, split_summary_path
    )
    baseline = json.loads(Path(baseline_lineage_path).read_text())
    if baseline.get("run_id") != BASELINE_RUN_ID:
        raise RuntimeError("Unexpected E2 baseline lineage")
    if baseline.get("test_loader_constructed") is not False:
        raise RuntimeError("Baseline lineage does not preserve the Test lock")

    selected_device = device or get_device()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    run_root = Path(runs_root) / RUN_ID / EXPERIMENT_ID
    output_root = Path(outputs_root) / RUN_ID
    if run_root.parent.exists() or output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2.5 lineage: {RUN_ID}")
    run_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    setup_reproducibility(SEED)
    train_dataset, validation_dataset = create_train_validation_datasets(
        dataset_root,
        manifest_path,
        image_size=E3_CONFIG["image_size"],
        augment_strength=E3_CONFIG["augment_strength"],
    )
    if len(train_dataset) != EXPECTED_MODEL_COUNTS["Train"]:
        raise RuntimeError("Unexpected E3 Train count")
    if len(validation_dataset) != EXPECTED_MODEL_COUNTS["Validation"]:
        raise RuntimeError("Unexpected E3 Validation count")
    train_loader, validation_loader = make_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=E3_CONFIG["batch_size"],
        num_workers=0,
        seed=SEED,
    )

    model = build_model(
        E3_CONFIG["model_name"],
        E3_CONFIG["training_mode"],
        num_classes=E3_CONFIG["num_classes"],
        dropout=E3_CONFIG["dropout"],
    ).to(selected_device)
    initial_state_hash = _state_dict_hash(model)
    baseline_initial_hash = baseline["initial_model_state_sha256"]
    if initial_state_hash != baseline_initial_hash:
        raise RuntimeError("E3 did not start from the canonical E2 initialization")

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer40_trainable = sum(
        p.numel() for p in model.network.layer4[0].parameters() if p.requires_grad
    )
    layer41_trainable = sum(
        p.numel() for p in model.network.layer4[1].parameters() if p.requires_grad
    )
    head_trainable = sum(p.numel() for p in model.network.fc.parameters() if p.requires_grad)
    if layer40_trainable != 0 or layer41_trainable == 0 or head_trainable == 0:
        raise RuntimeError("E3 trainable-layer contract failed")

    config = {
        **E3_CONFIG,
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "dataset_fingerprint_sha256": DATASET_FINGERPRINT,
        "split_fingerprint_sha256": dataset_verification["split_fingerprint_sha256"],
        "initial_model_state_sha256": initial_state_hash,
        "trainable_parameters": trainable_parameters,
        "e2_trainable_parameters": E2_BASELINE["trainable_parameters"],
        "layer4_0_trainable_parameters": layer40_trainable,
        "layer4_1_trainable_parameters": layer41_trainable,
        "head_trainable_parameters": head_trainable,
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
        raise FileNotFoundError("E3 latest.pt or best.pt is missing")

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
    best_index = best_epoch - 1
    recorded_loss = float(history["val_loss"][best_index])
    recorded_acc = float(history["val_acc"][best_index])
    if abs(val_loss - recorded_loss) > 1e-6 or abs(val_acc - recorded_acc) > 1e-6:
        raise RuntimeError("E3 checkpoint reload metrics do not match saved metrics")

    e3_result = {
        "experiment": EXPERIMENT_ID,
        "best_epoch": best_epoch,
        "epochs_trained": len(history["epoch"]),
        "train_accuracy": float(history["train_acc"][best_index]),
        "validation_accuracy": val_acc,
        "validation_loss": val_loss,
        "validation_macro_f1": val_f1,
        "generalization_gap": float(history["train_acc"][best_index] - val_acc),
        "trainable_parameters": trainable_parameters,
        "training_seconds": training_seconds,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": file_sha256(best_path),
        "reload_status": "PASS",
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    decision = choose_provisional_winner(e3_result)
    comparison = pd.DataFrame([E2_BASELINE, e3_result])
    comparison.to_csv(output_root / "validation_comparison_e2_e3.csv", index=False)
    (output_root / "training_history_e3.json").write_text(json.dumps(history, indent=2))
    report = {
        "run_id": RUN_ID,
        "baseline_run_id": BASELINE_RUN_ID,
        "dataset_verification": dataset_verification,
        "initial_model_state_sha256": initial_state_hash,
        "e2": E2_BASELINE,
        "e3": e3_result,
        "decision": decision,
        "test_loader_constructed": False,
        "test_evaluated": False,
        "test_metrics_recorded": False,
    }
    (output_root / "phase_2_5_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--baseline-lineage", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_e3(
        args.dataset_root,
        args.manifest,
        args.split_summary,
        args.baseline_lineage,
        args.runs_root,
        args.outputs_root,
    ), indent=2))


if __name__ == "__main__":
    main()
