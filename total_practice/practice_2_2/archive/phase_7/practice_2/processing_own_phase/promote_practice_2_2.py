"""Safely promote verified Practice 2.2 artifacts from staging."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


RUN_NAMES = ("E1_head_only", "E2_partial_finetune")
OUTPUT_FILES = (
    "controlled_experiment_comparison.csv",
    "controlled_experiment_selection.json",
    "controlled_run_manifest.json",
    "controlled_training_history.json",
    "overfitting_assessment.csv",
    "overfitting_assessment.json",
)


def promote_verified_staging(staging_root, project_root, backup_root):
    staging = Path(staging_root).resolve()
    project = Path(project_root).resolve()
    backup = Path(backup_root).resolve() / "pre_promotion_moved"
    canonical_runs = project / "runs" / "practice_2_2"
    canonical_outputs = project / "outputs" / "practice_2_2"
    canonical_reports = project / "reports" / "practice_2_2"

    selection = json.loads(
        (staging / "outputs" / "controlled_experiment_selection.json").read_text()
    )
    assessment = json.loads(
        (staging / "outputs" / "overfitting_assessment.json").read_text()
    )
    for artifact in (selection, assessment):
        if artifact.get("test_access_count") != 0:
            raise RuntimeError("Cannot promote artifacts that accessed Test")
        if artifact.get("test_evaluated") is not False:
            raise RuntimeError("Cannot promote artifacts that evaluated Test")
    if not selection.get("validation_verification_passed"):
        raise RuntimeError("Validation checkpoint verification did not pass")

    required = [staging / "runs" / name / "best.pt" for name in RUN_NAMES]
    required += [staging / "outputs" / name for name in OUTPUT_FILES]
    required += [staging / "reports" / "controlled_training_curves.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Staging is incomplete: {missing}")
    if backup.exists():
        raise FileExistsError(f"Promotion backup already exists: {backup}")

    (backup / "runs").mkdir(parents=True)
    (backup / "outputs").mkdir()
    (backup / "reports").mkdir()
    canonical_runs.mkdir(parents=True, exist_ok=True)
    canonical_outputs.mkdir(parents=True, exist_ok=True)
    canonical_reports.mkdir(parents=True, exist_ok=True)

    for name in RUN_NAMES:
        destination = canonical_runs / name
        if destination.exists():
            shutil.move(str(destination), str(backup / "runs" / name))
        shutil.move(str(staging / "runs" / name), str(destination))

    for name in OUTPUT_FILES:
        destination = canonical_outputs / name
        if destination.exists():
            shutil.move(str(destination), str(backup / "outputs" / name))
        shutil.move(str(staging / "outputs" / name), str(destination))

    report_name = "controlled_training_curves.png"
    report_destination = canonical_reports / report_name
    if report_destination.exists():
        shutil.move(
            str(report_destination),
            str(backup / "reports" / report_name),
        )
    shutil.move(str(staging / "reports" / report_name), str(report_destination))

    canonical_checkpoint_by_run = {
        name: str((canonical_runs / name / "best.pt").resolve())
        for name in RUN_NAMES
    }
    comparison_path = canonical_outputs / "controlled_experiment_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison["checkpoint_path"] = comparison["experiment"].map(
        canonical_checkpoint_by_run
    )
    comparison.to_csv(comparison_path, index=False)

    selection_path = canonical_outputs / "controlled_experiment_selection.json"
    selection["selected_checkpoint"] = canonical_checkpoint_by_run[
        selection["selected_experiment"]
    ]
    selection_path.write_text(json.dumps(selection, indent=2))

    for name in RUN_NAMES:
        validation_path = canonical_runs / name / "validation_result.json"
        validation = json.loads(validation_path.read_text())
        validation["checkpoint_path"] = canonical_checkpoint_by_run[name]
        validation_path.write_text(json.dumps(validation, indent=2))

    manifest_path = canonical_outputs / "controlled_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in RUN_NAMES:
        canonical_run_dir = str((canonical_runs / name).resolve())
        manifest.setdefault("run_directories", {})[name] = canonical_run_dir
        manifest["experiment_configurations"][name].pop("output_dir", None)
        manifest["experiment_configurations"][name].pop("experiment", None)
        manifest["experiment_configurations"][name].pop("config_id", None)
        manifest["experiment_configurations"][name].pop("dataset_root", None)
        manifest["experiment_configurations"][name].pop("split_manifest", None)
        manifest["experiment_configurations"][name].pop(
            "test_loader_constructed", None
        )
        manifest["experiment_configurations"][name].pop("test_evaluated", None)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "selected_experiment": selection["selected_experiment"],
        "selected_checkpoint": selection["selected_checkpoint"],
        "backup_root": str(backup),
        "test_access_count": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    args = parser.parse_args()
    result = promote_verified_staging(
        args.staging_root,
        args.project_root,
        args.backup_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
