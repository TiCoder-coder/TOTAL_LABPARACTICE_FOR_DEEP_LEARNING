import hashlib
import json
from pathlib import Path

import pytest

from scripts import sweep_results_to_csv


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_canonical_entrypoints_exclude_non_canonical_scripts() -> None:
    root = project_root()
    canonical_paths = (
        root / "src/course_work/experiments/phase_execution.py",
        root / "src/course_work/sweeps/dropout.py",
        root / "src/course_work/sweeps/heads.py",
        root / "src/course_work/sweeps/weight_decay.py",
        root / "src/course_work/sweeps/sweep_results.py",
        root / "scripts/run_single_condition.py",
        root / "scripts/run_all_pending.py",
        root / "scripts/run_phase_background.py",
        root / "scripts/run_phase_background/_notebook.py",
    )
    forbidden_names = (
        "_populate_sweep_results",
        "_fix_sweep_cells",
        "run_with_env_bypass",
        "run_all_sweeps_bypass",
        "setup_phases_0_to_15",
    )
    for path in canonical_paths:
        source = path.read_text(encoding="utf-8")
        assert all(name not in source for name in forbidden_names)


def test_missing_jsonl_cannot_create_header_only_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="source results are missing"):
        sweep_results_to_csv.convert_one("s8_learning_rate")
    assert not (tmp_path / "artifacts/sweeps/s8_learning_rate/results.csv").exists()


def test_empty_jsonl_cannot_create_header_only_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    source_path = tmp_path / "artifacts/sweeps/s8_learning_rate/live_sweep_results.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="without verified rows"):
        sweep_results_to_csv.convert_one("s8_learning_rate")
    assert not (tmp_path / "artifacts/sweeps/s8_learning_rate/results.csv").exists()


def test_phase_31_converter_uses_canonical_metrics_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    source_path = tmp_path / "artifacts/sweeps/S9_weight_decay/live_sweep_results.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "condition": "WD0",
                "sweep_id": "S9_WEIGHT_DECAY",
                "run_id": "RUN-WD0",
                "best_epoch": 4,
                "best_validation_rmse_wh": 60.0,
                "best_validation_mae_wh": 25.0,
                "best_validation_r2": 0.5,
                "config": {"weight_decay": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path, row_count = sweep_results_to_csv.convert_one("S9_weight_decay")
    assert path.name == "s9_weight_decay_metrics.csv"
    assert row_count == 1
    assert "weight_decay" in path.read_text(encoding="utf-8").splitlines()[0]


def test_phase_32_converter_uses_canonical_metrics_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    source_path = tmp_path / "artifacts/sweeps/S10_dropout/live_sweep_results.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "condition": "DR02",
                "sweep_id": "S10_DROPOUT",
                "run_id": "RUN-DR02",
                "best_epoch": 4,
                "best_validation_rmse_wh": 60.0,
                "best_validation_mae_wh": 25.0,
                "best_validation_r2": 0.5,
                "config": {"dropout": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path, row_count = sweep_results_to_csv.convert_one("S10_dropout")
    assert path.name == "s10_dropout_metrics.csv"
    assert row_count == 1
    assert "dropout" in path.read_text(encoding="utf-8").splitlines()[0]


def test_phase_33_converter_uses_canonical_metrics_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    source_path = tmp_path / "artifacts/sweeps/S11_d_model/live_sweep_results.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "condition": "D32",
                "sweep_id": "S11_D_MODEL",
                "run_id": "RUN-D32",
                "best_epoch": 4,
                "best_validation_rmse_wh": 60.0,
                "best_validation_mae_wh": 25.0,
                "best_validation_r2": 0.5,
                "config": {"d_model": 32},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path, row_count = sweep_results_to_csv.convert_one("S11_d_model")
    assert path.name == "s11_d_model_metrics.csv"
    assert row_count == 1
    assert "d_model" in path.read_text(encoding="utf-8").splitlines()[0]


def test_phase_34_converter_uses_canonical_metrics_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep_results_to_csv, "ROOT", tmp_path)
    source_path = tmp_path / "artifacts/sweeps/S12_heads/live_sweep_results.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "condition": "H2",
                "sweep_id": "S12_HEADS",
                "run_id": "RUN-H2",
                "best_epoch": 4,
                "best_validation_rmse_wh": 60.0,
                "best_validation_mae_wh": 25.0,
                "best_validation_r2": 0.5,
                "config": {"d_model": 64, "num_heads": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path, row_count = sweep_results_to_csv.convert_one("S12_heads")
    assert path.name == "s12_head_metrics.csv"
    assert row_count == 1
    assert "num_heads" in path.read_text(encoding="utf-8").splitlines()[0]


def test_notebook_selective_cells_are_static_orchestration_only() -> None:
    root = project_root()
    notebook = json.loads((root / "notebook_course_work/CourseWork.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    phase_cell = cells["3fe4478c"]
    configuration_cell = cells["phase-33-config-display"]
    assert "".join(phase_cell["source"]) == (
        "display(render_frozen_phase_evidence(30, PROJECT_ROOT))\n"
    )
    assert "".join(configuration_cell["source"]) == (
        "from course_work.reporting.phase_summary import render_phase_33_transformer_configuration\n"
        "display(render_phase_33_transformer_configuration())\n"
    )
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert cell_ids.index("phase-32-resume") < cell_ids.index("phase-33-config-heading")
    assert cell_ids.index("phase-33-config-heading") < cell_ids.index("phase-33-config-display")
    assert cell_ids.index("phase-33-config-display") < cell_ids.index("phase-34-heading")
    assert cell_ids.index("phase-34-heading") < cell_ids.index("phase-34-resume")
    assert cell_ids.index("phase-34-resume") < cell_ids.index("phase-35-heading")
    assert cell_ids.index("phase-35-heading") < cell_ids.index("phase-35-resume")
    assert cell_ids.index("phase-35-resume") < cell_ids.index("phase-36-heading")
    assert cell_ids.index("phase-36-heading") < cell_ids.index("phase-36-resume")
    assert cell_ids.index("phase-36-resume") < cell_ids.index("phase-37-heading")
    assert cell_ids.index("phase-37-heading") < cell_ids.index("phase-37-resume")
    assert cell_ids.index("phase-37-resume") < cell_ids.index("17dc5a66")
    phase_cells = {
        22: "cd4716af",
        23: "0881cfe3",
        24: "9b2d6e88",
        25: "51bc5965",
        26: "585b390d",
        27: "86f4ac0c",
        28: "b70c9707",
        29: "eb4f1b80",
        30: "3fe4478c",
        31: "phase-31-resume",
        32: "phase-32-resume",
        34: "phase-34-resume",
        35: "phase-35-resume",
        36: "phase-36-resume",
        37: "phase-37-resume",
        38: "17dc5a66",
        39: "f0fcdce7",
        40: "d407f95c",
        41: "7e86663d",
    }
    assert "".join(cells[phase_cells[22]]["source"]) == (
        "display(render_frozen_phase_evidence(22, PROJECT_ROOT))\n"
    )
    for phase_id in [*range(23, 33), *range(34, 42)]:
        assert "".join(cells[phase_cells[phase_id]]["source"]) == (
            f"display(render_frozen_phase_evidence({phase_id}, PROJECT_ROOT))\n"
        )
    assert len(phase_cell["outputs"]) == 1
    assert len(configuration_cell["outputs"]) == 1
    for phase_id in phase_cells:
        assert len(cells[phase_cells[phase_id]]["outputs"]) == 1
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            assert "application/vnd.jupyter.widget-view+json" not in data
    for cell in (phase_cell, configuration_cell):
        rendered = json.dumps(cell["outputs"], ensure_ascii=False).lower()
        assert "<script" not in rendered
        assert "jupyter.widget" not in rendered


def test_notebook_non_target_outputs_match_preservation_baseline() -> None:
    root = project_root()
    notebook = json.loads((root / "notebook_course_work/CourseWork.ipynb").read_text(encoding="utf-8"))
    baseline = json.loads(
        (root / "docs/save_log_in_processing/notebook_output_content_preservation_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {row["cell_id"]: row["outputs_sha256"] for row in baseline["cells"]}
    excluded = {
        "0caba13e",
        "phase-1-orchestration",
        "b0f56164",
        "phase-2-orchestration",
        "phase-3-orchestration",
        "phase-4-orchestration",
        "phase-5-orchestration",
        "phase-7-orchestration",
        "phase-8-orchestration",
        "phase-9-orchestration",
        "phase-10-orchestration",
        "phase-11-orchestration",
        "phase-12-orchestration",
        "phase-13-orchestration",
        "phase-14-orchestration",
        "499bc911",
        "518fca61",
        "efc7f487",
        "d60e1951",
        "2acc38f9",
        "2eb13c9c",
        "03db9e36",
        "3fe4478c",
        "phase-31-heading",
        "phase-31-resume",
        "phase-32-heading",
        "phase-32-resume",
        "phase-33-config-heading",
        "phase-33-config-display",
        "phase-34-heading",
        "phase-34-resume",
        "phase-35-heading",
        "phase-35-resume",
        "phase-36-heading",
        "phase-36-resume",
        "phase-37-heading",
        "phase-37-resume",
        "cd4716af",
        "0881cfe3",
        "9b2d6e88",
        "51bc5965",
        "585b390d",
        "86f4ac0c",
        "b70c9707",
        "eb4f1b80",
        "17dc5a66",
        "f0fcdce7",
        "d407f95c",
        "7e86663d",
        "ec7b2ea5",
        "05ec07a2",
        "5e9eec01",
        "4293c1aa",
        "b3d754ee",
        "d0833283",
        "8a7873a9",
        "5acc48a2",
        "fa11ad6a",
        "994bbdb9",
        "ed1e48a7",
        "6a082d01",
        "9eacc9e3",
        "2f0130ea",
        "88d07113",
        "4e6b273d",
        "b388dcd1",
        "753ab0f1",
        "62f722b6",
        "4dea0c14",
        "b496ac6f",
        "1451ccf0",
        "7cc29214",
        "bbcf08f2",
        "9a2f1b89",
        "1d5b5327",
        "2e8f083e",
        "c9df346a",
        "870efbe4",
        "7b48186f",
        "afeb48a2",
        "1c098fa6",
        "3c915fcc",
        "644dbb41",
        "de13c9d0",
        "5627f983",
        "332dd9c8",
        "9f864b91",
        "251da856",
        "76de7c69",
        "v2closure",
        "v2closedash",
        "a375b9ca",
    }
    for cell in notebook["cells"]:
        if cell["id"] in excluded:
            continue
        if cell["id"] not in expected:
            continue
        payload = json.dumps(
            cell.get("outputs", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == expected[cell["id"]]
