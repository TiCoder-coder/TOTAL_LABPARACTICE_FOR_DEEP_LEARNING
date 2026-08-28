"""Notebook entrypoints for coursework phases 42 through 46."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from runpy import run_path
from typing import Any

from course_work.utils.artifacts import read_json


def _materialize(phase_id: int, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    script = root / "scripts" / f"phase{phase_id}_{_SCRIPT_NAMES[phase_id]}.py"
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        run_path(str(script), run_name="__main__")
    return read_json(root / "artifacts" / _SIGNOFF_PATHS[phase_id])


_SCRIPT_NAMES = {
    42: "candidate_synthesis",
    43: "lstm_tuning",
    44: "rolling_origin",
    45: "final_model_lock",
    46: "three_seed_runs",
}

_SIGNOFF_PATHS = {
    42: "candidate_synthesis/phase_42_signoff.json",
    43: "lstm_tuning/phase_43_signoff.json",
    44: "rolling_origin/phase_44_signoff.json",
    45: "final_model_lock/phase_45_signoff.json",
    46: "three_seed_final_runs/phase_46_signoff.json",
}


def materialize_phase_42(project_root: Path) -> dict[str, Any]:
    return _materialize(42, project_root)


def materialize_phase_43(project_root: Path) -> dict[str, Any]:
    return _materialize(43, project_root)


def materialize_phase_44(project_root: Path) -> dict[str, Any]:
    return _materialize(44, project_root)


def materialize_phase_45(project_root: Path) -> dict[str, Any]:
    return _materialize(45, project_root)


def materialize_phase_46(project_root: Path) -> dict[str, Any]:
    return _materialize(46, project_root)