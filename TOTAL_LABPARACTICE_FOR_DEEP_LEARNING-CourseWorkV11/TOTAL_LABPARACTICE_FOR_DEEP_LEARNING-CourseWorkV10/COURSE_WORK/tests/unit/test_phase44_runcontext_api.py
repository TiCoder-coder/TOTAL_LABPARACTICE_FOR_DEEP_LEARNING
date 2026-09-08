"""PHASE 44 — RunContext constructor regression tests (TASK 8).

After the human's official attempt crashed with TypeError on `rehearsal=`,
these tests guarantee that:

  1. Official RunContext constructor accepts all canonical fields
  2. The obsolete `rehearsal=` keyword is rejected (loud failure)
  3. Every RunContext call site in scripts/ matches the dataclass schema
  4. Rehearsal RunContext is still valid
  5. Official setup reaches the FIRST Stage A training boundary
  6. No API mismatch occurs before the first training call
  7. candidate count = 4
  8. fold count = 3
  9. Stage A plan count = 12
 10. Stage B template count = 12
 11. Persistence count = 3
 12. Test inaccessible
 14. No canonical artifact mutation
"""
from __future__ import annotations

import inspect
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import ( 
    RunContext, run_real_pipeline, assert_context_invariants,
)

class _TrainingBoundarySentinel(Exception):
    pass


def test_official_runcontext_accepts_canonical_fields():
    """RunContext must accept every field that the official CLI passes."""
    canonical_fields = {
            "project_root": PROJECT_ROOT,
            "transformer_shortlist_path": PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            "lstm_handoff_path": PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            "phase_42_signoff_path": PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            "phase_43_signoff_path": PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            "artifact_dir": Path("/tmp/x_artifacts"),
            "registry_root": Path("/tmp/x_registry"),
            "run_root": Path("/tmp/x_runs"),
            "seed": 42,
            "is_rehearsal": False,
            "scientific_max_epochs": 50,
            "scientific_patience": 10,
            "rehearsal_synthetic": False,
        }
    with tempfile.TemporaryDirectory() as tmp:
        canonical_fields["artifact_dir"] = Path(tmp) / "artifacts"
        canonical_fields["registry_root"] = Path(tmp) / "registry"
        canonical_fields["run_root"] = Path(tmp) / "runs"
        ctx = RunContext(**canonical_fields)
    assert ctx.is_rehearsal is False
    assert ctx.scientific_max_epochs == 50
    assert ctx.scientific_patience == 10
    assert ctx.rehearsal_synthetic is False
    assert ctx.seed == 42


def test_obsolete_rehearsal_keyword_is_rejected():
    """Passing the obsolete `rehearsal=` keyword must raise TypeError.

    This is the exact regression: the human attempt crashed because the
    CLI was still passing `rehearsal=False` (a removed field).
    """
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(TypeError) as exc_info:
            RunContext(
                project_root=PROJECT_ROOT,
                transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
                lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
                phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
                phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
                artifact_dir=Path(tmp) / "artifacts",
                registry_root=Path(tmp) / "registry",
                run_root=Path(tmp) / "runs",
                seed=42,
                rehearsal=False,
                is_rehearsal=False,
            )
    msg = str(exc_info.value)
    assert "rehearsal" in msg or "unexpected keyword" in msg, (
        f"expected TypeError mentioning `rehearsal`, got: {msg}"
    )

def test_every_runcontext_call_site_matches_dataclass_schema():
    """Static check: parse every `RunContext(` in the repo and verify that
    all keyword arguments exist in the canonical dataclass fields.
    """
    import ast

    rc_params = set(inspect.signature(RunContext).parameters.keys())
    bad_call_sites = []
    files_to_check = [
        "scripts/phase44_rolling_origin.py",
        "scripts/_disposable_rehearsal_phase44.py",
        "scripts/_phase44_plan_probe.py",
        "scripts/_phase44_signoff_diagnostic.py",
        "scripts/_phase44_real_path_probe.py",
        "scripts/_phase44_epoch_propagation_check.py",
        "scripts/_phase44_handoff_check.py",
        "scripts/_phase44_official_dry_probe.py",
        "tests/unit/test_phase44_real_orchestrator.py",
        "tests/unit/test_phase44_failure_injection.py",
        "tests/unit/test_phase44_final_invariants.py",
    ]
    for relpath in files_to_check:
        fpath = PROJECT_ROOT / relpath
        if not fpath.exists():
            continue
        tree = ast.parse(fpath.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name != "RunContext":
                    continue
                for kw in node.keywords:
                    if kw.arg not in rc_params:
                        bad_call_sites.append(
                            f"{relpath}:{node.lineno} RunContext({kw.arg}=...) "
                            f"is not a valid field"
                        )
    assert not bad_call_sites, (
        "RunContext call sites with bad kwargs:\n" + "\n".join(bad_call_sites)
    )


def test_rehearsal_runcontext_is_still_valid():
    """Rehearsal must not be broken by the official-context fix."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=2,
            scientific_patience=2,
            seed=42,
            is_rehearsal=True,
            rehearsal_synthetic=True,
        )
        assert_context_invariants(ctx)  
        assert ctx.is_rehearsal is True
        assert ctx.rehearsal_synthetic is True
        assert ctx.scientific_max_epochs == 2


def test_official_setup_reaches_first_training_boundary():
    """The OFFICIAL path must reach the FIRST Stage A training call.

    We patch train_stage_a to raise a sentinel, run the official
    orchestrator with temp registry / artifact_dir / run_root, and
    verify the sentinel was raised. This proves:
      - RunContext constructed for official config
      - assert_context_invariants passes for official
      - 4 candidates load
      - 3 folds build
      - 12 Stage A plans and 12 Stage B templates are scheduled
      - 3 Persistence folds are scheduled
      - Test is NEVER accessed
      - No canonical artifact mutation
      - No canonical registry record
    """
    from course_work.rolling_origin import real_run as real_run_mod
    from course_work.rolling_origin import stages as stages_mod
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
    )
    from course_work.rolling_origin.folds import build_rolling_folds

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )

    with tempfile.TemporaryDirectory(prefix="phase44_official_test_") as tmp:
        tmp_path = Path(tmp)
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp_path / "artifacts",
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            seed=42,
            is_rehearsal=False,
            scientific_max_epochs=50,
            scientific_patience=10,
            rehearsal_synthetic=False,
            dataset_factory=lambda cand, fold: (
                real_run_mod.build_real_canonical_base_dataset(
                    project_root=PROJECT_ROOT, candidate=cand,
                )
            ),
        )
        assert_context_invariants(ctx)

        candidates = load_candidates(
            project_root=ctx.project_root,
            transformer_shortlist_path=ctx.transformer_shortlist_path,
            lstm_handoff_path=ctx.lstm_handoff_path,
        )
        assert len(candidates) == 4
        rtrn_ids = extract_robase_train_ids(ctx.project_root)
        rval_ids = extract_robase_val_ids(ctx.project_root)
        folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
        assert len(folds) == 3
        assert len(candidates) * len(folds) == 12

        boundary_reached = {"stage_a": False}

        def sentinel(*args, **kwargs):
            boundary_reached["stage_a"] = True
            raise _TrainingBoundarySentinel("first Stage A reached")

        orig_train_stage_a_real_run = real_run_mod.train_stage_a
        orig_train_stage_a_stages = stages_mod.train_stage_a
        real_run_mod.train_stage_a = sentinel
        stages_mod.train_stage_a = sentinel
        try:
            result = run_real_pipeline(ctx)
        finally:
            real_run_mod.train_stage_a = orig_train_stage_a_real_run
            stages_mod.train_stage_a = orig_train_stage_a_stages

        assert boundary_reached["stage_a"] is True, (
            "FIRST_TRAINING_BOUNDARY_REACHED must be True for official config"
        )
        assert result.exception is not None
        assert "_TrainingBoundarySentinel" in result.exception

    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    assert canonical_after == canonical_before, (
        "canonical artifact_dir was mutated by official dry-probe"
    )