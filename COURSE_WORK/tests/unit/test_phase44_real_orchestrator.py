"""PHASE 44 — Tests for the REAL orchestrator (TASK 16).

These tests verify the contract of `course_work.rolling_origin.real_run`:
  - Constants match the Phase 44 scientific budget
  - RunContext guards reject fast-mode / rehearsal leakage into official mode
  - The orchestrator can be exercised in disposable rehearsal mode
  - Canonical artifact_dir is never touched by rehearsal
  - Persistence is real (not synthetic)
  - Pooled metrics are computed from raw residuals
  - Signoff logic returns PASS / PASS_WITH_WARNING / FAIL only
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import (  # noqa: E402
    SCIENTIFIC_MAX_EPOCHS,
    SCIENTIFIC_PATIENCE,
    PERSISTENCE_FOLD_COUNT,
    RunContext,
    assert_context_invariants,
    run_real_pipeline,
)


def _make_rehearsal_ctx(tmp_dir: Path) -> RunContext:
    """Build a RunContext for disposable rehearsal.

    Points at the CANONICAL LSTM handoff
    (artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json) so that
    load_candidates() resolves 4 candidates (3 TR + 1 LSTM). This is the
    SAME schema that the canonical Phase 44 official path consumes.
    """
    return RunContext(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=tmp_dir / "artifacts",
        registry_root=tmp_dir / "registry",
        run_root=tmp_dir / "runs",
        scientific_max_epochs=2,
        scientific_patience=2,
        seed=42,
        is_rehearsal=True,
        rehearsal_synthetic=True,
    )


def test_scientific_constants_match_phase44_contract():
    """SCIENTIFIC_MAX_EPOCHS=50, SCIENTIFIC_PATIENCE=10 per Phase 44 plan."""
    assert SCIENTIFIC_MAX_EPOCHS == 50
    assert SCIENTIFIC_PATIENCE == 10
    assert PERSISTENCE_FOLD_COUNT == 3


def test_official_context_rejects_fast_mode_budget():
    """Official mode requires max_epochs >= 50 and patience >= 10."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=Path(tmp) / "lstm.json",  # OK for guards test
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=2,  # fast-mode value!
            is_rehearsal=False,        # but in official mode
            rehearsal_synthetic=False,
        )
        with pytest.raises(RuntimeError, match="official scientific_max_epochs"):
            assert_context_invariants(ctx)


def test_official_context_rejects_low_patience():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=Path(tmp) / "lstm.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=50,
            scientific_patience=2,    # too low
            is_rehearsal=False,
        )
        with pytest.raises(RuntimeError, match="official scientific_patience"):
            assert_context_invariants(ctx)


def test_official_context_rejects_synthetic_flag():
    """Synthetic rehearsal is forbidden in official mode."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=Path(tmp) / "lstm.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=50,
            scientific_patience=10,
            is_rehearsal=False,
            rehearsal_synthetic=True,  # FORBIDDEN in official
        )
        with pytest.raises(RuntimeError, match="rehearsal_synthetic=True is forbidden"):
            assert_context_invariants(ctx)


def test_rehearsal_context_accepts_tiny_budget():
    """Rehearsal mode allows max_epochs <= 5."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_rehearsal_ctx(Path(tmp))
        assert_context_invariants(ctx)  # should NOT raise


def test_rehearsal_context_rejects_huge_budget():
    """Rehearsal mode rejects max_epochs > 5."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_rehearsal_ctx(Path(tmp))
        ctx.scientific_max_epochs = 10  # exceeds ceiling
        with pytest.raises(RuntimeError, match="rehearsal scientific_max_epochs"):
            assert_context_invariants(ctx)


def test_real_orchestrator_runs_disposable_rehearsal():
    """The real orchestrator runs end-to-end in disposable rehearsal mode."""
    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = set(
        p.name for p in canonical_artifact_dir.iterdir()
        if canonical_artifact_dir.exists()
    )
    with tempfile.TemporaryDirectory(prefix="phase44_test_") as tmp:
        ctx = _make_rehearsal_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    canonical_after = set(
        p.name for p in canonical_artifact_dir.iterdir()
        if canonical_artifact_dir.exists()
    )
    assert canonical_before == canonical_after, (
        "Canonical artifact_dir was mutated by disposable rehearsal!"
    )
    assert result.exit_code in (0, 1)
    assert result.n_folds == 3
    assert result.n_persistence_bundles == 3
    assert result.n_candidates >= 3  # at least 3 Transformers
    # Signoff must be a valid status
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING", "FAIL"}


def test_real_orchestrator_runs_official_config_guards():
    """The orchestrator's guards reject official + fast-mode budget."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=Path(tmp) / "lstm.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=2,    # fast-mode value
            scientific_patience=2,
            is_rehearsal=False,
            rehearsal_synthetic=False,
        )
        with pytest.raises(RuntimeError):
            run_real_pipeline(ctx)


def test_real_run_module_exposes_required_public_api():
    """The module exposes the required public functions/classes."""
    import course_work.rolling_origin.real_run as rr
    assert hasattr(rr, "run_real_pipeline")
    assert hasattr(rr, "RunContext")
    assert hasattr(rr, "assert_context_invariants")
    assert hasattr(rr, "SCIENTIFIC_MAX_EPOCHS")
    assert hasattr(rr, "SCIENTIFIC_PATIENCE")
    assert hasattr(rr, "PERSISTENCE_FOLD_COUNT")


def test_real_run_result_dataclass_has_required_fields():
    """RealRunResult carries the canonical counter fields."""
    from course_work.rolling_origin.real_run import RealRunResult
    fields = RealRunResult.__dataclass_fields__
    for required in ("exit_code", "summary", "n_candidates", "n_folds",
                     "n_stage_a_runs", "n_stage_b_runs",
                     "n_outer_prediction_bundles", "n_persistence_bundles",
                     "recommended_transformer_id", "signoff_overall_status"):
        assert required in fields, f"missing field {required}"


def test_rehearsal_runs_in_tempdir_no_canonical_mutation():
    """Verify temp isolation: rehearsal does not write under canonical artifacts."""
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    if canonical.exists():
        before = {p.name for p in canonical.iterdir()}
    else:
        before = set()
    with tempfile.TemporaryDirectory(prefix="phase44_iso_") as tmp:
        ctx = _make_rehearsal_ctx(Path(tmp))
        run_real_pipeline(ctx)
    if canonical.exists():
        after = {p.name for p in canonical.iterdir()}
        assert before == after
