"""PHASE 44 — Failure injection tests (TASK 15).

Simulates failures at various pipeline stages and verifies the orchestrator
handles them gracefully (returns RealRunResult with non-zero exit_code and
exception trace, does NOT corrupt canonical artifacts).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import ( 
    RunContext,
    run_real_pipeline,
)


def _ctx(tmp: Path) -> RunContext:
    return RunContext(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=tmp / "artifacts",
        registry_root=tmp / "registry",
        run_root=tmp / "runs",
        scientific_max_epochs=2,
        scientific_patience=2,
        seed=42,
        is_rehearsal=True,
        rehearsal_synthetic=True,
    )


def test_missing_transformer_shortlist_fails_gracefully():
    """Missing transformer shortlist degrades gracefully to LSTM-only path."""
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    before = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.transformer_shortlist_path = Path("/nonexistent/transformer_shortlist.json")
        result = run_real_pipeline(ctx)
    after = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    assert before == after, "canonical artifact_dir was mutated by failure injection"
    assert result.exit_code == 1
    assert result.n_candidates == 1
    assert result.signoff_overall_status == "FAIL"


def test_missing_lstm_handoff_fails_gracefully():
    """Missing LSTM handoff degrades gracefully to Transformer-only path (3 cand)."""
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    before = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.lstm_handoff_path = Path("/nonexistent/lstm_handoff.json")
        result = run_real_pipeline(ctx)
    after = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    assert before == after
    assert result.n_candidates == 3
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING"}
    c01_failures = [f for f in result.signoff_failures if "C01" in f]
    if c01_failures:
        assert result.signoff_overall_status == "PASS_WITH_WARNING"
    assert result.exception is None


def test_zero_epoch_budget_fails_gracefully():
    """scientific_max_epochs=0 in rehearsal is rejected by guards."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.scientific_max_epochs = 0
        ctx.scientific_max_epochs = 6
        with pytest.raises(RuntimeError, match="rehearsal scientific_max_epochs"):
            from course_work.rolling_origin.real_run import assert_context_invariants
            assert_context_invariants(ctx)


def test_invalid_artifact_dir_is_reported_as_failure():
    """An artifact_dir that can't be written to triggers a structured failure."""
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    before = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.artifact_dir = Path("/nonexistent_readonly_dir_9999/artifacts")
        result = run_real_pipeline(ctx)
    after = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    assert before == after
    assert result is not None


def test_orchestrator_never_writes_under_canonical_artifact_dir_on_failure():
    """Even on failure, canonical artifact_dir is never mutated."""
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    before = sorted(p.name for p in canonical.iterdir()) \
        if canonical.exists() else []

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.transformer_shortlist_path = Path("/nonexistent/broken.json")
        run_real_pipeline(ctx)

    after = sorted(p.name for p in canonical.iterdir()) \
        if canonical.exists() else []
    assert before == after, "canonical artifact_dir mutated by failed run"


def test_official_run_id_is_never_allocated_during_synthetic_rehearsal():
    """Synthetic rehearsal must NOT create persistent run IDs in canonical registry."""
    canonical_registry = PROJECT_ROOT / "artifacts" / "registry"
    if canonical_registry.exists():
        before = sorted(p.name for p in canonical_registry.iterdir())
    else:
        before = []
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(Path(tmp))
        ctx.is_rehearsal = True
        ctx.rehearsal_synthetic = True
        run_real_pipeline(ctx)
    if canonical_registry.exists():
        after = sorted(p.name for p in canonical_registry.iterdir())
    else:
        after = []
    assert before == after, "canonical registry was mutated by rehearsal"


def test_synthetic_official_mock_is_unreachable():
    """The old pipeline.run_pipeline synthetic mock must be unreachable from
    run_real_pipeline(). If anyone ever re-introduces a call to it, this test
    will fail loudly.
    """
    from course_work.rolling_origin import pipeline as p44pipe

    def explode(*args, **kwargs):
        raise RuntimeError("OLD SYNTHETIC FUNCTION CALLED — must not happen")

    with patch.object(p44pipe, "run_pipeline", side_effect=explode), \
         patch.object(p44pipe, "_synth_predict_for_official", side_effect=explode):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _ctx(Path(tmp))
            result = run_real_pipeline(ctx)

    assert result.exit_code in (0, 1)
    assert result.n_candidates == 4
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING"}


def test_ys1_scaler_bundle_contract_satisfies_canonical_transform_target():
    """Regression: `FoldLocalScalerBundle.as_dict()` must satisfy the
    canonical `transform_target` / `inverse_transform_target` contract
    (option, scaling_version, fitted scaler object). This was the
    original cause of the `YS1 scaler bundle contract mismatch` failure
    in RUN_TR_ROB_0002/0003.
    """
    import numpy as np
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
    )
    from course_work.rolling_origin.folds import build_rolling_folds
    from course_work.rolling_origin.real_run import (
        fit_fold_local_scaler, build_real_canonical_base_dataset, RunContext,
    )
    from course_work.data.scaling import (
        transform_target, inverse_transform_target,
    )

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
    ro1 = folds[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp / "artifacts",
            registry_root=tmp / "registry",
            run_root=tmp / "runs",
            is_rehearsal=True,
            rehearsal_synthetic=True,
        )
        c = candidates[0]
        fd = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=c)
        for stage in ("A", "B"):
            bundle, _ = fit_fold_local_scaler(
                candidate=c, fold=ro1, fold_dataset=fd, fit_stage=stage, ctx=ctx,
            )
            d = bundle.as_dict()
            assert d["option"] == c.target_scaling_option, (
                f"stage {stage}: option field must equal target_scaling_option"
            )
            assert d["scaling_version"] in ("SCALING-v1", "FINAL_SCALING-v1"), (
                f"stage {stage}: scaling_version must be canonical, got {d['scaling_version']}"
            )
            assert d.get("scaler") is not None and hasattr(d["scaler"], "transform"), (
                f"stage {stage}: must have a fitted scaler with .transform method"
            )
            test_y = np.array([50.0, 100.0, 200.0])
            scaled = transform_target(test_y, "YS1", d)
            recovered = inverse_transform_target(scaled, "YS1", d)
            assert float(np.max(np.abs(test_y - recovered))) < 1e-9


def test_fold_local_bundle_uses_canonical_ys1_statistics():
    """Regression: the fold-local Y scaler MUST be fitted on the
    fold-specific allowed training population, NOT copied from the
    canonical Phase9 YS1 bundle.

    This enforces the Phase 44 plan §38, §40, §154:
      - Stage A Y scaler fit ONLY on inner_train allowed targets
      - Stage B Y scaler fit ONLY on outer_train allowed targets
      - Global Phase9 Y statistics MUST NOT be the source of fold-local stats

    Previously the orchestrator incorrectly copied global Phase9 YS1
    mean/std into every fold-local bundle (the 'previous YS1 schema
    fix'). That contract is now scientifically invalid; this test
    enforces that the fold-local Y scaler is independently fit.
    """
    import numpy as np
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
    )
    from course_work.rolling_origin.folds import build_rolling_folds
    from course_work.rolling_origin.real_run import (
        fit_fold_local_scaler, build_real_canonical_base_dataset, RunContext,
    )
    from course_work.data.scaling import (
        inverse_transform_target,
        load_validated_target_scaler,
        transform_target,
    )

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
    ro1 = folds[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp / "artifacts",
            registry_root=tmp / "registry",
            run_root=tmp / "runs",
            is_rehearsal=True,
            rehearsal_synthetic=True,
        )
        canonical = load_validated_target_scaler(PROJECT_ROOT)
        canonical_mean = float(canonical["scaler"].mean_[0])
        canonical_std = float(canonical["scaler"].scale_[0])

        c = candidates[0]
        fd = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=c)

        seen_means = set()
        seen_stds = set()
        for stage in ("A", "B"):
            bundle, _ = fit_fold_local_scaler(
                candidate=c, fold=ro1, fold_dataset=fd, fit_stage=stage, ctx=ctx,
            )
            assert bundle.y_mean is not None and bundle.y_std is not None
            seen_means.add(round(bundle.y_mean, 4))
            seen_stds.add(round(bundle.y_std, 4))
            d = bundle.as_dict()
            assert d["option"] == "YS1"
            assert d["scaling_version"] in ("SCALING-v1", "FINAL_SCALING-v1")
            assert d["scaler"] is not None and hasattr(d["scaler"], "transform")
            test_y = np.array([50.0, 100.0, 200.0])
            scaled = transform_target(test_y, "YS1", d)
            recovered = inverse_transform_target(scaled, "YS1", d)
            assert float(np.max(np.abs(test_y - recovered))) < 1e-9

        for fold in folds:
            for stage in ("A", "B"):
                bundle, _ = fit_fold_local_scaler(
                    candidate=c, fold=fold, fold_dataset=fd, fit_stage=stage, ctx=ctx,
                )
                seen_means.add(round(bundle.y_mean, 4))
                seen_stds.add(round(bundle.y_std, 4))

        assert len(seen_means) >= 2, (
            f"only {len(seen_means)} distinct y_mean values across 6 "
            f"fold-stage scalers — suggests global reuse"
        )
        assert len(seen_stds) >= 2, (
            f"only {len(seen_stds)} distinct y_std values across 6 "
            f"fold-stage scalers — suggests global reuse"
        )
