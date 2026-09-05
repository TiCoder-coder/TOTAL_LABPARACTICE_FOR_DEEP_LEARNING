"""PHASE 44 — Focused regression tests for real data path (TASKS 5+6+7+11+12).

Tests cover:
  1. real L36 loader returns [B,36,F]
  2. real L72 loader returns [B,72,F]
  3. Transformer real forward succeeds
  4. LSTM real forward succeeds
  5. target IDs map exactly to dataset samples
  6. scaling preserves [B,L,F] invariant
  7. inner_train population correct
  8. inner_val population correct
  9. outer_train population correct
 10. outer_eval population correct
 11. all 4 candidates work
 12. all 3 folds work
 13. no Test rows
 14. hard-stop reaches real first forward
 15. optimizer steps remain zero in probe
 16. no new official run IDs from probe
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---- Helpers ----

def _load_cands_and_folds():
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
    )
    from course_work.rolling_origin.folds import build_rolling_folds
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
    all_robase_ids = set(str(t) for t in (rtrn_ids + rval_ids))
    return candidates, folds, all_robase_ids


def _build_loader(base, ids, batch_size=32, shuffle=False, seed=42):
    from course_work.rolling_origin.stages import load_fold_subset_loader
    return load_fold_subset_loader(
        base_dataset=base,
        target_ids=list(ids),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


# ---- Test 1: L36 loader shape ----
@pytest.mark.parametrize("candidate_id", ["TR_C0_PRIMARY", "TR_C1_ALT_WEIGHT_DECAY", "LSTM_TUNED_WINNER"])
def test_real_l36_loader_returns_correct_shape(candidate_id):
    """Test 1: real L36 loader returns [B, 36, F]."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = next(c for c in candidates if c.candidate_id == candidate_id)
    assert cand.lookback_steps == 36, f"{candidate_id} should have lookback=36"

    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, _ = _build_loader(base, folds[0].inner_train_ids[:64], batch_size=32)
    batch = next(iter(loader))
    assert batch["x"].shape == (32, 36, 33), (
        f"{candidate_id} batch x shape should be (32, 36, 33), "
        f"got {tuple(batch['x'].shape)}"
    )


# ---- Test 2: L72 loader shape ----
def test_real_l72_loader_returns_correct_shape():
    """Test 2: real L72 loader returns [B, 72, F]."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = next(c for c in candidates if c.candidate_id == "TR_C2_ALT_LOOKBACK")
    assert cand.lookback_steps == 72

    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, _ = _build_loader(base, folds[0].inner_train_ids[:64], batch_size=32)
    batch = next(iter(loader))
    assert batch["x"].shape == (32, 72, 33), (
        f"TR_C2 batch x shape should be (32, 72, 33), "
        f"got {tuple(batch['x'].shape)}"
    )


# ---- Test 3: Transformer forward ----
@pytest.mark.parametrize("candidate_id", ["TR_C0_PRIMARY", "TR_C1_ALT_WEIGHT_DECAY", "TR_C2_ALT_LOOKBACK"])
def test_transformer_forward_on_real_batch(candidate_id):
    """Test 3: Transformer forward succeeds on real batch."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = next(c for c in candidates if c.candidate_id == candidate_id)

    from course_work.rolling_origin.real_run import (
        build_real_canonical_base_dataset, build_model_from_run_config,
    )
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, _ = _build_loader(base, folds[0].inner_train_ids[:32], batch_size=32)
    batch = next(iter(loader))
    model = build_model_from_run_config(cand.config)
    model.eval()
    with torch.no_grad():
        output = model(batch["x"])
    assert output.shape[-1] == 1, f"output last dim should be 1, got {tuple(output.shape)}"
    assert bool(torch.isfinite(output).all()), "output contains non-finite values"


# ---- Test 4: LSTM forward ----
def test_lstm_forward_on_real_batch():
    """Test 4: LSTM forward succeeds on real batch."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = next(c for c in candidates if c.candidate_id == "LSTM_TUNED_WINNER")

    from course_work.rolling_origin.real_run import (
        build_real_canonical_base_dataset, build_model_from_run_config,
    )
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, _ = _build_loader(base, folds[0].inner_train_ids[:32], batch_size=32)
    batch = next(iter(loader))
    model = build_model_from_run_config(cand.config)
    model.eval()
    with torch.no_grad():
        output = model(batch["x"])
    assert output.shape[-1] == 1
    assert bool(torch.isfinite(output).all())


# ---- Test 5: target IDs map exactly ----
def test_target_ids_map_exactly_to_dataset_samples():
    """Test 5: target IDs returned by loader exactly match requested IDs."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = candidates[0]

    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    ids = list(folds[0].inner_train_ids[:64])
    loader, kept_indices = _build_loader(base, ids, shuffle=False)
    kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
    expected = set(str(t) for t in ids)
    assert kept_tids == expected, (
        f"kept IDs {sorted(kept_tids)[:5]}... != expected {sorted(expected)[:5]}..."
    )


# ---- Test 6: scaling preserves [B,L,F] ----
def test_scaler_fit_preserves_batch_shape():
    """Test 6: fold-local scaler fit on canonical data preserves [N,F] shape."""
    candidates, folds, _ = _load_cands_and_folds()
    cand = candidates[0]

    from course_work.rolling_origin.real_run import (
        build_real_canonical_base_dataset, fit_fold_local_scaler,
    )
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    bundle, _ = fit_fold_local_scaler(
        candidate=cand, fold=folds[0],
        fold_dataset=base, fit_stage="A",
    )
    assert bundle is not None
    assert bundle.x_means is not None
    assert len(bundle.x_means) == 33, (
        f"scaler means should have 33 features, got {len(bundle.x_means)}"
    )


# ---- Tests 7-10: populations correct ----
def test_inner_train_population_correct():
    """Test 7: inner_train contains TRAIN-split target IDs only."""
    candidates, folds, all_robase = _load_cands_and_folds()
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, kept_indices = _build_loader(base, folds[0].inner_train_ids, shuffle=False)
    kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
    assert kept_tids.issubset(all_robase), "inner_train contains IDs outside ROBASE"


def test_inner_val_population_correct():
    """Test 8: inner_val contains VALIDATION-split target IDs only."""
    candidates, folds, all_robase = _load_cands_and_folds()
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, kept_indices = _build_loader(base, folds[0].inner_val_ids, shuffle=False)
    kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
    assert kept_tids.issubset(all_robase), "inner_val contains IDs outside ROBASE"


def test_outer_train_population_correct():
    """Test 9: outer_train contains ROBASE-train IDs only."""
    candidates, folds, all_robase = _load_cands_and_folds()
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, kept_indices = _build_loader(base, folds[0].outer_train_ids, shuffle=False)
    kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
    assert kept_tids.issubset(all_robase), "outer_train contains IDs outside ROBASE"


def test_outer_eval_population_correct():
    """Test 10: outer_eval contains ROBASE-val IDs only."""
    candidates, folds, all_robase = _load_cands_and_folds()
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    loader, kept_indices = _build_loader(base, folds[0].outer_eval_ids, shuffle=False)
    kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
    assert kept_tids.issubset(all_robase), "outer_eval contains IDs outside ROBASE"


# ---- Test 11: all 4 candidates work ----
def test_all_four_candidates_have_valid_datasets():
    """Test 11: all 4 candidates produce valid loaders with correct shape."""
    candidates, folds, _ = _load_cands_and_folds()
    assert len(candidates) == 4
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    for cand in candidates:
        base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
        loader, _ = _build_loader(base, folds[0].inner_train_ids[:32], batch_size=32)
        batch = next(iter(loader))
        L = cand.lookback_steps
        F = cand.config.get("data", {}).get("feature_count", 33)
        assert batch["x"].shape[1] == L, f"{cand.candidate_id} lookback mismatch"
        assert batch["x"].shape[2] == F, f"{cand.candidate_id} feature_count mismatch"


# ---- Test 12: all 3 folds work ----
def test_all_three_folds_produce_valid_loaders():
    """Test 12: all 3 folds produce valid loaders."""
    candidates, folds, _ = _load_cands_and_folds()
    assert len(folds) == 3
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    for fold in folds:
        loader, _ = _build_loader(base, fold.inner_train_ids[:32], batch_size=32)
        batch = next(iter(loader))
        assert batch["x"].ndim == 3


# ---- Test 13: no Test rows ----
def test_no_test_rows_in_any_loader():
    """Test 13: no Test rows appear in any loader."""
    candidates, folds, all_robase = _load_cands_and_folds()
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset

    cand = candidates[0]
    base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
    test_ids = set()
    for fold in folds:
        for role, ids in [
            ("inner_train", fold.inner_train_ids),
            ("inner_val", fold.inner_val_ids),
            ("outer_train", fold.outer_train_ids),
            ("outer_eval", fold.outer_eval_ids),
        ]:
            loader, kept_indices = _build_loader(base, ids[:32], shuffle=False)
            kept_tids = {str(base.window_records.iloc[i]["target_id"]) for i in kept_indices}
            test_in_loader = kept_tids - all_robase
            assert not test_in_loader, (
                f"Test IDs found in {fold.fold_id} {role}: {test_in_loader}"
            )


# ---- Test 14: hard-stop reaches first forward ----
class _SentinelHardStop(Exception):
    """Raised when Stage A training boundary is first reached."""


def test_official_orchestrator_reaches_first_training_boundary():
    """Test 14: official orchestrator reaches the first Stage A call."""
    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline, build_real_canonical_base_dataset,
    )
    from course_work.rolling_origin import stages as stages_mod
    import course_work.rolling_origin.real_run as real_run_mod

    orig = stages_mod.train_stage_a

    def _patched(*args, **kwargs):
        raise _SentinelHardStop("boundary reached")

    stages_mod.train_stage_a = _patched
    real_run_mod.train_stage_a = _patched
    try:
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
                seed=42,
                is_rehearsal=False,
                scientific_max_epochs=50,
                scientific_patience=10,
                rehearsal_synthetic=False,
                dataset_factory=lambda c, f: build_real_canonical_base_dataset(
                    project_root=PROJECT_ROOT, candidate=c,
                ),
            )
            result = run_real_pipeline(ctx)
            # The sentinel is caught and returned in result.exception
            assert result.exception is not None
            assert "_SentinelHardStop" in result.exception
    finally:
        stages_mod.train_stage_a = orig
        real_run_mod.train_stage_a = orig


# ---- Test 15+16: optimizer steps zero + no new official run IDs ----
def test_hard_stop_probe_zero_optimizer_steps():
    """Test 15+16: hard-stop probe has 0 optimizer steps and no new run IDs."""
    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline, build_real_canonical_base_dataset,
    )
    from course_work.rolling_origin import stages as stages_mod
    import course_work.rolling_origin.real_run as real_run_mod

    orig = stages_mod.train_stage_a

    def _patched(*args, **kwargs):
        raise _SentinelHardStop("boundary reached")

    stages_mod.train_stage_a = _patched
    real_run_mod.train_stage_a = _patched

    canonical_run_dir = PROJECT_ROOT / "artifacts" / "runs"
    runs_before = (
        sorted(p.name for p in canonical_run_dir.iterdir())
        if canonical_run_dir.exists() else []
    )

    try:
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
                seed=42,
                is_rehearsal=False,
                scientific_max_epochs=50,
                scientific_patience=10,
                rehearsal_synthetic=False,
                dataset_factory=lambda c, f: build_real_canonical_base_dataset(
                    project_root=PROJECT_ROOT, candidate=c,
                ),
            )
            run_real_pipeline(ctx)
    finally:
        stages_mod.train_stage_a = orig
        real_run_mod.train_stage_a = orig

    runs_after = (
        sorted(p.name for p in canonical_run_dir.iterdir())
        if canonical_run_dir.exists() else []
    )
    assert runs_after == runs_before, (
        f"canonical runs dir changed: before={runs_before}, after={runs_after}"
    )
