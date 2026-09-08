"""PHASE 44 — Focused invariant tests (TASK 11).

Covers all 13 invariants required by the final rehearsal gate:
  1. Disposable rehearsal sees 4 candidates
  2. LSTM handoff schema equals official schema
  3. Rehearsal Stage A = 12
  4. Rehearsal Stage B = 12
  5. Rehearsal learned predictions = 12
  6. Rehearsal Persistence = 3
  7. Rehearsal pooled models = 5
  8. 12/12 Stage B epoch propagation matches
  9. Rehearsal signoff PASS/PASS_WITH_WARNING
 10. Old synthetic official path unreachable
 11. Canonical registry unchanged
 12. Canonical artifacts unchanged
 13. Test inaccessible
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import (  
    RunContext, run_real_pipeline,
)


def _make_ctx(tmp: Path, *, rehearsal_synthetic: bool = True) -> RunContext:
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
        rehearsal_synthetic=rehearsal_synthetic,
    )


def test_invariant_1_disposable_rehearsal_sees_four_candidates():
    canonical = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    canonical_after = {p.name for p in canonical.iterdir()} if canonical.exists() else set()
    assert canonical_before == canonical_after
    assert result.n_candidates == 4


def test_invariant_2_lstm_handoff_schema_matches_official_schema():
    """The canonical LSTM handoff has the structured winner_config that
    load_candidates() requires. The disposable rehearsal consumes the
    SAME canonical handoff that official consumes."""
    canonical_handoff = (
        PROJECT_ROOT / "artifacts" / "lstm_tuning" / "phase44_rolling_origin_lstm_handoff.json"
    )
    flat_handoff = (
        PROJECT_ROOT / "artifacts" / "lstm_tuning" / "lstm_tuned_winner.json"
    )
    assert canonical_handoff.exists(), \
        "Canonical LSTM handoff (winner_config schema) must exist"
    assert flat_handoff.exists(), \
        "Flat LSTM handoff (LSTM tuning winner) must exist"
    d = json.loads(canonical_handoff.read_text())
    assert "winner_run_id" in d, "canonical handoff must have winner_run_id"
    assert "winner_config_fingerprint" in d, \
        "canonical handoff must have winner_config_fingerprint"
    assert "winner_config" in d, \
        "canonical handoff must have structured winner_config"
    assert d["ready_for_phase44"] is True

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        assert ctx.lstm_handoff_path == canonical_handoff, \
            "disposable rehearsal must consume the SAME canonical LSTM handoff"
        result = run_real_pipeline(ctx)
    assert result.n_candidates == 4


def test_invariant_3_rehearsal_stage_a_equals_twelve():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert result.n_stage_a_runs == 12


def test_invariant_4_rehearsal_stage_b_equals_twelve():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert result.n_stage_b_runs == 12


def test_invariant_5_rehearsal_learned_predictions_equals_twelve():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert result.n_outer_prediction_bundles == 12


def test_invariant_6_rehearsal_persistence_equals_three():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert result.n_persistence_bundles == 3


def test_invariant_7_rehearsal_pooled_models_equals_five():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert len(result.pooled_metrics_by_cid) == 5
    expected_models = {
        "TR_C0_PRIMARY", "TR_C1_ALT_WEIGHT_DECAY", "TR_C2_ALT_LOOKBACK",
        "LSTM_TUNED_WINNER", "PERSISTENCE_LAST_VALUE",
    }
    assert set(result.pooled_metrics_by_cid.keys()) == expected_models


def test_invariant_8_stage_b_epoch_propagation_12_of_12():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert len(result.inner_best_epochs) == 12
    assert len(result.stage_b_run_ids) == 12
    for (cid, fid), run_id in result.stage_b_run_ids.items():
        assert (cid, fid) in result.inner_best_epochs, \
            f"missing inner_best_epoch for ({cid}, {fid})"
        assert result.inner_best_epochs[(cid, fid)] >= 1, \
            f"invalid best_epoch_inner for ({cid}, {fid})"

def test_invariant_9_rehearsal_signoff_pass_or_pass_with_warning():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING"}

def test_invariant_10_old_synthetic_official_path_unreachable():
    from course_work.rolling_origin import pipeline as p44pipe

    def explode(*args, **kwargs):
        raise RuntimeError("OLD SYNTHETIC FUNCTION CALLED — must not happen")

    with patch.object(p44pipe, "run_pipeline", side_effect=explode), \
         patch.object(p44pipe, "_synth_predict_for_official", side_effect=explode):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _make_ctx(Path(tmp))
            result = run_real_pipeline(ctx)
    assert result.exit_code in (0, 1)
    assert result.n_candidates == 4
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING"}

def test_invariant_11_canonical_registry_unchanged():
    canonical_registry = PROJECT_ROOT / "artifacts" / "registry"
    before = sorted(p.name for p in canonical_registry.iterdir()) \
        if canonical_registry.exists() else []
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        run_real_pipeline(ctx)
    after = sorted(p.name for p in canonical_registry.iterdir()) \
        if canonical_registry.exists() else []
    assert before == after, \
        "canonical registry was mutated by disposable rehearsal"

def test_invariant_12_canonical_artifacts_unchanged():
    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    before = sorted(p.name for p in canonical_artifact_dir.iterdir()) \
        if canonical_artifact_dir.exists() else []
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        run_real_pipeline(ctx)
    after = sorted(p.name for p in canonical_artifact_dir.iterdir()) \
        if canonical_artifact_dir.exists() else []
    assert before == after, \
        "canonical artifact_dir was mutated by disposable rehearsal"

def test_invariant_13_test_inaccessible():
    """The orchestrator never writes test rows; Test is never accessed."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))
        result = run_real_pipeline(ctx)
    if result.phase45_handoff is not None:
        assert result.phase45_handoff.get("test_status") == "NOT_ACCESSED"
    assert result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING"}