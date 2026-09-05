"""Focused regression tests for Phase 43 canonical-state recovery.

These tests never train, never register runs, and never read Test target
values.  Phase 44 preflight is exercised only through its audit-only path.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_FILES = (
    "phase_43_signoff.json",
    "lstm_tuned_winner.json",
    "phase44_rolling_origin_lstm_handoff.json",
    "lstm_stage_lineage.csv",
    "phase43_active_reuse_set.json",
)


def _active_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts/lstm_tuning"
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ACTIVE_FILES
    }


@pytest.fixture(scope="module")
def phase44_preflight_result():
    from course_work.rolling_origin.preflight import run_preflight
    from course_work.training.engine import TrainingEngine

    train_calls: list[tuple] = []
    original_train = TrainingEngine.train

    def _forbid_train(*args, **kwargs):
        train_calls.append((args, kwargs))
        raise AssertionError("Phase 44 preflight attempted training")

    TrainingEngine.train = _forbid_train
    try:
        result = run_preflight(
            project_root=PROJECT_ROOT,
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            lstm_winner_path=PROJECT_ROOT / "artifacts/lstm_tuning/lstm_tuned_winner.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            include_official_config_audit=True,
        )
    finally:
        TrainingEngine.train = original_train
    return result, train_calls


def _gate(result, prefix: str):
    return next(gate for gate in result.gates if gate.gate_id.startswith(prefix))


def test_finalized_pass_cannot_be_downgraded_by_prepare():
    from scripts.phase43_lstm_tuning import Phase43LifecycleError, prepare_phase43

    before = _active_hashes()
    with pytest.raises(Phase43LifecycleError):
        prepare_phase43(PROJECT_ROOT)
    assert _active_hashes() == before


def test_prepared_snapshot_cannot_replace_pass(tmp_path: Path):
    from scripts.phase43_lstm_tuning import (
        Phase43LifecycleError,
        _snapshot_existing_prep_artifacts,
    )

    artifact_dir = tmp_path / "artifacts/lstm_tuning"
    artifact_dir.mkdir(parents=True)
    signoff = artifact_dir / "phase_43_signoff.json"
    signoff.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    winner = artifact_dir / "lstm_tuned_winner.json"
    winner.write_text(json.dumps({"winner_run_id": "RUN_KEEP"}), encoding="utf-8")
    before = winner.read_bytes()
    with pytest.raises(Phase43LifecycleError):
        _snapshot_existing_prep_artifacts(tmp_path)
    assert winner.read_bytes() == before


def test_dry_run_uses_staging_and_preserves_active_artifacts():
    from scripts.phase43_dry_run import run

    before = _active_hashes()
    result = run(PROJECT_ROOT)
    assert result["status"] == "PASS"
    assert result["artifact_scope"] == "TEMPORARY_STAGING"
    assert result["active_artifacts_modified"] is False
    assert result["staging_cleaned"] is True
    assert _active_hashes() == before


def test_rehearsal_prepare_is_staging_only():
    from scripts.phase43_lstm_tuning import _rehearse_production_path

    tree = ast.parse(inspect.getsource(_rehearse_production_path))
    prepare_args = [
        call.args[0].id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "prepare_phase43"
        and call.args
        and isinstance(call.args[0], ast.Name)
    ]
    assert prepare_args == ["staging_root"]


def test_handoff_winner_config_is_non_null():
    handoff = json.loads(
        (PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json").read_text()
    )
    assert handoff["winner_config"]
    assert handoff["winner_run_id"] == "RUN_LS_LST_0175_BCE5A2CD"


def test_handoff_fingerprint_equals_canonical_winner_fingerprint():
    from course_work.experiments.registry import compute_config_fingerprint

    root = PROJECT_ROOT / "artifacts/lstm_tuning"
    winner = json.loads((root / "lstm_tuned_winner.json").read_text())
    handoff = json.loads((root / "phase44_rolling_origin_lstm_handoff.json").read_text())
    run_config = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/runs"
            / winner["winner_run_id"]
            / "config.json"
        ).read_text()
    )
    recomputed = compute_config_fingerprint(handoff["winner_config"])
    assert winner["config_fingerprint"] == handoff["winner_config_fingerprint"]
    assert winner["config_fingerprint"] == run_config["config_fingerprint"]
    assert winner["config_fingerprint"] == recomputed


def test_reconstructed_winner_matches_active_winner():
    from scripts.phase43_lstm_tuning import _reconstruct_phase43_from_run_evidence

    runs, stages = _reconstruct_phase43_from_run_evidence(PROJECT_ROOT)
    active = json.loads(
        (PROJECT_ROOT / "artifacts/lstm_tuning/lstm_tuned_winner.json").read_text()
    )
    assert len(runs) == 10
    assert stages["LT5"].winner.run_id == active["winner_run_id"]
    assert stages["LT5"].winner.validation_rmse_wh == active["validation_rmse_wh"]


def test_phase44_sees_four_learned_candidates(phase44_preflight_result):
    result, _ = phase44_preflight_result
    gate = _gate(result, "G18_")
    assert gate.passed
    assert gate.details["n_candidates"] == 4


def test_phase44_builds_twelve_stage_a_payloads(phase44_preflight_result):
    result, _ = phase44_preflight_result
    gate = _gate(result, "G20_")
    assert gate.passed
    assert gate.details["stage_a_payloads"] == 12


def test_phase44_builds_twelve_stage_b_payloads(phase44_preflight_result):
    result, _ = phase44_preflight_result
    gate = _gate(result, "G20_")
    assert gate.passed
    assert gate.details["stage_b_payloads"] == 12


def test_phase44_preflight_has_zero_test_rows(phase44_preflight_result):
    result, _ = phase44_preflight_result
    for prefix in ("G06_", "G15_", "G19_", "G24_"):
        assert _gate(result, prefix).passed


def test_phase44_preflight_executes_zero_training_steps(phase44_preflight_result):
    result, train_calls = phase44_preflight_result
    assert result.all_passed
    assert train_calls == []
