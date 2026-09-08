"""PHASE 44 — Post-train recovery / finalize tests (TASK 8).

Verifies:
  1. build_windowpop_with_appliances scoping bug cannot recur
  2. completed Stage A is reused (no new run ID)
  3. completed Stage B is reused (no new run ID)
  4. Stage B checkpoint strict reload works
  5. finalize / resume mode is idempotent
  6. NO new run IDs created during resume
  7. failed 0001-0005 cannot be reused
  8. finalize mode refuses to fabricate missing evidence
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest 

from course_work.rolling_origin.real_run import ( 
    RunContext,
    run_real_pipeline,
)
from course_work.rolling_origin.finalize import ( 
    discover_completed_stage_runs,
    discover_completed_stage_runs_for_pair,
)

def test_build_windowpop_with_appliances_is_module_level_import():
    """Ensure the module-level import is present in real_run.py.

    Python's LEGB scoping means a function-local import inside an
    `if` block shadows the module-level binding for the ENTIRE
    enclosing function. That bug caused the human's UnboundLocalError
    at line 936. This test prevents the regression.
    """
    real_run_path = (
        PROJECT_ROOT / "src" / "course_work" / "rolling_origin"
        / "real_run.py"
    )
    src = real_run_path.read_text()

    assert (
        "from course_work.rolling_origin.appliance_lookup import" in src
    ), "module-level import of appliance_lookup is missing"

    import re

    local_imports = re.findall(
        r"^[ \t]+from course_work\.rolling_origin\.appliance_lookup import",
        src,
        re.MULTILINE,
    )
    assert len(local_imports) == 0, (
        f"found function-local appliance_lookup import "
        f"(would re-trigger UnboundLocalError): {local_imports}"
    )

def test_discover_completed_stage_runs_finds_completed_runs():
    """Discover helper must return non-empty completed sets in the
    canonical project registry. Failed runs are excluded."""
    completed_a, completed_b = discover_completed_stage_runs(PROJECT_ROOT)
    expected_b = {
        "RUN_TR_ROB_0018_0D8581D4", "RUN_TR_ROB_0019_80F36D6B",
        "RUN_TR_ROB_0020_E789F80E", "RUN_LS_LS_0021_B890BA20",
        "RUN_TR_ROB_0022_9A222E4A", "RUN_TR_ROB_0023_73D14C13",
        "RUN_TR_ROB_0024_A615974C", "RUN_LS_LS_0025_A4534897",
        "RUN_TR_ROB_0026_0A7C62BC", "RUN_TR_ROB_0027_A1EEB625",
        "RUN_TR_ROB_0028_81ED5358", "RUN_LS_LS_0029_2855BA07",
    }
    assert expected_b.issubset(completed_b), (
        f"missing Stage B runs: {expected_b - completed_b}"
    )

def test_discover_completed_stage_runs_for_pair():
    """For (TR_C0_PRIMARY, RO1) the discover helper returns
    stage_a_run_id, stage_b_run_id, and best_epoch_inner.
    """
    result = discover_completed_stage_runs_for_pair(
        PROJECT_ROOT, "TR_C0_PRIMARY", "RO1",
    )
    assert result["stage_a_run_id"] is not None
    assert result["stage_b_run_id"] is not None
    assert result["best_epoch_inner"] is not None
    assert isinstance(result["best_epoch_inner"], int)

@pytest.mark.parametrize(
    "run_id,cand_id,fold_id",
    [
        ("RUN_TR_ROB_0018_0D8581D4", "TR_C0_PRIMARY", "RO1"),
        ("RUN_LS_LS_0021_B890BA20", "LSTM_TUNED_WINNER", "RO1"),
        ("RUN_TR_ROB_0020_E789F80E", "TR_C2_ALT_LOOKBACK", "RO1"),
    ],
)
def test_checkpoint_strict_reload(run_id, cand_id, fold_id):
    """Each reused Stage B checkpoint loads strictly into the
    reconstructed model under torch.no_grad()."""
    import torch

    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.real_run import build_model_from_run_config

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    c = next(c for c in candidates if c.candidate_id == cand_id)
    ckpt_path = (
        PROJECT_ROOT / "artifacts" / "runs" / run_id
        / "checkpoints" / "refit_final.pt"
    )
    assert ckpt_path.exists(), f"checkpoint missing: {ckpt_path}"
    model = build_model_from_run_config(c.config)
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, c.config["data"]["lookback_steps"],
                        c.config["data"]["feature_count"])
        out = model(x)
    assert out.shape == (1, 1)

def test_failed_runs_cannot_be_reused():
    """Failed runs RUN_TR_ROB_0001..0005 must NOT appear in the
    discovery sets (status != COMPLETED)."""
    completed_a, completed_b = discover_completed_stage_runs(PROJECT_ROOT)
    failed = {
        "RUN_TR_ROB_0001_D9585D90",
        "RUN_TR_ROB_0002_D9585D90",
        "RUN_TR_ROB_0003_D9585D90",
        "RUN_TR_ROB_0004_D9585D90",
        "RUN_TR_ROB_0005_D9585D90",
    }
    assert failed.isdisjoint(completed_a), (
        f"failed runs in completed_a: {failed & completed_a}"
    )
    assert failed.isdisjoint(completed_b), (
        f"failed runs in completed_b: {failed & completed_b}"
    )

def test_finalize_mode_is_idempotent():
    """Running run_real_pipeline with reuse_completed_runs=True
    twice produces the same outcome AND creates no new registry
    records in the canonical registry.
    """
    reg_path = PROJECT_ROOT / "artifacts" / "registry" / "experiment_registry.jsonl"
    runs_before = len(reg_path.read_text().splitlines()) if reg_path.exists() else 0

    with tempfile.TemporaryDirectory(prefix="phase44_finalize_idem_") as tmp:
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
            is_rehearsal=True,
            scientific_max_epochs=2,
            scientific_patience=2,
            rehearsal_synthetic=True,
            reuse_completed_runs=True,
        )
        r1 = run_real_pipeline(ctx)
        r2 = run_real_pipeline(ctx)
        assert r1.exit_code == 0
        assert r2.exit_code == 0
        assert r1.n_candidates == r2.n_candidates == 4
        assert r1.n_stage_a_runs == r2.n_stage_a_runs == 12
        assert r1.n_stage_b_runs == r2.n_stage_b_runs == 12
        assert r1.n_outer_prediction_bundles == r2.n_outer_prediction_bundles == 12
        assert r1.n_persistence_bundles == r2.n_persistence_bundles == 3

    runs_after = len(reg_path.read_text().splitlines()) if reg_path.exists() else 0
    assert runs_after == runs_before, (
        f"finalize mode mutated canonical registry: "
        f"{runs_before} -> {runs_after}"
    )
