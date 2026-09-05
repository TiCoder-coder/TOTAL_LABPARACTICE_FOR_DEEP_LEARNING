"""TASK 3 — Diagnostic: dump all signoff checks with status + actual/expected.

Runs the real orchestrator in disposable rehearsal mode and prints every
consistency check the signoff builder evaluates.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from course_work.rolling_origin.real_run import (
    RunContext, run_real_pipeline, SCIENTIFIC_MAX_EPOCHS,
)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — SIGNOFF CHECK DIAGNOSTIC (TASK 3)")
    print("=" * 78)

    # Build a staging copy of the canonical LSTM handoff.
    # The canonical handoff (artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json)
    # has the structured winner_config that load_candidates() requires.
    canonical_lstm = PROJECT_ROOT / "artifacts" / "lstm_tuning" / "phase44_rolling_origin_lstm_handoff.json"
    flat_handoff_path = PROJECT_ROOT / "artifacts" / "lstm_tuning" / "lstm_tuned_winner.json"
    assert canonical_lstm.exists(), (
        f"Canonical LSTM handoff missing: {canonical_lstm}"
    )
    flat = json.loads(flat_handoff_path.read_text())
    canonical_lstm_data = json.loads(canonical_lstm.read_text())
    # NOTE: This is a DERIVED staging copy of the canonical handoff for
    # disposable rehearsal ONLY. The canonical artifact is NOT modified.
    # We copy the canonical structured handoff (which has the correct
    # winner_config + recorded fingerprint match) to the staging root.

    with tempfile.TemporaryDirectory(prefix="phase44_diag_") as tmp:
        staging_root = Path(tmp)
        staging_lstm_handoff = staging_root / "phase44_rolling_origin_lstm_handoff.json"
        staging_lstm_handoff.write_text(json.dumps(canonical_lstm_data, indent=2))

        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=staging_lstm_handoff,
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=staging_root / "artifacts",
            registry_root=staging_root / "registry",
            run_root=staging_root / "runs",
            scientific_max_epochs=2,
            scientific_patience=2,
            seed=42,
            is_rehearsal=True,
            rehearsal_synthetic=True,
        )
        print(f"\nStaging LSTM handoff at: {staging_lstm_handoff}")

        result = run_real_pipeline(ctx)
        print(f"\nexit_code: {result.exit_code}")
        print(f"summary: {result.summary}")
        print(f"signoff_overall_status: {result.signoff_overall_status}")
        print(f"\nsignoff_failures ({len(result.signoff_failures)}):")
        for f in result.signoff_failures:
            print(f"  - {f}")
        print(f"\npooled_metrics_by_cid ({len(result.pooled_metrics_by_cid)}):")
        for cid, m in result.pooled_metrics_by_cid.items():
            rmse = m.get('rmse_wh')
            mae = m.get('mae_wh')
            r2 = m.get('r2')
            rmse_s = f"{rmse:.4f}" if rmse is not None else "NA"
            mae_s = f"{mae:.4f}" if mae is not None else "NA"
            r2_s = f"{r2:.4f}" if r2 is not None else "NA"
            print(f"  {cid}: rmse={rmse_s} mae={mae_s} r2={r2_s}")

        if result.exception:
            print(f"\nexception:\n{result.exception[-2000:]}")

    return 0 if result.exit_code == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)
