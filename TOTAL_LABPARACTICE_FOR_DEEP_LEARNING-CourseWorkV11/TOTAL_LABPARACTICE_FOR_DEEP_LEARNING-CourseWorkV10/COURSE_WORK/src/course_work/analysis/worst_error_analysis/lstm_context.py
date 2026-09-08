"""Phase 51-E — LSTM eligibility context.

Reads canonical LSTM eligibility JSON and emits a Phase 51-E context row.
No inference. No checkpoint loading. No retraining.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root


LSTM_ELIGIBILITY_REL = "artifacts/final_test/final_test_lstm_eligibility.json"


def build_lstm_eligibility_context(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    fp = root / LSTM_ELIGIBILITY_REL
    payload = json.loads(fp.read_text())

    return {
        "model_id": payload.get("model_id"),
        "source_phase": payload.get("source_phase"),
        "source_run_id": payload.get("source_run_id"),
        "checkpoint_exists": payload.get("checkpoint_exists"),
        "checkpoint_frozen_pretest": payload.get("checkpoint_frozen_pretest"),
        "test_previously_accessed": payload.get("test_previously_accessed"),
        "config_valid": payload.get("config_valid"),
        "scaler_artifacts_available": payload.get("scaler_artifacts_available"),
        "final_test_pop_supported": payload.get("FINAL_TEST_POP_supported"),
        "common_target_comparison_possible": payload.get("common_target_comparison_possible"),
        "training_protocol_symmetric_with_transformer": payload.get(
            "training_protocol_symmetric_with_transformer"
        ),
        "eligible": payload.get("eligible"),
        "eligibility_status": payload.get("eligibility_status"),
        "reason": payload.get("reason"),
        "fairness_caveat": payload.get("fairness_caveat"),
        "status": payload.get("status"),
        "phase51_e_action": "NOT_APPLICABLE — no inference, no checkpoint loading",
    }
