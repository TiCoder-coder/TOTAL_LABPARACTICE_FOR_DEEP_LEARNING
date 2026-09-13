from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from course_work.model_improvement_v2 import step17_mape_addendum as addendum
from course_work.model_improvement_v2 import step17_post_hoc_benchmark as step17


ROOT = Path(__file__).resolve().parents[2]


def test_metric_computation_includes_standard_mape():
    metrics = step17._metrics(np.asarray([10.0, 20.0]), np.asarray([12.0, 18.0]))
    assert metrics["mape_pct"] == pytest.approx(15.0)
    assert metrics["mape_status"] == "DEFINED"


def test_addendum_preflight_verifies_sources_without_reading_prediction_rows(monkeypatch):
    monkeypatch.setattr(addendum.pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Test-derived CSV opened")))
    result = addendum.run_preflight(ROOT)
    assert result["status"] == "PASS"
    assert result["raw_test_source_opened"] is False
    assert result["checkpoint_payloads_loaded"] == 0
    assert result["training_executed"] is False
    assert result["inference_executed"] is False
    assert result["test_derived_prediction_rows_read"] == 0
    assert len(result["prediction_sources_verified"]) == 5


def test_evaluation_refuses_without_both_human_flags(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(addendum, "run_evaluation", forbidden)
    base = ["--experiment", "STEP17_MAPE_ADDENDUM", "--mode", "evaluate"]
    assert addendum.main(base) == 3
    assert addendum.main(base + ["--authorize-test-derived-evidence-access"]) == 3
    assert addendum.main(base + ["--acknowledge-post-hoc-not-unseen-test"]) == 3
    assert called is False


def test_preflight_rejects_authorization_flags(monkeypatch):
    monkeypatch.setattr(addendum, "run_preflight", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("preflight called")))
    assert addendum.main([
        "--experiment", "STEP17_MAPE_ADDENDUM",
        "--mode", "preflight",
        "--authorize-test-derived-evidence-access",
    ]) == 2
