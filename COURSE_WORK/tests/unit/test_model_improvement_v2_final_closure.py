from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from course_work.reporting.model_improvement_v2_dashboard import render_model_improvement_v2_closure_dashboard


ROOT = Path(__file__).resolve().parents[2]
CLOSURE = ROOT / "artifacts/model_improvement_v2/model_improvement_v2_final_closure.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_closure_policy_and_governance_are_locked():
    document = json.loads(CLOSURE.read_text())
    assert document["model_improvement_v2_status"] == "COMPLETE"
    assert document["final_policy"]["seeds"] == [42, 123, 2026]
    assert document["final_policy"]["weights"] == [1 / 3, 1 / 3, 1 / 3]
    assert document["post_hoc_benchmark"]["label"] == "POST_HOC_V2_BENCHMARK"
    assert "NOT_UNBIASED_UNSEEN_TEST" in document["post_hoc_benchmark"]["interpretation"]
    assert document["governance"]["post_test_retuning"] is False
    assert document["governance"]["best_seed_selected"] is False


def test_closure_provenance_hashes_match_immutable_sources():
    document = json.loads(CLOSURE.read_text())
    provenance = document["provenance"]
    assert _sha(ROOT / provenance["final_lock"]["path"]) == provenance["final_lock"]["sha256"]
    assert _sha(ROOT / provenance["step17_manifest"]["path"]) == provenance["step17_manifest"]["sha256"]
    assert _sha(ROOT / provenance["step17_config_snapshot"]["path"]) == provenance["step17_config_snapshot"]["sha256"]
    assert _sha(ROOT / provenance["step17_source_plan"]["path"]) == provenance["step17_source_plan"]["sha256"]
    for seed, expected in provenance["checkpoint_sha256"].items():
        lock = json.loads((ROOT / provenance["final_lock"]["path"]).read_text())
        record = next(row for row in lock["final_runs"] if str(row["seed"]) == seed)
        assert _sha(ROOT / record["checkpoint_path"]) == expected


def test_closure_metrics_match_existing_predictions_without_inference():
    document = json.loads(CLOSURE.read_text())
    base = ROOT / "artifacts/model_improvement_v2/post_hoc_v2_benchmark/predictions"
    frame = pd.read_csv(base / "ensemble.csv")
    error = frame["y_true_wh"] - frame["y_pred_wh"]
    expected = document["post_hoc_benchmark"]["ensemble_metrics"]
    assert len(frame) == document["post_hoc_benchmark"]["sample_count"] == 2961
    assert float((error.pow(2).mean()) ** 0.5) == pytest.approx(expected["rmse_wh"], abs=1e-12)
    assert float(error.abs().mean()) == pytest.approx(expected["mae_wh"], abs=1e-12)


def test_dashboard_is_read_only_closure_view():
    rendered = render_model_improvement_v2_closure_dashboard(ROOT)
    text = rendered.data
    assert "Model Improvement V2 — Final Closure" in text
    assert "POST_HOC_V2_BENCHMARK" in text
    assert "not an unbiased unseen-Test result" in text
