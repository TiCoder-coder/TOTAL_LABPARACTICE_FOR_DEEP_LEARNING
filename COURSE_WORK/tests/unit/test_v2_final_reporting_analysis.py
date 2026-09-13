import hashlib
import json
from pathlib import Path

from course_work.reporting.results_rebuild import render_verified_phase_result


ROOT = Path(__file__).resolve().parents[2]
REPORTING = ROOT / "artifacts/model_improvement_v2/final_reporting_analysis"


def _read(name: str) -> dict:
    return json.loads((REPORTING / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_reporting_manifest_binds_frozen_sources_and_outputs():
    manifest = _read("manifest.json")
    assert manifest["status"] == "PASS"
    assert manifest["lineage"] == {
        "analysis": "REPORTING_ANALYSIS_FROM_FROZEN_PREDICTIONS",
        "benchmark": "POST_HOC_V2_BENCHMARK",
        "best_seed_selection": False,
        "final_policy": "V2 Equal-weight Ensemble",
        "inference_executed": False,
        "project": "MODEL_IMPROVEMENT_V2",
        "raw_test_source_opened": False,
        "training_executed": False,
    }
    assert manifest["source_population_fingerprint"] == (
        "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
    )
    for relative, expected in manifest["source_prediction_sha256"].items():
        assert _sha(ROOT / relative) == expected
    for filename, expected in manifest["output_sha256"].items():
        assert _sha(REPORTING / filename) == expected


def test_phase48_51_artifacts_are_final_policy_reporting_only():
    for filename in (
        "phase48_prediction_analysis.json",
        "phase49_residual_analysis.json",
        "phase50_error_by_regime.json",
        "phase51_worst_error_analysis.json",
    ):
        document = _read(filename)
        assert document["status"] == "PASS"
        assert document["sample_count"] == 2961
        assert document["lineage"]["final_policy"] == "V2 Equal-weight Ensemble"
        assert document["lineage"]["training_executed"] is False
        assert document["lineage"]["inference_executed"] is False
        assert document["lineage"]["raw_test_source_opened"] is False

    phase50 = _read("phase50_error_by_regime.json")
    assert phase50["population_equality"] == "PASS"
    assert phase50["historical_assignment_csv_reused"] is False
    assert phase50["assignment_mode"] == (
        "REBUILT_FROM_FROZEN_PREDICTION_COLUMNS_USING_VERIFIED_TRAIN_THRESHOLDS"
    )
    assert len(phase50["rows"]) == 16

    phase51 = _read("phase51_worst_error_analysis.json")
    assert phase51["historical_v1_context_reused"] is False
    assert len(phase51["rows"]) == 20
    errors = [row["absolute_error_wh"] for row in phase51["rows"]]
    assert errors == sorted(errors, reverse=True)


def test_phase47_51_render_final_v2_without_v1_mixing():
    for phase in range(47, 52):
        html = render_verified_phase_result(phase, ROOT).data
        assert "V2" in html
        assert "Final V1" not in html
        assert "historical Transformer" not in html
    assert "V2 Equal-weight Ensemble" in render_verified_phase_result(48, ROOT).data
    assert "Train-only regime thresholds" in render_verified_phase_result(50, ROOT).data
    assert "no V1 context reuse" in render_verified_phase_result(51, ROOT).data


def test_phase52_57_are_explicitly_historical_v1_attention():
    marker = (
        "Historical V1 attention analysis — not recomputed for final V2 because "
        "V2 Test attention tensors were not generated."
    )
    for phase in range(52, 58):
        html = render_verified_phase_result(phase, ROOT).data
        assert marker in html


def test_phase58_59_use_v2_final_summary_and_make_no_attention_claim():
    phase58 = render_verified_phase_result(58, ROOT).data
    assert "V2 Equal-weight Ensemble" in phase58
    assert "FINAL_LOCKED_POLICY" in phase58
    assert "Final V1 tables" not in phase58
    phase59 = render_verified_phase_result(59, ROOT).data
    assert "MEAN_E14_M1_SEEDS_42_123_2026" in phase59
    assert "NO_V2_TEST_ATTENTION_TENSORS" in phase59
    assert "Post-Test retuning" in phase59
    assert ">NO<" in phase59
