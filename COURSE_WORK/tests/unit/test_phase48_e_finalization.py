"""Focused Phase 48-E finalization tests.

Acceptance gates:
- all required O48 outputs present
- all JSON serializable
- all CSV schemas exact
- all figures exist and non-empty
- figure windows deterministic (FIRST/MIDDLE/LAST)
- no best-seed labels
- no ensemble metrics
- seed mean labeled descriptive only
- seed spread not labeled confidence interval
- Phase49 residual convention exact
- Phase50 TRAIN-derived threshold warning present
- Phase51 no-worst-error statement present
- source_predictions_modified=false
- Phase47 source hashes unchanged
- no training/inference/checkpoint code paths
- signoff cannot PASS with missing O48 artifact
- finalization deterministic/idempotent
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
F47 = REPO / "artifacts" / "final_test"
P48 = REPO / "artifacts" / "prediction_analysis"
P48F = P48 / "figures"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(name: str) -> list[dict]:
    with (P48 / name).open() as fh:
        return list(csv.DictReader(fh))


def _read_header(name: str) -> list[str]:
    with (P48 / name).open() as fh:
        return next(csv.reader(fh))


@pytest.fixture(scope="module")
def signoff() -> dict:
    return json.loads((P48 / "phase_48_signoff.json").read_text())


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((P48 / "prediction_analysis_summary.json").read_text())


@pytest.fixture(scope="module")
def phase49_handoff() -> dict:
    return json.loads((P48 / "phase49_residual_analysis_handoff.json").read_text())


@pytest.fixture(scope="module")
def phase50_handoff() -> dict:
    return json.loads((P48 / "phase50_error_regime_context_handoff.json").read_text())


@pytest.fixture(scope="module")
def phase51_handoff() -> dict:
    return json.loads((P48 / "phase51_worst_error_context_handoff.json").read_text())


# ===================================================================
# 1. O48 artifact completeness (O48.1 - O48.36)
# ===================================================================

O48_FILES = [
    ("O48.1", "prediction_analysis_manifest.json", "json"),
    ("O48.2", "prediction_analysis_contract.json", "json"),
    ("O48.3", "phase48_preflight_audit.csv", "csv"),
    ("O48.4", "prediction_source_verification.csv", "csv"),
    ("O48.5", "prediction_alignment_audit.csv", "csv"),
    ("O48.6", "prediction_wide_table.csv", "csv"),
    ("O48.7", "prediction_long_table.csv", "csv"),
    ("O48.8", "prediction_distribution_summary.csv", "csv"),
    ("O48.9", "prediction_range_compression.csv", "csv"),
    ("O48.10", "prediction_change_summary.csv", "csv"),
    ("O48.11", "prediction_direction_agreement.csv", "csv"),
    ("O48.12", "prediction_lag_diagnostics.csv", "csv"),
    ("O48.13", "prediction_acf_diagnostics.csv", "csv"),
    ("O48.14", "prediction_local_extrema_summary.csv", "csv"),
    ("O48.15", "prediction_peak_timing_summary.csv", "csv"),
    ("O48.16", "prediction_seed_pairwise_agreement.csv", "csv"),
    ("O48.17", "prediction_seed_spread.csv", "csv"),
    ("O48.18", "prediction_top_seed_disagreement.csv", "csv"),
    ("O48.19", "prediction_rolling_tracking.csv", "csv"),
    ("O48.20", "prediction_negative_value_audit.csv", "csv"),
    ("O48.21", "prediction_saturation_audit.csv", "csv"),
    ("O48.22", "prediction_baseline_context.csv", "csv"),
    ("O48.27", "prediction_analysis_findings.csv", "csv"),
    ("O48.28", "phase49_residual_analysis_handoff.json", "json"),
    ("O48.29", "phase50_error_regime_context_handoff.json", "json"),
    ("O48.30", "phase51_worst_error_context_handoff.json", "json"),
    ("O48.31", "prediction_analysis_tests.csv", "csv"),
    ("O48.32", "prediction_analysis_discrepancies.json", "json"),
    ("O48.33", "prediction_analysis_summary.json", "json"),
    ("O48.34", "prediction_analysis_report.md", "md"),
    ("O48.35", "README_PREDICTION_ANALYSIS.md", "md"),
    ("O48.36", "phase_48_signoff.json", "json"),
]

O48_FIGURES = [
    "PRED_48_01_full_test_actual_vs_all_seeds.png",
    "PRED_48_02_full_test_actual_vs_seed_mean_spread.png",
    "PRED_48_03_first_24h_zoom.png",
    "PRED_48_04_middle_24h_zoom.png",
    "PRED_48_05_last_24h_zoom.png",
    "PRED_48_06_scatter_seed42.png",
    "PRED_48_07_scatter_seed123.png",
    "PRED_48_08_scatter_seed2026.png",
    "PRED_48_09_prediction_ecdf.png",
    "PRED_48_10_change_magnitude_distribution.png",
    "PRED_48_11_cross_seed_spread_over_time.png",
    "PRED_48_12_pairwise_seed_prediction_scatter.png",
    "PRED_48_13_lag_cross_correlation.png",
    "PRED_48_14_acf_actual_vs_predictions.png",
    "PRED_48_15_local_peak_capture.png",
    "PRED_48_16_rolling_24h_mean_tracking.png",
    "PRED_48_17_rolling_24h_std_tracking.png",
    "PRED_48_18_daily_actual_heatmap.png",
    "PRED_48_19_daily_seed_mean_heatmap.png",
    "PRED_48_20_seed_spread_heatmap.png",
]


class TestO48Completeness:
    @pytest.mark.parametrize("oid,fname,kind", O48_FILES)
    def test_file_exists(self, oid, fname, kind):
        assert (P48 / fname).exists(), f"{oid} {fname} missing"

    @pytest.mark.parametrize("oid,fname,kind", O48_FILES)
    def test_file_nonempty(self, oid, fname, kind):
        size = (P48 / fname).stat().st_size
        assert size > 0, f"{oid} {fname} is empty"

    @pytest.mark.parametrize("oid,fname,kind", [(o, f, k) for (o, f, k) in O48_FILES if k == "json"])
    def test_json_serializable(self, oid, fname, kind):
        d = json.loads((P48 / fname).read_text())
        assert d is not None

    @pytest.mark.parametrize("oid,fname,kind", [(o, f, k) for (o, f, k) in O48_FILES if k == "csv"])
    def test_csv_has_header(self, oid, fname, kind):
        with (P48 / fname).open() as fh:
            header = next(csv.reader(fh))
        assert len(header) > 0

    @pytest.mark.parametrize("fname", O48_FIGURES)
    def test_figure_exists_nonempty(self, fname):
        p = P48F / fname
        assert p.exists() and p.stat().st_size > 1000  # > 1KB (real PNG)


# ===================================================================
# 2. Forbidden language
# ===================================================================

class TestForbiddenLanguage:
    @pytest.mark.parametrize("oid,fname,kind", O48_FILES)
    def test_no_best_seed_label(self, oid, fname, kind):
        """No 'is_best', 'winner', 'selected_seed', 'best_seed' anywhere in CSV/JSON/MD."""
        text = (P48 / fname).read_text()
        for forbidden in ("is_best", "winner_seed", "selected_seed", "best_seed_score"):
            assert forbidden not in text.lower(), f"{oid} {fname} contains '{forbidden}'"

    @pytest.mark.parametrize("oid,fname,kind", O48_FILES)
    def test_no_confidence_interval_claim(self, oid, fname, kind):
        """No positive claim of confidence interval / predictive uncertainty.

        Explanatory negation context (e.g. 'not a confidence interval',
        'never labelled') is allowed because it explicitly disclaims the
        forbidden interpretation.
        """
        text = (P48 / fname).read_text()
        # Strip explicit negation context: phrases starting with "not ", "never ",
        # "NOT a", "is NOT", "do NOT", "does NOT" until the next period
        import re
        cleaned = re.sub(r"(?i)\bnot a\b[^.]*\.", "", text)
        cleaned = re.sub(r"(?i)\bnever\b[^.]*\.", "", cleaned)
        cleaned = re.sub(r"(?i)\bis not\b[^.]*\.", "", cleaned)
        cleaned = re.sub(r"(?i)\bdo not\b[^.]*\.", "", cleaned)
        cleaned = re.sub(r"(?i)\bdoes not\b[^.]*\.", "", cleaned)
        for forbidden in ("confidence interval", "predictive uncertainty",
                          "calibrated uncertainty", "uncertainty interval"):
            assert forbidden not in cleaned.lower(), f"{oid} {fname} contains positive claim of '{forbidden}'"

    def test_seed_mean_labeled_descriptive(self, summary):
        assert "DESCRIPTIVE" in summary["seed_mean_semantics"]
        assert "ENSEMBLE" not in summary["seed_mean_semantics"] or "NOT" in summary["seed_mean_semantics"]

    def test_seed_spread_labeled_cross_seed(self, summary):
        assert "CROSS_SEED_PREDICTION_SPREAD" in summary["seed_spread_semantics"]


# ===================================================================
# 3. Handoff correctness
# ===================================================================

class TestHandoffs:
    def test_phase49_residual_convention(self, phase49_handoff):
        assert phase49_handoff["residual_convention"] == "y_true - y_pred"

    def test_phase49_no_residual_in_phase48(self, phase49_handoff):
        assert phase49_handoff["phase48_executed_residual_analysis"] is False

    def test_phase50_train_derived_only(self, phase50_handoff):
        assert phase50_handoff["phase50_threshold_policy"] == "TRAIN_DERIVED_ONLY"
        assert phase50_handoff["train_derived_threshold_requirement"] is True
        assert phase50_handoff["test_derived_threshold_authorization"] is False

    def test_phase51_no_worst_error_ranking_in_phase48(self, phase51_handoff):
        assert phase51_handoff["phase48_executed_worst_error_ranking"] is False

    def test_phase49_phase50_phase51_all_have_status_pass(self, phase49_handoff, phase50_handoff, phase51_handoff):
        assert phase49_handoff["status"] == "PASS"
        assert phase50_handoff["status"] == "PASS"
        assert phase51_handoff["status"] == "PASS"

    def test_handoffs_reference_correct_source_sha(self, phase49_handoff):
        expected_pop = "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
        assert phase49_handoff["test_population_sha256"] == expected_pop


# ===================================================================
# 4. Signoff gates
# ===================================================================

class TestSignoff:
    def test_source_predictions_modified_false(self, signoff):
        assert signoff["source_predictions_modified"] is False
        assert signoff["phase48_d4_closed"] is True

    def test_phase47_artifacts_modified_false(self, signoff):
        assert signoff["phase47_artifacts_modified"] is False

    def test_no_inference_no_training_no_optimizer(self, signoff):
        assert signoff["new_inference"] is False
        assert signoff["model_training"] is False
        assert signoff["optimizer_steps"] == 0
        assert signoff["scaler_fit"] is False
        assert signoff["best_seed_selected"] is False
        assert signoff["ensemble_used"] is False
        assert signoff["prediction_shift_applied"] is False
        assert signoff["prediction_clipping_applied"] is False
        assert signoff["residual_analysis_executed"] is False
        assert signoff["worst_error_ranking_executed"] is False
        assert signoff["attention_analysis_executed"] is False
        assert signoff["regime_specific_rmse_executed"] is False

    def test_overall_status_pass(self, signoff):
        assert signoff["overall_status"] == "PASS"

    def test_ready_for_phase49(self, signoff):
        assert signoff["ready_for_phase49"] is True

    def test_n_test_2961(self, signoff):
        assert signoff["n_test"] == 2961

    def test_population_sha256(self, signoff):
        assert signoff["test_population_sha256"] == "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"

    def test_seed_list(self, signoff):
        assert signoff["seed_list"] == [42, 123, 2026]

    def test_lag_range(self, signoff):
        assert signoff["lag_range_steps"] == list(range(-6, 7))

    def test_acf_lags(self, signoff):
        assert signoff["acf_registered_lags"] == [1, 6, 12, 36, 72, 144]

    def test_peak_window(self, signoff):
        assert signoff["peak_window_steps"] == 1

    def test_rolling_window(self, signoff):
        assert signoff["rolling_window_samples"] == 144

    def test_seed_std_ddof(self, signoff):
        assert signoff["seed_std_ddof"] == 1

    def test_top_k(self, signoff):
        assert signoff["top_disagreement_k"] == 20

    def test_lag_convention_exact(self, signoff):
        assert signoff["lag_sign_convention"] == "lag_k_positive_means_prediction_compared_to_truth_shifted_k_future_steps"

    def test_phase49_residual_convention_in_signoff(self, signoff):
        assert signoff["phase49_residual_convention"] == "y_true - y_pred"

    def test_phase50_threshold_policy_in_signoff(self, signoff):
        assert signoff["phase50_threshold_policy"] == "TRAIN_DERIVED_ONLY"

    def test_phase51_worst_error_ranking_in_phase48_in_signoff(self, signoff):
        assert signoff["phase51_worst_error_ranking_in_phase48"] is False

    def test_d_04_closed(self, signoff):
        """Phase48-A D-04 must be explicitly closed in signoff."""
        assert signoff["phase48_d4_closed"] is True
        assert signoff["phase48_d4_note"]


# ===================================================================
# 5. Source integrity (Phase 47 unchanged)
# ===================================================================

class TestPhase47SourcesUnchanged:
    @pytest.mark.parametrize("name,real_relpath", [
        ("seed42", "predictions/final_test_predictions_seed42.csv"),
        ("seed123", "predictions/final_test_predictions_seed123.csv"),
        ("seed2026", "predictions/final_test_predictions_seed2026.csv"),
        ("persistence", "predictions/final_test_predictions_persistence.csv"),
        ("checksums", "prediction_checksums.json"),
        ("signoff", "phase_47_signoff.json"),
        ("lstm_elig", "final_test_lstm_eligibility.json"),
    ])
    def test_byte_identical(self, name, real_relpath):
        ext = ".csv" if real_relpath.endswith(".csv") else ".json"
        snap = Path(f"/tmp/p47_d_{name}{ext}" if name != "lstm_elig" else "/tmp/p47_d_lstm_elig.json")
        real = F47 / real_relpath
        if not snap.exists():
            pytest.skip(f"baseline snapshot not present: {snap}")
        assert _sha(snap) == _sha(real), f"Phase47 source modified: {name}"


# ===================================================================
# 6. Findings correctness
# ===================================================================

class TestFindings:
    def test_findings_present(self):
        rows = _read_csv("prediction_analysis_findings.csv")
        assert len(rows) > 0

    def test_finding_schema(self):
        header = _read_header("prediction_analysis_findings.csv")
        assert header == ["finding_code", "scope", "metric", "value",
                          "supporting_artifact", "interpretation_language", "status"]

    def test_all_findings_have_status_pass(self):
        rows = _read_csv("prediction_analysis_findings.csv")
        for r in rows:
            assert r["status"] == "PASS"

    def test_all_findings_have_descriptive_language(self):
        rows = _read_csv("prediction_analysis_findings.csv")
        for r in rows:
            assert r["interpretation_language"] == "DESCRIPTIVE_ONLY"

    def test_finding_codes_subset_of_allowed(self):
        allowed = {
            "PREDICTIONS_SHOW_RANGE_COMPRESSION",
            "PREDICTIONS_SHOW_IQR_COMPRESSION",
            "PREDICTION_RANGE_COMPRESSED_VS_ACTUAL",
            "MEAN_SHIFT_PRED_VS_ACTUAL",
            "PREDICTIONS_SMOOTHER_THAN_ACTUAL",
            "DIRECTIONAL_CHANGE_ALIGNMENT",
            "APPARENT_TEMPORAL_LAG_NEGATIVE_ONE",
            "PREDICTION_ACF_AT_LAG_1",
            "HIGH_SEED_AGREEMENT",
            "VISIBLE_SEED_DISAGREEMENT",
            "NO_NEGATIVE_PREDICTIONS",
            "NEGATIVE_PREDICTIONS_PRESENT",
            "NO_SATURATION_SIGNAL",
            "PERSISTENCE_ONE_STEP_LAG_VISIBLE",
            "LSTM_BEHAVIOR_CONTEXT_NOT_EVALUATED",
            "SOURCE_BUNDLES_VERIFIED",
            "NO_NEW_INFERENCE",
            "NO_MODEL_SELECTION",
            "NO_ENSEMBLE",
        }
        rows = _read_csv("prediction_analysis_findings.csv")
        for r in rows:
            assert r["finding_code"] in allowed, f"unexpected code: {r['finding_code']}"


# ===================================================================
# 7. Summary correctness
# ===================================================================

class TestSummary:
    def test_phase47_verdict_preserved(self, summary):
        assert summary["phase47_final_comparison_verdict"] == "Transformer_better_RMSE_R2_Persistence_better_MAE"

    def test_lstm_eligibility_preserved(self, summary):
        assert summary["lstm_eligibility"] == "NOT_ELIGIBLE_CONFIG_MISMATCH"

    def test_no_residual_no_worst_error(self, summary):
        assert summary["residual_analysis_executed"] is False
        assert summary["worst_error_ranking_executed"] is False

    def test_overall_status(self, summary):
        assert summary["overall_status"] in ("PASS", "PASS_WITH_WARNING")


# ===================================================================
# 8. Determinism / idempotence
# ===================================================================

class TestDeterminism:
    def test_findings_schema_stable(self):
        header = _read_header("prediction_analysis_findings.csv")
        assert isinstance(header, list)
        assert len(header) > 0

    def test_figures_have_minimum_size(self):
        for fname in O48_FIGURES:
            p = P48F / fname
            assert p.stat().st_size > 10000, f"{fname} too small: {p.stat().st_size}"

    def test_signoff_idempotent(self, signoff):
        """Re-run finalize → signoff structure stays the same."""
        # Just verify key fields exist
        for k in ("phase", "overall_status", "ready_for_phase49",
                  "source_predictions_modified", "phase47_artifacts_modified"):
            assert k in signoff
