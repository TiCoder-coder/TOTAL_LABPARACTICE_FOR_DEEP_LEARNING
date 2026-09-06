from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from course_work.residual_analysis.autocorrelation import (
    PHASE49_ACF_KEY_LAGS,
    PHASE49_ACF_LAG_RANGE,
    PHASE49_CADENCE_MINUTES,
    compute_acf_for_segment,
    compute_acf_for_seed,
    contiguous_test_segments_per_seed,
    detect_contiguous_segments,
    parse_timestamp,
)
from course_work.residual_analysis.distributions import (
    load_phase49_b_long_table,
    residuals_for_seed,
)
from course_work.residual_analysis.ljung_box import (
    PHASE49_LJUNG_BOX_LAGS,
    PHASE49_LJUNG_BOX_POLICY,
    ljung_box_for_seed,
)
from course_work.residual_analysis.materialize_d import SEED_LIST, materialize_phase49_d
from course_work.residual_analysis.rolling import (
    PHASE49_ROLLING_WINDOW_SAMPLES,
    rolling_residual_diagnostics,
)
from course_work.residual_analysis.sign_runs import sign_runs_for_seed
from course_work.residual_analysis.sign_transitions import (
    SIGN_LABELS,
    sign_transitions_for_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def phase49_d_manifest():
    return materialize_phase49_d(project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def long_table():
    return load_phase49_b_long_table(project_root=PROJECT_ROOT)


def _seed_rows_chronological(long_table, seed):
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: parse_timestamp(r["target_timestamp"]),
    )
    signs = [r["residual_sign"] for r in rows]
    timestamps = [parse_timestamp(r["target_timestamp"]) for r in rows]
    residuals = residuals_for_seed(long_table, seed)
    return rows, signs, timestamps, residuals


def test_residual_convention_unchanged(long_table):
    sample = long_table[0]
    expected = float(sample["y_true_wh"]) - float(sample["y_pred_wh"])
    assert abs(float(sample["residual_wh"]) - expected) == 0.0


def test_acf_lag_range_is_1_to_144(phase49_d_manifest):
    assert phase49_d_manifest["definition_freeze"]["acf_lag_range"] == [1, 144]


def test_key_acf_lags_are_exactly_1_6_12_36_72_144(phase49_d_manifest):
    assert phase49_d_manifest["definition_freeze"]["acf_key_lags"] == [1, 6, 12, 36, 72, 144]
    assert PHASE49_ACF_KEY_LAGS == (1, 6, 12, 36, 72, 144)
    assert PHASE49_ACF_LAG_RANGE == (1, 144)


def test_acf_valid_pair_count_correct(long_table, phase49_d_manifest):
    for seed in SEED_LIST:
        _, _, timestamps, residuals = _seed_rows_chronological(long_table, seed)
        acf_rows = compute_acf_for_seed(timestamps, residuals)
        for r in acf_rows:
            expected = 2961 - r["lag_steps"]
            assert r["valid_pair_count"] == expected, f"seed {seed} lag {r['lag_steps']}"


def test_no_acf_across_temporal_gaps():
    timestamps = [
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 10, 0),
        datetime(2026, 1, 1, 0, 30, 0),
        datetime(2026, 1, 1, 0, 40, 0),
    ]
    residuals = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    segments = detect_contiguous_segments(timestamps)
    assert len(segments) == 2
    acf = compute_acf_for_seed(timestamps, residuals, max_lag=1)
    assert acf[0]["valid_pair_count"] == 2


def test_chronological_order_preserved(phase49_d_manifest):
    contig = phase49_d_manifest["contiguity_per_seed"]["42"]
    assert contig[0]["start_timestamp"] < contig[0]["end_timestamp_last_inclusive"]


def test_ljung_box_lags_exactly_6_36_144(phase49_d_manifest):
    assert phase49_d_manifest["definition_freeze"]["ljung_box_lags"] == [6, 36, 144]
    assert PHASE49_LJUNG_BOX_LAGS == (6, 36, 144)


def test_ljung_box_contiguous_only(long_table, phase49_d_manifest):
    ljung_per_seed = phase49_d_manifest["ljung_box"]["per_seed"]
    for seed in SEED_LIST:
        lj = ljung_per_seed[seed]
        for seg in lj["per_segment"]:
            assert seg["status"] in {"SECONDARY_DIAGNOSTIC", "NOT_APPLICABLE_SEGMENT_TOO_SHORT"}


def test_ljung_box_policy_secondary_diagnostic(phase49_d_manifest):
    assert phase49_d_manifest["ljung_box"]["policy"] == "SECONDARY_DIAGNOSTIC"
    assert PHASE49_LJUNG_BOX_POLICY == "SECONDARY_DIAGNOSTIC"


def test_ljung_box_not_used_for_pass_fail(phase49_d_manifest):
    assert phase49_d_manifest["ljung_box"]["used_for_pass_fail"] is False
    assert phase49_d_manifest["contract_invariants"]["ljung_box_used_for_pass_fail"] is False
    assert phase49_d_manifest["status"] != "PASS" and phase49_d_manifest["status"] != "FAIL"
    assert "PENDING_HUMAN" in phase49_d_manifest["status"]


def test_sign_run_breaks_at_sign_change():
    signs = ["UNDERPREDICTION", "UNDERPREDICTION", "OVERPREDICTION", "OVERPREDICTION"]
    timestamps = [datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=10 * i) for i in range(4)]
    result = sign_runs_for_seed(signs, timestamps)
    assert result["run_count"] == 2
    assert result["run_lengths"] == [2, 2]


def test_sign_run_breaks_at_exact_zero():
    signs = ["UNDERPREDICTION", "EXACT", "UNDERPREDICTION"]
    timestamps = [datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=10 * i) for i in range(3)]
    result = sign_runs_for_seed(signs, timestamps)
    assert result["run_count"] == 3
    assert result["exact_zero_count"] == 1


def test_sign_run_breaks_at_temporal_gap():
    signs = ["UNDERPREDICTION", "UNDERPREDICTION", "UNDERPREDICTION"]
    timestamps = [
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 10, 0),
        datetime(2026, 1, 1, 0, 40, 0),
    ]
    result = sign_runs_for_seed(signs, timestamps)
    assert result["run_count"] == 2


def test_transition_pairs_require_exact_10_minute_cadence():
    signs = ["UNDERPREDICTION", "OVERPREDICTION", "UNDER"]
    timestamps = [
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 10, 0),
        datetime(2026, 1, 1, 0, 40, 0),
    ]
    with pytest.raises(ValueError):
        sign_transitions_for_seed(signs, timestamps)


def test_transition_counts_deterministic():
    signs = ["UNDERPREDICTION", "OVERPREDICTION", "UNDERPREDICTION", "EXACT"]
    timestamps = [datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=10 * i) for i in range(4)]
    result = sign_transitions_for_seed(signs, timestamps)
    counts = result["transition_count_table"]
    assert counts["UNDERPREDICTION->OVERPREDICTION"] == 1
    assert counts["OVERPREDICTION->UNDERPREDICTION"] == 1
    assert counts["UNDERPREDICTION->EXACT"] == 1
    a = materialize_phase49_d(project_root=PROJECT_ROOT)
    b = materialize_phase49_d(project_root=PROJECT_ROOT)
    for s in SEED_LIST:
        a_counts = {r["from_sign"]: {rr["to_sign"]: rr["count"] for rr in a["sign_transitions"][s]["rows"]} for r in a["sign_transitions"][s]["rows"]}
        b_counts = {r["from_sign"]: {rr["to_sign"]: rr["count"] for rr in b["sign_transitions"][s]["rows"]} for r in b["sign_transitions"][s]["rows"]}
        assert a_counts == b_counts


def test_transition_probabilities_normalized():
    signs = ["UNDERPREDICTION", "UNDERPREDICTION", "OVERPREDICTION", "OVERPREDICTION"]
    timestamps = [datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=10 * i) for i in range(4)]
    result = sign_transitions_for_seed(signs, timestamps)
    for from_sign in SIGN_LABELS:
        rows = [r for r in result["transition_rows"] if r["from_sign"] == from_sign]
        probs = [r["probability_given_from_a"] for r in rows]
        total = sum(probs)
        if any(r["count"] > 0 for r in rows):
            assert abs(total - 1.0) < 1e-12


def test_rolling_window_exactly_144(phase49_d_manifest):
    assert phase49_d_manifest["rolling"]["window_size"] == 144
    assert PHASE49_ROLLING_WINDOW_SAMPLES == 144


def test_rolling_windows_contiguous(phase49_d_manifest):
    assert phase49_d_manifest["rolling"]["all_windows_valid_count_144"] is True


def test_no_partial_rolling_windows(phase49_d_manifest):
    assert phase49_d_manifest["rolling"]["partial_windows_emitted"] is False
    assert phase49_d_manifest["rolling"]["n_total_windows"] == phase49_d_manifest["rolling"]["n_total_windows_expected"]
    for s in SEED_LIST:
        n_expected = 2961 - 144 + 1
        assert phase49_d_manifest["rolling"]["per_seed"][s]["n_windows"] == n_expected


def test_no_forward_fill_or_interpolation(phase49_d_manifest):
    assert phase49_d_manifest["rolling"]["forward_fill_used"] is False
    assert phase49_d_manifest["rolling"]["interpolation_used"] is False
    assert phase49_d_manifest["rolling"]["padding_used"] is False
    assert phase49_d_manifest["contract_invariants"]["rolling_no_partial_windows"] is True
    assert phase49_d_manifest["contract_invariants"]["rolling_no_interpolation"] is True
    assert phase49_d_manifest["contract_invariants"]["rolling_no_padding"] is True
    assert phase49_d_manifest["contract_invariants"]["rolling_no_forward_fill"] is True


def test_rolling_metrics_deterministic():
    signs = ["UNDERPREDICTION"] * 200
    timestamps = [datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=10 * i) for i in range(200)]
    residuals = np.linspace(0, 10, 200, dtype=np.float64)
    rows_a = rolling_residual_diagnostics(residuals, signs, timestamps)
    rows_b = rolling_residual_diagnostics(residuals, signs, timestamps)
    assert rows_a == rows_b


def test_no_best_period_or_worst_window_selection(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["no_best_period_selected"] is True
    assert phase49_d_manifest["contract_invariants"]["no_worst_window_ranking"] is True
    assert phase49_d_manifest["contract_invariants"]["worst_error_ranking_executed"] is False


def test_no_phase50_regime_creation(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["deciles_used_as_phase50_regimes"] is False
    assert phase49_d_manifest["contract_invariants"]["target_regime_analysis_deferred_to_phase50"] is True
    assert phase49_d_manifest["phase50_authorized"] is False


def test_no_best_seed_selection(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["best_seed_selected"] is False


def test_no_ensemble(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["ensemble_promoted"] is False


def test_no_training_inference_or_checkpoint_loading(phase49_d_manifest):
    inv = phase49_d_manifest["contract_invariants"]
    assert inv["new_inference"] is False
    assert inv["training"] is False
    assert inv["checkpoint_loading"] is False
    assert inv["scaler_fit"] is False
    assert inv["optimizer_steps"] == 0


def test_phase47_unchanged(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["phase47_modified"] is False
    from course_work.residual_analysis.sources import compute_seed_bundle_sha256, load_prediction_checksums
    checksums = load_prediction_checksums(project_root=PROJECT_ROOT)
    for seed in (42, 123, 2026):
        key = {"seed42": "seed_42", "seed123": "seed_123", "seed2026": "seed_2026"}[f"seed{seed}"]
        expected = checksums["predictions"][key]["sha256"]
        observed = compute_seed_bundle_sha256(seed, project_root=PROJECT_ROOT)
        assert observed == expected


def test_phase48_unchanged(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["phase48_modified"] is False


def test_phase49_b_c_unchanged(phase49_d_manifest):
    assert phase49_d_manifest["contract_invariants"]["phase49_b_modified"] is False
    assert phase49_d_manifest["contract_invariants"]["phase49_c_modified"] is False


def test_outputs_deterministic(phase49_d_manifest):
    again = materialize_phase49_d(project_root=PROJECT_ROOT)
    for s in SEED_LIST:
        a_runs = phase49_d_manifest["sign_runs"][s]["summary_row"]
        b_runs = again["sign_runs"][s]["summary_row"]
        assert a_runs == b_runs


def test_phase49_d_artifacts_exist(phase49_d_manifest):
    for k in (
        "residual_acf_csv",
        "acf_key_lags_csv",
        "ljung_box_csv",
        "sign_runs_csv",
        "sign_transitions_csv",
        "rolling_residual_diagnostics_csv",
    ):
        p = Path(phase49_d_manifest["artifacts"][k])
        assert p.exists(), k
        assert p.stat().st_size > 0


def test_per_seed_contiguity_single_segment(phase49_d_manifest):
    contig = phase49_d_manifest["contiguity_per_seed"]
    for s in SEED_LIST:
        assert len(contig[s]) == 1
        assert contig[s][0]["length"] == 2961


def test_sign_run_zero_breaks_run(long_table, phase49_d_manifest):
    for s in SEED_LIST:
        sr = phase49_d_manifest["sign_runs"][s]
        assert sr["summary_row"]["breaks_at_zero"] is True
        assert sr["summary_row"]["breaks_at_sign_change"] is True
        assert sr["summary_row"]["breaks_at_gap"] is True


def test_ljung_box_per_seed_segments_correctly_classified(phase49_d_manifest):
    for s in SEED_LIST:
        lj = phase49_d_manifest["ljung_box"]["per_seed"][s]
        for row in lj["per_segment"]:
            assert "Q_lag6" in row
            assert "Q_lag36" in row
            assert "Q_lag144" in row
            assert "p_value_lag6" in row
            assert "p_value_lag36" in row
            assert "p_value_lag144" in row
