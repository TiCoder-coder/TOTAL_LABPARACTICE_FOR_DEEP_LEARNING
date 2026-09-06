from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .autocorrelation import (
    PHASE49_ACF_KEY_LAGS,
    PHASE49_ACF_LAG_RANGE,
    PHASE49_CADENCE_MINUTES,
    acf_audit_summary,
    compute_acf_for_seed,
    contiguous_test_segments_per_seed,
    detect_contiguous_segments,
    parse_timestamp,
)
from .d_writers import (
    snapshot_phase49_b_c_artifacts,
    write_acf_key_lags_csv,
    write_ljung_box_csv,
    write_phase49_d_manifest,
    write_residual_acf_csv,
    write_rolling_diagnostics_csv,
    write_sign_run_table_csv,
    write_sign_runs_csv,
    write_sign_transitions_csv,
)
from .distributions import load_phase49_b_long_table, residuals_for_seed
from .ljung_box import (
    PHASE49_LJUNG_BOX_LAGS,
    PHASE49_LJUNG_BOX_POLICY,
    ljung_box_for_seed,
)
from .rolling import (
    PHASE49_ROLLING_WINDOW_SAMPLES,
    rolling_residual_diagnostics,
)
from .sign_runs import sign_run_summary_row, sign_runs_for_seed
from .sign_transitions import (
    SIGN_LABELS,
    sign_transition_summary_row,
    sign_transitions_for_seed,
)
from course_workutils.artifacts import get_project_root


SEED_LIST = ("42", "123", "2026")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _seed_rows_chronological(
    long_table: list[dict[str, Any]],
    seed: str,
) -> tuple[list[str], list[datetime], np.ndarray]:
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: parse_timestamp(r["target_timestamp"]),
    )
    signs = [r["residual_sign"] for r in rows]
    timestamps = [parse_timestamp(r["target_timestamp"]) for r in rows]
    residuals = residuals_for_seed(long_table, seed)
    return signs, timestamps, residuals


def materialize_phase49_d(project_root=None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    long_table = load_phase49_b_long_table(root)

    if len(long_table) != 8883:
        raise ValueError(
            f"Phase49-D expects Phase49-B long table with 8883 rows; got {len(long_table)}"
        )

    acf_rows_all: list[dict[str, Any]] = []
    acf_key_rows_all: list[dict[str, Any]] = []
    ljung_rows: list[dict[str, Any]] = []
    sign_run_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    rolling_rows_all: list[dict[str, Any]] = []
    contiguity_per_seed: dict[str, Any] = {}
    acf_audits: dict[str, Any] = {}
    ljung_per_seed: dict[str, Any] = {}
    sign_run_per_seed: dict[str, Any] = {}
    sign_transition_per_seed: dict[str, Any] = {}
    rolling_per_seed: dict[str, Any] = {}

    for seed in SEED_LIST:
        signs, timestamps, residuals = _seed_rows_chronological(long_table, seed)
        segments = detect_contiguous_segments(timestamps)
        contiguity_per_seed[seed] = contiguous_test_segments_per_seed(timestamps)

        acf_rows = compute_acf_for_seed(timestamps, residuals, max_lag=PHASE49_ACF_LAG_RANGE[1])
        for r in acf_rows:
            r2 = dict(r)
            r2["seed"] = seed
            acf_rows_all.append(r2)
            if r["lag_steps"] in PHASE49_ACF_KEY_LAGS:
                acf_key_rows_all.append(dict(r2))
        acf_audits[seed] = acf_audit_summary(acf_rows, seed)

        ljung = ljung_box_for_seed(segments, residuals)
        for seg_row in ljung["per_segment"]:
            full_row = {"seed": seed, **seg_row}
            ljung_rows.append(full_row)
        ljung_per_seed[seed] = ljung

        sign_run_result = sign_runs_for_seed(signs, timestamps)
        sign_run_row = sign_run_summary_row(seed, sign_run_result)
        sign_run_rows.append(sign_run_row)
        sign_run_per_seed[seed] = {
            "run_count": sign_run_result["run_count"],
            "summary_row": sign_run_row,
            "run_table": sign_run_result["run_table"],
        }
        write_sign_run_table_csv(sign_run_result["run_table"], seed, root)

        transition_result = sign_transitions_for_seed(signs, timestamps)
        for r in transition_result["transition_rows"]:
            transition_rows.append({"seed": seed, **r})
        sign_transition_per_seed[seed] = {
            "eligible_pairs": transition_result["eligible_pairs"],
            "summary_row": sign_transition_summary_row(seed, transition_result),
            "rows": transition_result["transition_rows"],
        }

        rolling_rows = rolling_residual_diagnostics(residuals, signs, timestamps)
        for r in rolling_rows:
            r2 = dict(r)
            rolling_rows_all.append(r2)
        rolling_per_seed[seed] = {
            "n_windows": len(rolling_rows),
            "rows": rolling_rows,
        }

    acf_path = write_residual_acf_csv(acf_rows_all, root)
    acf_key_path = write_acf_key_lags_csv(acf_key_rows_all, root)
    ljung_path = write_ljung_box_csv(ljung_rows, root)
    sign_run_path = write_sign_runs_csv(sign_run_rows, root)
    transition_path = write_sign_transitions_csv(transition_rows, root)
    rolling_path = write_rolling_diagnostics_csv(rolling_rows_all, root)

    phase49_bc_sha = snapshot_phase49_b_c_artifacts(root)

    all_seeds_single_segment = all(
        contiguity_per_seed[s].__len__() == 1 for s in SEED_LIST
    )
    all_windows_valid_count_144 = all(
        r["window_valid_count"] == PHASE49_ROLLING_WINDOW_SAMPLES
        for r in rolling_rows_all
    )
    total_rolling_windows = sum(rolling_per_seed[s]["n_windows"] for s in SEED_LIST)
    expected_rolling_windows = sum(
        contiguity_per_seed[s][0]["length"] - PHASE49_ROLLING_WINDOW_SAMPLES + 1
        for s in SEED_LIST
    )

    manifest = {
        "phase": "49",
        "phase_sub_letter": "D",
        "scope": "Temporal residual diagnostics: gap-safe ACF + Ljung-Box secondary + sign runs + sign transitions + rolling 144",
        "executed_at_utc": _utc_now_iso(),
        "sources": {
            "phase49_b_long_table": "artifacts/residual_analysis/residual_long_table.csv",
            "phase49_b_long_table_rows": 8883,
            "phase49_c_manifest": "artifacts/residual_analysis/phase49_c_manifest.json",
        },
        "definition_freeze": {
            "residual_convention": "y_true - y_pred",
            "positive_semantics": "UNDERPREDICTION",
            "negative_semantics": "OVERPREDICTION",
            "zero_policy": "EXACT_ZERO",
            "cadence_minutes": PHASE49_CADENCE_MINUTES,
            "acf_lag_range": list(PHASE49_ACF_LAG_RANGE),
            "acf_key_lags": list(PHASE49_ACF_KEY_LAGS),
            "ljung_box_lags": list(PHASE49_LJUNG_BOX_LAGS),
            "ljung_box_policy": PHASE49_LJUNG_BOX_POLICY,
            "rolling_window_samples": PHASE49_ROLLING_WINDOW_SAMPLES,
            "sign_run_breaks_at": ["sign_change", "ZERO", "temporal_gap"],
            "transition_cadence_minutes": PHASE49_CADENCE_MINUTES,
            "no_epsilon_zero": True,
            "no_smoothing": True,
        },
        "contiguity_per_seed": contiguity_per_seed,
        "all_seeds_single_contiguous_segment": all_seeds_single_segment,
        "acf": {
            "n_rows": len(acf_rows_all),
            "key_lag_rows": len(acf_key_rows_all),
            "per_seed_audit": acf_audits,
        },
        "ljung_box": {
            "lags": list(PHASE49_LJUNG_BOX_LAGS),
            "policy": PHASE49_LJUNG_BOX_POLICY,
            "used_for_pass_fail": False,
            "per_seed": ljung_per_seed,
        },
        "sign_runs": sign_run_per_seed,
        "sign_transitions": sign_transition_per_seed,
        "rolling": {
            "window_size": PHASE49_ROLLING_WINDOW_SAMPLES,
            "all_windows_valid_count_144": all_windows_valid_count_144,
            "partial_windows_emitted": False,
            "forward_fill_used": False,
            "interpolation_used": False,
            "padding_used": False,
            "n_total_windows": total_rolling_windows,
            "n_total_windows_expected": expected_rolling_windows,
            "per_seed": {s: {"n_windows": rolling_per_seed[s]["n_windows"]} for s in SEED_LIST},
        },
        "seed_mean_residual_semantics": "SEED_MEAN_RESIDUAL_DESCRIPTIVE",
        "artifacts": {
            "residual_acf_csv": str(acf_path),
            "acf_key_lags_csv": str(acf_key_path),
            "ljung_box_csv": str(ljung_path),
            "sign_runs_csv": str(sign_run_path),
            "sign_run_table_per_seed_csv": "artifacts/residual_analysis/phase49_sign_run_table_seed{42,123,2026}.csv",
            "sign_transitions_csv": str(transition_path),
            "rolling_residual_diagnostics_csv": str(rolling_path),
            "phase49_d_manifest_json": "artifacts/residual_analysis/phase49_d_manifest.json",
        },
        "contract_invariants": {
            "best_seed_selected": False,
            "ensemble_promoted": False,
            "three_n_iid_interpretation": False,
            "new_inference": False,
            "training": False,
            "checkpoint_loading": False,
            "scaler_fit": False,
            "optimizer_steps": 0,
            "phase47_modified": False,
            "phase48_modified": False,
            "phase49_b_modified": False,
            "phase49_c_modified": False,
            "residual_correction_applied": False,
            "bias_correction_applied": False,
            "worst_error_ranking_executed": False,
            "attention_analysis_executed": False,
            "ljung_box_used_for_pass_fail": False,
            "deciles_used_as_phase50_regimes": False,
            "target_regime_analysis_deferred_to_phase50": True,
            "no_epsilon_around_zero": True,
            "no_smoothing": True,
            "rolling_no_partial_windows": True,
            "rolling_no_interpolation": True,
            "rolling_no_padding": True,
            "rolling_no_forward_fill": True,
            "no_best_period_selected": True,
            "no_worst_window_ranking": True,
        },
        "phase49_b_c_artifact_sha256_snapshot": phase49_bc_sha,
        "phase50_authorized": False,
        "phase51_authorized": False,
        "phase52_plus_authorized": False,
        "ready_for_phase49_e": (
            all_seeds_single_segment
            and all_windows_valid_count_144
            and total_rolling_windows == expected_rolling_windows
            and all(
                all(
                    np.isfinite(row[f"Q_lag{k}"]) and np.isfinite(row[f"p_value_lag{k}"])
                    for k in PHASE49_LJUNG_BOX_LAGS
                )
                for row in ljung_rows
            )
        ),
        "status": "PENDING_HUMAN_APPROVAL_FOR_PHASE49_E",
        "next_step_letter_after_d": "E (magnitude associations + prediction-decile diagnostics + cross-seed residual agreement + baseline residual context); requires explicit human approval",
    }

    write_phase49_d_manifest(manifest, root)
    return manifest


if __name__ == "__main__":
    out = materialize_phase49_d()
    print(
        "phase49-d status:",
        out["status"],
        "ready_for_phase49_e:",
        out["ready_for_phase49_e"],
        "n_rolling_windows_total:",
        out["rolling"]["n_total_windows"],
    )
