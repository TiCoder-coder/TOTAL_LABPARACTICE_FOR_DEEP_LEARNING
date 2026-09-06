from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cross_seed import (
    SEED_PAIRS,
    build_residual_vectors_by_target_id,
    compute_cross_seed_sign_consensus,
    compute_pair_residual_agreement,
)
from .distributions import (
    load_phase49_b_long_table,
    residuals_for_seed,
)
from .e_writers import (
    snapshot_phase49_b_c_d_artifacts,
    write_cross_seed_residual_agreement_csv,
    write_cross_seed_sign_consensus_csv,
    write_magnitude_associations_csv,
    write_persistence_context_csv,
    write_phase49_e_manifest,
    write_prediction_deciles_csv,
)
from .magnitude_associations import compute_magnitude_associations
from .persistence_context import (
    compute_persistence_residual_context,
    load_persistence_rows,
    verify_persistence_against_phase47,
)
from .prediction_deciles import compute_decile_diagnostics
from .sources import (
    load_phase47_signoff,
    load_prediction_checksums,
)
from ..utils.artifacts import get_project_root, sha256_file


SEED_LIST = ("42", "123", "2026")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _y_pred_for_seed(long_table, seed):
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: r["target_id"],
    )
    return [float(r["y_pred_wh"]) for r in rows]


def _y_true_for_seed(long_table, seed):
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: r["target_id"],
    )
    return [float(r["y_true_wh"]) for r in rows]


def _signs_for_seed(long_table, seed):
    rows = sorted(
        [r for r in long_table if r["seed"] == seed],
        key=lambda r: r["target_id"],
    )
    return [r["residual_sign"] for r in rows]


def materialize_phase49_e(project_root=None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    long_table = load_phase49_b_long_table(root)

    if len(long_table) != 8883:
        raise ValueError(
            f"Phase49-E expects Phase49-B long table with 8883 rows; got {len(long_table)}"
        )

    magnitude_rows: list[dict[str, Any]] = []
    decile_rows_all: list[dict[str, Any]] = []
    for seed in SEED_LIST:
        residuals = residuals_for_seed(long_table, seed)
        y_true = np.asarray(_y_true_for_seed(long_table, seed), dtype=np.float64)
        y_pred = np.asarray(_y_pred_for_seed(long_table, seed), dtype=np.float64)
        signs = _signs_for_seed(long_table, seed)

        for row in compute_magnitude_associations(residuals, y_true, y_pred):
            row_with_seed = {"seed": seed, **row}
            magnitude_rows.append(row_with_seed)

        decile_rows, _ = compute_decile_diagnostics(residuals, signs, y_pred, seed)
        decile_rows_all.extend(decile_rows)

    pair_rows: list[dict[str, Any]] = []
    seed_resid = {
        seed: np.asarray(_y_true_for_seed(long_table, seed), dtype=np.float64)
        - np.asarray(_y_pred_for_seed(long_table, seed), dtype=np.float64)
        for seed in SEED_LIST
    }
    seed_resid_ids = {
        seed: [r["target_id"] for r in long_table if r["seed"] == seed]
        for seed in SEED_LIST
    }
    for seed_a, seed_b in SEED_PAIRS:
        if seed_resid_ids[seed_a] != seed_resid_ids[seed_b]:
            raise ValueError(
                f"Target ID misalignment between seed {seed_a} and seed {seed_b}"
            )
        pair = compute_pair_residual_agreement(
            seed_resid[seed_a], seed_resid[seed_b], seed_a, seed_b
        )
        pair_rows.append(pair)

    consensus = compute_cross_seed_sign_consensus(long_table)
    consensus_rows = consensus["rows"]

    persistence_rows, persistence_meta = load_persistence_rows(root)
    expected_persistence_sha = load_prediction_checksums(root)["predictions"]["persistence"]["sha256"]
    persistence_sha_match = persistence_meta["observed_sha256"] == expected_persistence_sha
    if not persistence_sha_match:
        raise ValueError(
            "Persistence sha256 mismatch: expected "
            f"{expected_persistence_sha}, observed {persistence_meta['observed_sha256']}"
        )

    verification = verify_persistence_against_phase47(
        persistence_rows, long_table, persistence_meta
    )
    if verification["status"] != "PASS":
        raise ValueError(
            f"Persistence verification failed: {verification}"
        )

    persistence_context = compute_persistence_residual_context(persistence_rows)

    magnitude_path = write_magnitude_associations_csv(magnitude_rows, root)
    decile_path = write_prediction_deciles_csv(decile_rows_all, root)
    pair_path = write_cross_seed_residual_agreement_csv(pair_rows, root)
    consensus_path = write_cross_seed_sign_consensus_csv(consensus_rows, root)
    persistence_path = write_persistence_context_csv(persistence_context, root)

    phase47_signoff = load_phase47_signoff(root)
    lstm_eligibility = phase47_signoff.get("lstm_eligibility")
    lstm_metrics = phase47_signoff.get("lstm_metrics")

    phase49_bcd_sha = snapshot_phase49_b_c_d_artifacts(root)

    manifest = {
        "phase": "49",
        "phase_sub_letter": "E",
        "scope": "Magnitude associations + prediction-decile diagnostics + cross-seed residual agreement + cross-seed sign consensus + persistence baseline residual context",
        "executed_at_utc": _utc_now_iso(),
        "sources": {
            "phase49_b_long_table": "artifacts/residual_analysis/residual_long_table.csv",
            "phase49_b_wide_table": "artifacts/residual_analysis/residual_wide_table.csv",
            "phase49_distribution_summary": "artifacts/residual_analysis/phase49_residual_distribution_summary.csv",
            "phase49_signed_bias": "artifacts/residual_analysis/phase49_signed_bias.csv",
            "phase49_tail_diagnostics": "artifacts/residual_analysis/phase49_tail_diagnostics.csv",
            "phase49_residual_acf": "artifacts/residual_analysis/phase49_residual_acf.csv",
            "phase49_rolling": "artifacts/residual_analysis/phase49_rolling_residual_diagnostics.csv",
            "persistence_source": "artifacts/final_test/predictions/final_test_predictions_persistence.csv",
        },
        "definition_freeze": {
            "residual_convention": "y_true - y_pred",
            "positive_semantics": "UNDERPREDICTION",
            "negative_semantics": "OVERPREDICTION",
            "zero_policy": "EXACT_ZERO",
            "decile_count": 10,
            "decile_source": "y_pred_only",
            "decile_use": "PREDICTION-DECILE DESCRIPTIVE DIAGNOSTICS ONLY",
            "phase50_regime_use": False,
            "cross_seed_scope": "CROSS_SEED_RESIDUAL_AGREEMENT_DIAGNOSTIC",
            "seed_pairs": [list(p) for p in SEED_PAIRS],
            "causal_interpretation": "NONE",
            "lstm_status": "NOT_ELIGIBLE_CONFIG_MISMATCH",
            "phase47_baseline_interpretation": {
                "transformer_better_metrics": ["RMSE", "R2"],
                "persistence_better_metric": ["MAE"],
            },
        },
        "magnitude_associations": {
            "n_rows": len(magnitude_rows),
            "rows_by_seed": {
                seed: [r for r in magnitude_rows if r["seed"] == seed] for seed in SEED_LIST
            },
        },
        "prediction_deciles": {
            "n_rows": len(decile_rows_all),
            "decile_count": 10,
            "decile_source": "y_pred_only",
            "rows_by_seed": {
                seed: [r for r in decile_rows_all if r["seed"] == seed] for seed in SEED_LIST
            },
        },
        "cross_seed_residual_agreement": {
            "n_pairs": len(pair_rows),
            "seed_pairs": [list(p) for p in SEED_PAIRS],
            "rows": pair_rows,
            "scope": "CROSS_SEED_RESIDUAL_AGREEMENT_DIAGNOSTIC",
            "purpose": "NOT model-performance comparison",
        },
        "cross_seed_sign_consensus": {
            "n_targets": consensus["n_targets"],
            "fractions_sum_to_one": consensus["fractions_sum_to_one"],
            "zero_policy": consensus["zero_policy"],
            "rows": consensus_rows,
            "n_per_target_class_rows": len(consensus["per_target_class"]),
        },
        "persistence_context": {
            "source_sha256_match": persistence_sha_match,
            "verification": verification,
            "context": persistence_context,
            "phase47_signoff_persistence_metrics": {
                "mae_wh": phase47_signoff.get("persistence_mae_wh"),
                "rmse_wh": phase47_signoff.get("persistence_rmse_wh"),
                "r2": phase47_signoff.get("persistence_r2"),
            },
        },
        "lstm_policy": {
            "lstm_tuned_dev": "NOT_ELIGIBLE_CONFIG_MISMATCH",
            "lstm_eligibility_from_signoff": lstm_eligibility,
            "lstm_metrics_from_signoff": lstm_metrics,
            "lstm_residual_table_created": False,
            "lstm_placeholder_residuals_created": False,
        },
        "seed_mean_residual_semantics": "SEED_MEAN_RESIDUAL_DESCRIPTIVE",
        "artifacts": {
            "magnitude_associations_csv": str(magnitude_path),
            "prediction_deciles_csv": str(decile_path),
            "cross_seed_residual_agreement_csv": str(pair_path),
            "cross_seed_sign_consensus_csv": str(consensus_path),
            "persistence_context_csv": str(persistence_path),
            "phase49_e_manifest_json": "artifacts/residual_analysis/phase49_e_manifest.json",
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
            "phase49_d_modified": False,
            "residual_correction_applied": False,
            "bias_correction_applied": False,
            "worst_error_ranking_executed": False,
            "attention_analysis_executed": False,
            "ljung_box_used_for_pass_fail": False,
            "phase50_regimes_created": False,
            "target_regime_analysis_deferred_to_phase50": True,
            "deciles_used_as_phase50_regimes": False,
            "causal_interpretation_emitted": False,
            "lstm_residual_artifact_created": False,
            "lstm_placeholder_residuals_created": False,
            "no_epsilon_around_zero": True,
        },
        "phase49_b_c_d_artifact_sha256_snapshot": phase49_bcd_sha,
        "phase47_persistence_sha256_first16": expected_persistence_sha[:16],
        "phase50_authorized": False,
        "phase51_authorized": False,
        "phase52_plus_authorized": False,
        "ready_for_phase49_f": (
            persistence_sha_match
            and verification["status"] == "PASS"
            and consensus["fractions_sum_to_one"]
            and len(pair_rows) == 3
            and all(decile["N"] > 0 for decile in decile_rows_all)
        ),
        "status": "PENDING_HUMAN_APPROVAL_FOR_PHASE49_F",
        "next_step_letter_after_e": "F (final figures + final findings + final Phase49 signoff); requires explicit human approval",
    }

    write_phase49_e_manifest(manifest, root)
    return manifest


if __name__ == "__main__":
    out = materialize_phase49_e()
    print(
        "phase49-e status:",
        out["status"],
        "ready_for_phase49_f:",
        out["ready_for_phase49_f"],
    )
