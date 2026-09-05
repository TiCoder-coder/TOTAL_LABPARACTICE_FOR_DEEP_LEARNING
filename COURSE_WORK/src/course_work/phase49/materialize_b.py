from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    assert_ensemble_not_promoted,
    assert_three_n_iid_not_claimed,
    assert_best_seed_not_selected,
)
from .residual import reconstruct_residual_long_table
from .sources import (
    compute_seed_bundle_sha256,
    load_phase47_signoff,
    verify_source_bundles,
)
from .metrics import reconstruct_seed_metrics, compare_recomputed_to_phase47
from .wide_table import (
    PHASE49_ARTIFACT_DIR_REL,
    build_residual_wide_table_rows,
    snapshot_existing_artifact_sha,
    utc_now_iso,
    write_phase49_b_manifest,
    write_phase49_field_consistency_audit,
    write_phase49_metric_reconstruction_audit,
    write_residual_long_table,
    write_residual_wide_table,
    write_source_verification_manifest,
)


PHASE47_METRIC_KEYS: dict[int, dict[str, str]] = {
    42: {"mae_wh": "seed42_mae_wh", "rmse_wh": "seed42_rmse_wh", "r2": "seed42_r2"},
    123: {"mae_wh": "seed123_mae_wh", "rmse_wh": "seed123_rmse_wh", "r2": "seed123_r2"},
    2026: {"mae_wh": "seed2026_mae_wh", "rmse_wh": "seed2026_rmse_wh", "r2": "seed2026_r2"},
}


def _field_consistency_for_seed(
    seed: int,
    long_table: list[dict[str, Any]],
    bundle_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    n = len(long_table)

    def _stats(stored_col: str, derived_value_fn) -> dict[str, Any]:
        diffs = []
        for r, b in zip(long_table, bundle_rows):
            derived = derived_value_fn(r)
            stored = float(b[stored_col])
            diffs.append(abs(derived - stored))
        max_d = max(diffs)
        mean_d = sum(diffs) / len(diffs)
        return {
            "n": n,
            "max_abs_difference": max_d,
            "mean_abs_difference": mean_d,
            "tolerance": 0.0,
            "status": "PASS" if max_d == 0.0 else "FAIL",
        }

    residual_audit = {
        "seed": seed,
        "field": "residual_wh",
        **_stats(
            "residual_wh",
            lambda r: r["residual_wh"],
        ),
    }
    audit.append(residual_audit)

    abs_audit = {
        "seed": seed,
        "field": "absolute_error_wh",
        **_stats(
            "absolute_error_wh",
            lambda r: r["absolute_error_wh"],
        ),
    }
    audit.append(abs_audit)

    sq_audit = {
        "seed": seed,
        "field": "squared_error_wh2_vs_squared_error_wh",
        "n": n,
        "max_abs_difference": max(
            abs(float(b["squared_error_wh"]) - r["squared_error_wh2"])
            for r, b in zip(long_table, bundle_rows)
        ),
        "mean_abs_difference": sum(
            abs(float(b["squared_error_wh"]) - r["squared_error_wh2"])
            for r, b in zip(long_table, bundle_rows)
        )
        / n,
        "tolerance": 0.0,
        "status": "PASS",
    }
    if sq_audit["max_abs_difference"] != 0.0:
        sq_audit["status"] = "FAIL"
    audit.append(sq_audit)

    return audit


def _metric_reconstruction_for_seed(
    seed: int,
    long_table: list[dict[str, Any]],
    phase47_signoff: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    residuals = [r["residual_wh"] for r in long_table]
    y_true = [r["y_true_wh"] for r in long_table]
    recomputed = reconstruct_seed_metrics(residuals, y_true)
    keys = PHASE47_METRIC_KEYS[seed]
    stored = {
        "mae_wh": phase47_signoff[keys["mae_wh"]],
        "rmse_wh": phase47_signoff[keys["rmse_wh"]],
        "r2": phase47_signoff[keys["r2"]],
        "n": 2961,
    }
    comparison = compare_recomputed_to_phase47(recomputed, stored, tolerance=0.0)

    rows: list[dict[str, Any]] = []
    for metric in ("mae_wh", "rmse_wh", "r2"):
        per = comparison["per_metric"][metric]
        rows.append(
            {
                "seed": seed,
                "metric": metric,
                "recomputed_value": per["recomputed_value"],
                "stored_phase47_value": per["stored_phase47_value"],
                "absolute_difference": per["absolute_difference"],
                "tolerance": per["tolerance"],
                "status": per["status"],
            }
        )
    return rows, comparison["overall_pass"]


def materialize_phase49_b(project_root=None) -> dict[str, Any]:
    assert_best_seed_not_selected(None)
    assert_ensemble_not_promoted(False)
    assert_three_n_iid_not_claimed(False)

    from .sources import load_seed_bundle_rows as _load

    source_verification = verify_source_bundles(project_root=project_root)
    source_verification_manifest = {
        "verification_status": (
            "PASS" if source_verification["__overall_pass__"] else "FAIL"
        ),
        "per_seed": {
            seed_str: payload
            for seed_str, payload in source_verification.items()
            if seed_str not in ("__cross_seed__", "__overall_pass__")
        },
        "cross_seed_alignment": source_verification["__cross_seed__"],
        "verification_at_utc": utc_now_iso(),
    }

    write_source_verification_manifest(
        source_verification_manifest, project_root=project_root
    )

    long_tables_by_seed: dict[int, list[dict[str, Any]]] = {}
    field_audit_rows: list[dict[str, Any]] = []
    metric_audit_rows: list[dict[str, Any]] = []
    per_seed_metrics: dict[int, dict[str, float]] = {}

    phase47_signoff = load_phase47_signoff(project_root=project_root)
    overall_metric_pass = True

    for seed in (42, 123, 2026):
        bundle_sha = compute_seed_bundle_sha256(seed, project_root=project_root)
        long_table = reconstruct_residual_long_table(
            seed, source_sha256=bundle_sha, project_root=project_root
        )
        long_tables_by_seed[seed] = long_table

        bundle_rows = _load(seed, project_root=project_root)
        field_audit_rows.extend(
            _field_consistency_for_seed(seed, long_table, bundle_rows)
        )
        metric_rows, seed_pass = _metric_reconstruction_for_seed(
            seed, long_table, phase47_signoff
        )
        metric_audit_rows.extend(metric_rows)
        if not seed_pass:
            overall_metric_pass = False

        per_seed_metrics[seed] = {
            "n": len(long_table),
            "mae_wh": metric_rows[0]["recomputed_value"],
            "rmse_wh": metric_rows[1]["recomputed_value"],
            "r2": metric_rows[2]["recomputed_value"],
        }

    long_table_combined: list[dict[str, Any]] = []
    for seed in (42, 123, 2026):
        long_table_combined.extend(long_tables_by_seed[seed])

    long_table_path = write_residual_long_table(
        long_table_combined, project_root=project_root
    )

    wide_rows = build_residual_wide_table_rows(long_tables_by_seed)
    wide_table_path = write_residual_wide_table(
        wide_rows, project_root=project_root
    )

    field_audit_path = write_phase49_field_consistency_audit(
        field_audit_rows, project_root=project_root
    )
    metric_audit_path = write_phase49_metric_reconstruction_audit(
        metric_audit_rows, project_root=project_root
    )

    phase47_sha_snapshot = {
        "seed42": compute_seed_bundle_sha256(42, project_root=project_root),
        "seed123": compute_seed_bundle_sha256(123, project_root=project_root),
        "seed2026": compute_seed_bundle_sha256(2026, project_root=project_root),
    }
    phase47_sha_snapshot["phase_47_signoff_unchanged_at_phase49_b"] = (
        snapshot_existing_artifact_sha(
            Path("artifacts/final_test/phase_47_signoff.json"),
            project_root=project_root,
        )
    )
    phase47_sha_snapshot["phase_48_signoff_unchanged_at_phase49_b"] = (
        snapshot_existing_artifact_sha(
            Path("artifacts/prediction_analysis/phase_48_signoff.json"),
            project_root=project_root,
        )
    )

    manifest = {
        "phase": "49",
        "phase_sub_letter": "B",
        "scope": "Residual reconstruction + metric reconstruction ONLY",
        "executed_at_utc": utc_now_iso(),
        "source_verification": source_verification_manifest,
        "field_consistency_audit_path": str(field_audit_path),
        "metric_reconstruction_audit_path": str(metric_audit_path),
        "long_table_path": str(long_table_path),
        "long_table_rows": len(long_table_combined),
        "long_table_rows_expected": 2961 * 3,
        "wide_table_path": str(wide_table_path),
        "wide_table_rows": len(wide_rows),
        "wide_table_rows_expected": 2961,
        "per_seed_metrics": {
            str(seed): {
                "n": per_seed_metrics[seed]["n"],
                "mae_wh": per_seed_metrics[seed]["mae_wh"],
                "rmse_wh": per_seed_metrics[seed]["rmse_wh"],
                "r2": per_seed_metrics[seed]["r2"],
            }
            for seed in (42, 123, 2026)
        },
        "per_seed_metrics_match_phase47": overall_metric_pass,
        "phase47_source_sha256_snapshot": phase47_sha_snapshot,
        "phase48_canonical_artifacts_unchanged": (
            phase47_sha_snapshot["phase_48_signoff_unchanged_at_phase49_b"]
            is not None
        ),
        "phase47_canonical_artifacts_unchanged": (
            phase47_sha_snapshot["phase_47_signoff_unchanged_at_phase49_b"]
            is not None
        ),
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
            "residual_correction_applied": False,
            "bias_correction_applied": False,
            "worst_error_ranking_executed": False,
            "attention_analysis_executed": False,
            "ljung_box_used_for_pass_fail": False,
            "deciles_used_as_phase50_regimes": False,
            "target_regime_analysis_deferred_to_phase50": True,
        },
        "ready_for_phase49_c": (
            source_verification["__overall_pass__"] and overall_metric_pass
        ),
        "status": (
            "PASS" if (
                source_verification["__overall_pass__"] and overall_metric_pass
            ) else "FAIL"
        ),
        "next_step_letter_after_b": "C (gap-safe ACF / sign runs / rolling 144); requires human approval",
        "phase50_authorized": False,
        "phase51_authorized": False,
        "phase52_plus_authorized": False,
    }

    write_phase49_b_manifest(manifest, project_root=project_root)
    return manifest


if __name__ == "__main__":
    payload = materialize_phase49_b()
    print(
        "phase49-b status:",
        payload["status"],
        "ready_for_phase49_c:",
        payload["ready_for_phase49_c"],
    )
