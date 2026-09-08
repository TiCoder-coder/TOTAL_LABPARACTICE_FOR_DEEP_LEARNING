from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .bias import (
    BIAS_FIELDS,
    SIGN_BALANCE_FIELDS,
    signed_bias_for_residuals,
    sign_balance_rows,
)
from .c_writers import (
    snapshot_phase49_b_artifacts,
    write_distribution_summary_csv,
    write_histogram_bin_edges_csv,
    write_histogram_csv,
    write_phase49_c_manifest,
    write_signed_bias_csv,
    write_sign_balance_csv,
    write_tail_diagnostics_csv,
)
from .contract import (
    assert_best_seed_not_selected,
    assert_ensemble_not_promoted,
    assert_three_n_iid_not_claimed,
)
from .distributions import (
    DISTRIBUTION_FIELDS,
    distribution_summary_rows,
    load_phase49_b_long_table,
    residuals_for_seed,
)
from .histogram import (
    PHASE49_HISTOGRAM_BINS,
    compute_common_bin_edges,
    compute_histogram_rows,
    histogram_invariants,
)
from .tails import TAIL_FIELDS, compute_tail_diagnostics
from .sources import load_seed_bundle_rows
from course_workutils.artifacts import get_project_root


SEED_LIST = ("42", "123", "2026")


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def materialize_phase49_c(project_root=None) -> dict[str, Any]:
    assert_best_seed_not_selected(None)
    assert_ensemble_not_promoted(False)
    assert_three_n_iid_not_claimed(False)

    root = project_root if project_root is not None else get_project_root()
    long_table = load_phase49_b_long_table(root)

    if len(long_table) != 8883:
        raise ValueError(
            f"Phase49-C expects Phase49-B long table with 8883 rows; got {len(long_table)}"
        )

    distribution_rows = distribution_summary_rows(long_table, SEED_LIST)
    bias_rows: list[dict[str, Any]] = []
    for seed in SEED_LIST:
        residuals = residuals_for_seed(long_table, seed)
        bias = signed_bias_for_residuals(residuals)
        bias_row: dict[str, Any] = {"seed": seed}
        for k in BIAS_FIELDS:
            if k == "seed":
                continue
            bias_row[k] = bias[k]
        bias_rows.append(bias_row)

    sign_rows = sign_balance_rows(long_table, SEED_LIST)

    tail_rows: list[dict[str, Any]] = []
    for seed in SEED_LIST:
        seed_rows = [r for r in long_table if r["seed"] == seed]
        abs_errors = np.asarray(
            [float(r["absolute_error_wh"]) for r in seed_rows], dtype=np.float64
        )
        tail = compute_tail_diagnostics(abs_errors)
        tail_row: dict[str, Any] = {"seed": seed}
        tail_row.update(tail)
        tail_rows.append(tail_row)

    residuals_by_seed = {
        seed: residuals_for_seed(long_table, seed) for seed in SEED_LIST
    }
    edges = compute_common_bin_edges(residuals_by_seed)
    if edges.size - 1 != PHASE49_HISTOGRAM_BINS:
        raise ValueError(
            f"Phase49-C histogram edges must produce {PHASE49_HISTOGRAM_BINS} bins; "
            f"got {edges.size - 1}"
        )
    histogram_rows = compute_histogram_rows(residuals_by_seed, edges)
    hist_invariants = histogram_invariants(histogram_rows)

    dist_path = write_distribution_summary_csv(distribution_rows, root)
    bias_path = write_signed_bias_csv(bias_rows, root)
    sign_path = write_sign_balance_csv(sign_rows, root)
    tail_path = write_tail_diagnostics_csv(tail_rows, root)
    hist_path = write_histogram_csv(histogram_rows, root)
    edges_path = write_histogram_bin_edges_csv(edges, root)

    phase49_b_sha = snapshot_phase49_b_artifacts(root)
    phase47_sha = {}
    for seed in (42, 123, 2026):
        phase47_sha[f"seed{seed}"] = (
            Path("artifacts/final_test") / "predictions"
            / f"final_test_predictions_seed{seed}.csv"
        )

    seed_shas = {}
    for seed in (42, 123, 2026):
        try:
            bundle_rows = load_seed_bundle_rows(seed, project_root=root)
            n_check = len(bundle_rows)
        except Exception:
            n_check = -1
        from course_workutils.artifacts import sha256_file
        seed_shas[f"seed{seed}"] = {
            "sha256_first16": sha256_file(root / phase47_sha[f"seed{seed}"])[:16],
            "n_rows": n_check,
        }

    manifest = {
        "phase": "49",
        "phase_sub_letter": "C",
        "scope": "Residual distribution + signed bias + sign balance + tail diagnostics + 50-common-bin histogram preparation",
        "executed_at_utc": _utc_now_iso(),
        "sources": {
            "phase49_b_long_table": "artifacts/residual_analysis/residual_long_table.csv",
            "phase49_b_long_table_rows": 8883,
            "phase49_b_wide_table": "artifacts/residual_analysis/residual_wide_table.csv",
            "phase49_b_manifest": "artifacts/residual_analysis/phase49_b_manifest.json",
        },
        "artifacts": {
            "distribution_summary_csv": str(dist_path),
            "signed_bias_csv": str(bias_path),
            "sign_balance_csv": str(sign_path),
            "tail_diagnostics_csv": str(tail_path),
            "histogram_csv": str(hist_path),
            "histogram_bin_edges_csv": str(edges_path),
            "phase49_c_manifest_json": "artifacts/residual_analysis/phase49_c_manifest.json",
        },
        "definition_freeze": {
            "residual_convention": "y_true - y_pred",
            "positive_semantics": "UNDERPREDICTION",
            "negative_semantics": "OVERPREDICTION",
            "zero_policy": "EXACT_ZERO",
            "mad_definition": "median(abs(residual - median(residual)))",
            "skewness": "sample skewness bias=False",
            "kurtosis": "Fisher excess kurtosis bias=False normal_reference=0",
            "std_ddof": 1,
            "histogram_bins": 50,
            "histogram_common_range": True,
            "cross_seed_summary_ddof": 1,
        },
        "n_per_seed": 2961,
        "seeds": list(SEED_LIST),
        "distribution_summary": {r["seed"]: {k: r[k] for k in DISTRIBUTION_FIELDS} for r in distribution_rows},
        "signed_bias": {r["seed"]: {k: r[k] for k in BIAS_FIELDS if k != "seed"} for r in bias_rows},
        "sign_balance": {r["seed"]: {k: r[k] for k in SIGN_BALANCE_FIELDS if k != "seed"} for r in sign_rows},
        "tail_diagnostics": {r["seed"]: {k: r[k] for k in TAIL_FIELDS if k != "seed"} for r in tail_rows},
        "histogram": {
            "n_bins": PHASE49_HISTOGRAM_BINS,
            "n_rows_total": len(histogram_rows),
            "shared_bin_edges_across_seeds": hist_invariants["shared_bin_edges_across_seeds"],
            "all_seeds_count_to_2961": hist_invariants["all_seeds_count_to_2961"],
            "all_seeds_fraction_to_1": hist_invariants["all_seeds_fraction_to_1"],
            "n_seeds": hist_invariants["n_seeds"],
            "common_residual_range": {
                "min": float(np.min(edges)),
                "max": float(np.max(edges)),
            },
        },
        "seed_mean_residual_semantics": "SEED_MEAN_RESIDUAL_DESCRIPTIVE",
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
            "residual_correction_applied": False,
            "bias_correction_applied": False,
            "worst_error_ranking_executed": False,
            "attention_analysis_executed": False,
            "ljung_box_used_for_pass_fail": False,
            "deciles_used_as_phase50_regimes": False,
            "target_regime_analysis_deferred_to_phase50": True,
            "no_epsilon_around_zero": True,
            "no_top_k_worst_ranking": True,
        },
        "phase49_b_artifact_sha256_snapshot": phase49_b_sha,
        "phase47_source_sha256_snapshot_first16": {
            f"seed{seed}": seed_shas[f"seed{seed}"]["sha256_first16"]
            for seed in (42, 123, 2026)
        },
        "phase47_source_n_rows": {
            f"seed{seed}": seed_shas[f"seed{seed}"]["n_rows"]
            for seed in (42, 123, 2026)
        },
        "phase50_authorized": False,
        "phase51_authorized": False,
        "phase52_plus_authorized": False,
        "ready_for_phase49_d": (
            hist_invariants["shared_bin_edges_across_seeds"]
            and hist_invariants["all_seeds_count_to_2961"]
            and hist_invariants["all_seeds_fraction_to_1"]
            and all(r["fractions_sum_to_one"] for r in sign_rows)
            and all(
                r["n"] == 2961
                for r in (
                    *distribution_rows,
                    *bias_rows,
                    *sign_rows,
                    *tail_rows,
                )
            )
        ),
        "status": "PENDING_HUMAN_APPROVAL_FOR_PHASE49_D",
        "next_step_letter_after_c": "D (ACF + sign runs + Ljung-Box + rolling 144); requires explicit human approval",
    }

    manifest_path = write_phase49_c_manifest(manifest, root)
    return manifest


if __name__ == "__main__":
    out = materialize_phase49_c()
    print(
        "phase49-c distribution rows:",
        len(out["distribution_summary"]),
        "ready_for_phase49_d:",
        out["ready_for_phase49_d"],
    )
