"""Phase 51-E orchestrator — regime / enrichment / persistence / LSTM /
sign-consensus context.

Pre-execution gates:
  - Phase 51-B/51-C/51-D frozen SHAs must match exactly.
  - Phase 50 regime assignment SHA must match exactly.
  - Phase 47 Persistence SHA must match exactly.

Phase 51-E does NOT:
  - inspect ±6 temporal context
  - inspect raw input windows
  - build case narratives
  - build figures
  - handoff to Phase 52
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    contract,
    e_writers,
    lstm_context,
    persistence_context,
    regime_context,
    sign_consensus_context,
)
from ..utils.artifacts import get_project_root


CANONICAL_SHAS: dict[str, str] = {
    # Phase 51-B
    "artifacts/worst_error_analysis/worst_error_selection_contract.json":
        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4",
    "artifacts/worst_error_analysis/phase51_target_level_working_table.csv":
        "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f",
    # Phase 51-C
    "artifacts/worst_error_analysis/worst_per_seed_top20.csv":
        "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716",
    "artifacts/worst_error_analysis/worst_shared_top20.csv":
        "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a",
    "artifacts/worst_error_analysis/worst_underprediction_top10.csv":
        "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8",
    "artifacts/worst_error_analysis/worst_overprediction_top10.csv":
        "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa",
    "artifacts/worst_error_analysis/shared_all_under_top10.csv":
        "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637",
    "artifacts/worst_error_analysis/shared_all_over_top10.csv":
        "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be",
    # Phase 51-D
    "artifacts/worst_error_analysis/seed_overlap_table.csv":
        "6fb4c9cd5055763ae77cad6af2627c7439295d7eda084c55b41688076f3c402e",
    "artifacts/worst_error_analysis/worst_case_membership_matrix.csv":
        "036442da9f282b7e9008304672da25c25470e4443ff130f4dc1065636e60db3e",
    "artifacts/worst_error_analysis/error_concentration_table.csv":
        "8913a7dd1d43f5f58b1d0db86900ccf0815504fd087e9669dcff26678781565e",
    "artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv":
        "33052a40551e28c91762ea4fff28975edb027ade0fecdef47b63be13e6baa9aa",
    "artifacts/worst_error_analysis/hardness_group_summary.csv":
        "98a2da44cdf059e35d852139fff9af3f2a06bc0dfc5158919b4667f51afbb3e4",
}

PHASE50_ASSIGNMENT_SHA = (
    "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
)
PERSISTENCE_SHA = "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_phase51_e(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()

    # ── Gate 1: verify frozen Phase 51-B/C/D SHAs ───────────────────────────
    diffs: list[str] = []
    for rel, expected in CANONICAL_SHAS.items():
        actual = _sha256(root / rel)
        if actual != expected:
            diffs.append(f"{rel}: expected {expected}, actual {actual}")
    if diffs:
        raise RuntimeError(
            "Phase 51-B/C/D frozen SHA drift detected. Refusing to run.\n"
            + "\n".join(diffs)
        )

    # ── Gate 2: verify Phase 50 assignment SHA ──────────────────────────────
    actual_p50 = _sha256(root / "artifacts/error_by_regime/test_regime_assignment.csv")
    if actual_p50 != PHASE50_ASSIGNMENT_SHA:
        raise RuntimeError(
            f"Phase 50 assignment SHA mismatch: expected "
            f"{PHASE50_ASSIGNMENT_SHA}, got {actual_p50}"
        )

    # ── Gate 3: verify Phase 47 Persistence SHA ─────────────────────────────
    actual_p47 = _sha256(
        root / "artifacts/final_test/predictions/final_test_predictions_persistence.csv"
    )
    if actual_p47 != PERSISTENCE_SHA:
        raise RuntimeError(
            f"Phase 47 Persistence SHA mismatch: expected "
            f"{PERSISTENCE_SHA}, got {actual_p47}"
        )

    # ── Build O51.18 worst-case regime context ──────────────────────────────
    regime_ctx = regime_context.build_regime_context_table(root)
    rc_shas: dict[str, str] = {}
    rc_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "mean_abs_error_wh",
        "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
        "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
    ]
    for family, rows in regime_ctx.items():
        rel = _family_to_rel(family)
        rc_shas[family] = e_writers.write_csv(
            rows, rc_fieldnames, rel, root,
        )

    # ── Build O51.19 regime prevalence + enrichment ─────────────────────────
    enrich_rows = regime_context.build_enrichment_table(root)
    enrich_fieldnames = [
        "selection_family", "seed", "selection_k",
        "regime_family", "regime_label",
        "selected_count", "selected_prevalence",
        "global_count", "global_prevalence",
        "enrichment_ratio", "prevalence_difference",
        "status", "phase50_assignment_sha256", "selection_contract_sha256",
    ]
    enrich_sha = e_writers.write_csv(
        enrich_rows, enrich_fieldnames, "regime_overrepresentation.csv", root,
    )

    # ── Build O51.21 worst-case persistence context (Phase 51-E ownership) ──
    pers_ctx_rows = persistence_context.build_persistence_context_table(root)
    pers_fieldnames = [
        "selection_family", "seed", "rank", "target_id", "target_timestamp",
        "y_true_wh", "transformer_metric_field",
        "transformer_abs_error_wh",
        "persistence_y_pred_wh", "persistence_residual_wh",
        "persistence_absolute_error_wh", "persistence_squared_error_wh2",
    ]
    pers_sha = e_writers.write_csv(
        pers_ctx_rows, pers_fieldnames,
        "baseline_context.csv", root,
    )
    pers_summary = persistence_context.build_persistence_summary(root, pers_ctx_rows)
    pers_sum_sha = e_writers.write_csv(
        pers_summary,
        [
            "selection_family", "seed", "n_selected",
            "persistence_mean_abs_error_wh_on_selected",
            "persistence_min_abs_error_wh_on_selected",
            "persistence_max_abs_error_wh_on_selected",
        ],
        "baseline_context_summary.csv", root,
    )

    # ── LSTM eligibility context ─────────────────────────────────────────────
    lstm_ctx = lstm_context.build_lstm_eligibility_context(root)
    lstm_sha = e_writers.write_json(
        lstm_ctx, "lstm_eligibility_context.json", root,
    )

    # ── Sign-consensus context for selected targets ─────────────────────────
    selected_ids = sorted({
        r["target_id"]
        for rows in regime_ctx.values()
        for r in rows
    })
    sign_rows = sign_consensus_context.build_sign_consensus_context(root, selected_ids)
    sign_sha = e_writers.write_csv(
        sign_rows,
        [
            "target_id",
            "seed42_residual_sign", "seed123_residual_sign",
            "seed2026_residual_sign",
            "cross_seed_consensus_class",
        ],
        "regime_composition.csv", root,  # Phase51-E membership-vs-regime composition
    )

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "phase": 51,
        "subphase": "51-E",
        "version": "WORST_ERROR_ANALYSIS_REGIME_CONTEXT-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_shas": {
            "selection_contract_sha256": CANONICAL_SHAS[
                "artifacts/worst_error_analysis/worst_error_selection_contract.json"
            ],
            "working_table_sha256": CANONICAL_SHAS[
                "artifacts/worst_error_analysis/phase51_target_level_working_table.csv"
            ],
            "phase50_assignment_sha256": PHASE50_ASSIGNMENT_SHA,
            "persistence_sha256": PERSISTENCE_SHA,
        },
        "phase51_d_canonical_sha256s": {
            k.split("/")[-1]: v for k, v in CANONICAL_SHAS.items()
            if "worst_error_analysis" in k and k.split("/")[-1] not in (
                "worst_error_selection_contract.json",
                "phase51_target_level_working_table.csv",
            )
        },
        "deliverables": {
            "regime_context": {
                family: {
                    "path": f"artifacts/worst_error_analysis/{_family_to_rel(family)}",
                    "sha256": sha,
                }
                for family, sha in rc_shas.items()
            },
            "regime_overrepresentation": {
                "path": "artifacts/worst_error_analysis/regime_overrepresentation.csv",
                "sha256": enrich_sha,
            },
            "baseline_context": {
                "path": "artifacts/worst_error_analysis/baseline_context.csv",
                "sha256": pers_sha,
            },
            "baseline_context_summary": {
                "path": "artifacts/worst_error_analysis/baseline_context_summary.csv",
                "sha256": pers_sum_sha,
            },
            "lstm_eligibility_context": {
                "path": "artifacts/worst_error_analysis/lstm_eligibility_context.json",
                "sha256": lstm_sha,
            },
            "regime_composition": {
                "path": "artifacts/worst_error_analysis/regime_composition.csv",
                "sha256": sign_sha,
            },
        },
        "forbidden_actions_status": {
            "regime_enrichment_executed": True,
            "persistence_context_executed": True,
            "lstm_context_executed": True,
            "individual_case_inspection_executed": False,
            "temporal_context_executed": False,
            "input_context_executed": False,
            "casebook_created": False,
            "figures_created": False,
            "best_seed_selected": False,
            "ensemble": False,
            "three_n_iid_interpretation": False,
            "new_test_inference": False,
            "checkpoint_loading": False,
            "training": False,
            "scaler_fit": False,
            "prediction_correction": False,
        },
        "phase47_global_interpretation_preserved": (
            "Persistence better globally on MAE. "
            "Transformer better globally on RMSE and R^2."
        ),
        "ready_for_phase51_f": False,
    }
    manifest_sha = e_writers.write_json(
        manifest, "phase51_e_manifest.json", root
    )

    return {
        "phase": 51,
        "subphase": "51-E",
        "status": "PASS",
        "selection_contract_sha256": CANONICAL_SHAS[
            "artifacts/worst_error_analysis/worst_error_selection_contract.json"
        ],
        "phase50_assignment_sha256": PHASE50_ASSIGNMENT_SHA,
        "persistence_sha256": PERSISTENCE_SHA,
        "phase51_b_c_d_unchanged": True,
        "phase50_unchanged": True,
        "phase47_persistence_unchanged": True,
        "artifacts": {
            "regime_overrepresentation": enrich_sha,
            "baseline_context": pers_sha,
            "baseline_context_summary": pers_sum_sha,
            "lstm_eligibility_context": lstm_sha,
            "regime_composition": sign_sha,
            "phase51_e_manifest": manifest_sha,
            **{f"regime_context__{k}": v for k, v in rc_shas.items()},
        },
        "n_rows_enrichment": len(enrich_rows),
        "n_rows_persistence_context": len(pers_ctx_rows),
        "n_rows_persistence_summary": len(pers_summary),
        "n_rows_sign_consensus": len(sign_rows),
        "n_selected_unique_targets": len(selected_ids),
    }


def _family_to_rel(family: str) -> str:
    return {
        "W1_PER_SEED_WORST": "regime_composition.csv",  # shared composition
        "W2_SHARED_WORST": "shared_worst_regime_context.csv",
        "W3_UNDERPREDICTION_WORST": "regime_composition.csv",
        "W4_OVERPREDICTION_WORST": "regime_composition.csv",
        "W3_SH_SHARED_ALL_UNDER": "regime_composition.csv",
        "W4_SH_SHARED_ALL_OVER": "regime_composition.csv",
    }.get(family, "regime_composition.csv")


def main(project_root=None) -> dict[str, Any]:
    return materialize_phase51_e(project_root)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
