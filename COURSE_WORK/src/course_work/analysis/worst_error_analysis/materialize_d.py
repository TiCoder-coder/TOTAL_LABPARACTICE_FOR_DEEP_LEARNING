"""Phase 51-D orchestrator — execute cross-seed overlap, concentration,
hardness/seed-spread context, and write all O51.D artifacts.

Pre-execution gate: verify every canonical Phase 51-B/51-C SHA matches the
human-recorded canonical SHAs exactly. If any SHA differs unexpectedly,
STOP and DO NOT regenerate.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import concentration as concentration_mod
from . import hardness as hardness_mod
from . import overlap as overlap_mod
from . import d_writers
from course_work.utils.artifacts import get_project_root


# Canonical SHAs (must match the ones recorded in the user's spec)
CANONICAL_SHAS: dict[str, str] = {
    "worst_error_selection_contract.json":
        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4",
    "phase51_target_level_working_table.csv":
        "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f",
    "worst_per_seed_top20.csv":
        "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716",
    "worst_shared_top20.csv":
        "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a",
    "worst_underprediction_top10.csv":
        "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8",
    "worst_overprediction_top10.csv":
        "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa",
    "shared_all_under_top10.csv":
        "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637",
    "shared_all_over_top10.csv":
        "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be",
}

# Upstream frozen artifacts (verbatim copy from Phase 47–50 verification)
UPSTREAM_FROZEN_SHAS: dict[str, str] = {
    "artifacts/final_test/predictions/final_test_predictions_seed42.csv":
        "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2",
    "artifacts/prediction_analysis/phase_48_signoff.json":
        "e8c102d582a35dd2d86a8c275f4cb1bf2d24f4f841404aad33c4b0e29161fa9b",
    "artifacts/residual_analysis/residual_long_table.csv":
        "8418a99110bfda7047bd27c49c1c7a9769313b1ce1fa6dc66925f5286d120038",
    "artifacts/error_by_regime/test_regime_assignment.csv":
        "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_phase51_d(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()

    # ── Gate 1: verify all canonical Phase 51-B/51-C SHAs ───────────────────
    sha_diffs: list[str] = []
    for rel, expected in CANONICAL_SHAS.items():
        fp = root / "artifacts/worst_error_analysis" / rel
        actual = _sha256(fp)
        if actual != expected:
            sha_diffs.append(
                f"{rel}: expected {expected}, actual {actual}"
            )
    if sha_diffs:
        raise RuntimeError(
            "Canonical Phase 51-B/51-C SHA drift detected. Refusing to run.\n"
            + "\n".join(sha_diffs)
        )

    # ── Gate 2: verify upstream frozen SHAs (read-only check, no mutation) ──
    upstream_diffs: list[str] = []
    for rel, expected in UPSTREAM_FROZEN_SHAS.items():
        fp = root / rel
        actual = _sha256(fp)
        if actual != expected:
            upstream_diffs.append(
                f"{rel}: expected {expected}, actual {actual}"
            )
    if upstream_diffs:
        raise RuntimeError(
            "Upstream frozen SHA drift detected. Refusing to run.\n"
            + "\n".join(upstream_diffs)
        )

    # ── Build O51.15 (cross-seed Top-K overlap) ────────────────────────────
    overlap_rows = overlap_mod.build_seed_overlap_table(root)
    overlap_header = [
        "section",
        "group_label",
        "set_a",
        "set_b",
        "intersection_count",
        "union_count",
        "jaccard",
        "extra",
    ]
    overlap_sha = d_writers.write_csv(
        overlap_rows, overlap_header, "seed_overlap_table.csv", root
    )

    # ── Build O51.16 (binary membership matrix) ────────────────────────────
    mm_header, mm_rows = overlap_mod.build_membership_matrix(root)
    mm_sha = d_writers.write_csv(
        mm_rows, mm_header, "worst_case_membership_matrix.csv", root
    )

    # ── Build O51.17 (error concentration) ──────────────────────────────────
    conc_rows = concentration_mod.build_error_concentration_table(root)
    conc_header = [
        "seed",
        "k",
        "sae_top_k",
        "sse_top_k",
        "sae_share",
        "sse_share",
        "global_sae_seed",
        "global_sse_seed",
        "n_population",
        "top_k_source",
    ]
    conc_sha = d_writers.write_csv(
        conc_rows, conc_header, "error_concentration_table.csv", root
    )

    # ── Build O51.20 (hardness vs seed-disagreement + group summary) ─────────
    per_target, summaries = hardness_mod.build_hardness_table(root)
    hardness_header = list(per_target[0].keys()) if per_target else [
        "target_id", "group_label",
        "in_W1_seed42", "in_W1_seed123", "in_W1_seed2026", "in_W2",
        "in_shared_three_seed_worst",
        "computed_mean_abs_error_wh", "computed_median_abs_error_wh",
        "computed_min_abs_error_wh", "computed_max_abs_error_wh",
        "computed_sample_sd_abs_error_wh",
        "phase48_seed_mean_prediction", "phase48_seed_std_prediction",
        "phase48_seed_range_prediction",
    ]
    hardness_sha = d_writers.write_csv(
        per_target, hardness_header, "hardness_vs_seed_disagreement.csv", root
    )

    summary_header = [
        "group_label",
        "source",
        "n_targets",
        "mean_of_mean_abs_error_wh",
        "median_of_mean_abs_error_wh",
        "mean_of_sample_sd_abs_error_wh",
        "mean_of_seed_std_prediction",
        "mean_of_seed_range_prediction",
    ]
    summary_sha = d_writers.write_csv(
        summaries, summary_header, "hardness_group_summary.csv", root
    )

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "phase": 51,
        "subphase": "51-D",
        "version": "WORST_ERROR_ANALYSIS_DIAGNOSTICS-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract_sha256": CANONICAL_SHAS["worst_error_selection_contract.json"],
        "source_working_table_sha256": CANONICAL_SHAS["phase51_target_level_working_table.csv"],
        "phase51_c_canonical_sha256s": {
            k: v for k, v in CANONICAL_SHAS.items()
            if k not in ("worst_error_selection_contract.json",
                         "phase51_target_level_working_table.csv")
        },
        "deliverables": {
            "O51.15_seed_overlap_table": {
                "path": "artifacts/worst_error_analysis/seed_overlap_table.csv",
                "sha256": overlap_sha,
            },
            "O51.16_worst_case_membership_matrix": {
                "path": "artifacts/worst_error_analysis/worst_case_membership_matrix.csv",
                "sha256": mm_sha,
            },
            "O51.17_error_concentration_table": {
                "path": "artifacts/worst_error_analysis/error_concentration_table.csv",
                "sha256": conc_sha,
            },
            "O51.20_hardness_vs_seed_disagreement": {
                "path": "artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv",
                "sha256": hardness_sha,
            },
            "O51.20_hardness_group_summary": {
                "path": "artifacts/worst_error_analysis/hardness_group_summary.csv",
                "sha256": summary_sha,
            },
        },
        "forbidden_actions_status": {
            "overlap_analysis_executed": True,
            "error_concentration_executed": True,
            "regime_enrichment_executed": False,
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
        "ready_for_phase51_e": False,
    }
    manifest_sha = d_writers.write_json(
        manifest, "phase51_d_manifest.json", root
    )

    return {
        "phase": 51,
        "subphase": "51-D",
        "status": "PASS",
        "selection_contract_sha256": CANONICAL_SHAS["worst_error_selection_contract.json"],
        "source_working_table_sha256": CANONICAL_SHAS["phase51_target_level_working_table.csv"],
        "canonical_inputs_unchanged": True,
        "upstream_frozen_unchanged": True,
        "artifacts": {
            "O51.15_seed_overlap_table": overlap_sha,
            "O51.16_worst_case_membership_matrix": mm_sha,
            "O51.17_error_concentration_table": conc_sha,
            "O51.20_hardness_vs_seed_disagreement": hardness_sha,
            "O51.20_hardness_group_summary": summary_sha,
            "phase51_d_manifest.json": manifest_sha,
        },
        "n_rows_overlap": len(overlap_rows),
        "n_rows_membership_matrix": len(mm_rows),
        "n_rows_concentration": len(conc_rows),
        "n_rows_hardness_per_target": len(per_target),
        "n_groups_hardness_summary": len(summaries),
    }


def main(project_root=None) -> dict[str, Any]:
    return materialize_phase51_d(project_root)


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, default=str))
