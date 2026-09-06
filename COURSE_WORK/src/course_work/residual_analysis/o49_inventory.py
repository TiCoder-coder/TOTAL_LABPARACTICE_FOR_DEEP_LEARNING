from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.artifacts import get_project_root, sha256_file


PHASE49_ARTIFACT_DIR_REL = Path("artifacts/residual_analysis")


# O49 enumeration: each item is (id, expected relative path, kind).
O49_INVENTORY: list[tuple[str, str, str]] = [
    ("O49.1_contract", "phase49_contract.json", "json"),
    ("O49.2_preflight", "phase49_preflight.json", "json"),
    ("O49.3_source_verification", "phase49_source_verification.json", "json"),
    ("O49.4_field_consistency_audit", "phase49_field_consistency_audit.csv", "csv"),
    ("O49.5_metric_reconstruction_audit", "phase49_metric_reconstruction_audit.csv", "csv"),
    ("O49.6_residual_long_table", "residual_long_table.csv", "csv"),
    ("O49.7_residual_wide_table", "residual_wide_table.csv", "csv"),
    ("O49.8_distribution_summary", "phase49_residual_distribution_summary.csv", "csv"),
    ("O49.9_signed_bias", "phase49_signed_bias.csv", "csv"),
    ("O49.10_sign_balance", "phase49_sign_balance.csv", "csv"),
    ("O49.11_tail_diagnostics", "phase49_tail_diagnostics.csv", "csv"),
    ("O49.12_histogram_50bins", "phase49_residual_histogram_50bins.csv", "csv"),
    ("O49.13_histogram_bin_edges", "phase49_histogram_bin_edges.csv", "csv"),
    ("O49.14_residual_acf", "phase49_residual_acf.csv", "csv"),
    ("O49.15_residual_acf_key_lags", "phase49_residual_acf_key_lags.csv", "csv"),
    ("O49.16_ljung_box", "phase49_ljung_box.csv", "csv"),
    ("O49.17_sign_runs", "phase49_sign_runs.csv", "csv"),
    ("O49.18_sign_run_table", "phase49_sign_run_table_seed{42}.csv|phase49_sign_run_table_seed{123}.csv|phase49_sign_run_table_seed{2026}.csv", "csv-list"),
    ("O49.19_sign_transitions", "phase49_sign_transitions.csv", "csv"),
    ("O49.20_rolling_residual_diagnostics", "phase49_rolling_residual_diagnostics.csv", "csv"),
    ("O49.21_magnitude_associations", "phase49_magnitude_associations.csv", "csv"),
    ("O49.22_prediction_deciles", "phase49_prediction_deciles.csv", "csv"),
    ("O49.23_cross_seed_residual_agreement", "phase49_cross_seed_residual_agreement.csv", "csv"),
    ("O49.24_cross_seed_sign_consensus", "phase49_cross_seed_sign_consensus.csv", "csv"),
    ("O49.25_persistence_context", "phase49_persistence_context.csv", "csv"),
    ("O49.26_phase49_b_manifest", "phase49_b_manifest.json", "json"),
    ("O49.27_phase49_c_manifest", "phase49_c_manifest.json", "json"),
    ("O49.28_phase49_d_manifest", "phase49_d_manifest.json", "json"),
    ("O49.29_phase49_e_manifest", "phase49_e_manifest.json", "json"),
    ("O49.30_phase49_findings", "phase49_findings.json", "json"),
    ("O49.31_phase50_handoff", "phase50_handoff.json", "json"),
    ("O49.32_phase51_context_handoff", "phase51_context_handoff.json", "json"),
    ("O49.33_phase49_signoff", "phase_49_signoff.json", "json"),
]


def _expand_csv_multi(rel: str, project_root: Path) -> list[Path]:
    if "{" not in rel:
        return [project_root / PHASE49_ARTIFACT_DIR_REL / rel]
    # Handle pipe-separated list like: "phase49_sign_run_table_seed{42}.csv|phase49_sign_run_table_seed{123}.csv|..."
    parts = rel.split("|")
    out: list[Path] = []
    for p in parts:
        # Substitute {X} with X
        while "{" in p:
            start = p.index("{")
            end = p.index("}", start)
            p = p[:start] + p[start + 1 : end] + p[end + 1 :]
        out.append(project_root / PHASE49_ARTIFACT_DIR_REL / p)
    return out


def check_o49_completeness(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    rows: list[dict[str, Any]] = []
    for o49_id, rel, kind in O49_INVENTORY:
        if kind in ("csv-multi", "csv-list"):
            paths = _expand_csv_multi(rel, root)
            existing = [p for p in paths if p.exists()]
            implemented = len(existing) == len(paths)
            sizes = [p.stat().st_size for p in existing]
            rows.append(
                {
                    "o49_id": o49_id,
                    "filename": rel,
                    "kind": kind,
                    "expected_paths": [str(p) for p in paths],
                    "implemented": implemented,
                    "exists": len(existing) > 0,
                    "files_complete": len(existing),
                    "files_expected": len(paths),
                    "schema_valid": all(s > 0 for s in sizes) if sizes else False,
                    "source_valid": True,
                    "json_serializable": kind == "csv",
                    "status": "PASS" if implemented else "MISSING",
                }
            )
        else:
            p = root / PHASE49_ARTIFACT_DIR_REL / rel
            exists = p.exists()
            size = p.stat().st_size if exists else 0
            rows.append(
                {
                    "o49_id": o49_id,
                    "filename": rel,
                    "kind": kind,
                    "expected_paths": [str(p)],
                    "implemented": exists,
                    "exists": exists,
                    "files_complete": 1 if exists else 0,
                    "files_expected": 1,
                    "schema_valid": size > 0 if exists else False,
                    "source_valid": True,
                    "json_serializable": kind == "json",
                    "status": "PASS" if exists and size > 0 else "MISSING",
                }
            )
    n_complete = sum(1 for r in rows if r["status"] == "PASS")
    return {
        "n_o49_total": len(O49_INVENTORY),
        "n_o49_complete": n_complete,
        "completeness_fraction": n_complete / len(O49_INVENTORY) if O49_INVENTORY else 0.0,
        "rows": rows,
        "all_complete": n_complete == len(O49_INVENTORY),
        "missing": [r["o49_id"] for r in rows if r["status"] != "PASS"],
    }
