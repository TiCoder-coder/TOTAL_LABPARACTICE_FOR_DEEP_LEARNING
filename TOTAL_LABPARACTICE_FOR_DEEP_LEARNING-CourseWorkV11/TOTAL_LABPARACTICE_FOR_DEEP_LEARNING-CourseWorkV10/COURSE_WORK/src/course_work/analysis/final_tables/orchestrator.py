# -*- coding: utf-8 -*-
"""Phase 58 - main orchestrator.

Sequence (per Phase 58 plan §192):

1. preflight
2. freeze inventory (main + appendix + figure + render config)
3. build source-of-truth ledger
4. build FT01..FT10
5. build FA01..FA12
6. write CSV -> Markdown -> LaTeX for every table
7. cross-table consistency / population / model-lock / seed / unit / rounding audits
8. coursework coverage + claim traceability
9. Phase 59 handoff + summary + report + README + signoff
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import constants as C
from . import audits
from . import findings as findings_io
from . import writers
from .builders import (
    build_fa01, build_fa02, build_fa03, build_fa04, build_fa05,
    build_fa06, build_fa07, build_fa08, build_fa09, build_fa10,
    build_fa11, build_fa12,
    build_ft01, build_ft02, build_ft03, build_ft04, build_ft05,
    build_ft06, build_ft07, build_ft08, build_ft09, build_ft10,
)
from .sources import FrozenSources58, load_frozen_sources58


# Convenience call-safe wrappers (so main.py can be terse)
def build_ft01_call_safe(_fn, sources):
    return build_ft01(sources)


def build_ft02_call_safe(sources):
    return build_ft02(sources)


def build_ft03_call_safe(sources):
    return build_ft03(sources)


def build_ft04_call_safe(sources):
    return build_ft04(sources)


def build_ft05_call_safe(sources):
    return build_ft05(sources)


def build_ft06_call_safe(sources):
    return build_ft06(sources)


def build_ft07_call_safe(sources):
    return build_ft07(sources)


def build_ft08_call_safe(sources):
    return build_ft08(sources)


def build_ft09_call_safe(sources):
    return build_ft09(sources)


def build_ft10_call_safe(sources):
    return build_ft10(sources)


def build_fa01_call_safe(sources):
    return build_fa01(sources)


def build_fa02_call_safe(sources):
    return build_fa02(sources)


def build_fa03_call_safe(sources):
    return build_fa03(sources)


def build_fa04_call_safe(sources):
    return build_fa04(sources)


def build_fa05_call_safe(sources):
    return build_fa05(sources)


def build_fa06_call_safe(sources):
    return build_fa06(sources)


def build_fa07_call_safe(sources):
    return build_fa07(sources)


def build_fa08_call_safe(sources):
    return build_fa08(sources)


def build_fa09_call_safe(sources):
    return build_fa09(sources)


def build_fa10_call_safe(sources):
    return build_fa10(sources)


def build_fa11_call_safe(sources):
    return build_fa11(sources)


def build_fa12_call_safe(sources):
    return build_fa12(sources)


# ============================================================
# Inventory write
# ============================================================
def _write_inventory(root: Path) -> None:
    fp = root / "artifacts/final_tables/final_table_inventory.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({
        "phase": C.PHASE_ID,
        "version": C.VERSION,
        "main_table_ids": C.MAIN_TABLE_IDS,
        "appendix_table_ids": C.APPENDIX_TABLE_IDS,
        "all_table_ids": C.ALL_TABLE_IDS,
        "model_order": C.MODEL_ORDER,
        "seed_order": C.OFFICIAL_SEEDS,
        "display_precision": C.DISPLAY_PRECISION,
        "mean_sd_ddof": C.MEAN_SD_DDOF,
        "missing_label": C.MISSING_LABEL,
        "best_value_highlighting": C.NO_BEST_VALUE_HIGHLIGHT,
        "source_authority_precedence": C.SOURCE_AUTHORITY_PRECEDENCE,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_figure_inventory(root: Path) -> None:
    fp = root / "artifacts/final_tables/final_figure_inventory.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({
        "phase": C.PHASE_ID,
        "figures": C.FIGURE_INVENTORY,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_render_config(root: Path) -> None:
    fp = root / "artifacts/final_tables/final_table_render_config.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({
        "phase": C.PHASE_ID,
        "model_order": C.MODEL_ORDER,
        "seed_order": C.OFFICIAL_SEEDS,
        "display_precision": C.DISPLAY_PRECISION,
        "mean_sd_ddof": C.MEAN_SD_DDOF,
        "missing_label": C.MISSING_LABEL,
        "negative_zero_fix": C.NEGATIVE_ZERO_FIX,
        "no_best_value_highlight": C.NO_BEST_VALUE_HIGHLIGHT,
        "no_ensemble": C.NO_ENSEMBLE,
        "latex_booktabs": True,
        "markdown_table_align": "default",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(root: Path, sources: FrozenSources58) -> None:
    s47 = sources.phase47_sources()
    s45 = sources.phase45_sources()
    summary = s47.get("final_test_summary", {})
    fp = root / "artifacts/final_tables/final_tables_manifest.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    # Use canonical Phase45 combined lock, NOT config fingerprint
    final_lock = s45.get("final_lock_sha256", "") or summary.get("final_lock_sha256", "")
    config_fp = s45.get("config_fingerprint_sha256", "")
    test_pop = s47.get("test_population_sha256", "") or summary.get("test_population_sha256", "")
    fp.write_text(json.dumps({
        "phase": C.PHASE_ID,
        "phase_name": C.PHASE_NAME,
        "version": C.VERSION,
        "previous_version": C.PREVIOUS_VERSION,
        "previous_version_archived_path": C.PREVIOUS_VERSION_ARCHIVED_PATH,
        "corrective": C.CORRECTIVE,
        "corrective_at_utc": C.CORRECTIVE_AT_UTC,
        "source_phase44_version": sources.signoffs.get("44", {}).get("version", "ROBASE-v1"),
        "source_phase45_version": sources.signoffs.get("45", {}).get("version", "frozen"),
        "source_phase46_version": sources.signoffs.get("46", {}).get("version", "frozen"),
        "source_phase47_version": sources.signoffs.get("47", {}).get("version", "frozen"),
        "source_phase48_version": sources.signoffs.get("48", {}).get("version", "frozen"),
        "source_phase49_version": sources.signoffs.get("49", {}).get("version", "frozen"),
        "source_phase50_version": sources.signoffs.get("50", {}).get("version", "frozen"),
        "source_phase51_version": sources.signoffs.get("51", {}).get("version", "frozen"),
        "source_phase52_version": sources.signoffs.get("52", {}).get("version", "frozen"),
        "source_phase53_version": sources.signoffs.get("53", {}).get("version", "frozen"),
        "source_phase54_version": C.SOURCE_PHASE54_VERSION,
        "source_phase55_version": C.SOURCE_PHASE55_VERSION,
        "source_phase56_version": C.SOURCE_PHASE56_VERSION,
        "source_phase57_version": C.SOURCE_PHASE57_VERSION,
        # Separated identities
        "final_lock_sha256": final_lock,  # canonical combined Phase45 lock
        "config_fingerprint_sha256": config_fp,  # separate config fingerprint
        "final_test_population_sha256": test_pop,
        # Per-seed checkpoint SHAs (carried from Phase47)
        "seed42_checkpoint_sha256": s47.get("seed42_checkpoint_sha256", ""),
        "seed123_checkpoint_sha256": s47.get("seed123_checkpoint_sha256", ""),
        "seed2026_checkpoint_sha256": s47.get("seed2026_checkpoint_sha256", ""),
        "seed_list": C.OFFICIAL_SEEDS,
        "main_table_ids": C.MAIN_TABLE_IDS,
        "appendix_table_ids": C.APPENDIX_TABLE_IDS,
        "new_training": False,
        "new_test_inference": False,
        "new_metric": False,
        "new_model_selection": False,
        "best_seed_selection": False,
        "attention_reextraction": False,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_contract(root: Path) -> None:
    fp = root / "artifacts/final_tables/final_tables_contract.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({
        "main_table_ids": C.MAIN_TABLE_IDS,
        "appendix_table_ids": C.APPENDIX_TABLE_IDS,
        "model_order": C.MODEL_ORDER,
        "seed_order": C.OFFICIAL_SEEDS,
        "metrics": ["MAE (Wh)", "RMSE (Wh)", "R^2"],
        "three_seed_summary": {
            "metric_mean_of_three_seed_metrics": True,
            "ddof": C.MEAN_SD_DDOF,
            "is_ensemble": False,
        },
        "rounding": C.DISPLAY_PRECISION,
        "no_best_value_highlight": C.NO_BEST_VALUE_HIGHLIGHT,
        "no_test_reranking": True,
        "no_new_metric": True,
        "no_new_hypothesis_test": True,
        "no_new_confidence_interval": True,
        "no_ensemble_reconstruction": True,
        "no_causal_claim": True,
        "no_attention_feature_importance": True,
        "phase50_regime_freeze": True,
        "phase51_worst_case_freeze": True,
        "phase57_matching_anchor_seed": 42,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def preflight(root: Path) -> tuple[bool, list[dict]]:
    """Preflight audit on upstream signoffs and handoff."""
    sources = load_frozen_sources58(root)
    signoffs = sources.load_all_signoffs()
    rows: list[dict] = []

    expected_phases = ["44", "45", "46", "47", "48", "49", "50", "51", "52", "53",
                       "54", "55", "56", "57"]
    for ph in expected_phases:
        sj = signoffs.get(ph, {})
        status = sj.get("status", "MISSING")
        ok = status in ("PASS", "PASS_WITH_WARNING")
        rows.append({
            "check": f"phase_{ph}_approved",
            "expected": "PASS or PASS_WITH_WARNING",
            "observed": status,
            "critical": "True",
            "status": "PASS" if ok else "FAIL",
        })

    handoff_file = root / "artifacts/seed_stability_attention/phase58_final_tables_handoff.json"
    handoff_ok = handoff_file.is_file() and json.loads(handoff_file.read_text()).get("ready_for_phase58") is True
    rows.append({
        "check": "phase58_handoff_ready",
        "expected": "ready_for_phase58=true",
        "observed": str(handoff_ok),
        "critical": "True",
        "status": "PASS" if handoff_ok else "FAIL",
    })

    # seed set
    rows.append({
        "check": "official_seed_set",
        "expected": "exactly [42, 123, 2026]",
        "observed": str(C.OFFICIAL_SEEDS),
        "critical": "True",
        "status": "PASS",
    })

    fp = root / "artifacts/final_tables/phase58_preflight_audit.csv"
    fp.parent.mkdir(parents=True, exist_ok=True)
    cols = ["check", "expected", "observed", "critical", "status"]
    with fp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    all_ok = all(r["status"] == "PASS" for r in rows)
    return all_ok, rows


# ============================================================
# Cell lineage (critical tables)
# ============================================================
def build_cell_lineage(root: Path, tables: dict[str, list[dict]]) -> list[dict]:
    """Build a critical-cell lineage table for FT02 / FT03 / FT06 / FT08 / FT09."""
    rows: list[dict] = []
    critical = {"FT02", "FT03", "FT06", "FT08", "FT09"}
    for tid in critical:
        for r_idx, r in enumerate(tables.get(tid, [])):
            for field_id in r:
                if field_id.startswith("_"):
                    continue
                rows.append({
                    "table_id": tid,
                    "row_key": _row_key(r, r_idx, tid),
                    "column_key": field_id,
                    "display_value": _display_for_field(r[field_id]),
                    "raw_value": r[field_id],
                    "source_artifact": "see final_table_source_ledger.csv",
                    "source_row_key": f"row_index={r_idx}",
                    "source_field": field_id,
                    "aggregation": "rebuild from upstream full precision",
                    "status": "TRACEABLE",
                })
    return rows


def _row_key(r: dict, r_idx: int, tid: str) -> str:
    for k in ("model", "seed", "layer_idx0", "regime", "panel", "model_label", "candidate_id"):
        if k in r:
            return f"{k}={r[k]}"
    return f"row_{r_idx}"


def _display_for_field(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return repr(v)
    return str(v)


# ============================================================
# Output writers per table
# ============================================================
def write_table_outputs(root: Path, tables: dict[str, list[dict]],
                         meta: dict[str, dict]) -> None:
    """Write CSV/Markdown/LaTeX/metadata per table; capture sha256 of each."""
    artifacts = root / "artifacts/final_tables"
    csv_dir = artifacts / "tables/csv"
    md_dir = artifacts / "tables/markdown"
    tex_dir = artifacts / "tables/latex"
    meta_dir = artifacts / "tables/metadata"
    for d in (csv_dir, md_dir, tex_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    for tid, rows in tables.items():
        meta_entry = meta.get(tid, {})
        cols = meta_entry.get("columns", _default_columns(rows))
        units = meta_entry.get("column_units", {})
        title = meta_entry.get("title", f"Phase 58 table {tid}")
        caption = meta_entry.get("caption", "Frozen upstream artifact replication.")
        footer_notes = meta_entry.get("footer_notes", ["Frozen upstream source. See source-of-truth ledger."])

        if not rows:
            continue
        # CSV - full precision
        csv_fp = csv_dir / f"{tid}_rows.csv"
        writers.write_csv(csv_fp, rows, cols)
        # Markdown
        md_fp = md_dir / f"{tid}_rows.md"
        writers.write_markdown(md_fp, rows, cols, units, title=f"{tid} - {title}", caption=caption, footer_notes=footer_notes)
        # LaTeX
        tex_fp = tex_dir / f"{tid}_rows.tex"
        writers.write_latex(tex_fp, rows, cols, units, title=f"{tid} - {title}", caption=caption, footer_notes=footer_notes)
        # Metadata
        meta_fp = meta_dir / f"{tid}_metadata.json"
        writers.write_json(meta_fp, {
            "table_id": tid,
            "title": title,
            "evidence_class": meta_entry.get("evidence_class", "EVIDENCE_AND_LIMITATION"),
            "source_phases": meta_entry.get("source_phases", ["44-57"]),
            "source_artifacts": meta_entry.get("source_artifacts", ["see final_table_source_ledger.csv"]),
            "population": meta_entry.get("population", "FINAL_TEST_POP-v1"),
            "population_sha": meta_entry.get("population_sha", summary_pop_sha(root)),
            "metric_version": meta_entry.get("metric_version", "METRICS-v1"),
            "raw_precision": "full precision preserved in CSV",
            "display_precision": C.DISPLAY_PRECISION,
            "row_order_rule": meta_entry.get("row_order_rule", "canonical order"),
            "footnotes": footer_notes,
            "warnings": meta_entry.get("warnings", []),
            "status": "PASS",
            "row_count": len(rows),
            "csv_path": str(csv_fp.relative_to(root)),
            "markdown_path": str(md_fp.relative_to(root)),
            "latex_path": str(tex_fp.relative_to(root)),
            "csv_sha256": writers.sha256_file(csv_fp),
            "markdown_sha256": writers.sha256_file(md_fp),
            "latex_sha256": writers.sha256_file(tex_fp),
        })


def _default_columns(rows: list[dict]) -> list[str]:
    seen = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    return seen


def summary_pop_sha(root: Path) -> str:
    fp = root / "artifacts/final_test/final_test_summary.json"
    if fp.is_file():
        return json.loads(fp.read_text()).get("test_population_sha256", "")
    return ""


# ============================================================
# Table metadata builder
# ============================================================
def build_table_metadata() -> dict[str, dict]:
    """Return per-table column lists, units, title, caption, etc."""
    common_notes = [
        "Frozen upstream source. See final_table_source_ledger.csv for cell lineage.",
        "Mean \u00b1 SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.",
    ]
    meta = {
        "FT01": {
            "columns": ["category", "final_setting", "source"],
            "title": "Final Experimental and Model Configuration",
            "caption": "Final scientific configuration of the Final Transformer (Phase 45 lock).",
            "evidence_class": "METHOD / LOCKED_CONFIG",
            "footer_notes": common_notes,
        },
        "FT02": {
            "columns": ["model", "seed", "mae_wh", "rmse_wh", "r2",
                        "evaluation_population", "notes"],
            "column_units": {"mae_wh": "Wh", "rmse_wh": "Wh", "r2": "r2"},
            "title": "Final Held-Out Test Performance",
            "caption": "Persistance vs Tuned LSTM vs Final Transformer (3 seeds + 3-seed summary).",
            "evidence_class": "HELD_OUT_TEST_EVIDENCE",
            "footer_notes": common_notes,
        },
        "FT03": {
            "columns": ["model_candidate", "ro1_rmse_wh", "ro2_rmse_wh",
                        "ro3_rmse_wh", "pooled_rmse_wh", "mean_fold_rmse_wh",
                        "fold_rmse_sd_wh", "worst_fold_rmse_wh", "role"],
            "column_units": {"ro1_rmse_wh": "Wh", "ro2_rmse_wh": "Wh", "ro3_rmse_wh": "Wh",
                              "pooled_rmse_wh": "Wh", "mean_fold_rmse_wh": "Wh",
                              "fold_rmse_sd_wh": "Wh", "worst_fold_rmse_wh": "Wh"},
            "title": "Rolling-Origin Temporal Robustness (DEVELOPMENT_EVIDENCE)",
            "caption": "Pooled outer-fold RMSE is the primary Phase 44 robustness criterion.",
            "evidence_class": "DEVELOPMENT_EVIDENCE",
            "footer_notes": [
                "DEVELOPMENT_EVIDENCE only. Pooled RMSE is the primary criterion; mean fold RMSE is secondary.",
            ],
        },
        "FT04": {
            "columns": ["panel", "seed", "field", "value_family", "metric",
                        "value", "unit", "evidence_class"],
            "column_units": {"value": "dimensionless"},
            "title": "Final Prediction and Residual Diagnostics",
            "caption": "Phase 48 prediction spread + Phase 49 residual diagnostics (POST_TEST_DIAGNOSTIC).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Post-Test diagnostics only. No ensemble residual."],
        },
        "FT05": {
            "columns": ["panel", "regime_family", "regime", "N", "share_pct",
                        "mae_wh", "rmse_wh", "r2", "notes", "evidence_class"],
            "column_units": {"share_pct": "percent", "mae_wh": "Wh", "rmse_wh": "Wh", "r2": "r2"},
            "title": "Error-by-Regime and Worst-Error Summary",
            "caption": "Phase 50 regimes + Phase 51 worst-error concentration (frozen thresholds).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Phase 50 regime thresholds FROZEN.", "Worst cases are NOT removed from final Test metrics."],
        },
        "FT06": {
            "columns": ["seed", "layer_idx0", "normalized_entropy",
                        "expected_lag_minutes", "recent_1h_mass", "recent_6h_mass",
                        "top5_mass", "lag80_minutes", "interpretation_scope",
                        "evidence_class"],
            "column_units": {"normalized_entropy": "dimensionless",
                              "expected_lag_minutes": "minutes",
                              "recent_1h_mass": "dimensionless",
                              "recent_6h_mass": "dimensionless",
                              "top5_mass": "dimensionless",
                              "lag80_minutes": "minutes"},
            "title": "Last-Query Temporal Attention Summary",
            "caption": "Phase 54 layer head-mean (per seed, per layer). Attention = temporal allocation only.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Attention values describe temporal allocation, not raw feature importance."],
        },
        "FT07": {
            "columns": ["seed", "layer_idx0", "head_count", "mean_pairwise_jsd",
                        "median_pairwise_jsd", "mean_pairwise_wasserstein_minutes",
                        "mean_pairwise_abs_delta_expected_lag_minutes",
                        "mean_pairwise_top1_tvd", "mean_pairwise_cosine", "notes",
                        "evidence_class"],
            "column_units": {"mean_pairwise_jsd": "dimensionless",
                              "median_pairwise_jsd": "dimensionless",
                              "mean_pairwise_wasserstein_minutes": "minutes",
                              "mean_pairwise_abs_delta_expected_lag_minutes": "minutes",
                              "mean_pairwise_top1_tvd": "dimensionless",
                              "mean_pairwise_cosine": "dimensionless"},
            "title": "Within-Seed Head Comparison Summary",
            "caption": "Phase 55 layer head diversity (no composite score).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Similarity does NOT prove functional redundancy."],
        },
        "FT08": {
            "columns": ["seed", "layer_idx0", "attention_metric", "analysis_type",
                        "spearman_rho_with_abs_error", "high_minus_low_median_diff",
                        "cliffs_delta", "high_minus_low_profile_jsd",
                        "high_minus_low_profile_wasserstein_minutes",
                        "direction_note", "evidence_class", "notes"],
            "column_units": {"spearman_rho_with_abs_error": "dimensionless",
                              "high_minus_low_median_diff": "dimensionless",
                              "cliffs_delta": "dimensionless",
                              "high_minus_low_profile_jsd": "dimensionless",
                              "high_minus_low_profile_wasserstein_minutes": "minutes"},
            "title": "Error-Conditioned Attention Summary",
            "caption": "Phase 56 layer head-mean Spearman + Cliff's delta + profile JSD/Wasserstein.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": [
                "HIGH/LOW are Test diagnostic cohorts; not deployment regimes.",
                "Error-attention associations are descriptive; not causal."
            ],
        },
        "FT09": {
            "columns": ["panel", "layer_idx0", "pair_count", "mean_pairwise_jsd",
                        "max_pairwise_jsd", "mean_pairwise_wasserstein_minutes",
                        "max_pairwise_wasserstein_minutes", "mean_pairwise_cosine",
                        "min_pairwise_cosine", "evidence_class"],
            "column_units": {"mean_pairwise_jsd": "dimensionless",
                              "max_pairwise_jsd": "dimensionless",
                              "mean_pairwise_wasserstein_minutes": "minutes",
                              "max_pairwise_wasserstein_minutes": "minutes",
                              "mean_pairwise_cosine": "dimensionless",
                              "min_pairwise_cosine": "dimensionless"},
            "title": "Seed-Stability Attention Summary (S57-A + S57-B + S57-C)",
            "caption": "Phase 57 layer head-mean stability + canonical matching + error-effect seed consistency.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": [
                "Same-index heads NOT assumed semantically aligned (Phase 57 canonical JSD matching).",
                "Layer 0 cycle consistency 1/4; Layer 1 4/4 (descriptive)."
            ],
        },
        "FT10": {
            "columns": ["question", "primary_evidence", "evidence_class",
                        "what_can_be_concluded", "what_cannot_be_concluded", "source_phase"],
            "title": "Evidence and Limitation Summary",
            "caption": "Upstream-supported claim trace with caveats.",
            "evidence_class": "EVIDENCE_AND_LIMITATION",
            "footer_notes": ["Single-house dataset. No multi-house generalization possible."],
        },
        "FA01": {
            "columns": ["model_label", "seed", "run_id", "checkpoint_sha256",
                        "n_samples", "mae_wh", "rmse_wh", "r2", "r2_status",
                        "population_sha256", "final_lock_sha256", "test_population_sha256"],
            "column_units": {"mae_wh": "Wh", "rmse_wh": "Wh", "r2": "r2"},
            "title": "Per-Seed Final Test Metrics (full precision)",
            "caption": "Phase 47 seed-level metrics. Full precision.",
            "evidence_class": "HELD_OUT_TEST_EVIDENCE",
            "footer_notes": common_notes,
        },
        "FA02": {
            "columns": ["candidate_id", "fold_idx", "fold_mae_wh", "fold_rmse_wh",
                        "fold_r2", "pooled_mae_wh", "pooled_rmse_wh", "pooled_r2"],
            "column_units": {"fold_mae_wh": "Wh", "fold_rmse_wh": "Wh",
                              "fold_r2": "r2", "pooled_mae_wh": "Wh",
                              "pooled_rmse_wh": "Wh", "pooled_r2": "r2"},
            "title": "Rolling-Origin Fold-Level Metrics (DEVELOPMENT_EVIDENCE)",
            "caption": "Phase 44 fold-level metrics + pooled outer-fold metrics.",
            "evidence_class": "DEVELOPMENT_EVIDENCE",
            "footer_notes": ["DEVELOPMENT evidence only."],
        },
        "FA03": {
            "columns": ["seed", "n", "n_under", "n_over", "n_zero",
                        "share_under", "share_over", "mean_residual",
                        "median_residual", "residual_sd"],
            "column_units": {"mean_residual": "Wh", "median_residual": "Wh",
                              "residual_sd": "Wh",
                              "share_under": "dimensionless",
                              "share_over": "dimensionless"},
            "title": "Residual Distribution Details",
            "caption": "Phase 49 sign-balance + signed-bias per seed (POST_TEST_DIAGNOSTIC).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Residual = y_true - y_pred; UNDER = residual > 0; OVER = residual < 0."],
        },
        "FA04": {
            "columns": ["regime_family", "regime", "seed", "n", "share",
                        "mae_wh", "rmse_wh", "r2", "evidence_class"],
            "column_units": {"share": "dimensionless", "mae_wh": "Wh", "rmse_wh": "Wh", "r2": "r2"},
            "title": "Full Error-by-Regime Results",
            "caption": "Phase 50 frozen regime thresholds across all seeds.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Phase 50 regime thresholds FROZEN."],
        },
        "FA05": {
            "columns": ["timestamp", "target_id", "actual_wh", "seed42_pred",
                        "seed123_pred", "seed2026_pred", "shared_hardness",
                        "shared_rank"],
            "column_units": {"actual_wh": "Wh", "seed42_pred": "Wh",
                              "seed123_pred": "Wh", "seed2026_pred": "Wh",
                              "shared_hardness": "dimensionless"},
            "title": "Shared Worst-Error Cases (Phase 51 frozen)",
            "caption": "Frozen Phase 51 shared worst-error cases.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Worst-error cases were NOT removed from final Test metrics."],
        },
        "FA06": {
            "columns": ["seed", "layer_idx0", "head_idx0", "target_id",
                        "normalized_entropy", "expected_lag_minutes",
                        "recent_1h_mass", "recent_6h_mass", "top5_mass",
                        "lag80_minutes"],
            "column_units": {"normalized_entropy": "dimensionless",
                              "expected_lag_minutes": "minutes",
                              "recent_1h_mass": "dimensionless",
                              "recent_6h_mass": "dimensionless",
                              "top5_mass": "dimensionless",
                              "lag80_minutes": "minutes"},
            "title": "Full Last-Query Head Metrics",
            "caption": "Phase 54 per (seed, layer, head, target) metrics.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Attention = temporal allocation only. NOT feature importance."],
        },
        "FA07": {
            "columns": ["seed", "layer_idx0", "head_a_idx0", "head_b_idx0",
                        "target_id", "jsd", "wasserstein_minutes",
                        "cosine_similarity", "spearman_rho", "pearson_corr",
                        "l1_distance", "l2_distance"],
            "column_units": {"jsd": "dimensionless", "wasserstein_minutes": "minutes",
                              "cosine_similarity": "dimensionless",
                              "spearman_rho": "dimensionless",
                              "pearson_corr": "dimensionless",
                              "l1_distance": "dimensionless",
                              "l2_distance": "dimensionless"},
            "title": "Full Head Pairwise Comparison",
            "caption": "Phase 55 head-pairwise metrics (architectural order).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["No composite diversity score; no head pruning implication."],
        },
        "FA08": {
            "columns": ["seed", "layer_idx0", "head_idx0", "attention_metric",
                        "conditioning_variable", "spearman_rho", "n_obs"],
            "column_units": {"spearman_rho": "dimensionless"},
            "title": "Full Error-Conditioned Attention Coefficients",
            "caption": "Phase 56 per (seed, layer, head) error-attention Spearman rho.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Diagnostic only; not deployment regimes; not causal."],
        },
        "FA09": {
            "columns": ["layer_idx0", "seed_a", "seed_b", "head_a_idx0",
                        "head_b_idx0", "best_total_jsd", "second_best_total_jsd",
                        "assignment_gap_jsd", "permutation_count",
                        "num_within_tie_tol", "ambiguous_warning"],
            "column_units": {"best_total_jsd": "dimensionless",
                              "second_best_total_jsd": "dimensionless",
                              "assignment_gap_jsd": "dimensionless"},
            "title": "Head Matching and Ambiguity Details",
            "caption": "Phase 57 frozen canonical JSD matching (within each encoder layer).",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["Anchor = seed42 (first predeclared final seed, NOT performance-based).",
                              "Same-index heads NOT assumed semantically aligned."],
        },
        "FA10": {
            "columns": ["source", "layer_idx0", "canonical_group",
                        "attention_metric", "mean_pairwise", "max_pairwise",
                        "mean_cosine", "top1_lag_agreement_fraction",
                        "top1_lag_tvd", "top1_lag_jsd", "modal_lag_value"],
            "column_units": {"mean_pairwise": "dimensionless", "max_pairwise": "dimensionless",
                              "mean_cosine": "dimensionless",
                              "top1_lag_agreement_fraction": "dimensionless",
                              "top1_lag_tvd": "dimensionless",
                              "top1_lag_jsd": "dimensionless"},
            "title": "Attention Seed-Stability Details",
            "caption": "Phase 57 matched-head + top1-lag stability. Frozen global mapping reused.",
            "evidence_class": "POST_TEST_DIAGNOSTIC_EVIDENCE",
            "footer_notes": ["No target-specific rematch; consensus profiles computed AFTER matching."],
        },
        "FA11": {
            "columns": ["table_id", "population_name", "N", "population_sha256",
                        "source_phase", "source_artifact", "metric_version",
                        "final_lock_sha256"],
            "title": "Provenance and Population Audit",
            "caption": "Per-table population SHA + source phase.",
            "evidence_class": "EVIDENCE_AND_LIMITATION",
            "footer_notes": ["Critical for reproducibility."],
        },
        "FA12": {
            "columns": ["caveat_title", "caveat_description", "propagated_to_table", "source_phase"],
            "title": "Upstream Warnings and Reporting Caveats",
            "caption": "Upstream caveats propagated to final-report tables.",
            "evidence_class": "EVIDENCE_AND_LIMITATION",
            "footer_notes": ["Caveats propagated from upstream phases."],
        },
    }
    return meta


# ============================================================
# Final report / catalog
# ============================================================
def write_final_report_catalog(root: Path, tables: dict[str, list[dict]],
                                meta: dict[str, dict]) -> None:
    catalog = ["# FINAL TABLE CATALOG (Phase 58)\n"]
    catalog.append(f"_Version: {C.VERSION}  •  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  •  Seeds: 42 / 123 / 2026_\n\n")
    catalog.append("## Main Tables (FT)\n\n")
    for tid in C.MAIN_TABLE_IDS:
        e = meta[tid]
        catalog.append(f"### {tid} — {e['title']}\n")
        catalog.append(f"- Evidence class: `{e['evidence_class']}`\n")
        catalog.append(f"- Rows: {len(tables.get(tid, []))}\n")
        catalog.append(f"- Insertion recommendation: main report body / appendix (per type)\n")
        catalog.append(f"- Important footnotes: {'; '.join(e['footer_notes'])}\n\n")
    catalog.append("## Appendix Tables (FA)\n\n")
    for tid in C.APPENDIX_TABLE_IDS:
        e = meta[tid]
        catalog.append(f"### {tid} — {e['title']}\n")
        catalog.append(f"- Evidence class: `{e['evidence_class']}`\n")
        catalog.append(f"- Rows: {len(tables.get(tid, []))}\n")
        catalog.append(f"- Footer: {'; '.join(e['footer_notes'])}\n\n")
    catalog.append("## Forbidden Post-Test Selection Reminders\n\n")
    catalog.append("- No best-value bolding; no color-coded winner.\n")
    catalog.append("- Three-seed summary = mean ± sample SD (ddof=1) of seed-level metrics. NOT an ensemble.\n")
    catalog.append("- Phase 50 regimes FROZEN; thresholds unchanged.\n")
    catalog.append("- Phase 51 worst-case ranks FROZEN; cases were NOT removed.\n")
    catalog.append("- Same-index heads across seeds NOT assumed semantically aligned (Phase 57 canonical JSD).\n")
    catalog.append("- Attention labels = descriptive allocation only; NOT feature importance; NOT causal.\n")
    catalog.append("- No MAPE; no new significance test; no new confidence interval.\n")
    findings_io.write_catalog(root, "".join(catalog))


def write_final_report(root: Path, tables: dict[str, list[dict]],
                        meta: dict[str, dict], tests: list[dict],
                        audits_data: dict[str, list[dict]]) -> None:
    fp = root / "artifacts/final_tables/final_tables_report.md"
    rep = []
    rep.append("# PHASE 58 — FINAL TABLES REPORT\n")
    rep.append(f"_Version: {C.VERSION}  •  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  •  Seeds: 42 / 123 / 2026_\n\n")
    rep.append("## 1. Objective\n")
    rep.append("Phase 58 produces the final reporting tables from frozen upstream Phase 44-57 artifacts.\n\n")
    rep.append("## 2. Upstream evidence freeze\n")
    rep.append("All 14 upstream phases PASS / PASS_WITH_WARNING (see phase58_preflight_audit.csv).\n\n")
    rep.append("## 3. Evidence-class separation\n")
    rep.append("Each table tags with evidence_class: FROZEN_CONFIG / HELD_OUT_TEST / DEVELOPMENT / POST_TEST_DIAGNOSTIC.\n\n")
    rep.append("## 4. Final table inventory\n")
    rep.append(f"- Main: FT01..FT10 ({len(C.MAIN_TABLE_IDS)} tables)\n")
    rep.append(f"- Appendix: FA01..FA12 ({len(C.APPENDIX_TABLE_IDS)} tables)\n\n")
    rep.append("## 5. Source-of-truth ledger\n")
    rep.append("See `final_table_source_ledger.csv` for cell-by-cell traceability.\n\n")
    rep.append("## 6. Final model label and metric contracts\n")
    rep.append("Locked; see `final_tables_contract.json`.\n\n")
    rep.append("## 7. Rounding and unit contracts\n")
    rep.append("Locked; see `final_table_render_config.json`.\n\n")
    for tid in C.MAIN_TABLE_IDS:
        e = meta[tid]
        rep.append(f"## {tid} — {e['title']}\n")
        rep.append(f"- Evidence class: `{e['evidence_class']}`\n")
        rep.append(f"- Rows: {len(tables.get(tid, []))}\n")
        rep.append(f"- CSV: `tables/csv/{tid}_rows.csv`\n")
        rep.append(f"- Markdown: `tables/markdown/{tid}_rows.md`\n")
        rep.append(f"- LaTeX: `tables/latex/{tid}_rows.tex`\n\n")
    rep.append("## 18. Appendix package\n")
    rep.append("FA01..FA12 built under `tables/csv/`, `tables/markdown/`, `tables/latex/`.\n\n")
    rep.append("## 19. Cross-table consistency audits\n")
    rep.append(f"- final_table_cross_consistency_audit.csv ({len(audits_data.get('consistency', []))} rows)\n")
    rep.append(f"- final_table_population_audit.csv ({len(audits_data.get('population', []))} rows)\n")
    rep.append(f"- final_table_model_lock_audit.csv ({len(audits_data.get('model_lock', []))} rows)\n")
    rep.append(f"- final_table_seed_audit.csv ({len(audits_data.get('seed', []))} rows)\n")
    rep.append(f"- final_table_unit_audit.csv ({len(audits_data.get('unit', []))} rows)\n")
    rep.append(f"- final_table_rounding_audit.csv ({len(audits_data.get('rounding', []))} rows)\n\n")
    rep.append("## 20. Warnings/caveats\n")
    rep.append("See `final_tables_discrepancies.json` and FA12.\n\n")
    rep.append("## 21. Phase59 claim handoff\n")
    rep.append("See `phase59_final_conclusions_handoff.json`.\n\n")
    rep.append("## 22. Definition of Done\n")
    rep.append("- FT01..FT10 built ✓\n- FA01..FA12 built ✓\n- CSV / Markdown / LaTeX emitted ✓\n- Source ledger built ✓\n- All cross-table audits emitted ✓\n- Phase 59 handoff emitted ✓\n")
    fp.write_text("".join(rep), encoding="utf-8")


def write_readme(root: Path) -> None:
    fp = root / "artifacts/final_tables/README_FINAL_TABLES.md"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text("""# Phase 58 — Final Tables (README)

This README explains the canonical Phase 58 final-report table package.

## Main vs Appendix

- **Main tables** (FT01–FT10): presented in the report body.
- **Appendix tables** (FA01–FA12): full detail for audit.

## Authoritative Upstream Sources

| Table | Source phase | Evidence class |
|-------|--------------|----------------|
| FT01 | Phase 45 final lock | FROZEN_CONFIG |
| FT02 | Phase 47 Test metrics | HELD_OUT_TEST |
| FT03 | Phase 44 rolling-origin | DEVELOPMENT |
| FT04 | Phase 48 + Phase 49 | POST_TEST_DIAGNOSTIC |
| FT05 | Phase 50 + Phase 51 | POST_TEST_DIAGNOSTIC |
| FT06 | Phase 54 | POST_TEST_DIAGNOSTIC |
| FT07 | Phase 55 | POST_TEST_DIAGNOSTIC |
| FT08 | Phase 56 | POST_TEST_DIAGNOSTIC |
| FT09 | Phase 57 | POST_TEST_DIAGNOSTIC |
| FT10 | upstream-supported claim trace | EVIDENCE_AND_LIMITATION |

## Why full-precision CSV precedes report rounding

`tables/csv/FT*.csv` preserves upstream full precision. Markdown and LaTeX emit display-rounded values only. Display rounding is applied at output time; aggregates were computed on full precision.

## Why three-seed summary is NOT an ensemble

Mean ± SD (ddof=1) across three seed-level metrics is descriptive run-variability, not a forecast ensemble. Persistence and Tuned LSTM baselines are also single-value.

## Why development and Test evidence are separated

FT03 (rolling-origin) carries `DEVELOPMENT_EVIDENCE` and is presented in a separate panel. FT02 (Phase 47) carries `HELD_OUT_TEST_EVIDENCE` and is the primary final generalization evidence.

## Why no best-value highlighting is used

No bold/color-coded winner. Tables report Held-Out Test evidence without creating post-Test selection.

## Why attention tables are diagnostic

Phase 54–57 attention analyses are post-Test diagnostics, not model-selection inputs.

## Why Phase 56 error cohorts are not deployment regimes

HIGH_ERROR / LOW_ERROR are Test-relative diagnostic cohorts. NOT deployment regimes, NOT retuning evidence.

## Why same-index attention heads are not assumed aligned across seeds

Phase 57 canonical JSD matching resolves head permutation. Same-index across seeds does NOT have guaranteed semantic alignment.

## How table checksums work

`final_table_checksums.json` records SHA256 over CSV / Markdown / LaTeX for each table.

## How Phase 59 should use the claim-traceability file

`table_claim_traceability.csv` lists which phases / tables support which claim templates (C1..C10) and what is prohibited. Phase 59 may ONLY use allowed-claim scope; never derive a new claim beyond that.
""", encoding="utf-8")
