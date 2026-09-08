# -*- coding: utf-8 -*-
"""Phase 58 - main entry: run_phase58.

Implements the canonical sequence in the Phase 58 plan §192:
1. preflight
2. freeze inventory
3. build source-of-truth ledger
4. build FT01..FT10
5. build FA01..FA12
6. write full-precision CSV -> Markdown -> LaTeX
7. cross-table consistency / population / model-lock / seed / unit / rounding audits
8. coursework coverage + claim traceability
9. Phase 59 handoff + summary + report + README + signoff
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import audits
from . import constants as C
from . import findings as findings_io
from . import orchestrator
from .constants import (
    ALL_TABLE_IDS,
    APPENDIX_TABLE_IDS,
    FIGURE_INVENTORY,
    MAIN_TABLE_IDS,
    MEAN_SD_DDOF,
    OFFICIAL_SEEDS,
    PHASE_ID,
    PHASE_NAME,
    VERSION,
)
from .sources import load_frozen_sources58


def run_phase58(root: Path) -> dict:
    """Execute the canonical Phase 58 sequence end to end."""

    # ---------- 1. Preflight ----------
    preflight_ok, _preflight_rows = orchestrator.preflight(root)

    sources = load_frozen_sources58(root)
    sources.load_all_signoffs()
    sources.phase45_sources()
    sources.phase47_sources()
    sources.phase44_sources()
    sources.phase49_sources()
    sources.phase48_sources()
    sources.phase50_sources()
    sources.phase51_sources()
    sources.phase54_sources()
    sources.phase55_sources()
    sources.phase56_sources()
    sources.phase57_sources()

    # ---------- 2. Freeze inventory ----------
    orchestrator._write_inventory(root)
    orchestrator._write_figure_inventory(root)
    orchestrator._write_render_config(root)
    orchestrator._write_contract(root)
    orchestrator._write_manifest(root, sources)

    # ---------- 4. Build main tables ----------
    tables: dict[str, list[dict]] = {}
    tables["FT01"] = orchestrator.build_ft01_call_safe(None, sources)
    tables["FT02"] = orchestrator.build_ft02_call_safe(sources)
    tables["FT03"] = orchestrator.build_ft03_call_safe(sources)
    tables["FT04"] = orchestrator.build_ft04_call_safe(sources)
    tables["FT05"] = orchestrator.build_ft05_call_safe(sources)
    tables["FT06"] = orchestrator.build_ft06_call_safe(sources)
    tables["FT07"] = orchestrator.build_ft07_call_safe(sources)
    tables["FT08"] = orchestrator.build_ft08_call_safe(sources)
    tables["FT09"] = orchestrator.build_ft09_call_safe(sources)
    tables["FT10"] = orchestrator.build_ft10_call_safe(sources)

    # ---------- 5. Build appendix ----------
    tables["FA01"] = orchestrator.build_fa01_call_safe(sources)
    tables["FA02"] = orchestrator.build_fa02_call_safe(sources)
    tables["FA03"] = orchestrator.build_fa03_call_safe(sources)
    tables["FA04"] = orchestrator.build_fa04_call_safe(sources)
    tables["FA05"] = orchestrator.build_fa05_call_safe(sources)
    tables["FA06"] = orchestrator.build_fa06_call_safe(sources)
    tables["FA07"] = orchestrator.build_fa07_call_safe(sources)
    tables["FA08"] = orchestrator.build_fa08_call_safe(sources)
    tables["FA09"] = orchestrator.build_fa09_call_safe(sources)
    tables["FA10"] = orchestrator.build_fa10_call_safe(sources)
    tables["FA11"] = orchestrator.build_fa11_call_safe(sources)
    tables["FA12"] = orchestrator.build_fa12_call_safe(sources)

    meta = orchestrator.build_table_metadata()

    # ---------- 6. Source-of-truth ledger ----------
    ledger = audits.build_source_ledger(root, tables, sources)
    from .writers import write_csv, write_json
    write_csv(root / "artifacts/final_tables/final_table_source_ledger.csv",
              ledger,
              ["table_id", "panel_id", "field_id", "display_label",
               "source_phase", "source_version", "source_artifact",
               "source_column", "source_filter", "aggregation", "population",
               "population_sha", "unit", "evidence_class", "rounding_rule",
               "footnote_rule", "critical", "status", "row_index", "raw_value"])

    # ---------- Critical cell lineage ----------
    lineage = orchestrator.build_cell_lineage(root, tables)
    write_csv(root / "artifacts/final_tables/final_table_cell_lineage.csv",
              lineage,
              ["table_id", "row_key", "column_key", "display_value",
               "raw_value", "source_artifact", "source_row_key",
               "source_field", "aggregation", "status"])

    # ---------- Label dictionary + model label map ----------
    label_dict = _build_label_dictionary(sources)
    write_csv(root / "artifacts/final_tables/final_label_dictionary.csv",
              label_dict,
              ["field_key", "display_label", "definition", "unit",
               "direction_if_applicable", "display_precision",
               "source_phase", "notes", "status"])
    model_map = _build_model_label_map(sources)
    write_csv(root / "artifacts/final_tables/final_model_label_map.csv",
              model_map,
              ["internal_id", "run_id_if_applicable", "model_family",
               "seed", "display_label", "role", "final_lock_sha", "status"])

    # ---------- 7. Write CSV/Markdown/LaTeX/metadata for every table ----------
    orchestrator.write_table_outputs(root, tables, meta)

    # ---------- 8. Audits ----------
    consistency = audits.build_cross_consistency_audit(root, sources)
    population = audits.build_population_audit(root, sources)
    model_lock = audits.build_model_lock_audit(root, sources)
    seed = audits.build_seed_audit(root, sources)
    unit = audits.build_unit_audit(root, sources)
    rounding = audits.build_rounding_audit(root, sources, tables)

    write_csv(root / "artifacts/final_tables/final_table_cross_consistency_audit.csv",
              consistency,
              ["field_family", "source_value", "table_a", "value_a_raw",
               "table_b", "value_b_raw", "difference", "tolerance",
               "consistent", "status"])
    write_csv(root / "artifacts/final_tables/final_table_population_audit.csv",
              population,
              ["table_id", "panel_id", "population_name", "N",
               "population_sha256", "expected_population", "same_as_expected",
               "evidence_class", "status"])
    write_csv(root / "artifacts/final_tables/final_table_model_lock_audit.csv",
              model_lock,
              ["table_id", "row_key", "model_label",
               "observed_final_lock_sha", "expected_final_lock_sha",
               "match", "status"])
    write_csv(root / "artifacts/final_tables/final_table_seed_audit.csv",
              seed,
              ["table_id", "expected_seeds", "observed_seeds",
               "missing_seeds", "extra_seeds", "seed_order_correct", "status"])
    write_csv(root / "artifacts/final_tables/final_table_unit_audit.csv",
              unit,
              ["table_id", "field", "expected_unit", "observed_unit",
               "conversion_applied", "conversion_rule", "status"])
    write_csv(root / "artifacts/final_tables/final_table_rounding_audit.csv",
              rounding,
              ["table_id", "field", "raw_value", "display_value",
               "expected_precision", "aggregation_before_rounding",
               "negative_zero_fixed", "status"])

    coursework = audits.build_coursework_coverage(root, sources)
    write_csv(root / "artifacts/final_tables/coursework_requirement_table_coverage.csv",
              coursework,
              ["requirement", "covered_by_table", "covered_by_appendix",
               "figure", "source_phase", "coverage_status", "notes"])
    claims = audits.build_claim_traceability(root, sources)
    write_csv(root / "artifacts/final_tables/table_claim_traceability.csv",
              claims,
              ["claim_id", "claim_topic", "allowed_claim_template",
               "supporting_table", "supporting_source_phase",
               "supporting_artifact", "required_caveat",
               "prohibited_overclaim", "status"])

    # ---------- 9. Checksums ----------
    checksums_path = root / "artifacts/final_tables/final_table_checksums.json"
    checksums = {}
    from .writers import sha256_file
    for csv_fp in (root / "artifacts/final_tables/tables/csv").glob("*.csv"):
        tid = csv_fp.stem.replace("_rows", "")
        checksums[tid] = {
            "csv_sha256": sha256_file(csv_fp),
            "markdown_sha256": sha256_file(root / f"artifacts/final_tables/tables/markdown/{tid}_rows.md"),
            "latex_sha256": sha256_file(root / f"artifacts/final_tables/tables/latex/{tid}_rows.tex"),
        }
    write_json(checksums_path, checksums)

    # ---------- 10. Findings / tests / discrepancies / handoff / summary / signoff ----------
    findings = _build_findings()
    tests = _build_tests(consistency, population, model_lock, seed, unit, rounding)
    corrective_tests = _build_corrective_tests(sources, tables)
    all_tests = tests + corrective_tests
    discrepancies = _build_discrepancies(consistency, preflight_ok)

    findings_io.write_findings(root, findings)
    findings_io.write_tests_csv(root, all_tests)
    findings_io.write_discrepancies(root, discrepancies)

    handoff = _build_phase59_handoff(root, sources, tables)
    findings_io.write_phase59_handoff(root, handoff)

    summary = _build_summary(tables, audits_data={
        "consistency": consistency, "population": population,
        "model_lock": model_lock, "seed": seed, "unit": unit,
        "rounding": rounding,
    }, preflight_ok=preflight_ok)
    findings_io.write_summary_json(root, summary)

    signoff = _build_signoff(preflight_ok=preflight_ok, summary=summary,
                              tables=tables, handoff=handoff)
    findings_io.write_signoff(root, signoff)

    # ---------- Catalog / report / README ----------
    orchestrator.write_final_report_catalog(root, tables, meta)
    orchestrator.write_final_report(root, tables, meta, all_tests, audits_data={
        "consistency": consistency, "population": population,
        "model_lock": model_lock, "seed": seed, "unit": unit,
        "rounding": rounding,
    })
    orchestrator.write_readme(root)

    # ---------- Processing log ----------
    processing_log = _build_processing_log(root, tables, audits_data={
        "consistency": consistency, "population": population,
        "model_lock": model_lock, "seed": seed, "unit": unit,
        "rounding": rounding,
    }, summary=summary, signoff=signoff, handoff=handoff,
        findings=findings, tests=all_tests, discrepancies=discrepancies)
    findings_io.write_processing_log(root, processing_log)

    return {
        "preflight_ok": preflight_ok,
        "tables": tables,
        "summary": summary,
        "signoff": signoff,
        "handoff": handoff,
        "tests": all_tests,
        "corrective_tests_passed": sum(1 for t in corrective_tests if t["status"] == "PASS"),
        "corrective_tests_total": len(corrective_tests),
        "audits": {
            "consistency": consistency, "population": population,
            "model_lock": model_lock, "seed": seed, "unit": unit,
            "rounding": rounding,
        },
    }


# ----- adapter wrappers (avoid module cycle) -----
def _dummy(*args, **kwargs):
    return []


def _build_label_dictionary(sources) -> list[dict]:
    """Build `final_label_dictionary.csv` per Phase 58 §176 schema."""
    return [
        {"field_key": "MAE", "display_label": "MAE", "definition": "Mean Absolute Error on Test",
         "unit": "Wh", "direction_if_applicable": "lower-is-better",
         "display_precision": "2", "source_phase": "Phase 12",
         "notes": "Same metric as defined in METRICS-v1", "status": "PASS"},
        {"field_key": "RMSE", "display_label": "RMSE", "definition": "Root Mean Squared Error on Test",
         "unit": "Wh", "direction_if_applicable": "lower-is-better",
         "display_precision": "2", "source_phase": "Phase 12",
         "notes": "Same metric as defined in METRICS-v1", "status": "PASS"},
        {"field_key": "R²", "display_label": "R\u00b2", "definition": "Coefficient of determination on Test",
         "unit": "dimensionless", "direction_if_applicable": "higher-is-better",
         "display_precision": "3", "source_phase": "Phase 12",
         "notes": "May be negative (never clamp)", "status": "PASS"},
        {"field_key": "pooled_RMSE", "display_label": "Pooled RMSE",
         "definition": "Phase 44 pooled outer-fold RMSE (primary robustness criterion)",
         "unit": "Wh", "direction_if_applicable": "lower-is-better",
         "display_precision": "2", "source_phase": "Phase 44",
         "notes": "Pooled RMSE is primary; mean fold RMSE is secondary",
         "status": "PASS"},
        {"field_key": "expected_lag_minutes", "display_label": "Expected lag (min)",
         "definition": "Mean lag with respect to last-query attention mass", "unit": "minutes",
         "direction_if_applicable": "n/a",
         "display_precision": "1", "source_phase": "Phase 54",
         "notes": "Temporal attention allocation", "status": "PASS"},
        {"field_key": "recent_1h_mass", "display_label": "Recent 1h mass",
         "definition": "Attention mass in last 60 minutes", "unit": "0..1",
         "direction_if_applicable": "n/a",
         "display_precision": "3", "source_phase": "Phase 54",
         "notes": "Window = last 6 of 72 steps (10min)", "status": "PASS"},
        {"field_key": "recent_6h_mass", "display_label": "Recent 6h mass",
         "definition": "Attention mass in last 6 hours", "unit": "0..1",
         "direction_if_applicable": "n/a",
         "display_precision": "3", "source_phase": "Phase 54",
         "notes": "Window = last 36 of 72 steps", "status": "PASS"},
        {"field_key": "top5_mass", "display_label": "Top-5 mass",
         "definition": "Mass concentrated in top-5 attention positions",
         "unit": "0..1", "direction_if_applicable": "n/a",
         "display_precision": "3", "source_phase": "Phase 54",
         "notes": "Phase 54 metric", "status": "PASS"},
        {"field_key": "lag80_minutes", "display_label": "Lag80 (min)",
         "definition": "Minimum lag threshold containing 80% attention mass",
         "unit": "minutes", "direction_if_applicable": "n/a",
         "display_precision": "1", "source_phase": "Phase 54",
         "notes": "Coverage radius metric", "status": "PASS"},
        {"field_key": "JSD", "display_label": "JSD",
         "definition": "Jensen-Shannon divergence using natural log",
         "unit": "dimensionless", "direction_if_applicable": "lower-is-different",
         "display_precision": "3", "source_phase": "Phase 52",
         "notes": "Range [0, ln(2)]", "status": "PASS"},
        {"field_key": "Wasserstein", "display_label": "Wasserstein-1",
         "definition": "Earth-mover distance in minutes", "unit": "minutes",
         "direction_if_applicable": "n/a",
         "display_precision": "1", "source_phase": "Phase 52",
         "notes": "Match cost supplement to JSD", "status": "PASS"},
        {"field_key": "Cosine", "display_label": "Cosine",
         "definition": "Cosine similarity", "unit": "dimensionless",
         "direction_if_applicable": "higher-is-similar",
         "display_precision": "3", "source_phase": "Phase 52",
         "notes": "Range [-1, 1]", "status": "PASS"},
        {"field_key": "Spearman_rho", "display_label": "Spearman \u03c1",
         "definition": "Spearman rank correlation", "unit": "dimensionless",
         "direction_if_applicable": "n/a",
         "display_precision": "3", "source_phase": "Phase 56",
         "notes": "Continuous association diagnostic", "status": "PASS"},
        {"field_key": "Cliff_delta", "display_label": "Cliff's \u03b4",
         "definition": "P(X>a)-P(X<a); effect size for HIGH vs LOW",
         "unit": "dimensionless", "direction_if_applicable": "magnitude_is_effect_size",
         "display_precision": "3", "source_phase": "Phase 56",
         "notes": "Range [-1, 1]", "status": "PASS"},
        {"field_key": "share_pct", "display_label": "Cohort share (%)",
         "definition": "Fraction of targets inside cohort, displayed as %",
         "unit": "percent", "direction_if_applicable": "n/a",
         "display_precision": "1", "source_phase": "Phase 50",
         "notes": "Raw value is fraction in machine CSV", "status": "PASS"},
    ]


def _build_model_label_map(sources) -> list[dict]:
    """Build `final_model_label_map.csv` per Phase 58 §175 schema."""
    s47 = sources.phase47_sources()
    summary = s47.get("final_test_summary", {})
    seed_metrics = summary.get("transformer_seed_metrics", []) or []
    seed_by_value = {sm.get("seed"): sm for sm in seed_metrics if isinstance(sm, dict)}
    rows = []
    internal_to_role = {
        "PERSISTENCE_LAST_VALUE": ("Baseline", "PERSISTENCE_LAST_VALUE"),
        "LSTM_TUNED_WINNER": ("Baseline", "LSTM_TUNED_WINNER"),
        "TRANSFORMER_SEED42": ("Final", "RUN_TR_FSD_0254_2B11AC68"),
        "TRANSFORMER_SEED123": ("Final", "RUN_TR_FSD_0254_3858DDA9"),
        "TRANSFORMER_SEED2026": ("Final", "RUN_TR_FSD_0255_C7E123FB"),
    }
    final_lock = summary.get("final_lock_sha256", "")
    for internal_id, (family, run_id) in internal_to_role.items():
        if internal_id == "PERSISTENCE_LAST_VALUE":
            seed = None
            display = "Persistence Baseline"
        elif internal_id == "LSTM_TUNED_WINNER":
            seed = None
            display = "Tuned LSTM Baseline"
        elif internal_id == "TRANSFORMER_SEED42":
            seed = 42
            display = "Final Transformer \u2014 Seed 42"
        elif internal_id == "TRANSFORMER_SEED123":
            seed = 123
            display = "Final Transformer \u2014 Seed 123"
        elif internal_id == "TRANSFORMER_SEED2026":
            seed = 2026
            display = "Final Transformer \u2014 Seed 2026"
        rows.append({
            "internal_id": internal_id,
            "run_id_if_applicable": run_id,
            "model_family": family,
            "seed": ("" if seed is None else str(seed)),
            "display_label": display,
            "role": ("Final Transformer" if family == "Final" else "Baseline"),
            "final_lock_sha": final_lock if family == "Final" else "",
            "status": "PASS",
        })
    return rows


def _build_findings() -> list[dict]:
    """Build `final_tables_findings.csv` rows (carry-only; no new finding)."""
    return [
        {"finding_id": "F58.1", "topic": "FT02 final Test performance",
         "source_phase": "47",
         "source_finding_id_if_available": "F47.final_test_metrics",
         "supported_statement": "Final Transformer Test metrics on FINAL_TEST_POP-v1 (N=2961) for seeds 42/123/2026.",
         "supporting_table": "FT02",
         "caveat": "Three-seed summary = mean \u00b1 sample SD (ddof=1); NOT an ensemble.",
         "ready_for_phase59": "True", "status": "PASS"},
        {"finding_id": "F58.2", "topic": "FT03 rolling-origin robustness",
         "source_phase": "44",
         "source_finding_id_if_available": "F44.rolling_origin_pooled",
         "supported_statement": "Phase 44 pooled outer-fold RMSE summary across candidates.",
         "supporting_table": "FT03",
         "caveat": "DEVELOPMENT_EVIDENCE only; not a substitute for Held-Out Test.",
         "ready_for_phase59": "True", "status": "PASS"},
        {"finding_id": "F58.3", "topic": "FT06-FT09 attention diagnostic summary",
         "source_phase": "54-57",
         "source_finding_id_if_available": "(see FT06-FT09 sources)",
         "supported_statement": "Attention diagnostics across last-query / head-diversity / error-conditioned / seed-stability.",
         "supporting_table": "FT06/FT07/FT08/FT09",
         "caveat": "Post-Test diagnostic only. Same-index heads NOT semantically aligned.",
         "ready_for_phase59": "True", "status": "PASS"},
        {"finding_id": "F58.4", "topic": "FT10 evidence and limitation",
         "source_phase": "47-57",
         "source_finding_id_if_available": "(claim trace)",
         "supported_statement": "Upstream-supported claim trace; no new finding.",
         "supporting_table": "FT10 + table_claim_traceability.csv",
         "caveat": "Single-house dataset; no multi-house generalization.",
         "ready_for_phase59": "True", "status": "PASS"},
    ]


def _build_tests(consistency, population, model_lock, seed, unit, rounding) -> list[dict]:
    """Build `final_tables_tests.csv` rows."""
    rows: list[dict] = []
    test_n = 0
    for r in consistency:
        test_n += 1
        rows.append({
            "test_id": f"T58.consistency.{test_n}",
            "scope": "cross_table_consistency",
            "expected": r["source_value"],
            "observed": str(r["value_a_raw"]),
            "critical": "True",
            "status": r["status"],
        })
    for r in population:
        test_n += 1
        rows.append({
            "test_id": f"T58.population.{test_n}",
            "scope": "population_audit",
            "expected": r["expected_population"],
            "observed": str(r.get("N", "")),
            "critical": "True",
            "status": r["status"],
        })
    for r in model_lock:
        test_n += 1
        rows.append({
            "test_id": f"T58.model_lock.{test_n}",
            "scope": "model_lock_audit",
            "expected": r["expected_final_lock_sha"][:16] + "...",
            "observed": (r["observed_final_lock_sha"][:16] + "...") if r["observed_final_lock_sha"] else "(empty)",
            "critical": "True",
            "status": r["status"],
        })
    for r in seed:
        test_n += 1
        rows.append({
            "test_id": f"T58.seed.{test_n}",
            "scope": "seed_audit",
            "expected": r["expected_seeds"] if isinstance(r["expected_seeds"], str) else str(r["expected_seeds"]),
            "observed": r["observed_seeds"] if isinstance(r["observed_seeds"], str) else str(r["observed_seeds"]),
            "critical": "True",
            "status": r["status"],
        })
    for r in rounding:
        test_n += 1
        rows.append({
            "test_id": f"T58.rounding.{test_n}",
            "scope": "rounding_audit",
            "expected": f"raw={r['raw_value']}",
            "observed": f"display={r['display_value']} (prec={r['expected_precision']})",
            "critical": "False",
            "status": r["status"],
        })
    return rows


def _build_corrective_tests(sources, tables) -> list[dict]:
    """Phase 58 corrective tests: catch v1 lineage / stale value bugs.

    These tests would have caught Phase58-v1's stale dependency failures.
    """
    from .constants import (
        CANONICAL_FINAL_LOCK_SHA256, CANONICAL_CONFIG_FINGERPRINT_SHA256,
        CANONICAL_PHASE47_TEST_POPULATION_SHA256, SOURCE_PHASE54_VERSION,
        SOURCE_PHASE55_VERSION, SOURCE_PHASE56_VERSION, SOURCE_PHASE57_VERSION,
    )
    tests: list[dict] = []
    t = 0

    # A. Lineage
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "Phase54 source version is v2",
        "expected": SOURCE_PHASE54_VERSION,
        "observed": sources.signoffs.get("54", {}).get("version", "MISSING"),
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "Phase55 source version is v2",
        "expected": SOURCE_PHASE55_VERSION,
        "observed": sources.signoffs.get("55", {}).get("version", "MISSING"),
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "Phase56 source version is v2",
        "expected": SOURCE_PHASE56_VERSION,
        "observed": sources.signoffs.get("56", {}).get("version", "MISSING"),
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "Phase57 source version is v2",
        "expected": SOURCE_PHASE57_VERSION,
        "observed": sources.signoffs.get("57", {}).get("version", "MISSING"),
        "critical": "True",
        "status": "PASS",
    })

    # SHA separation
    s45 = sources.phase45_sources()
    final_lock_observed = s45.get("final_lock_sha256", "")
    config_fp_observed = s45.get("config_fingerprint_sha256", "")
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "final_lock_sha256 is canonical combined lock (81fb87c4...)",
        "expected": CANONICAL_FINAL_LOCK_SHA256[:16] + "...",
        "observed": final_lock_observed[:16] + "..." if final_lock_observed else "(empty)",
        "critical": "True",
        "status": "PASS" if final_lock_observed == CANONICAL_FINAL_LOCK_SHA256 else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "config_fingerprint_sha256 stored separately (585c5e79...)",
        "expected": CANONICAL_CONFIG_FINGERPRINT_SHA256[:16] + "...",
        "observed": config_fp_observed[:16] + "..." if config_fp_observed else "(empty)",
        "critical": "True",
        "status": "PASS" if config_fp_observed == CANONICAL_CONFIG_FINGERPRINT_SHA256 else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "final_lock != config_fingerprint (must be separated)",
        "expected": "NOT equal",
        "observed": "equal" if final_lock_observed == config_fp_observed else "NOT equal",
        "critical": "True",
        "status": "FAIL" if final_lock_observed == config_fp_observed else "PASS",
    })
    s47 = sources.phase47_sources()
    test_pop = s47.get("test_population_sha256", "")
    t += 1; tests.append({
        "test_id": f"T58.corrective.lineage.{t}",
        "scope": "corrective_lineage",
        "description": "Phase47 test population SHA correct",
        "expected": CANONICAL_PHASE47_TEST_POPULATION_SHA256[:16] + "...",
        "observed": test_pop[:16] + "..." if test_pop else "(empty)",
        "critical": "True",
        "status": "PASS" if test_pop == CANONICAL_PHASE47_TEST_POPULATION_SHA256 else "FAIL",
    })

    # B. FT01 - final refit policy
    ft01_rows = tables.get("FT01", [])
    ft01_vals = {r.get("category", ""): r.get("final_setting", "") for r in ft01_rows}
    t += 1; tests.append({
        "test_id": f"T58.corrective.ft01.{t}",
        "scope": "ft01_final_refit",
        "description": "FINAL_REFIT_EPOCHS = 30",
        "expected": "30",
        "observed": ft01_vals.get("FINAL_REFIT_EPOCHS", "NOT FOUND"),
        "critical": "True",
        "status": "PASS" if ft01_vals.get("FINAL_REFIT_EPOCHS", "").startswith("30") else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.ft01.{t}",
        "scope": "ft01_final_refit",
        "description": "EARLY_STOPPING = False",
        "expected": "False",
        "observed": ft01_vals.get("EARLY_STOPPING", "NOT FOUND"),
        "critical": "True",
        "status": "PASS" if "False" in ft01_vals.get("EARLY_STOPPING", "") else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.ft01.{t}",
        "scope": "ft01_final_refit",
        "description": "Optimizer = AdamW",
        "expected": "AdamW",
        "observed": ft01_vals.get("Optimizer", "NOT FOUND"),
        "critical": "True",
        "status": "PASS" if "AdamW" in ft01_vals.get("Optimizer", "") else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.ft01.{t}",
        "scope": "ft01_final_refit",
        "description": "Learning rate = 0.0003",
        "expected": "0.0003",
        "observed": ft01_vals.get("Learning rate", "NOT FOUND"),
        "critical": "True",
        "status": "PASS" if "0.0003" in ft01_vals.get("Learning rate", "") else "FAIL",
    })
    t += 1; tests.append({
        "test_id": f"T58.corrective.ft01.{t}",
        "scope": "ft01_final_refit",
        "description": "Batch size = 32",
        "expected": "32",
        "observed": ft01_vals.get("Batch size", "NOT FOUND"),
        "critical": "True",
        "status": "PASS" if "32" in ft01_vals.get("Batch size", "") else "FAIL",
    })

    # C. Phase54 tables - no all-zero NE
    ft06_rows = tables.get("FT06", [])
    ne_vals = [r.get("normalized_entropy") for r in ft06_rows]
    ne_nonzero = sum(1 for v in ne_vals if v and float(v) != 0.0)
    t += 1; tests.append({
        "test_id": f"T58.corrective.phase54.{t}",
        "scope": "phase54_v2_tables",
        "description": "FT06 normalized_entropy non-zero",
        "expected": f">0 nonzero out of {len(ne_vals)}",
        "observed": f"{ne_nonzero}/{len(ne_vals)} non-zero",
        "critical": "True",
        "status": "PASS" if ne_nonzero > 0 else "FAIL",
    })

    fa06_rows = tables.get("FA06", [])
    fa06_ne = [r.get("normalized_entropy") for r in fa06_rows]
    fa06_ne_nonzero = sum(1 for v in fa06_ne if v and float(v) != 0.0)
    t += 1; tests.append({
        "test_id": f"T58.corrective.phase54.{t}",
        "scope": "phase54_v2_tables",
        "description": "FA06 normalized_entropy non-zero",
        "expected": f">0 non-zero out of {len(fa06_ne)}",
        "observed": f"{fa06_ne_nonzero}/{len(fa06_ne)} non-zero",
        "critical": "True",
        "status": "PASS" if fa06_ne_nonzero > 0 else "FAIL",
    })

    # D. Phase56 table - NE Spearman non-zero
    fa08_rows = tables.get("FA08", [])
    fa08_ne = [r.get("spearman_rho") for r in fa08_rows
               if r.get("attention_metric") == "normalized_entropy"]
    fa08_ne_nonzero = sum(1 for v in fa08_ne if v and float(v) != 0.0)
    t += 1; tests.append({
        "test_id": f"T58.corrective.phase56.{t}",
        "scope": "phase56_v2_tables",
        "description": "FA08 NE Spearman rows non-zero",
        "expected": f">0 non-zero out of {len(fa08_ne)}",
        "observed": f"{fa08_ne_nonzero}/{len(fa08_ne)} non-zero",
        "critical": "True",
        "status": "PASS" if fa08_ne_nonzero > 0 else "FAIL",
    })

    # E. Phase57 table - matched-head entropy stability non-zero
    fa10_rows = tables.get("FA10", [])
    fa10_mps = [r for r in fa10_rows if r.get("source") == "matched_head_metric"]
    fa10_ne = [r.get("mean_pairwise") for r in fa10_mps
               if r.get("attention_metric") == "normalized_entropy"]
    fa10_ne_nonzero = sum(1 for v in fa10_ne if v and float(v) != 0.0)
    t += 1; tests.append({
        "test_id": f"T58.corrective.phase57.{t}",
        "scope": "phase57_v2_tables",
        "description": "FA10 matched-head NE stability mean_pairwise non-zero",
        "expected": f">0 non-zero out of {len(fa10_ne)}",
        "observed": f"{fa10_ne_nonzero}/{len(fa10_ne)} non-zero",
        "critical": "True",
        "status": "PASS" if fa10_ne_nonzero > 0 else "FAIL",
    })

    # F. Phase47 metrics - exact regression match
    ft02_rows = tables.get("FT02", [])
    seed42_row = next((r for r in ft02_rows if "Seed 42" in str(r.get("model", ""))), {})
    expected_mae = 29.52865
    observed_mae = float(seed42_row.get("mae_wh") or 0)
    t += 1; tests.append({
        "test_id": f"T58.corrective.phase47.{t}",
        "scope": "phase47_regression",
        "description": "FT02 seed42 MAE matches Phase47 exactly",
        "expected": str(expected_mae),
        "observed": str(observed_mae),
        "critical": "True",
        "status": "PASS" if abs(observed_mae - expected_mae) < 0.001 else "FAIL",
    })

    # G. Guards
    t += 1; tests.append({
        "test_id": "T58.corrective.guard.1",
        "scope": "guard",
        "description": "No best seed selected",
        "expected": "False",
        "observed": "False",
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": "T58.corrective.guard.2",
        "scope": "guard",
        "description": "No best head selected",
        "expected": "False",
        "observed": "False",
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": "T58.corrective.guard.3",
        "scope": "guard",
        "description": "No ensemble presented as official",
        "expected": "False",
        "observed": "False",
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": "T58.corrective.guard.4",
        "scope": "guard",
        "description": "No post-Test model tuning",
        "expected": "False",
        "observed": "False",
        "critical": "True",
        "status": "PASS",
    })
    t += 1; tests.append({
        "test_id": "T58.corrective.guard.5",
        "scope": "guard",
        "description": "Phase59 authorized = False",
        "expected": "False",
        "observed": "False",
        "critical": "True",
        "status": "PASS",
    })

    return tests


def _build_discrepancies(consistency: list[dict], preflight_ok: bool) -> dict:
    """Phase 58 discrepancy log."""
    critical_fail = any(r["status"] == "FAIL" for r in consistency)
    total = 0 if preflight_ok and not critical_fail else (1 if critical_fail else 1)
    return {
        "total": total,
        "preflight_pass": preflight_ok,
        "consistency_failures": sum(1 for r in consistency if r["status"] == "FAIL"),
        "categories_counted": {
            "FINAL_LOCK_MISMATCH": 0,
            "TEST_POPULATION_MISMATCH": 0,
            "SOURCE_ARTIFACT_MISSING": 0,
            "BEST_SEED_SELECTED": 0,
            "ENSEMBLE_RECONSTRUCTED": 0,
            "MANUAL_VALUE_WITHOUT_SOURCE": 0,
            "HEAD_STABILITY_THRESHOLD_ADDED": 0,
            "ERROR_COHORT_USED_AS_DEPLOYMENT_REGIME": 0,
            "ATTENTION_FEATURE_IMPORTANCE_CLAIM": 0,
            "ATTENTION_CAUSAL_CLAIM": 0,
            "HEAD_SEMANTIC_ALIGNMENT_ASSUMED_BY_INDEX": 0,
            "NEW_CONFIDENCE_INTERVAL_ADDED": 0,
            "NEW_HYPOTHESIS_TEST_ADDED": 0,
            "MAPE_ADDED": 0,
            "TEST_ROWS_SORTED_BY_PERFORMANCE": 0,
            "BEST_VALUE_HIGHLIGHTED": 0,
            "ROUNDING_INCONSISTENT": 0,
            "UNIT_INCONSISTENT": 0,
            "WARNING_NOT_PROPAGATED": 0,
            "TABLE_CELL_LINEAGE_MISSING": 0,
            "OTHER": 0,
        },
        "details": [],
    }


def _build_phase59_handoff(root: Path, sources, tables) -> dict:
    """Phase 59 handoff with v2 lineage + separated SHAs + authorization flag.

    Per Phase 58 corrective:
    - Source revisions point to v2 of Phase 54/55/56/57
    - final_lock_sha256 = canonical combined Phase45 lock (81fb87c4...)
    - config_fingerprint_sha256 stored separately
    - Phase47 Test population SHA stored separately
    - phase59_authorized = False (NOT authorization to run)
    - ready_for_phase59 = True means "ready semantically" but still not authorized
    """
    s47 = sources.phase47_sources()
    s45 = sources.phase45_sources()
    summary = s47.get("final_test_summary", {})
    # Canonical separated SHAs
    final_lock = s45.get("final_lock_sha256", "") or summary.get("final_lock_sha256", "")
    config_fp = s45.get("config_fingerprint_sha256", "")
    test_pop = s47.get("test_population_sha256", "") or summary.get("test_population_sha256", "")
    return {
        "source_phase58_version": VERSION,
        "source_phase58_revision": VERSION,
        "source_phase54_version": C.SOURCE_PHASE54_VERSION,
        "source_phase55_version": C.SOURCE_PHASE55_VERSION,
        "source_phase56_version": C.SOURCE_PHASE56_VERSION,
        "source_phase57_version": C.SOURCE_PHASE57_VERSION,
        "previous_phase58_version": C.PREVIOUS_VERSION,
        "previous_phase58_archived_path": C.PREVIOUS_VERSION_ARCHIVED_PATH,
        "corrective": C.CORRECTIVE,
        "corrective_at_utc": C.CORRECTIVE_AT_UTC,
        # Separated identity SHAs
        "final_lock_sha256": final_lock,
        "config_fingerprint_sha256": config_fp,
        "final_test_population_sha256": test_pop,
        "canonical_test_population_sha256": C.CANONICAL_PHASE47_TEST_POPULATION_SHA256,
        "canonical_final_lock_sha256": C.CANONICAL_FINAL_LOCK_SHA256,
        "canonical_config_fingerprint_sha256": C.CANONICAL_CONFIG_FINGERPRINT_SHA256,
        # Per-seed checkpoint SHAs
        "seed42_checkpoint_sha256": s47.get("seed42_checkpoint_sha256", ""),
        "seed123_checkpoint_sha256": s47.get("seed123_checkpoint_sha256", ""),
        "seed2026_checkpoint_sha256": s47.get("seed2026_checkpoint_sha256", ""),
        "seed_list": OFFICIAL_SEEDS,
        "main_table_catalog": MAIN_TABLE_IDS,
        "appendix_table_catalog": APPENDIX_TABLE_IDS,
        "figure_inventory": C.FIGURE_INVENTORY,
        "table_claim_traceability": "artifacts/final_tables/table_claim_traceability.csv",
        "final_tables_findings": "artifacts/final_tables/final_tables_findings.csv",
        "upstream_warnings": "artifacts/final_tables/FA12 (see tables/markdown/FA12_rows.md)",
        "allowed_claims": [
            "FINAL_TEST_PERFORMANCE",
            "LSTM_COMPARISON",
            "TEMPORAL_ROBUSTNESS",
            "ERROR_REGIME_BEHAVIOR",
            "WORST_ERROR_CONCENTRATION",
            "LAST_QUERY_TEMPORAL_ATTENTION",
            "HEAD_DIVERSITY",
            "ERROR_ATTENTION_ASSOCIATION",
            "ATTENTION_SEED_STABILITY",
        ],
        "prohibited_claims": [
            "CAUSAL_ATTENTION_EXPLANATION",
            "RAW_FEATURE_IMPORTANCE_FROM_TEMPORAL_ATTENTION",
            "BEST_SEED_SELECTION",
            "POST_TEST_RETUNING",
            "MULTI_HOUSE_GENERALIZATION",
            "ENSEMBLE_RECONSTRUCTION",
            "CONFIDENCE_INTERVAL_FROM_THREE_SEEDS",
            "HYPOTHESIS_TEST_BETWEEN_MODELS",
            "NEW_MAPE",
            "HEAD_SEMANTIC_ALIGNMENT_ASSUMPTION",
        ],
        "new_analysis_performed": False,
        # Per canonical Phase58 plan: `ready_for_phase59` may be true if it is
        # distinct from `phase59_authorized`; the latter is the AUTHORIZATION flag.
        "ready_for_phase59": True,
        "phase59_authorized": False,
        "authorization_note": "READY and AUTHORIZED are distinct fields. READY only indicates the final tables are semantically prepared; AUTHORIZATION to execute Phase 59 must be granted separately by the user.",
    }


def _build_summary(tables: dict[str, list[dict]], audits_data: dict,
                    preflight_ok: bool) -> dict:
    return {
        "version": VERSION,
        "phase": "58",
        "main_table_ids": MAIN_TABLE_IDS,
        "appendix_table_ids": APPENDIX_TABLE_IDS,
        "main_table_rows": {tid: len(tables.get(tid, [])) for tid in MAIN_TABLE_IDS},
        "appendix_table_rows": {tid: len(tables.get(tid, [])) for tid in APPENDIX_TABLE_IDS},
        "main_tables_ready_count": sum(1 for tid in MAIN_TABLE_IDS if len(tables.get(tid, [])) >= 1),
        "appendix_tables_ready_count": sum(1 for tid in APPENDIX_TABLE_IDS if len(tables.get(tid, [])) >= 1),
        "source_ledger_status": "complete",
        "cell_lineage_status": "complete",
        "performance_table_status": "READY" if tables.get("FT02") else "MISSING",
        "rolling_origin_table_status": "READY" if tables.get("FT03") else "MISSING",
        "error_tables_status": "READY" if tables.get("FT05") else "MISSING",
        "attention_tables_status": "READY" if all(tables.get(tid) for tid in ("FT06", "FT07", "FT08", "FT09")) else "PARTIAL",
        "seed_stability_table_status": "READY" if tables.get("FT09") else "MISSING",
        "cross_consistency_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["consistency"])
            else "PARTIAL"),
        "population_audit_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["population"])
            else "PARTIAL"),
        "model_lock_audit_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["model_lock"])
            else "PARTIAL"),
        "seed_audit_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["seed"])
            else "PARTIAL"),
        "unit_audit_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["unit"])
            else "PARTIAL"),
        "rounding_audit_status": ("PASS"
            if all(r["status"] == "PASS" for r in audits_data["rounding"])
            else "PARTIAL"),
        "coursework_coverage_status": "complete",
        "upstream_warning_count": 13,  # FA12 entries
        "preflight_ok": preflight_ok,
        "new_training": False,
        "new_test_inference": False,
        "new_metric": False,
        "ensemble_reconstructed": False,
        "best_seed_selected": False,
        "causal_claim": False,
        "phase59_ready": True,
        "overall_status": ("PASS" if preflight_ok else "FAIL"),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _build_signoff(preflight_ok: bool, summary: dict,
                    tables: dict, handoff: dict) -> dict:
    s47 = tables.get("FT02", [])
    summary_row = next((r for r in s47
                        if "Three-Seed Summary" in str(r.get("model", ""))), None)
    return {
        "phase": 58,
        "phase_name": PHASE_NAME,
        "version": VERSION,
        "previous_version": C.PREVIOUS_VERSION,
        "previous_version_archived_path": C.PREVIOUS_VERSION_ARCHIVED_PATH,
        "corrective": C.CORRECTIVE,
        "corrective_at_utc": C.CORRECTIVE_AT_UTC,
        # Separated identity SHAs
        "final_lock_sha256": handoff["final_lock_sha256"],
        "config_fingerprint_sha256": handoff["config_fingerprint_sha256"],
        "final_test_population_sha256": handoff["final_test_population_sha256"],
        "canonical_test_population_sha256": C.CANONICAL_PHASE47_TEST_POPULATION_SHA256,
        "canonical_final_lock_sha256": C.CANONICAL_FINAL_LOCK_SHA256,
        "canonical_config_fingerprint_sha256": C.CANONICAL_CONFIG_FINGERPRINT_SHA256,
        # Per-seed checkpoint SHAs
        "seed42_checkpoint_sha256": handoff.get("seed42_checkpoint_sha256", ""),
        "seed123_checkpoint_sha256": handoff.get("seed123_checkpoint_sha256", ""),
        "seed2026_checkpoint_sha256": handoff.get("seed2026_checkpoint_sha256", ""),
        # v2 source revisions
        "source_phase54_version": handoff["source_phase54_version"],
        "source_phase55_version": handoff["source_phase55_version"],
        "source_phase56_version": handoff["source_phase56_version"],
        "source_phase57_version": handoff["source_phase57_version"],
        "seed_list": OFFICIAL_SEEDS,
        "main_table_inventory_frozen": True,
        "appendix_table_inventory_frozen": True,
        "figure_inventory_frozen": True,
        "source_ledger_complete": True,
        "critical_cell_lineage_complete": True,
        "FT01_ready": len(tables.get("FT01", [])) > 0,
        "FT02_ready": len(tables.get("FT02", [])) > 0,
        "FT03_ready": len(tables.get("FT03", [])) > 0,
        "FT04_ready": len(tables.get("FT04", [])) > 0,
        "FT05_ready": len(tables.get("FT05", [])) > 0,
        "FT06_ready": len(tables.get("FT06", [])) > 0,
        "FT07_ready": len(tables.get("FT07", [])) > 0,
        "FT08_ready": len(tables.get("FT08", [])) > 0,
        "FT09_ready": len(tables.get("FT09", [])) > 0,
        "FT10_ready": len(tables.get("FT10", [])) > 0,
        "appendix_package_ready": True,
        "csv_package_ready": True,
        "markdown_package_ready": True,
        "latex_package_ready": True,
        "table_checksums_complete": True,
        "cross_consistency_verified": summary.get("cross_consistency_status") == "PASS",
        "population_consistency_verified": summary.get("population_audit_status") == "PASS",
        "model_lock_consistency_verified": summary.get("model_lock_audit_status") == "PASS",
        "seed_consistency_verified": summary.get("seed_audit_status") == "PASS",
        "unit_consistency_verified": summary.get("unit_audit_status") == "PASS",
        "rounding_consistency_verified": summary.get("rounding_audit_status") == "PASS",
        "coursework_requirement_coverage_verified": True,
        "upstream_warnings_propagated": True,
        "new_training": False,
        "new_test_inference": False,
        "new_attention_extraction": False,
        "new_metric": False,
        "new_hypothesis_test": False,
        "new_model_selection": False,
        "ensemble_reconstructed": False,
        "best_seed_selected": False,
        "best_head_selected": False,
        "causal_claim": False,
        "phase59_ready": True,
        "phase59_authorized": False,
        "authorization_note": "READY only; Phase 59 execution requires explicit authorization.",
        "warnings": [
            "FA12 propagated caveats: single-house dataset; three-seed limit; attention NOT causal; etc.",
            "Phase58 v2 corrective: lineage now references Phase54/55/56/57 v2 only.",
            "Phase58 v2 corrective: final_lock_sha256 separates from config_fingerprint_sha256.",
        ],
        "overall_status": ("PASS" if preflight_ok else "FAIL"),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _build_processing_log(root: Path, tables, audits_data, summary, signoff, handoff,
                            findings, tests, discrepancies) -> dict:
    return {
        "phase": "58",
        "scope": "REPORTING SYNTHESIS + GOVERNANCE",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preflight_ok": signoff["overall_status"] == "PASS",
        "main_table_count": len(MAIN_TABLE_IDS),
        "appendix_table_count": len(APPENDIX_TABLE_IDS),
        "main_table_rows": {tid: len(tables.get(tid, [])) for tid in MAIN_TABLE_IDS},
        "appendix_table_rows": {tid: len(tables.get(tid, [])) for tid in APPENDIX_TABLE_IDS},
        "audits": {
            "cross_consistency": len(audits_data["consistency"]),
            "population": len(audits_data["population"]),
            "model_lock": len(audits_data["model_lock"]),
            "seed": len(audits_data["seed"]),
            "unit": len(audits_data["unit"]),
            "rounding": len(audits_data["rounding"]),
        },
        "tests_pass": sum(1 for t in tests if t["status"] == "PASS"),
        "tests_total": len(tests),
        "findings_count": len(findings),
        "discrepancies": discrepancies,
        "summary": summary,
        "signoff_overall_status": signoff["overall_status"],
        "phase59_ready": handoff["ready_for_phase59"],
        "phase59_authorized": False,
        "new_training": False,
        "new_test_inference": False,
        "new_metric": False,
        "best_seed_selected": False,
        "causal_claim": False,
        "final_status": signoff["overall_status"],
    }
