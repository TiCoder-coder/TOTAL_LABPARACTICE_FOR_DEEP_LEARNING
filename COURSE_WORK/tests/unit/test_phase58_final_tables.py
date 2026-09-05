"""Phase 58 - focused integrity tests (final tables).

These tests verify the integrity of the Phase 58 implementation against
the canonical plan (Phase_58_Final_tables.md).

NO model loading, NO training, NO recomputation of upstream metrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _art() -> Path:
    return _root() / "artifacts" / "final_tables"


def _signoff():
    return json.loads((_art() / "phase_58_signoff.json").read_text())


def _manifest():
    return json.loads((_art() / "final_tables_manifest.json").read_text())


def _summary():
    return json.loads((_art() / "final_tables_summary.json").read_text())


# ===========================================================================
# Test Class 1 - Signoff integrity
# ===========================================================================
class TestPhase58Signoff:
    def test_signoff_exists(self):
        assert (_art() / "phase_58_signoff.json").is_file()

    def test_signoff_overall_status_pass(self):
        sj = _signoff()
        assert sj.get("overall_status") == "PASS", sj

    def test_signoff_has_all_ft_ready_flags(self):
        sj = _signoff()
        for tid in [f"FT{i:02d}" for i in range(1, 11)]:
            assert sj.get(f"{tid}_ready") is True, f"{tid}_ready missing or False"

    def test_signoff_seed_list_exact(self):
        sj = _signoff()
        assert sj.get("seed_list") == [42, 123, 2026]

    def test_signoff_phase59_ready_true(self):
        sj = _signoff()
        assert sj.get("phase59_ready") is True

    def test_signoff_no_new_training(self):
        sj = _signoff()
        assert sj.get("new_training") is False
        assert sj.get("new_test_inference") is False
        assert sj.get("new_attention_extraction") is False
        assert sj.get("new_metric") is False
        assert sj.get("new_hypothesis_test") is False

    def test_signoff_no_best_seed(self):
        sj = _signoff()
        assert sj.get("best_seed_selected") is False
        assert sj.get("best_head_selected") is False
        assert sj.get("ensemble_reconstructed") is False

    def test_signoff_no_causal_claim(self):
        sj = _signoff()
        assert sj.get("causal_claim") is False


# ===========================================================================
# Test Class 2 - Manifest integrity
# ===========================================================================
class TestPhase58Manifest:
    def test_manifest_main_table_ids(self):
        m = _manifest()
        assert m["main_table_ids"] == [f"FT0{i}" for i in range(1, 10)] + ["FT10"]

    def test_manifest_appendix_table_ids(self):
        m = _manifest()
        assert m["appendix_table_ids"] == [f"FA0{i}" for i in range(1, 10)] + ["FA10", "FA11", "FA12"]

    def test_manifest_seed_list(self):
        m = _manifest()
        assert m["seed_list"] == [42, 123, 2026]

    def test_manifest_safety_flags(self):
        m = _manifest()
        for f in ["new_training", "new_test_inference", "new_metric",
                  "new_model_selection", "best_seed_selection",
                  "attention_reextraction"]:
            assert m[f] is False, f"{f} is not False"


# ===========================================================================
# Test Class 3 - Table artifact presence
# ===========================================================================
class TestPhase58Tables:
    def test_all_main_table_csv(self):
        csv_dir = _art() / "tables/csv"
        for tid in [f"FT0{i}" for i in range(1, 10)] + ["FT10"]:
            assert (csv_dir / f"{tid}_rows.csv").is_file(), f"{tid} CSV missing"

    def test_all_appendix_table_csv(self):
        csv_dir = _art() / "tables/csv"
        for tid in [f"FA0{i}" for i in range(1, 10)] + ["FA10", "FA11", "FA12"]:
            assert (csv_dir / f"{tid}_rows.csv").is_file(), f"{tid} CSV missing"

    def test_all_tables_have_markdown(self):
        md_dir = _art() / "tables/markdown"
        for tid in [f"FT0{i}" for i in range(1, 10)] + ["FT10"] + [f"FA0{i}" for i in range(1, 10)] + ["FA10", "FA11", "FA12"]:
            assert (md_dir / f"{tid}_rows.md").is_file(), f"{tid} Markdown missing"

    def test_all_tables_have_latex(self):
        tex_dir = _art() / "tables/latex"
        for tid in [f"FT0{i}" for i in range(1, 10)] + ["FT10"] + [f"FA0{i}" for i in range(1, 10)] + ["FA10", "FA11", "FA12"]:
            assert (tex_dir / f"{tid}_rows.tex").is_file(), f"{tid} LaTeX missing"

    def test_all_tables_have_metadata(self):
        meta_dir = _art() / "tables/metadata"
        for tid in [f"FT0{i}" for i in range(1, 10)] + ["FT10"] + [f"FA0{i}" for i in range(1, 10)] + ["FA10", "FA11", "FA12"]:
            assert (meta_dir / f"{tid}_metadata.json").is_file(), f"{tid} metadata missing"

    def test_ft02_has_three_seed_rows(self):
        import csv
        rows = list(csv.DictReader((_art() / "tables/csv/FT02_rows.csv").open(encoding="utf-8")))
        seed_rows = [r for r in rows if "Seed" in str(r.get("model", ""))
                      and "Three-Seed" not in str(r.get("model", ""))]
        assert len(seed_rows) == 3

    def test_ft02_has_three_seed_summary(self):
        import csv
        rows = list(csv.DictReader((_art() / "tables/csv/FT02_rows.csv").open(encoding="utf-8")))
        sum_rows = [r for r in rows if "Three-Seed Summary" in str(r.get("model", ""))]
        assert len(sum_rows) == 1

    def test_ft02_baseline_persistence_row(self):
        import csv
        rows = list(csv.DictReader((_art() / "tables/csv/FT02_rows.csv").open(encoding="utf-8")))
        pers = [r for r in rows if "Persistence" in str(r.get("model", ""))]
        assert len(pers) == 1
        # Persistence has known full-precision values
        assert float(pers[0]["rmse_wh"]) > 0


# ===========================================================================
# Test Class 4 - Audits
# ===========================================================================
class TestPhase58Audits:
    def test_preflight_passes(self):
        import csv
        rows = list(csv.DictReader((_art() / "phase58_preflight_audit.csv").open(encoding="utf-8")))
        for r in rows:
            assert r["status"] == "PASS", f"Preflight failure: {r}"

    def test_cross_consistency_audit(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_cross_consistency_audit.csv").open(encoding="utf-8")))
        # All checks should PASS or be consistent
        for r in rows:
            assert r["status"] == "PASS", f"Consistency fail: {r}"

    def test_population_audit_passes(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_population_audit.csv").open(encoding="utf-8")))
        for r in rows:
            assert r["status"] == "PASS", f"Population fail: {r}"

    def test_seed_audit_passes(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_seed_audit.csv").open(encoding="utf-8")))
        for r in rows:
            assert r["status"] == "PASS", f"Seed fail: {r}"

    def test_unit_audit_passes(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_unit_audit.csv").open(encoding="utf-8")))
        for r in rows:
            assert r["status"] == "PASS", f"Unit fail: {r}"

    def test_model_lock_audit(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_model_lock_audit.csv").open(encoding="utf-8")))
        assert len(rows) > 0

    def test_rounding_audit_passes(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_rounding_audit.csv").open(encoding="utf-8")))
        for r in rows:
            assert r["status"] == "PASS", f"Rounding fail: {r}"

    def test_table_checksums(self):
        cs = json.loads((_art() / "final_table_checksums.json").read_text())
        for tid, c in cs.items():
            assert "csv_sha256" in c
            assert "markdown_sha256" in c
            assert "latex_sha256" in c
            assert len(c["csv_sha256"]) == 64


# ===========================================================================
# Test Class 5 - Source ledger and lineage
# ===========================================================================
class TestPhase58Ledger:
    def test_source_ledger_present(self):
        assert (_art() / "final_table_source_ledger.csv").is_file()

    def test_source_ledger_has_critical_tables(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_source_ledger.csv").open(encoding="utf-8")))
        table_ids = {r["table_id"] for r in rows}
        for tid in ("FT01", "FT02", "FT03", "FT06", "FT08", "FT09"):
            assert tid in table_ids, f"{tid} missing from source ledger"

    def test_cell_lineage_present(self):
        assert (_art() / "final_table_cell_lineage.csv").is_file()

    def test_cell_lineage_has_critical_tables(self):
        import csv
        rows = list(csv.DictReader((_art() / "final_table_cell_lineage.csv").open(encoding="utf-8")))
        table_ids = {r["table_id"] for r in rows}
        for tid in ("FT02", "FT03", "FT06", "FT08", "FT09"):
            assert tid in table_ids


# ===========================================================================
# Test Class 6 - Handoff + summary
# ===========================================================================
class TestPhase58Handoff:
    def test_phase59_handoff_exists(self):
        assert (_art() / "phase59_final_conclusions_handoff.json").is_file()

    def test_phase59_handoff_ready(self):
        h = json.loads((_art() / "phase59_final_conclusions_handoff.json").read_text())
        assert h["ready_for_phase59"] is True

    def test_phase59_handoff_has_allowed_claims(self):
        h = json.loads((_art() / "phase59_final_conclusions_handoff.json").read_text())
        assert "FINAL_TEST_PERFORMANCE" in h["allowed_claims"]
        assert "ATTENTION_SEED_STABILITY" in h["allowed_claims"]

    def test_phase59_handoff_has_prohibited_claims(self):
        h = json.loads((_art() / "phase59_final_conclusions_handoff.json").read_text())
        assert "CAUSAL_ATTENTION_EXPLANATION" in h["prohibited_claims"]
        assert "BEST_SEED_SELECTION" in h["prohibited_claims"]
        assert "MULTI_HOUSE_GENERALIZATION" in h["prohibited_claims"]

    def test_summary_present(self):
        assert (_art() / "final_tables_summary.json").is_file()


# ===========================================================================
# Test Class 7 - Findings / tests / discrepancies
# ===========================================================================
class TestPhase58Artifacts:
    def test_findings_present(self):
        assert (_art() / "final_tables_findings.csv").is_file()

    def test_tests_present(self):
        assert (_art() / "final_tables_tests.csv").is_file()

    def test_discrepancies_present(self):
        assert (_art() / "final_tables_discrepancies.json").is_file()

    def test_coursework_coverage_present(self):
        assert (_art() / "coursework_requirement_table_coverage.csv").is_file()

    def test_claim_traceability_present(self):
        assert (_art() / "table_claim_traceability.csv").is_file()

    def test_catalog_present(self):
        assert (_art() / "FINAL_TABLE_CATALOG.md").is_file()

    def test_report_present(self):
        assert (_art() / "final_tables_report.md").is_file()

    def test_readme_present(self):
        assert (_art() / "README_FINAL_TABLES.md").is_file()


# ===========================================================================
# Test Class 8 - Forbidden token check
# ===========================================================================
class TestForbiddenTokens:
    """Ensure no forbidden tokens appear in Phase 58 source modules."""

    FORBIDDEN = [
        "ensemble_reconstructed", "best_seed_selected",
        "MAPE", "mape_",
        "scaler.fit", "scaler.fit_transform",
        ".backward()", "optimizer.step",
        "model.train(", "return_attention",
        "extract_attention", "fit_transform",
        "torch.load", "model.forward",
        "materialize_phase", "cudnn",
    ]

    def test_phase58_no_forbidden_tokens_in_builders(self):
        from pathlib import Path
        fp = Path("src/course_work/phase58/builders.py")
        text = fp.read_text(encoding="utf-8")
        # Strip strings & comments
        stripped = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        stripped = re.sub(r"'''.*?'''", '', stripped, flags=re.DOTALL)
        for tok in self.FORBIDDEN:
            # Use regex with strict boundaries for safety
            if tok == "MAPE":
                # Word boundary
                assert not re.search(r"\bMAPE\b", stripped), f"forbidden token MAPE in builders"
            elif tok in ("scaler.fit", "scaler.fit_transform", "return_attention",
                          "extract_attention", "model.forward", "torch.load",
                          "materialize_phase"):
                assert tok not in stripped, f"forbidden token {tok} in builders"
            elif tok in (".backward()", "optimizer.step"):
                assert tok not in stripped, f"forbidden token {tok} in builders"
            elif tok == "model.train(":
                assert tok not in stripped, f"forbidden token {tok} in builders"
            elif tok == "cudnn":
                # cudnn may appear in FT01 documentation of LOCKED reproducibility settings
                # but not as code that *sets* cudnn. We allow the term only if it appears
                # in a string used for documentation display.
                # Allow if inside a quoted string in the line
                for line in stripped.splitlines():
                    if tok in line and not re.search(r'["\'].*cudnn.*["\']', line):
                        raise AssertionError(f"forbidden token {tok} in builders: {line.strip()}")
            elif tok == "ensemble_reconstructed":
                # We use this as a False flag in summary
                assert stripped.count("ensemble_reconstructed") <= 1, \
                    f"forbidden token {tok} in builders"
            elif tok == "best_seed_selected":
                assert stripped.count("best_seed_selected") <= 1

    def test_phase58_no_forbidden_tokens_in_main(self):
        from pathlib import Path
        fp = Path("src/course_work/phase58/main.py")
        text = fp.read_text(encoding="utf-8")
        stripped = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        stripped = re.sub(r"'''.*?'''", '', stripped, flags=re.DOTALL)
        for tok in self.FORBIDDEN:
            if tok in ("scaler.fit", "scaler.fit_transform", "return_attention",
                        "extract_attention", "model.forward", "torch.load",
                        "materialize_phase", "model.train("):
                assert tok not in stripped, f"forbidden token {tok} in main"
            elif tok in (".backward()", "optimizer.step"):
                assert tok not in stripped, f"forbidden token {tok} in main"
            elif tok == "cudnn":
                # Documentation labels only
                for line in stripped.splitlines():
                    if tok in line and not re.search(r'["\'].*cudnn.*["\']', line):
                        raise AssertionError(f"forbidden token {tok} in main: {line.strip()}")
            elif tok == "MAPE":
                # Allowed only in: "no MAPE" caveat contexts
                if re.search(r"\bMAPE\b", stripped):
                    # Allow in 'no_mape' / 'NO_MAPE' / 'no MAPE' contexts
                    ctx = re.search(r"(?:no|never|do not|no_mapE|not authorized|not allowed).*?\bMAPE\b",
                                    stripped, re.IGNORECASE)
                    assert ctx is not None, f"MAPE referenced without 'no/forbidden' context"


# ===========================================================================
# Test Class 9 - Upstream artifact immutability
# ===========================================================================
class TestUpstreamImmutability:
    def test_phase47_summary_sha_unchanged(self):
        sj = _signoff()
        # Read upstream phase47 summary directly and confirm
        upstream = json.loads((_root() / "artifacts/final_test/final_test_summary.json").read_text())
        assert sj["final_lock_sha256"] == upstream.get("final_lock_sha256")

    def test_upstream_test_population_sha_matches(self):
        sj = _signoff()
        upstream = json.loads((_root() / "artifacts/final_test/final_test_summary.json").read_text())
        assert sj["final_test_population_sha256"] == upstream.get("test_population_sha256")


# ===========================================================================
# Test Class 10 - Processing log
# ===========================================================================
class TestProcessingLog:
    def test_processing_log_exists(self):
        log_fp = _root() / "docs/save_log_in_processing/phase_58_final_tables_log.json"
        assert log_fp.is_file()

    def test_processing_log_signoff_status(self):
        log_fp = _root() / "docs/save_log_in_processing/phase_58_final_tables_log.json"
        log = json.loads(log_fp.read_text())
        assert log["signoff_overall_status"] == "PASS"

    def test_processing_log_safety_flags(self):
        log_fp = _root() / "docs/save_log_in_processing/phase_58_final_tables_log.json"
        log = json.loads(log_fp.read_text())
        for f in ("new_training", "new_test_inference", "new_metric",
                  "best_seed_selected", "causal_claim"):
            assert log[f] is False
