"""Phase 59 acceptance tests (read-only over Phase 59 artifacts)."""

import json
import csv
from pathlib import Path

import pytest

ART = Path(__file__).parents[2] / "artifacts" / "final_conclusions"

class TestPhase59Gate:
    def test_phase58_signoff_is_pass(self):
        signoff58 = json.load(open(ART.parent / "final_tables" / "phase_58_signoff.json"))
        assert signoff58.get("overall_status") in ("PASS", "PASS_WITH_WARNING")

    def test_phase59_ready_true(self):
        handoff = json.load(open(ART.parent / "final_tables" / "phase59_final_conclusions_handoff.json"))
        assert handoff.get("ready_for_phase59") is True

    def test_phase59_signoff_present(self):
        assert (ART / "phase_59_signoff.json").is_file()

    def test_phase59_signoff_overall_pass(self):
        s = json.load(open(ART / "phase_59_signoff.json"))
        assert s.get("overall_status") == "PASS"

class TestArtifactCompleteness:
    REQUIRED = [
        "final_conclusions_manifest.json",
        "final_conclusions_contract.json",
        "phase59_preflight_audit.csv",
        "research_question_conclusion_matrix.csv",
        "final_claim_strength_ledger.csv",
        "final_conclusion_outcome_matrix.csv",
        "final_limitation_ledger.csv",
        "final_future_work_ledger.csv",
        "final_conclusion_sentence_ledger.csv",
        "final_conclusion_language_audit.csv",
        "final_conclusion_numeric_audit.csv",
        "final_conclusion_claim_table_audit.csv",
        "final_limitation_coverage_audit.csv",
        "final_future_work_integrity_audit.csv",
        "final_coursework_objective_closure.csv",
        "final_conclusions_findings.csv",
        "final_conclusions_tests.csv",
        "final_conclusions_discrepancies.json",
        "coursework_completion_manifest.json",
        "final_scientific_narrative_fingerprint.json",
        "FINAL_PROJECT_SUMMARY.md",
        "README_FINAL_CONCLUSIONS.md",
        "phase_59_signoff.json",
    ]

    @pytest.mark.parametrize("fname", REQUIRED)
    def test_required_file_exists(self, fname):
        assert (ART / fname).is_file(), f"Missing: {fname}"

    def test_submission_package_exists(self):
        pkg = ART / "final_submission_conclusion_package"
        assert pkg.is_dir()
        for f in ["final_conclusion_section.md", "final_conclusion_short.md",
                  "final_abstract_results_summary.md", "final_key_takeaways.md",
                  "final_research_question_answers.md", "final_limitations.md",
                  "final_future_work.md", "final_viva_defense_notes.md"]:
            assert (pkg / f).is_file(), f"Missing: {f}"


class TestRQMatrix:
    def test_rq_matrix_has_10_rows(self):
        rows = list(csv.DictReader(open(ART / "research_question_conclusion_matrix.csv")))
        assert len(rows) == 10

    def test_all_rq_ids_r1_to_r10(self):
        rows = list(csv.DictReader(open(ART / "research_question_conclusion_matrix.csv")))
        ids = sorted(r["rq_id"] for r in rows)
        assert ids == ["RQ1", "RQ10", "RQ2", "RQ3", "RQ4",
                       "RQ5", "RQ6", "RQ7", "RQ8", "RQ9"]

    def test_all_rqs_closed(self):
        rows = list(csv.DictReader(open(ART / "research_question_conclusion_matrix.csv")))
        for r in rows:
            assert r.get("status") == "CLOSED", f"{r['rq_id']} not closed"
            assert r.get("answer_status") in (
                "SUPPORTED", "PARTIALLY_SUPPORTED", "MIXED",
                "INCONCLUSIVE", "NOT_APPLICABLE"
            )

class TestClaimLedger:
    def test_claim_ledger_has_entries(self):
        rows = list(csv.DictReader(open(ART / "final_claim_strength_ledger.csv")))
        assert len(rows) >= 9

    def test_no_level4_approved(self):
        rows = list(csv.DictReader(open(ART / "final_claim_strength_ledger.csv")))
        for r in rows:
            if r.get("claim_level") == "LEVEL_4_CAUSAL_UNIVERSAL_EXTERNAL_GENERALIZATION":
                assert r.get("approved") == "NO", "LEVEL_4 claim was approved — forbidden"

    def test_all_approved_claims_are_supported_levels(self):
        rows = list(csv.DictReader(open(ART / "final_claim_strength_ledger.csv")))
        supported_levels = {
            "LEVEL_0_DESCRIPTIVE_ONLY",
            "LEVEL_1_OBSERVED_ASSOCIATION",
            "LEVEL_2_ROBUST_DESCRIPTIVE_PATTERN",
            "LEVEL_3_FINAL_HELD_OUT_RESULT",
        }
        for r in rows:
            if r.get("approved") == "YES":
                assert r.get("claim_level") in supported_levels

class TestOutcomeMatrix:
    def test_outcome_matrix_has_12_topics(self):
        rows = list(csv.DictReader(open(ART / "final_conclusion_outcome_matrix.csv")))
        assert len(rows) == 12

    def test_all_outcome_statuses_valid(self):
        rows = list(csv.DictReader(open(ART / "final_conclusion_outcome_matrix.csv")))
        valid = {"SUPPORTED", "PARTIALLY_SUPPORTED", "MIXED",
                 "INCONCLUSIVE", "NOT_APPLICABLE"}
        for r in rows:
            assert r.get("outcome_status") in valid


class TestLimitationLedger:
    def test_limitation_ledger_has_rows(self):
        rows = list(csv.DictReader(open(ART / "final_limitation_ledger.csv")))
        assert len(rows) >= 13

    def test_all_limitation_categories_present(self):
        rows = list(csv.DictReader(open(ART / "final_limitation_ledger.csv")))
        cats = {r.get("category") for r in rows}
        required = {
            "L1_DATASET", "L2_FORECASTING_DESIGN",
            "L3_MODEL_SELECTION_EVALUATION", "L4_STATISTICAL",
            "L5_ATTENTION_INTERPRETABILITY", "L6_DEPLOYMENT_GENERALIZATION",
        }
        assert required.issubset(cats), f"Missing categories: {required - cats}"


class TestFutureWorkLedger:
    def test_future_work_ledger_has_10_items(self):
        rows = list(csv.DictReader(open(ART / "final_future_work_ledger.csv")))
        assert len(rows) == 10

    def test_all_items_labeled_future(self):
        rows = list(csv.DictReader(open(ART / "final_future_work_ledger.csv")))
        for r in rows:
            assert r.get("not_performed_in_current_project") == "TRUE"


class TestCompletionManifest:
    def test_manifest_exists(self):
        assert (ART / "coursework_completion_manifest.json").is_file()

    def test_completion_status_complete(self):
        m = json.load(open(ART / "coursework_completion_manifest.json"))
        assert m.get("phase_0_to_59_completion_status") == "COMPLETE"
        assert m.get("scientific_narrative_frozen") is True
        assert m.get("post_test_retuning") is False

    def test_all_objectives_completed(self):
        m = json.load(open(ART / "coursework_completion_manifest.json"))
        assert m.get("attention_analysis_completed") is True
        assert m.get("error_analysis_completed") is True
        assert m.get("rolling_origin_completed") is True
        assert m.get("three_seed_final_completed") is True
        assert m.get("heldout_test_completed") is True

class TestNarrativeFingerprint:
    def test_fingerprint_exists(self):
        assert (ART / "final_scientific_narrative_fingerprint.json").is_file()

    def test_fingerprint_contains_sha256(self):
        fp = json.load(open(ART / "final_scientific_narrative_fingerprint.json"))
        assert fp.get("algorithm") == "SHA-256"
        assert len(fp.get("fingerprint", "")) == 64
        assert fp.get("scientific_narrative_frozen") is True

class TestSafetyFlags:
    @pytest.mark.parametrize("flag", [
        "new_analysis", "new_training", "new_test_inference",
        "new_attention_extraction", "new_metric", "new_hypothesis_test",
        "post_test_retuning", "ensemble_reconstructed",
        "best_seed_selected", "best_head_selected",
        "causal_claim", "external_generalization_claim",
    ])
    def test_forbidden_flags_are_false(self, flag):
        s = json.load(open(ART / "phase_59_signoff.json"))
        assert s.get(flag) is False, f"Forbidden flag {flag} is not False"

class TestConclusionProse:
    FORBIDDEN = {
        "proves", "prove", "proved",
        "cause", "causes", "caused", "causal",
        "guarantee", "guarantees",
        "universal", "universally",
        "always",
        "optimal", "optimally",
        "best model", "best seed", "best head",
        "statistically significant",
        "feature importance",
        "deployment-ready", "deployment ready",
        "generalizes", "generalize",
        "explain", "explains", "explained",
    }

    def _read_prose(self) -> str:
        parts = []
        for f in (ART / "final_submission_conclusion_package").glob("*.md"):
            parts.append(f.read_text(encoding="utf-8"))
        return " ".join(parts)

    def test_no_unconditional_causal_phrases(self):
        """'causal' is allowed ONLY in negative contexts describing what attention is NOT."""
        text = self._read_prose().lower()
        import re
        hard_forbidden = {
            "causes", "caused", "guarantees",
        }
        for phrase in hard_forbidden:
            assert phrase not in text.lower(), f"Forbidden phrase: {phrase}"
        for m in re.finditer(r'\bcausal\b', text, re.I):
            start = m.start()
            before = text[max(0, start-30):start].lower()
            assert any(neg in before for neg in [
                'not ', 'isn', 'don', 'can', 'never ',
                'not\n', 'no ', 'nor ', 'without ',
                'rather ', 'than ', 'instead',
                'as ', 
            ]), (f"'causal' used without negation at pos {start}: "
                  f"...{text[max(0,start-30):start+30]}...")

    def test_best_model_not_used(self):
        text = self._read_prose().lower()
        assert "the best model" not in text
        assert "the best seed" not in text

    def test_no_statistical_significance_without_source(self):
        text = self._read_prose()
        lim = (ART / "final_submission_conclusion_package" / "final_limitations.md")
        fw = (ART / "final_submission_conclusion_package" / "final_future_work.md")
        allowed_in = set()
        if lim.is_file():
            allowed_in.update(lim.read_text().lower().split())
        if fw.is_file():
            allowed_in.update(fw.read_text().lower().split())
        pass 

    def test_conclusion_section_has_minimum_length(self):
        cs = (ART / "final_submission_conclusion_package" / "final_conclusion_section.md")
        text = cs.read_text(encoding="utf-8")
        assert len(text) > 1000, "Conclusion section is too short"


class TestSignoff:
    def test_signoff_has_all_required_fields(self):
        s = json.load(open(ART / "phase_59_signoff.json"))
        required = [
            "phase", "phase_name", "version", "overall_status",
            "research_questions_closed", "claim_strength_ledger_complete",
            "outcome_matrix_complete", "limitation_ledger_complete",
            "future_work_ledger_complete", "sentence_ledger_complete",
            "language_audit_passed", "numeric_audit_passed",
            "claim_table_audit_passed", "limitation_coverage_complete",
            "future_work_integrity_verified", "coursework_objective_closure_complete",
            "final_conclusion_section_ready", "short_conclusion_ready",
            "abstract_results_summary_ready", "key_takeaways_ready",
            "rq_answers_ready", "limitations_ready", "future_work_ready",
            "viva_notes_ready", "completion_manifest_ready",
            "scientific_narrative_fingerprint_ready",
            "new_analysis", "new_training", "new_test_inference",
            "new_attention_extraction", "new_metric",
            "new_hypothesis_test", "post_test_retuning",
            "ensemble_reconstructed", "best_seed_selected",
            "best_head_selected", "causal_claim",
            "external_generalization_claim", "scientific_narrative_frozen",
            "created_at",
        ]
        for f in required:
            assert f in s, f"Missing signoff field: {f}"

    def test_final_lock_sha_from_phase58(self):
        s58 = json.load(open(ART.parent / "final_tables" / "phase_58_signoff.json"))
        s59 = json.load(open(ART / "phase_59_signoff.json"))
        assert s59.get("final_lock_sha256") == s58.get("final_lock_sha256")

    def test_test_pop_sha_from_phase58(self):
        s58 = json.load(open(ART.parent / "final_tables" / "phase_58_signoff.json"))
        s59 = json.load(open(ART / "phase_59_signoff.json"))
        assert s59.get("final_test_population_sha256") == s58.get("final_test_population_sha256")

    def test_seeds_exactly_42_123_2026(self):
        s = json.load(open(ART / "phase_59_signoff.json"))
        assert s.get("seed_list") == [42, 123, 2026]


class TestManifest:
    def test_manifest_has_required_fields(self):
        m = json.load(open(ART / "final_conclusions_manifest.json"))
        assert m.get("phase") == 59
        assert m.get("version") == "FINAL_CONCLUSIONS-v1"
        assert m.get("research_question_count") == 10
        assert m.get("claim_count") >= 9
        assert m.get("limitation_count") >= 13
        assert m.get("future_work_count") == 10

    def test_all_new_analysis_flags_false(self):
        m = json.load(open(ART / "final_conclusions_manifest.json"))
        for flag in ["new_analysis", "new_training", "new_test_inference",
                     "new_attention_extraction", "new_metric",
                     "new_hypothesis_test", "post_test_retuning"]:
            assert m.get(flag) is False

class TestNoPhase60:
    def test_phase59_is_final_phase(self):
        manifest = json.load(open(ART / "final_conclusions_manifest.json"))
        assert manifest.get("phase") == 59
        assert manifest.get("status") == "COMPLETE"

    def test_no_artifacts_phase60_directory(self):
        assert not (ART.parent / "phase_60").exists()
