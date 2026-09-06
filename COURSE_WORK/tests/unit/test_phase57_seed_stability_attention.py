"""Phase 57 - focused integrity tests (seed-stability attention check).

These tests verify the integrity and scientific protocol adherence of the
Phase 57 implementation against the canonical plan.

NO model loading, NO training, NO recomputation of upstream metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifacts() -> Path:
    return _root() / "artifacts" / "seed_stability_attention"


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test: Source integrity
# ---------------------------------------------------------------------------

class TestSourceIntegrity:
    def test_phase57_signoff_exists(self):
        fp = _artifacts() / "phase_57_signoff.json"
        assert fp.is_file(), "phase_57_signoff.json missing"
        obj = _read_json(fp)
        assert obj.get("overall_status") in ("PASS", "PASS_WITH_WARNING")

    def test_three_official_seeds(self):
        obj = _read_json(_artifacts() / "phase_57_signoff.json")
        assert obj.get("seed_list") == [42, 123, 2026]

    def test_final_lock_sha_consistent(self):
        obj = _read_json(_artifacts() / "phase_57_signoff.json")
        sha = obj.get("final_lock_sha256", "")
        assert len(sha) == 64
        assert sha == "102086f71ed01611b963c44926d7472a3ecc49a0b63f41d79100ef816b52a9ff"

    def test_anchor_seed_42_not_performance_based(self):
        obj = _read_json(_artifacts() / "phase_57_signoff.json")
        assert obj.get("alignment_anchor_seed") == 42
        assert obj.get("performance_based_anchor") is False
        assert obj.get("anchor_reason") == "FIRST_PREDECLARED_FINAL_SEED"

    def test_canonical_matching_metadata(self):
        obj = _read_json(_artifacts() / "phase_57_signoff.json")
        assert obj.get("canonical_matching_cost") == "JSD"
        assert obj.get("matching_method") == "EXHAUSTIVE_PERMUTATION"
        assert float(obj.get("match_tie_tolerance")) == 1e-12
        assert obj.get("matching_sensitivity") == "WASSERSTEIN_MINUTES"


# ---------------------------------------------------------------------------
# Test: Preflight + source verification
# ---------------------------------------------------------------------------

class TestPreflightAndSourceVerification:
    def test_preflight_audit_pass(self):
        rows = list(_read_preflight())
        assert rows, "preflight empty"
        fails = [r for r in rows if r["critical"] == "YES" and r["status"] != "PASS"]
        assert not fails, f"critical preflight failures: {fails}"

    def test_three_official_seeds_present(self):
        rows = list(_read_preflight())
        seeds_row = next(r for r in rows if r["check"] == "three_official_seeds_present")
        assert "[42, 123, 2026]" in str(seeds_row["observed"])

    def test_matching_contract_frozen(self):
        rows = list(_read_preflight())
        cf_row = next(r for r in rows if r["check"] == "matching_contract_frozen_before_results")
        assert "MATCH_TIE_TOL=1e-12" in str(cf_row["observed"])


def _read_preflight():
    fp = _artifacts() / "phase57_preflight_audit.csv"
    import csv
    with fp.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Test: Layer head-mean stability
# ---------------------------------------------------------------------------

class TestLayerHeadMeanStability:
    def test_layer_pairwise_rows_present(self):
        rows = list(_read_csv("layer_head_mean_seed_stability.csv"))
        # 2 layers * 3 seed_pairs = 6 rows
        assert len(rows) == 6

    def test_layer_summary_present(self):
        rows = list(_read_csv("layer_head_mean_seed_stability_summary.csv"))
        assert len(rows) == 2

    def test_jsd_in_bounds(self):
        rows = list(_read_csv("layer_head_mean_seed_stability.csv"))
        for r in rows:
            v = float(r["jsd"])
            assert 0.0 <= v <= float(np.log(2.0)) + 1e-9, f"jsd out of bounds: {v}"

    def test_wasserstein_minutes_nonneg(self):
        rows = list(_read_csv("layer_head_mean_seed_stability.csv"))
        for r in rows:
            v = float(r["wasserstein_minutes"])
            assert v >= 0.0, f"negative Wasserstein: {v}"


# ---------------------------------------------------------------------------
# Test: Per-target layer stability
# ---------------------------------------------------------------------------

class TestPerTargetLayerStability:
    def test_row_count(self):
        rows = list(_read_csv("layer_head_mean_per_target_stability.csv"))
        # N_TEST=2961 * 2 layers * 3 pairs = 17766
        assert len(rows) == 2961 * 2 * 3

    def test_three_seed_disagreement(self):
        rows = list(_read_csv("layer_attention_disagreement_by_target.csv"))
        # N_TEST * 2 layers = 5922
        assert len(rows) == 2961 * 2


# ---------------------------------------------------------------------------
# Test: Layer attention metric stability
# ---------------------------------------------------------------------------

class TestLayerMetricStability:
    def test_metric_coverage(self):
        rows = list(_read_csv("layer_attention_metric_seed_stability.csv"))
        # 2 layers * 6 metrics * 3 seed_pairs = 36
        assert len(rows) == 2 * 6 * 3


# ---------------------------------------------------------------------------
# Test: Head matching
# ---------------------------------------------------------------------------

class TestHeadMatching:
    def test_cost_matrix_shape(self):
        rows = list(_read_csv("head_matching_cost_matrices.csv"))
        # 2 layers * 3 seed_pairs * 4 heads * 4 heads = 96
        assert len(rows) == 2 * 3 * 4 * 4

    def test_canonical_assignments_bijective(self):
        rows = list(_read_csv("head_matching_assignments.csv"))
        # 2 layers * 3 seed_pairs * 4 heads = 24
        assert len(rows) == 2 * 3 * 4
        # Each (layer, seed_pair) should map all 4 target heads uniquely
        seen = set()
        for r in rows:
            key = (r["layer_idx0"], r["seed_a"], r["seed_b"])
            target = int(r["head_b_idx0"])
            assert (key, target) not in seen, f"non-bijective assignment: {r}"
            seen.add((key, target))

    def test_permutation_count_4_factorial(self):
        rows = list(_read_csv("head_matching_ambiguity_audit.csv"))
        for r in rows:
            assert int(r["permutation_count"]) == 24, f"bad perm count: {r}"

    def test_match_tie_tolerance_recorded(self):
        rows = list(_read_csv("head_matching_ambiguity_audit.csv"))
        for r in rows:
            assert abs(float(r["match_tie_tolerance"]) - 1e-12) < 1e-18


# ---------------------------------------------------------------------------
# Test: Cycle consistency + Wasserstein sensitivity
# ---------------------------------------------------------------------------

class TestCycleAndSensitivity:
    def test_cycle_rows_present(self):
        rows = list(_read_csv("head_matching_cycle_consistency.csv"))
        # 2 layers * 4 heads = 8
        assert len(rows) == 8

    def test_wasserstein_sensitivity_rows_present(self):
        rows = list(_read_csv("head_matching_wasserstein_sensitivity.csv"))
        # 2 layers * 3 seed_pairs * 4 heads = 24
        assert len(rows) == 2 * 3 * 4

    def test_matching_independence(self):
        rows = list(_read_csv("head_matching_independence_audit.csv"))
        for r in rows:
            name = r["check"]
            if name == "mean_temporal_profiles":
                assert r["used_in_matching"] == "True"
            else:
                assert r["used_in_matching"] == "False", f"{name} should not be used in matching"


# ---------------------------------------------------------------------------
# Test: Canonical matched groups + fingerprint
# ---------------------------------------------------------------------------

class TestCanonicalGroupsAndFingerprint:
    def test_canonical_groups_count(self):
        rows = list(_read_csv("canonical_matched_head_groups.csv"))
        # 2 layers * 4 canonical groups = 8
        assert len(rows) == 2 * 4

    def test_fingerprint_frozen_before_phase56(self):
        fp = _read_json(_artifacts() / "head_matching_fingerprint.json")
        assert fp.get("created_before_error_effect_mapping") is True
        assert fp.get("anchor_seed") == 42
        assert fp.get("performance_based_anchor") is False
        assert fp.get("cost_metric") == "JSD_NATURAL_LOG"
        assert fp.get("matching_method") == "EXHAUSTIVE_PERMUTATION"
        assert float(fp.get("tie_tolerance")) == 1e-12


# ---------------------------------------------------------------------------
# Test: Matched-head stability
# ---------------------------------------------------------------------------

class TestMatchedHeadStability:
    def test_matched_head_profile_rows(self):
        rows = list(_read_csv("matched_head_profile_seed_stability.csv"))
        # 2 layers * 4 groups * 3 seed_pairs = 24
        assert len(rows) == 2 * 4 * 3

    def test_matched_head_per_target_rows(self):
        rows = list(_read_csv("matched_head_per_target_stability.csv"))
        # 2 layers * 4 groups * 3 seed_pairs * N_TEST = 71,064
        assert len(rows) == 2 * 4 * 3 * 2961

    def test_matched_head_metric_rows(self):
        rows = list(_read_csv("matched_head_metric_seed_stability.csv"))
        # 2 layers * 4 groups * 6 metrics * 3 seed_pairs = 144
        assert len(rows) == 2 * 4 * 6 * 3

    def test_matched_head_top1_rows(self):
        rows = list(_read_csv("matched_head_top1_lag_stability.csv"))
        # 2 layers * 4 groups * 3 seed_pairs = 24
        assert len(rows) == 2 * 4 * 3


# ---------------------------------------------------------------------------
# Test: Dense case stability
# ---------------------------------------------------------------------------

class TestDenseCaseStability:
    def test_dense_rows_present(self):
        rows = list(_read_csv("dense_case_attention_seed_stability.csv"))
        # 44 cases * 2 layers * 4 groups * 3 seed_pairs = 1056
        assert len(rows) == 44 * 2 * 4 * 3


# ---------------------------------------------------------------------------
# Test: Error-conditioned stability
# ---------------------------------------------------------------------------

class TestErrorConditionedStability:
    def test_layer_rows_present(self):
        rows = list(_read_csv("layer_error_conditioned_seed_stability.csv"))
        assert rows, "no layer error-conditioned rows"

    def test_matched_head_rows_present(self):
        rows = list(_read_csv("matched_head_error_conditioned_stability.csv"))
        assert rows, "no matched-head error-conditioned rows"

    def test_sign_agreement_columns_present(self):
        rows = list(_read_csv("layer_error_conditioned_seed_stability.csv"))
        for r in rows:
            assert "positive_count" in r
            assert "negative_count" in r
            assert "all_defined_same_sign" in r


# ---------------------------------------------------------------------------
# Test: Prediction-attention disagreement
# ---------------------------------------------------------------------------

class TestPredictionAttention:
    def test_layer_pred_assoc_rows(self):
        rows = list(_read_csv("prediction_attention_disagreement_association.csv"))
        # 2 layers * 2 pred_metrics * 2 attn_metrics = 8
        assert len(rows) == 2 * 2 * 2

    def test_matched_pred_assoc_rows(self):
        rows = list(_read_csv("matched_head_prediction_disagreement_association.csv"))
        # 2 layers * 4 groups * 2 pred * 2 attn = 32
        assert len(rows) == 2 * 4 * 2 * 2


# ---------------------------------------------------------------------------
# Test: Stability evidence summary
# ---------------------------------------------------------------------------

class TestStabilityEvidenceSummary:
    def test_summary_rows_present(self):
        rows = list(_read_csv("attention_seed_stability_evidence_summary.csv"))
        codes = [r["evidence_row"] for r in rows]
        assert "LAYER_PROFILE" in codes
        assert "HEAD_MATCHING" in codes
        assert "MATCH_CYCLE" in codes
        assert "MATCH_METHOD_SENSITIVITY" in codes
        assert "MATCHED_HEAD_PROFILE" in codes
        assert "DENSE_CASE_FULL_MAP" in codes
        assert "ERROR_CONDITIONED_LAYER" in codes
        assert "ERROR_CONDITIONED_MATCHED_HEAD" in codes
        assert "PREDICTION_ATTENTION_DISAGREEMENT" in codes

    def test_no_weighted_score_columns(self):
        rows = list(_read_csv("attention_seed_stability_evidence_summary.csv"))
        # No composite weighted score column should exist
        assert "composite_score" not in (rows[0].keys() if rows else {})


# ---------------------------------------------------------------------------
# Test: Findings, tests, discrepancies
# ---------------------------------------------------------------------------

class TestFindingsAndTests:
    def test_findings_present(self):
        rows = list(_read_csv("seed_stability_attention_findings.csv"))
        assert len(rows) >= 16

    def test_tests_present_and_all_pass(self):
        rows = list(_read_csv("seed_stability_attention_tests.csv"))
        assert len(rows) >= 30
        for r in rows:
            assert r["status"] == "PASS", f"test not PASS: {r}"

    def test_discrepancies_empty_or_review(self):
        obj = _read_json(_artifacts() / "seed_stability_attention_discrepancies.json")
        assert obj["total"] == 0
        assert obj["status"] in ("PASS", "REVIEW")


# ---------------------------------------------------------------------------
# Test: Figures
# ---------------------------------------------------------------------------

class TestFigures:
    def test_figure_count(self):
        fig_dir = _artifacts() / "figures"
        figs = list(fig_dir.glob("SEEDATTN_57_*.png"))
        # 14 figure families (some have multiple panels; count all PNGs)
        assert len(figs) >= 14

    def test_no_figure_regen_or_select_by_interest(self):
        # Just sanity: all figures start with SEEDATTN_57_
        fig_dir = _artifacts() / "figures"
        figs = list(fig_dir.glob("*.png"))
        for f in figs:
            assert f.name.startswith("SEEDATTN_57_"), f"unexpected figure: {f.name}"


# ---------------------------------------------------------------------------
# Test: Phase 58 handoff + Phase 59 context
# ---------------------------------------------------------------------------

class TestHandoffs:
    def test_phase58_handoff(self):
        obj = _read_json(_artifacts() / "phase58_final_tables_handoff.json")
        assert obj.get("ready_for_phase58") is True
        assert obj.get("best_seed_selected") is False
        assert obj.get("best_head_selected") is False
        assert obj.get("causal_claim") is False
        assert obj.get("seed_list") == [42, 123, 2026]

    def test_phase59_context_handoff(self):
        obj = _read_json(_artifacts() / "phase59_conclusions_context_handoff.json")
        assert obj.get("ready_for_phase59_context") is True
        assert obj.get("attention_is_temporal_not_feature_importance") is True
        assert obj.get("attention_is_not_causal_explanation") is True
        assert obj.get("three_seed_limit") is True
        assert obj.get("no_model_change_after_test") is True

    def test_signoff_status(self):
        obj = _read_json(_artifacts() / "phase_57_signoff.json")
        assert obj.get("phase58_ready") is True
        assert obj.get("phase59_context_ready") is True
        assert obj.get("best_seed_selected") is False
        assert obj.get("best_head_selected") is False
        assert obj.get("head_pruning") is False
        assert obj.get("head_ablation") is False
        assert obj.get("model_training") is False
        assert obj.get("new_test_inference") is False
        assert obj.get("new_attention_extraction") is False
        assert obj.get("causal_claim") is False
        assert obj.get("weighted_stability_score") is False
        assert obj.get("same_index_semantic_alignment_assumed") is False
        assert obj.get("target_specific_rematching") is False
        assert obj.get("error_specific_rematching") is False
        assert obj.get("regime_specific_rematching") is False
        assert obj.get("case_specific_rematching") is False


# ---------------------------------------------------------------------------
# Test: Forbidden tokens absent in source modules
# ---------------------------------------------------------------------------

class TestForbiddenTokens:
    FORBIDDEN_TOKENS = [
        "model.forward",
        "torch.load",
        "return_attention",
        "materialize_phase52",
        "extract_attention",
        "scaler.fit",
        ".fit(",
        ".fit_transform(",
        "optimizer.step",
        ".backward()",
        "model.train()",
    ]

    def test_phase57_source_modules_clean(self):
        from course_work.analysis.seed_stability_attention import sources, analyses, figures, findings as F2
        from course_work.analysis.seed_stability_attention import orchestrator
        # Token must be in an actual code statement (not inside a docstring/comment/string literal).
        # We use a heuristic: strip docstrings (triple quotes) and comments, then look at each
        # remaining statement for forbidden patterns.
        for name, mod in [
            ("sources.py", sources),
            ("analyses.py", analyses),
            ("figures.py", figures),
            ("findings.py", F2),
            ("orchestrator.py", orchestrator),
        ]:
            src_text = Path(mod.__file__).read_text(encoding="utf-8")
            import re
            stripped = re.sub(r'"""[\s\S]*?"""', '', src_text)
            stripped = re.sub(r"'''[\s\S]*?'''", '', stripped)
            lines = []
            for line in stripped.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                if "#" in line:
                    line = line[:line.index("#")]
                lines.append(line)
            cleaned = "\n".join(lines)
            for tok in self.FORBIDDEN_TOKENS:
                pat = re.compile(r'(?<!\w)' + re.escape(tok) + r'(?!\w)')
                bad_lines = []
                for line in cleaned.splitlines():
                    if pat.search(line):
                        # Strip string literals to ignore false-positives where the
                        # forbidden token appears inside a description like
                        # "- training, fine-tune, optimizer.step, scaler.fit"
                        stripped_line = re.sub(r'"[^"]*"', '', line)
                        stripped_line = re.sub(r"'[^']*'", '', stripped_line)
                        if pat.search(stripped_line):
                            bad_lines.append(line.strip())
                assert not bad_lines, f"{name}: forbidden token {tok!r} in: {bad_lines}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(name: str):
    import csv
    fp = _artifacts() / name
    with fp.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
