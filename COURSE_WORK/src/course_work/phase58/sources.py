# -*- coding: utf-8 -*-
"""Phase 58 - frozen upstream artifact loaders (read-only).

All loaders are pure readers; they MUST NOT mutate any upstream artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_csv(fp: Path) -> list[dict[str, str]]:
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def _sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenSources58:
    """Read-only container of all frozen upstream sources used by Phase 58."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.signoffs: dict[str, dict] = {}
        self.handoffs: dict[str, dict] = {}
        self.manifests: dict[str, dict] = {}

    # ---------- canonical signoffs ----------
    def load_all_signoffs(self) -> dict[str, dict]:
        """Return a {phase: signoff} dict for Phases 44-57.

        All status values are reported as PASS / PASS_WITH_WARNING / FAIL.
        """
        for phase, dirname in [
            (44, "rolling_origin"),
            (45, "final_model_lock"),
            (46, "three_seed_final_runs"),
            (47, "final_test"),
            (48, "prediction_analysis"),
            (49, "residual_analysis"),
            (50, "error_by_regime"),
            (51, "worst_error_analysis"),
            (52, "attention_extraction"),
            (53, "attention_heatmaps"),
            (54, "last_query_attention"),
            (55, "head_comparison"),
            (56, "error_conditioned_attention"),
            (57, "seed_stability_attention"),
        ]:
            candidates = list((self.artifacts / dirname).glob("phase_*_signoff.json")) \
                + list((self.artifacts / dirname).glob("phase*_signoff.json"))
            if not candidates:
                self.signoffs[str(phase)] = {"status": "MISSING", "phase": phase}
                continue
            sj = _read_json(candidates[0])
            status = (
                sj.get("overall_status")
                or sj.get("phase51_status")
                or sj.get("phase52_status")
                or sj.get("status")
                or ("PASS" if sj.get("all_gates_pass", sj.get("n_gates_passed", 0) > 0) else "FAIL")
            )
            sj["_path"] = str(candidates[0])
            sj["status"] = status
            self.signoffs[str(phase)] = sj
        return self.signoffs

    # ---------- Phase 45 (final lock) ----------
    def phase45_sources(self) -> dict[str, Any]:
        d = self.artifacts / "final_model_lock"
        out: dict[str, Any] = {}
        files = {
            "scientific_config": d / "final_model_scientific_config.json",
            "seed_contract": d / "final_seed_contract.json",
            "loss_contract": d / "final_loss_contract.json",
            "optimizer_contract": d / "final_optimizer_contract.json",
            "boundary_contract": d / "final_boundary_contract.json",
            "feature_contract": d / "final_feature_contract.json",
            "data_region_contract": d / "final_data_region_contract.json",
            "epoch_policy": d / "final_epoch_policy.json",
            "training_recipe": d / "final_training_recipe.json",
            "revin_contract": d / "final_revin_contract.json",
            "scaling_contract": d / "final_scaling_contract.json",
            "checkpoint_contract": d / "final_checkpoint_contract.json",
            "lock_fingerprint": d / "final_model_lock_fingerprint.json",
            "config_fingerprint": d / "final_model_config_fingerprint.json",
            "signoff": d / "phase_45_signoff.json",
        }
        for k, fp in files.items():
            if fp.is_file():
                out[k] = _read_json(fp)
        # Add canonical separated identities
        if "lock_fingerprint" in out:
            out["final_lock_sha256"] = out["lock_fingerprint"].get("combined_lock_sha256", "")
        if "config_fingerprint" in out:
            out["config_fingerprint_sha256"] = out["config_fingerprint"].get("canonical_json_sha256", "")
        return out

    # ---------- Phase 47 (final Test) ----------
    def phase47_sources(self) -> dict[str, Any]:
        d = self.artifacts / "final_test"
        out: dict[str, Any] = {}
        for name in [
            "final_test_summary.json",
            "final_test_summary_table.csv",
            "final_test_baseline_metrics.csv",
            "transformer_seed_aggregate_metrics.csv",
            "final_test_model_comparison.csv",
            "final_test_evaluation_manifest.json",
            "final_test_release_verification.json",
            "phase_47_signoff.json",
        ]:
            fp = d / name
            if fp.is_file():
                if name.endswith(".json"):
                    out[name.replace(".json", "")] = _read_json(fp)
                else:
                    out[name.replace(".csv", "")] = _read_csv(fp)
        # Pull the canonical Test identity directly from Phase47 signoff
        sg = out.get("phase_47_signoff", {})
        if sg:
            out["final_lock_sha256"] = sg.get("final_lock_sha256", "")
            out["test_population_sha256"] = sg.get("test_population_sha256", "")
            out["seed42_checkpoint_sha256"] = sg.get("seed42_checkpoint_sha256", "")
            out["seed123_checkpoint_sha256"] = sg.get("seed123_checkpoint_sha256", "")
            out["seed2026_checkpoint_sha256"] = sg.get("seed2026_checkpoint_sha256", "")
        # Also pull the canonical lock keys
        if "final_test_summary" in out:
            out["final_lock_sha256"] = out["final_test_summary"].get("final_lock_sha256") or out["final_lock_sha256"]
            out["test_population_sha256"] = out["final_test_summary"].get("test_population_sha256") or out["test_population_sha256"]
            out["n_test"] = out["final_test_summary"].get("n_test")
            out["transformer_seed_metrics"] = out["final_test_summary"].get("transformer_seed_metrics")
            out["persistence_metrics"] = out["final_test_summary"].get("persistence_metrics")
            out["lstm_metrics"] = out["final_test_summary"].get("lstm_metrics")
            out["transformer_mean_mae_wh"] = out["final_test_summary"].get("transformer_mean_mae_wh")
            out["transformer_sd_mae_wh"] = out["final_test_summary"].get("transformer_sd_mae_wh")
            out["transformer_mean_rmse_wh"] = out["final_test_summary"].get("transformer_mean_rmse_wh")
            out["transformer_sd_rmse_wh"] = out["final_test_summary"].get("transformer_sd_rmse_wh")
            out["transformer_mean_r2"] = out["final_test_summary"].get("transformer_mean_r2")
            out["transformer_sd_r2"] = out["final_test_summary"].get("transformer_sd_r2")
        return out

    # ---------- Phase 44 (rolling-origin) ----------
    def phase44_sources(self) -> dict[str, Any]:
        d = self.artifacts / "rolling_origin"
        out: dict[str, Any] = {}
        for name in [
            "rolling_origin_pooled_metrics.csv",
            "rolling_origin_fold_metrics.csv",
            "rolling_origin_transformer_robustness_ranking.csv",
            "rolling_origin_macro_robustness_metrics.csv",
            "rolling_origin_recommended_transformer.json",
            "rolling_origin_results.csv",
        ]:
            fp = d / name
            if fp.is_file():
                if name.endswith(".json"):
                    out[name.replace(".json", "")] = _read_json(fp)
                else:
                    out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 49 (residuals) ----------
    def phase49_sources(self) -> dict[str, Any]:
        d = self.artifacts / "residual_analysis"
        out: dict[str, Any] = {}
        for name in [
            "residual_long_table.csv",
            "phase49_residual_distribution_summary.csv",
            "phase49_signed_bias.csv",
            "phase49_sign_balance.csv",
            "phase49_tail_diagnostics.csv",
            "phase49_prediction_deciles.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 48 (prediction spread) ----------
    def phase48_sources(self) -> dict[str, Any]:
        d = self.artifacts / "prediction_analysis"
        out: dict[str, Any] = {}
        for name in [
            "prediction_seed_spread.csv",
            "prediction_range_compression.csv",
            "prediction_distribution_summary.csv",
            "prediction_change_summary.csv",
            "prediction_local_extrema_summary.csv",
            "prediction_peak_timing_summary.csv",
            "prediction_seed_pairwise_agreement.csv",
            "prediction_baseline_context.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 50 (error regimes) ----------
    def phase50_sources(self) -> dict[str, Any]:
        d = self.artifacts / "error_by_regime"
        out: dict[str, Any] = {}
        for name in [
            "regime_metrics_long.csv",
            "regime_cross_seed_summary.csv",
            "regime_pairwise_contrasts.csv",
            "regime_rank_stability.csv",
            "regime_metrics_persistence.csv",
            "regime_metrics_lstm.csv",
            "regime_error_join_audit.csv",
            "regime_reference_train_audit.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 51 (worst cases) ----------
    def phase51_sources(self) -> dict[str, Any]:
        d = self.artifacts / "worst_error_analysis"
        out: dict[str, Any] = {}
        for name in [
            "phase51_target_level_working_table.csv",
            "phase51_attention_handoff_cases.csv",
            "phase51_signoff.json",
        ]:
            fp = d / name
            if fp.is_file():
                if name.endswith(".json"):
                    out[name.replace(".json", "")] = _read_json(fp)
                else:
                    out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 54 (last-query attention) ----------
    def phase54_sources(self) -> dict[str, Any]:
        d = self.artifacts / "last_query_attention"
        out: dict[str, Any] = {}
        for name in [
            "last_query_metrics_long.csv",
            "last_query_profile_by_lag.csv",
            "last_query_layer_head_mean_profile.csv",
            "last_query_top1_lag_frequency.csv",
            "last_query_lag_bin_mass.csv",
            "last_query_metric_summary_by_head.csv",
            "last_query_seed_overall_profile.csv",
            "last_query_recent_mass_summary.csv",
            "last_query_coverage_radius_summary.csv",
            "last_query_lag_bin_mass.csv",
            "last_query_top1_lag_frequency.csv",
            "last_query_phase52_summary_reconstruction_audit.csv",
            "phase_54_signoff.json",
        ]:
            fp = d / name
            if fp.is_file():
                if name.endswith(".json"):
                    out[name.replace(".json", "")] = _read_json(fp)
                else:
                    out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 55 (head comparison) ----------
    def phase55_sources(self) -> dict[str, Any]:
        d = self.artifacts / "head_comparison"
        out: dict[str, Any] = {}
        for name in [
            "head_behavior_summary.csv",
            "head_pair_comparison_long.csv",
            "layer_head_diversity_summary.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 56 (error-conditioned attention) ----------
    def phase56_sources(self) -> dict[str, Any]:
        d = self.artifacts / "error_conditioned_attention"
        out: dict[str, Any] = {}
        for name in [
            "error_attention_association_long.csv",
            "error_attention_high_low_metric_comparison.csv",
            "error_attention_high_low_cliffs_delta_matrix.csv",
            "error_attention_signed_metric_comparison.csv",
            "error_attention_layer_head_mean_association.csv",
            "error_attention_layer_head_mean_high_low.csv",
            "error_attention_layer_cross_seed_summary.csv",
            "error_cohort_regime_composition.csv",
            "error_conditioning_assignment.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        return out

    # ---------- Phase 57 (seed stability) ----------
    def phase57_sources(self) -> dict[str, Any]:
        d = self.artifacts / "seed_stability_attention"
        out: dict[str, Any] = {}
        for name in [
            "head_matching_assignments.csv",
            "head_matching_cycle_consistency.csv",
            "head_matching_wasserstein_sensitivity.csv",
            "head_matching_independence_audit.csv",
            "layer_head_mean_seed_stability_summary.csv",
            "matched_head_profile_seed_stability.csv",
            "matched_head_metric_seed_stability.csv",
            "matched_head_top1_lag_stability.csv",
            "dense_case_attention_seed_stability.csv",
            "layer_error_conditioned_seed_stability.csv",
            "matched_head_error_conditioned_stability.csv",
            "prediction_attention_disagreement_association.csv",
            "canonical_matched_head_groups.csv",
            "attention_seed_stability_evidence_summary.csv",
        ]:
            fp = d / name
            if fp.is_file():
                out[name.replace(".csv", "")] = _read_csv(fp)
        # handoff + fingerprint from phase57
        handoff = d / "phase58_final_tables_handoff.json"
        if handoff.is_file():
            out["phase58_handoff"] = _read_json(handoff)
        fp_fp = d / "head_matching_fingerprint.json"
        if fp_fp.is_file():
            out["head_matching_fingerprint"] = _read_json(fp_fp)
        signoff = d / "phase_57_signoff.json"
        if signoff.is_file():
            out["phase_57_signoff"] = _read_json(signoff)
        return out

    # ---------- upstream SHAs (for provenance / cross-consistency) ----------
    def all_frozen_artifacts(self) -> dict[str, str]:
        """Map of canonical key -> sha256 of that upstream file."""
        out: dict[str, str] = {}
        roots = [
            "final_model_lock/final_model_scientific_config.json",
            "final_test/final_test_summary.json",
            "rolling_origin/rolling_origin_pooled_metrics.csv",
            "rolling_origin/rolling_origin_fold_metrics.csv",
            "residual_analysis/residual_long_table.csv",
            "prediction_analysis/prediction_seed_spread.csv",
            "error_by_regime/regime_metrics_long.csv",
            "worst_error_analysis/phase51_target_level_working_table.csv",
            "last_query_attention/last_query_metrics_long.csv",
            "last_query_attention/last_query_layer_head_mean_profile.csv",
            "head_comparison/head_behavior_summary.csv",
            "head_comparison/layer_head_diversity_summary.csv",
            "error_conditioned_attention/error_attention_layer_head_mean_association.csv",
            "error_conditioned_attention/error_attention_high_low_cliffs_delta_matrix.csv",
            "seed_stability_attention/layer_head_mean_seed_stability_summary.csv",
            "seed_stability_attention/head_matching_assignments.csv",
            "seed_stability_attention/canonical_matched_head_groups.csv",
        ]
        for p in roots:
            fp = self.artifacts / p
            if fp.is_file():
                out[p] = _sha256_file(fp)
        return out


def load_frozen_sources58(root: Path) -> FrozenSources58:
    return FrozenSources58(root)
