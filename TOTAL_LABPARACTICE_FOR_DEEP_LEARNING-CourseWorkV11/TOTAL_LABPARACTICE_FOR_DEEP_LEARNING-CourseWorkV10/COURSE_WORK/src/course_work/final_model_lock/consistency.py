"""Phase45 plan §183–§192 acceptance checks.

Each check returns ``CheckResult(status, code, description, actual, expected)``.
A FAIL on any CRITICAL check blocks Phase45 PASS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    code: str
    description: str
    status: str  
    actual: Any
    expected: Any
    severity: str  


def _r(code: str, desc: str, status: str, actual: Any, expected: Any, severity: str = "CRITICAL") -> CheckResult:
    return CheckResult(code=code, description=desc, status=status, actual=actual, expected=expected, severity=severity)


def check_phase44_signoff(signoff: dict[str, Any]) -> CheckResult:
    actual = signoff.get("overall_status")
    return _r(
        "phase44_signoff_overall_status",
        "Phase44 signoff overall_status must be PASS or PASS_WITH_WARNING",
        "PASS" if actual in {"PASS", "PASS_WITH_WARNING"} else "FAIL",
        actual,
        {"PASS", "PASS_WITH_WARNING"},
    )


def check_approved_for_phase45(signoff: dict[str, Any]) -> CheckResult:
    actual = signoff.get("approved_for_phase45")
    return _r(
        "approved_for_phase45_true",
        "Phase44 must approve Phase45",
        "PASS" if actual is True else "FAIL",
        actual,
        True,
    )


def check_test_not_accessed(signoff: dict[str, Any]) -> CheckResult:
    actual = signoff.get("test_status")
    return _r(
        "test_status_not_accessed",
        "Test must be NOT_ACCESSED in Phase44 signoff",
        "PASS" if actual == "NOT_ACCESSED" else "FAIL",
        actual,
        "NOT_ACCESSED",
    )


def check_transformer_only(candidate) -> CheckResult:
    actual = candidate.model_family
    return _r(
        "final_family_transformer_only",
        "Final model family must be TRANSFORMER_ENCODER",
        "PASS" if actual == "TRANSFORMER_ENCODER" else "FAIL",
        actual,
        "TRANSFORMER_ENCODER",
    )


def check_candidate_fingerprint(candidate, handoff: dict[str, Any]) -> CheckResult:
    actual = (candidate.config_fingerprint, handoff.get("recommended_transformer_fingerprint"))
    expected = (handoff["recommended_transformer_fingerprint"],) * 2
    ok = actual[0] == actual[1]
    return _r(
        "candidate_fingerprint_match",
        "Locked candidate fingerprint must match handoff",
        "PASS" if ok else "FAIL",
        actual,
        expected,
    )


def check_epoch_evidence(epoch_decision) -> CheckResult:
    actual = (epoch_decision.RO1, epoch_decision.RO2, epoch_decision.RO3)
    expected = ("positive int", "positive int", "positive int")
    ok = all(isinstance(v, int) and v > 0 for v in actual)
    return _r(
        "three_inner_epochs_present",
        "All three RO inner best epochs must be positive integers",
        "PASS" if ok else "FAIL",
        actual,
        expected,
    )


def check_median_rule(epoch_decision) -> CheckResult:
    actual = (epoch_decision.FINAL_REFIT_EPOCHS, epoch_decision.sorted_epochs)
    sorted_epochs = sorted([epoch_decision.RO1, epoch_decision.RO2, epoch_decision.RO3])
    expected = (sorted_epochs[1], tuple(sorted_epochs))
    ok = epoch_decision.FINAL_REFIT_EPOCHS == sorted_epochs[1]
    return _r(
        "median_rule_exact",
        "FINAL_REFIT_EPOCHS must equal median(RO1,RO2,RO3)",
        "PASS" if ok else "FAIL",
        actual,
        expected,
    )


def check_no_fallback_50(epoch_decision) -> CheckResult:
    actual = epoch_decision.FINAL_REFIT_EPOCHS
    return _r(
        "no_fallback_50",
        "FINAL_REFIT_EPOCHS must not be a placeholder/max_epochs fallback",
        "PASS" if actual != 50 and actual != epoch_decision.candidate_max_epochs else "FAIL",
        actual,
        "median of three RO inner epochs",
    )


def check_final_dev_excludes_test(final_dev_pop) -> CheckResult:
    actual = final_dev_pop.test_target_values_accessed
    return _r(
        "final_dev_excludes_test",
        "FINAL_DEV must not include TEST target values",
        "PASS" if actual is False else "FAIL",
        actual,
        False,
    )


def check_final_dev_assertion(final_dev_pop) -> CheckResult:
    actual = (final_dev_pop.last_target_timestamp, final_dev_pop.first_test_timestamp)
    ok = bool(actual[0] < actual[1])
    return _r(
        "last_final_dev_ts_before_first_test_ts",
        "FINAL_DEV last target timestamp must be strictly before TEST first target timestamp",
        "PASS" if ok else "FAIL",
        actual,
        "last < first_test",
    )


def check_scaler_no_test(scaling_contract) -> CheckResult:
    actual = scaling_contract.Test_rows_used
    return _r(
        "final_scaler_excludes_test",
        "FINAL_SCALING-v1 scaler fit must not include Test rows",
        "PASS" if actual is False else "FAIL",
        actual,
        False,
    )


def check_seeds_exact(seed_contract: dict[str, Any]) -> CheckResult:
    actual = seed_contract.get("seeds")
    expected = [42, 123, 2026]
    return _r(
        "final_seeds_exact",
        "FINAL_SEEDS-v1 must equal [42, 123, 2026]",
        "PASS" if actual == expected else "FAIL",
        actual,
        expected,
    )


def check_three_planned_runs(run_matrix: list[dict[str, Any]]) -> CheckResult:
    actual = len(run_matrix)
    expected = 3
    return _r(
        "three_planned_phase46_runs",
        "Exactly 3 Phase46 planned runs",
        "PASS" if actual == expected else "FAIL",
        actual,
        expected,
    )


def check_same_epochs_all_seeds(run_matrix: list[dict[str, Any]]) -> CheckResult:
    epochs = {r["final_refit_epochs"] for r in run_matrix}
    actual = list(epochs)
    return _r(
        "same_epochs_all_seeds",
        "All planned Phase46 runs must share the same FINAL_REFIT_EPOCHS",
        "PASS" if len(epochs) == 1 else "FAIL",
        actual,
        "single value",
    )


def check_validation_none(recipe_dict: dict[str, Any]) -> CheckResult:
    actual = recipe_dict.get("validation_loader")
    return _r(
        "phase46_validation_none",
        "Phase46 recipe must not use validation_loader",
        "PASS" if actual is None else "FAIL",
        actual,
        None,
    )


def check_early_stopping_false(recipe_dict: dict[str, Any]) -> CheckResult:
    actual = recipe_dict.get("early_stopping")
    return _r(
        "early_stopping_false",
        "Phase46 recipe must have early_stopping=false",
        "PASS" if actual is False else "FAIL",
        actual,
        False,
    )


def check_checkpoint_type_final_refit(recipe_dict: dict[str, Any]) -> CheckResult:
    actual = recipe_dict.get("checkpoint_type")
    return _r(
        "checkpoint_type_final_refit",
        "Phase46 recipe must use FINAL_REFIT checkpoint",
        "PASS" if actual == "FINAL_REFIT" else "FAIL",
        actual,
        "FINAL_REFIT",
    )


def check_no_best_semantics(recipe_dict: dict[str, Any]) -> CheckResult:
    actual = recipe_dict.get("final_refit_mode") == "FINAL_REFIT_MODE-v1"
    return _r(
        "no_best_semantics",
        "Phase46 must not use BEST-checkpoint selection",
        "PASS" if actual else "FAIL",
        actual,
        True,
    )


def check_attention_compatible(locked_config: dict[str, Any]) -> CheckResult:
    model = locked_config.get("model", {})
    actual = model.get("attention_aware", False) and model.get("model_family") == "TRANSFORMER_ENCODER"
    return _r(
        "attention_compatibility_retained",
        "Locked Transformer must support attention extraction",
        "PASS" if actual else "FAIL",
        actual,
        True,
    )


def check_lookback_72(candidate) -> CheckResult:
    actual = candidate.lookback_steps
    return _r(
        "lookback_72",
        "Locked candidate must use lookback_steps=72",
        "PASS" if actual == 72 else "FAIL",
        actual,
        72,
    )


def check_zero_training(evidence: dict[str, Any]) -> CheckResult:
    actual = (
        evidence.get("optimizer_steps", 0),
        evidence.get("new_scientific_run_ids", 0),
        evidence.get("new_validation_runs", 0),
    )
    ok = all(v == 0 for v in actual)
    return _r(
        "zero_training_zero_validation",
        "Phase45 must execute zero optimizer steps and create zero scientific/validation runs",
        "PASS" if ok else "FAIL",
        actual,
        (0, 0, 0),
    )


def check_o45_artifacts_complete(artifacts_dir) -> CheckResult:
    _expected = {n for n in ARTIFACT_NAMES if n != "phase_45_signoff.json"}
    present = {n for n in _expected if (artifacts_dir / n).exists()}
    missing = _expected - present
    actual = (len(present), len(_expected), sorted(missing))
    return _r(
        "o45_artifacts_38_of_38",
        f"All O45 artifacts (excluding phase_45_signoff.json which is written post-preflight) must be written",
        "PASS" if not missing else "FAIL",
        actual,
        (len(_expected), len(_expected), []),
    )


def check_phase46_handoff(handoff: dict[str, Any]) -> CheckResult:
    required = {
        "final_lock_sha256": str,
        "candidate_id": str,
        "config_fingerprint": str,
        "training_recipe": dict,
        "FINAL_REFIT_EPOCHS": int,
        "FINAL_DEV_REGION-v1": str,
        "target_ids_fingerprint": (str, type(None)),
        "seed_list": list,
        "final_refit_mode": str,
        "test_locked": bool,
        "no_validation": bool,
        "no_early_stopping": bool,
        "checkpoint_type": str,
        "test_status": str,
    }
    missing = []
    for k, t in required.items():
        if k not in handoff:
            missing.append(k)
        elif not isinstance(handoff[k], t):
            missing.append(f"{k}({type(handoff[k]).__name__}!={t.__name__})")
    actual = (len(handoff), len(required), missing)
    return _r(
        "phase46_handoff_complete",
        "phase46_three_seed_handoff.json must contain the lock contract fields",
        "PASS" if not missing else "FAIL",
        actual,
        (len(required), len(required), []),
    )


def check_phase47_guard(guard: dict[str, Any]) -> CheckResult:
    actual = (
        guard.get("test_access_first_allowed_phase"),
        guard.get("phase45_test_access"),
        guard.get("phase46_test_access"),
    )
    expected = (47, "forbidden", "forbidden")
    ok = actual == expected
    return _r(
        "phase47_guard_complete",
        "phase47_test_evaluation_guard.json must forbid Test before phase 47",
        "PASS" if ok else "FAIL",
        actual,
        expected,
    )


def check_wb0_locked(boundary_evidence: dict[str, Any]) -> CheckResult:
    actual = (
        boundary_evidence.get("wb0_primary"),
        boundary_evidence.get("protocol_amendment_required"),
    )
    expected = (True, False)
    ok = actual == expected
    return _r(
        "wb0_locked_no_amendment",
        "WB0 must remain primary and no pending protocol amendment",
        "PASS" if ok else "FAIL",
        actual,
        expected,
    )


ACCEPTANCE_CHECKS = [
    ("phase44_signoff_overall_status", lambda ctx: check_phase44_signoff(ctx["signoff"])),
    ("approved_for_phase45_true", lambda ctx: check_approved_for_phase45(ctx["signoff"])),
    ("test_status_not_accessed", lambda ctx: check_test_not_accessed(ctx["signoff"])),
    ("final_family_transformer_only", lambda ctx: check_transformer_only(ctx["candidate"])),
    ("candidate_fingerprint_match", lambda ctx: check_candidate_fingerprint(ctx["candidate"], ctx["handoff"])),
    ("three_inner_epochs_present", lambda ctx: check_epoch_evidence(ctx["epoch_decision"])),
    ("median_rule_exact", lambda ctx: check_median_rule(ctx["epoch_decision"])),
    ("no_fallback_50", lambda ctx: check_no_fallback_50(ctx["epoch_decision"])),
    ("lookback_72", lambda ctx: check_lookback_72(ctx["candidate"])),
    ("final_dev_excludes_test", lambda ctx: check_final_dev_excludes_test(ctx["final_dev"])),
    ("last_final_dev_ts_before_first_test_ts", lambda ctx: check_final_dev_assertion(ctx["final_dev"])),
    ("final_scaler_excludes_test", lambda ctx: check_scaler_no_test(ctx["scaling_contract"])),
    ("final_seeds_exact", lambda ctx: check_seeds_exact(ctx["seed_contract"])),
    ("three_planned_phase46_runs", lambda ctx: check_three_planned_runs(ctx["run_matrix"])),
    ("same_epochs_all_seeds", lambda ctx: check_same_epochs_all_seeds(ctx["run_matrix"])),
    ("phase46_validation_none", lambda ctx: check_validation_none(ctx["recipe_dict"])),
    ("early_stopping_false", lambda ctx: check_early_stopping_false(ctx["recipe_dict"])),
    ("checkpoint_type_final_refit", lambda ctx: check_checkpoint_type_final_refit(ctx["recipe_dict"])),
    ("no_best_semantics", lambda ctx: check_no_best_semantics(ctx["recipe_dict"])),
    ("attention_compatibility_retained", lambda ctx: check_attention_compatible(ctx["locked_config"])),
    ("wb0_locked_no_amendment", lambda ctx: check_wb0_locked(ctx["boundary_evidence"])),
    ("zero_training_zero_validation", lambda ctx: check_zero_training(ctx["training_evidence"])),
    ("o45_artifacts_38_of_38", lambda ctx: check_o45_artifacts_complete(ctx["artifacts_dir"])),
    ("phase46_handoff_complete", lambda ctx: check_phase46_handoff(ctx["phase46_handoff"])),
    ("phase47_guard_complete", lambda ctx: check_phase47_guard(ctx["phase47_guard"])),
]


ARTIFACT_NAMES = {
    "final_model_lock_manifest.json",       # O45.1
    "final_model_lock_contract.json",       # O45.2
    "phase45_preflight_audit.csv",          # O45.3
    "final_candidate_source_audit.csv",     # O45.4
    "final_lineage_audit.csv",              # O45.5
    # Frozen config
    "final_model_scientific_config.json",   # O45.6
    "final_feature_contract.json",          # O45.7
    "final_preprocessing_contract.json",    # O45.8
    "final_boundary_contract.json",         # O45.9
    "final_revin_contract.json",            # O45.10
    "final_optimizer_contract.json",        # O45.11
    "final_loss_contract.json",             # O45.12
    "final_epoch_policy.json",              # O45.13
    "final_epoch_source_audit.csv",         # O45.14
    "final_data_region_contract.json",      # O45.15
    "final_scaling_contract.json",          # O45.16
    "final_seed_contract.json",             # O45.17
    "final_training_recipe.json",           # O45.18
    "final_checkpoint_contract.json",       # O45.19
    "final_attention_compatibility_audit.csv", # O45.20
    "final_environment_contract.json",      # O45.21
    "final_three_seed_run_matrix.csv",      # O45.22
    # Fingerprints
    "final_model_config_fingerprint.json",  # O45.23
    "final_training_recipe_fingerprint.json", # O45.24
    "final_lineage_fingerprint.json",       # O45.25
    "final_model_lock_fingerprint.json",    # O45.26
    # Evidence
    "rolling_origin_selection_evidence.json", # O45.27
    "boundary_sensitivity_evidence.json",   # O45.28
    "baseline_context_evidence.json",       # O45.29
    # Handoff + guard
    "phase46_three_seed_handoff.json",      # O45.30
    "phase47_test_evaluation_guard.json",   # O45.31
    # Findings
    "final_model_lock_findings.csv",        # O45.32
    "final_model_lock_tests.csv",           # O45.33
    "final_model_lock_discrepancies.json",  # O45.34
    "final_model_lock_summary.json",        # O45.35
    "final_model_lock_report.md",           # O45.36
    "README_FINAL_MODEL_LOCK.md",           # O45.37
    "phase_45_signoff.json",                # O45.38
}


def run_acceptance_checks(ctx: dict[str, Any]) -> list[CheckResult]:
    """Run all acceptance checks with the given context dict.

    Returns a list of ``CheckResult``. Any CRITICAL FAIL blocks Phase45 PASS.
    """
    out: list[CheckResult] = []
    for _, fn in ACCEPTANCE_CHECKS:
        try:
            res = fn(ctx)
        except Exception as exc:  
            res = CheckResult(
                code="exception",
                description=str(fn),
                status="FAIL",
                actual=str(exc),
                expected="no exception",
                severity="CRITICAL",
            )
        out.append(res)
    return out


__all__ = [
    "CheckResult",
    "ACCEPTANCE_CHECKS",
    "ARTIFACT_NAMES",
    "run_acceptance_checks",
    "check_phase44_signoff",
    "check_approved_for_phase45",
    "check_test_not_accessed",
    "check_transformer_only",
    "check_candidate_fingerprint",
    "check_epoch_evidence",
    "check_median_rule",
    "check_no_fallback_50",
    "check_final_dev_excludes_test",
    "check_final_dev_assertion",
    "check_scaler_no_test",
    "check_seeds_exact",
    "check_three_planned_runs",
    "check_same_epochs_all_seeds",
    "check_validation_none",
    "check_early_stopping_false",
    "check_checkpoint_type_final_refit",
    "check_no_best_semantics",
    "check_attention_compatible",
    "check_zero_training",
    "check_o45_artifacts_complete",
    "check_phase46_handoff",
    "check_phase47_guard",
    "check_wb0_locked",
    "check_lookback_72",
]
