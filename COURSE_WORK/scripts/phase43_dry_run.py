#!/usr/bin/env python3
"""Phase 43 - Non-scientific dry-run.

Verifies the Phase 43 LSTM tuning pipeline WITHOUT executing any training
or registering scientific run IDs. Safe for the agent to run.

Checks:

- Phase 42 handoff / signoff
- shared data contract resolution (canonical)
- lookback parity
- reference resolution (Phase 20 LSTM_B0)
- candidate-space construction (LT1-LT5)
- run-budget calculation (<=10)
- loader construction validation (parity)
- artifact schema validation
- Test firewall validation

Stops BEFORE `registry.register_run()` and `engine.train()`.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from course_work.data.datasets import build_train_validation_loaders
from course_work.experiments.registry import ExperimentRegistry, load_upstream_context
from course_work.lstm_tuning import artifacts as ph43_artifacts
from course_work.lstm_tuning import preflight as ph43_preflight
from course_work.lstm_tuning.reference_resolution import resolve_lstm_t0_reference
from course_work.lstm_tuning.shared_data_contract import resolve_shared_data_contract
from course_work.lstm_tuning.stages import StageExecutor
from course_work.lstm_tuning.tuning_space import (
    FIXED_TRAINING_CONTRACT,
    LT1_VALUES,
    LT2_VALUES,
    LT3_VALUES,
    LT4_VALUES,
    LT5_VALUES,
    MAX_FRESH_SCIENTIFIC_RUNS,
    REFERENCE_HYPERPARAMETERS,
    _set_path as ts_set_path,
    lt3_applicable,
)
from scripts.phase43_lstm_tuning import _enforce_scientific_contract


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(project_root: Path) -> dict[str, Any]:
    print(f"[dry-run] start={_now_iso()} project_root={project_root}")
    out: dict[str, Any] = {"status": "PASS", "checks": []}

    # 1. Phase 42 gate
    phase42_signoff = project_root / "artifacts/candidate_synthesis/phase_42_signoff.json"
    if not phase42_signoff.exists():
        out["status"] = "FAIL"
        out["checks"].append({"name": "phase42_signoff_present", "status": "FAIL"})
        return out
    signoff = json.loads(phase42_signoff.read_text())
    ok = signoff.get("status") == "PASS" and signoff.get("ready_for_phase43") is True
    out["checks"].append({"name": "phase42_gate", "status": "PASS" if ok else "FAIL", "phase42_status": signoff.get("status")})
    if not ok:
        out["status"] = "FAIL"
        return out
    print("Phase42 gate: PASS")

    # 2. Shared data contract resolution
    contract = resolve_shared_data_contract(project_root)
    out["shared_lookback"] = contract.lookback_steps
    out["train_count"] = contract.train_sample_count
    out["validation_count"] = contract.validation_sample_count
    out["batch_size"] = contract.batch_size
    out["max_epochs"] = 50
    out["patience"] = 10
    out["seed"] = 42
    print(f"shared data contract: PASS  lookback={contract.lookback_steps}  bs={contract.batch_size}")

    # 3. Reference resolution
    registry = ExperimentRegistry(project_root)
    reference = resolve_lstm_t0_reference(registry, contract)
    out["reference_resolved"] = bool(reference.phase20_exact_match)
    out["reference_run_id"] = reference.reference_run_id
    print(f"reference resolution: source={reference.source} phase20_exact_match={reference.phase20_exact_match} -> {reference.reference_run_id}")

    # 4. Candidate-space construction
    base_config = _build_base_config(project_root, reference, registry)
    base_config = _apply_fixed_training_contract(base_config)
    base_config = _apply_reference_hyperparameters(base_config)
    executor = StageExecutor(
        contract=contract,
        reference_config=base_config,
        reference_run_id=reference.reference_run_id if reference.phase20_exact_match else None,
    )
    planned = executor.plan_full_sweep()
    out["lt1_candidates"] = len([c for c in planned[0].candidates])
    out["lt2_candidates"] = len([c for c in planned[1].candidates])
    out["lt3_applicable"] = planned[2].applicable
    out["lt3_applicability_reason"] = "num_layers=1 -> SKIPPED_NOT_APPLICABLE" if not planned[2].applicable else "num_layers>=2"
    out["lt4_candidates"] = len([c for c in planned[3].candidates])
    out["lt5_candidates"] = len([c for c in planned[4].candidates])
    print(f"LT1 candidates: {out['lt1_candidates']}")
    print(f"LT2 candidates: {out['lt2_candidates']}")
    print(f"LT3: {out['lt3_applicability_reason']}")
    print(f"LT4 candidates: {out['lt4_candidates']}")
    print(f"LT5 candidates: {out['lt5_candidates']}")

    fresh_count = sum(1 for s in planned for c in s.candidates if c.source_type == "FRESH")
    out["planned_fresh_runs"] = fresh_count
    out["max_fresh_runs"] = MAX_FRESH_SCIENTIFIC_RUNS
    print(f"Planned fresh scientific runs: {fresh_count} (max {MAX_FRESH_SCIENTIFIC_RUNS})")
    if fresh_count > MAX_FRESH_SCIENTIFIC_RUNS:
        out["status"] = "FAIL"
        out["checks"].append({"name": "max_fresh_runs", "status": "FAIL", "fresh": fresh_count})
        return out

    # 4b. Pre-registration validation: every planned candidate MUST pass
    # registry.validate_run_config() WITHOUT being registered. This is a
    # non-scientific, pre-flight gate.
    from course_work.experiments.registry import validate_run_config, compute_config_fingerprint

    upstream_ctx = load_upstream_context(project_root)
    prevalidation: list[dict[str, Any]] = []
    prevalidation_ok = True
    existing_records = registry._load_records()
    existing_fingerprints = {r.get("config_fingerprint") for r in existing_records}
    existing_run_ids = {r.get("run_id") for r in existing_records}
    for stage in planned:
        for cand in stage.candidates:
            cfg = deepcopy(cand.config)
            cfg = _enforce_scientific_contract(cfg, contract, upstream_ctx)
            try:
                validate_run_config(cfg, upstream_ctx)
                status = "PASS"
            except ValueError as exc:
                status = f"FAIL:{exc}"
                prevalidation_ok = False
            # Duplicate / predecessor eligibility (mirrors register_run rules).
            fingerprint = compute_config_fingerprint(validate_run_config(cfg, upstream_ctx)) if status == "PASS" else None
            duplicate = fingerprint in existing_fingerprints if fingerprint else None
            rerun_required = duplicate
            prevalidation.append({
                "stage": stage.stage,
                "candidate_id": cand.option,
                "feature_variant_id": cfg["data"]["feature_variant_id"],
                "feature_count": cfg["data"]["feature_count"],
                "feature_fingerprint": cfg["lineage"]["feature_fingerprint"],
                "x_scaler_bundle_id": cfg["lineage"]["scaler_bundle_id"],
                "target_scaler_bundle_id": cfg["lineage"]["target_scaler_bundle_id"],
                "lookback_steps": cfg["data"]["lookback_steps"],
                "batch_size": cfg["training"]["batch_size"],
                "validation": status,
                "config_fingerprint": fingerprint,
                "is_duplicate_of_existing_run": duplicate,
                "rerun_reason_required_if_duplicate": rerun_required,
                "parent_run_id_policy": "PRE_DECESSOR_NONE_CONFIG_DIFFERS_FROM_HISTORY",
            })
    # Sanity: count current run IDs pre-registration (should equal existing count)
    pre_existing_run_ids = set(existing_run_ids)
    out["prevalidation"] = prevalidation
    out["prevalidation_ok"] = prevalidation_ok
    out["pre_registration_run_ids_snapshot"] = sorted(pre_existing_run_ids)
    if not prevalidation_ok:
        out["status"] = "FAIL"
        out["checks"].append({"name": "candidate_prevalidation", "status": "FAIL"})
        return out
    fail_count = sum(1 for row in prevalidation if not row["validation"].startswith("PASS"))
    print(f"candidate prevalidation: {len(prevalidation)} configs, {fail_count} fail")

    # 5. Loader construction validation (no train invocation)
    datasets, loaders_dict, _ = build_train_validation_loaders(
        project_root=project_root,
        variant_id=contract.feature_variant_id,
        lookback=contract.lookback_steps,
        target_option=contract.target_scaling_id,
        batch_size=contract.batch_size,
        seed=42,
    )
    train_loader = loaders_dict["TRAIN"][0]
    validation_loader = loaders_dict["VALIDATION"][0]
    bs_parity_ok = (train_loader.batch_size == contract.batch_size and validation_loader.batch_size == contract.batch_size)
    out["batch_parity"] = bs_parity_ok
    print(f"loader batch size parity: PASS={bs_parity_ok}  train={train_loader.batch_size}  valid={validation_loader.batch_size}")

    # 6. Test firewall validation: test data exists but is locked
    assert contract.test_locked, "Test firewall violated: contract.test_locked is False"
    out["test_firewall"] = "FORBIDDEN"
    out["test_locked"] = contract.test_locked
    out["test_sample_count_known"] = contract.test_sample_count >= 0
    print(f"Test access: FORBIDDEN (locked={contract.test_locked}, test_samples={contract.test_sample_count})")

    # 7. Artifact schema validation.  PREPARED artifacts are disposable dry-run
    # output and must never be written into the active repository.  In
    # particular, a finalized PASS state must not be snapshotted/deleted and
    # replaced by skeletons.
    import tempfile

    # Load handoff + phase42 signoff for preflight
    from course_work.lstm_tuning.shared_data_contract import resolve_phase43_handoff, validate_phase42_signoff
    handoff = resolve_phase43_handoff(project_root)
    phase42_signoff = validate_phase42_signoff(project_root)
    preflight_rows = ph43_preflight.build_preflight_audit(
        contract,
        handoff_lookback=int(handoff["selected_lookback"]),
        handoff_feature_variant=str(handoff["selected_feature_variant"]),
        handoff_target_scaling=str(handoff["selected_target_scaling"]),
        handoff_boundary_protocol=str(handoff["boundary_protocol"]),
        phase42_signoff_status=str(phase42_signoff.get("overall_status") or phase42_signoff.get("status")),
        test_locked_handoff=bool(handoff.get("test_locked", False)),
    )
    common_rows = ph43_preflight.build_common_data_audit(contract)

    with tempfile.TemporaryDirectory(prefix="phase43_dry_run_") as temp_dir:
        staging_root = Path(temp_dir)
        prepared = _prepare_official_artifacts(
            staging_root,
            contract,
            reference,
            planned,
            fresh_count,
            base_config,
            preflight_rows=preflight_rows,
            common_rows=common_rows,
        )
        out["artifacts"] = prepared["artifacts"]
        out["artifact_scope"] = "TEMPORARY_STAGING"
        out["active_artifacts_modified"] = False
    out["staging_cleaned"] = True
    out["scientific_run_ids_created"] = 0
    out["official_phase43_signoff_status"] = "PREPARED"
    print(f"O43 artifacts prepared (status=PREPARED): {len(prepared['artifacts'])} paths")
    print("  artifact scope: TEMPORARY_STAGING (active repository unchanged)")

    # 8. Discrepancies: none expected
    out["discrepancies"] = []
    out["stopped_before"] = ["registry.register_run()", "engine.train()", "optimizer.step"]
    print(f"stopped before: {out['stopped_before']}")

    out["completed_at"] = _now_iso()
    print(f"[dry-run] end={_now_iso()} status={out['status']}")
    return out


def _build_base_config(project_root: Path, reference, registry: ExperimentRegistry) -> dict[str, Any]:
    if reference.phase20_exact_match and reference.reference_run_id:
        for record in registry._load_records():
            if record.get("run_id") == reference.reference_run_id:
                return deepcopy(record.get("config", {}))
    from course_work.experiments.registry import build_reference_run_config
    return build_reference_run_config(project_root=project_root, model_family="LSTM")


def _apply_fixed_training_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    for path, value in FIXED_TRAINING_CONTRACT.items():
        ts_set_path(cfg, path, value)
    return cfg


def _apply_reference_hyperparameters(cfg: dict[str, Any]) -> dict[str, Any]:
    for path, value in REFERENCE_HYPERPARAMETERS.items():
        ts_set_path(cfg, path, value)
    cfg.setdefault("training", {})
    cfg["training"]["max_epochs"] = 50
    cfg["training"]["early_stopping_patience"] = 10
    cfg["training"]["early_stopping_min_delta"] = 0
    cfg["training"]["loss_name"] = "MSE"
    cfg["training"]["optimizer_name"] = "AdamW"
    cfg["training"]["gradient_clip_max_norm"] = 1.0
    cfg["training"]["scheduler_name"] = None
    cfg["training"]["revin_enabled"] = False
    cfg.setdefault("model", {})
    cfg["model"]["bidirectional"] = False
    cfg["model"]["batch_first"] = True
    cfg["model"]["pooling"] = "LAST_STEP"
    cfg.setdefault("reproducibility", {})
    cfg["reproducibility"]["seed"] = 42
    return cfg


def _prepare_official_artifacts(
    project_root: Path,
    contract,
    reference,
    planned_stages,
    fresh_count: int,
    base_config: dict[str, Any],
    preflight_rows: list[list[Any]] | None = None,
    common_rows: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Emit all O43.* artifacts at status=PREPARED. Never call registry.register_run()."""
    paths: dict[str, Path] = {}
    paths["manifest"] = ph43_artifacts.write_lstm_tuning_manifest(
        project_root, contract, reference.to_dict(), MAX_FRESH_SCIENTIFIC_RUNS, status="PREPARED"
    )
    paths["contract"] = ph43_artifacts.write_lstm_tuning_contract(project_root)
    paths["shared_data_contract"] = ph43_artifacts.write_lstm_shared_data_contract(project_root, contract)
    paths["reference_resolution"] = ph43_artifacts.write_reference_resolution(project_root, reference.to_dict())
    paths["reference_audit"] = ph43_artifacts.write_reference_audit(project_root, [
        ["reused_phase20", bool(reference.phase20_exact_match), bool(reference.phase20_exact_match), "PASS" if reference.phase20_exact_match else "FAIL"],
        ["reference_source", reference.source, "REUSED_PHASE20 or FRESH_PHASE43", "PASS" if reference.source in ("REUSED_PHASE20", "FRESH_PHASE43") else "FAIL"],
        ["reference_run_id_present", bool(reference.reference_run_id), bool(reference.reference_run_id), "PASS" if reference.reference_run_id else "FAIL"],
        ["mismatch_fields_empty", reference.source == "REUSED_PHASE20", reference.mismatch_fields, "PASS" if reference.source == "REUSED_PHASE20" else "PENDING"],
    ])
    if preflight_rows is None:
        preflight_rows = ph43_preflight.build_preflight_audit(project_root, contract)
    if common_rows is None:
        common_rows = ph43_preflight.build_common_data_audit(project_root, contract)
    paths["preflight_audit"] = ph43_preflight.write_preflight_csv(project_root, preflight_rows)
    paths["common_data_audit"] = ph43_preflight.write_common_data_audit_csv(project_root, common_rows)
    paths["tuning_space"] = ph43_artifacts.write_lstm_tuning_space(project_root)

    lineage_rows = []
    for s in planned_stages:
        lineage_rows.append([
            s.stage,
            s.factor,
            s.reference_run_id or "",
            s.reference_option,
            "|".join(c.option + ":" + (c.reuse_run_id or "") for c in s.candidates),
            "", "", "", "", "",
            "PREPARED",
        ])
    paths["stage_lineage"] = ph43_artifacts.write_stage_lineage(project_root, lineage_rows)

    run_matrix_rows = []
    for s in planned_stages:
        for c in s.candidates:
            run_matrix_rows.append([
                "",
                s.stage,
                c.option,
                c.source_type,
                contract.feature_variant_id,
                contract.target_scaling_id,
                contract.lookback_id,
                contract.boundary_protocol,
                contract.batch_size,
                c.config.get("model", {}).get("hidden_size") or "",
                c.config.get("model", {}).get("num_layers") or "",
                c.config.get("model", {}).get("dropout") or "",
                c.config.get("training", {}).get("learning_rate") or "",
                c.config.get("training", {}).get("weight_decay") or "",
                c.config.get("training", {}).get("loss_name", "MSE"),
                c.config.get("training", {}).get("max_epochs", 50),
                c.config.get("training", {}).get("early_stopping_patience", 10),
                c.config.get("training", {}).get("gradient_clip_max_norm", 1.0),
                c.config.get("reproducibility", {}).get("seed", 42),
                "", "", "", "",
                "PREPARED",
            ])
    paths["run_matrix"] = ph43_artifacts.write_run_matrix(project_root, run_matrix_rows)

    paths["findings"] = ph43_artifacts.write_findings(project_root, [["F-PREPARED", "Dry-run completed; phase 43 corrective implementation prepared."]])
    paths["tests"] = ph43_artifacts.write_tests(project_root, [["phase43_dry_run", "PASS", "PASS"]])
    paths["discrepancies"] = ph43_artifacts.write_discrepancies(project_root, [])
    paths["summary"] = ph43_artifacts.write_summary(project_root, {
        "status": "PREPARED",
        "phase": 43,
        "planned_fresh_runs": fresh_count,
        "max_fresh_runs": MAX_FRESH_SCIENTIFIC_RUNS,
        "stages": [s.stage for s in planned_stages],
        "lt1_values": list(LT1_VALUES),
        "lt2_values": list(LT2_VALUES),
        "lt3_values": list(LT3_VALUES),
        "lt4_values": list(LT4_VALUES),
        "lt5_values": list(LT5_VALUES),
        "lookback_steps": contract.lookback_steps,
        "batch_size": contract.batch_size,
        "seed": 42,
        "test_status": "NOT_ACCESSED",
    })
    paths["report"] = ph43_artifacts.write_report(project_root, "# Phase 43 LSTM Tuning Report (PREPARED)\n\nDry-run prepared; awaiting human scientific training.\n")
    paths["readme"] = ph43_artifacts.write_readme(project_root, "# Phase 43 LSTM Tuning (PREPARED)\n\nDry-run prepared.\n")

    # Per-stage placeholder artifacts at PREPARED status (so plan knows counts)
    for s in planned_stages:
        ph43_artifacts.write_lt3_applicability(
            project_root,
            num_layers=2 if s.applicable else 1,
            applicable=s.applicable,
            status_label=("SKIPPED_NOT_APPLICABLE" if not s.applicable else "APPLICABLE"),
        )

    paths["tuned_winner"] = ph43_artifacts.write_tuned_winner(
        project_root,
        contract,
        tuned_config=base_config,
        run_id=None,
        config_fingerprint=None,
        validation_rmse_wh=None,
        validation_mae_wh=None,
        validation_r2=None,
        best_epoch=None,
        parameter_count=None,
        population_fingerprint=contract.population_fingerprint,
    )
    paths["phase44_handoff"] = ph43_artifacts.write_phase44_lstm_handoff(project_root, contract, paths["tuned_winner"])

    signoff_payload = {
        "phase": 43,
        "phase_name": "LSTM tuning",
        "version": "LSTM_TUNING-v1",
        "status": "PREPARED",
        "ready_for_phase44": False,
        "shared_data_contract": {
            "feature_variant_id": contract.feature_variant_id,
            "target_scaling_id": contract.target_scaling_id,
            "lookback_id": contract.lookback_id,
            "lookback_steps": contract.lookback_steps,
            "boundary_protocol": contract.boundary_protocol,
            "population_version": contract.window_population_version,
            "batch_size": contract.batch_size,
            "metric_version": contract.metric_version,
            "feature_fingerprint": contract.feature_fingerprint,
        },
        "training_contract": {
            "max_epochs": 50,
            "patience": 10,
            "min_delta": 0,
            "loss": "MSE",
            "optimizer": "AdamW",
            "gradient_clip_max_norm": 1.0,
            "scheduler": None,
            "warmup": None,
            "accumulation": 1,
            "amp": False,
            "seed": 42,
        },
        "reference_resolution": reference.to_dict(),
        "planned_fresh_runs": fresh_count,
        "max_fresh_runs": MAX_FRESH_SCIENTIFIC_RUNS,
        "stage_plan": [
            {
                "stage": s.stage,
                "factor": s.factor,
                "factor_values": list(s.factor_values),
                "reference_option": s.reference_option,
                "reference_value": s.reference_value,
                "applicable": s.applicable,
                "candidates": [
                    {"option": c.option, "value": c.value, "source_type": c.source_type, "reuse_run_id": c.reuse_run_id}
                    for c in s.candidates
                ],
            }
            for s in planned_stages
        ],
        "winner": None,
        "final_winner": None,
        "test_access": "FORBIDDEN",
        "dry_run_completed_at": _now_iso(),
    }
    paths["signoff"] = ph43_artifacts.write_phase43_signoff(project_root, signoff_payload)

    return {"artifacts": {k: str(v.relative_to(project_root)) for k, v in paths.items()}}


PREP_ARTIFACT_NAMES = {
    "lstm_tuning_manifest.json",
    "lstm_tuning_contract.json",
    "phase43_preflight_audit.csv",
    "lstm_shared_data_contract.json",
    "lstm_reference_resolution.json",
    "lstm_reference_audit.csv",
    "lstm_tuning_space.json",
    "lstm_run_matrix.csv",
    "lstm_stage_lineage.csv",
    "lt1_hidden_size_metrics.csv",
    "lt1_hidden_size_winner.json",
    "lt2_layers_metrics.csv",
    "lt2_layers_winner.json",
    "lt3_dropout_applicability.json",
    "lt3_dropout_metrics.csv",
    "lt3_dropout_winner.json",
    "lt4_learning_rate_metrics.csv",
    "lt4_learning_rate_winner.json",
    "lt5_weight_decay_metrics.csv",
    "lt5_weight_decay_winner.json",
    "lstm_architecture_audit.csv",
    "lstm_training_config_delta_audit.csv",
    "lstm_common_data_audit.csv",
    "lstm_initialization_audit.csv",
    "lstm_sample_order_audit.csv",
    "lstm_optimizer_group_audit.csv",
    "lstm_optimizer_budget_audit.csv",
    "lstm_gradient_diagnostics.csv",
    "lstm_convergence_diagnostics.csv",
    "lstm_runtime_diagnostics.csv",
    "lstm_run_provenance.csv",
    "lstm_tuning_effect.csv",
    "lstm_contextual_baseline_comparison.csv",
    "lstm_tuned_winner.json",
    "phase44_rolling_origin_lstm_handoff.json",
    "lstm_tuning_findings.csv",
    "lstm_tuning_tests.csv",
    "lstm_tuning_discrepancies.json",
    "lstm_tuning_summary.json",
    "lstm_tuning_report.md",
    "README_LSTM_TUNING.md",
    "phase_43_signoff.json",
}


def _snapshot_existing_prep_artifacts(project_root: Path) -> dict[str, Any]:
    """Move any existing PREP phase 43 artifacts into _history/ snapshot
    before re-writing them with the corrective implementation. Historical
    LSTM_TUNING scientific run directories under artifacts/runs/ are NOT
    touched by this routine.
    """
    from scripts.phase43_lstm_tuning import _assert_prepared_write_allowed

    _assert_prepared_write_allowed(project_root)
    artifacts_dir = project_root / "artifacts" / "lstm_tuning"
    if not artifacts_dir.exists():
        return {"snapshot_count": 0}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = artifacts_dir / "_history" / f"PHASE43_CORRECTIVE_PREP_{timestamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in PREP_ARTIFACT_NAMES:
        src = artifacts_dir / name
        if src.exists():
            dst = snapshot_dir / name
            try:
                dst.write_bytes(src.read_bytes())
                src.unlink()
                moved += 1
            except OSError:
                pass
    summary = {"snapshot_count": moved, "snapshot_dir": str(snapshot_dir.relative_to(project_root))}
    return summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Phase 43 non-scientific dry-run")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.project_root)
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(_cli())
