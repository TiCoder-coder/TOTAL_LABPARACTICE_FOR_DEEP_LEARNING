from __future__ import annotations

import argparse
import json

from course_work.diagnostics.learning_diagnostics import recover_phase_22
from course_work.experiments.phase_execution import inspect_phase_state, plan_phase_resume
from course_work.experiments.sweep_recovery import inspect_phase_22_recovery, inspect_recovery_chain
from course_work.reporting.phase_summary import (
    build_phase_processing_log,
    materialize_phase_resume_log,
    save_phase_processing_log,
)
from course_work.sweeps.sweep_results import finalize_verified_sweep
from course_work.utils.environment import recover_environment_revision

if __package__:
    from scripts.run_single_condition import ROOT, run_condition
else:
    from run_single_condition import ROOT, run_condition


PHASE_SWEEP_IDS = {
    23: "S1_FEATURE_SET",
    24: "S2_TIME_FEATURE",
    25: "S3_TARGET_SCALING",
    26: "S4_LOOKBACK",
    27: "S5_POOLING",
    28: "S6_ACTIVATION",
    29: "S7_BATCH_SIZE",
    30: "S8_LEARNING_RATE",
    31: "S9_WEIGHT_DECAY",
    32: "S10_DROPOUT",
    33: "S11_D_MODEL",
    34: "S12_HEADS",
}


def selected_phases(target_phase: int, with_dependencies: bool) -> tuple[int, ...]:
    if target_phase not in PHASE_SWEEP_IDS:
        raise ValueError(f"Unsupported sweep Phase {target_phase}")
    if with_dependencies and target_phase <= 30:
        return tuple(range(23, target_phase + 1))
    return (target_phase,)


def recovery_preview(target_phase: int) -> dict:
    if target_phase <= 30:
        return inspect_recovery_chain(target_phase, ROOT)
    decision = plan_phase_resume(target_phase, ROOT)
    return {
        "target_phase": target_phase,
        "audit_only": True,
        "scientific_artifacts_written": False,
        "earliest_invalid_phase": target_phase if decision["state"] != "VALID_REUSABLE" else None,
        "required_phases": [target_phase] if decision["state"] != "VALID_REUSABLE" else [],
        "direct_target_allowed": decision["readiness"]["ready"],
        "execution_ready": decision["readiness"]["ready"],
        "phases": [
            {
                "phase_id": target_phase,
                "state": decision["state"],
                "missing_conditions": decision["inspection"]["conditions"]["missing_conditions"],
            }
        ],
    }


def _refresh_log(phase_id: int) -> str:
    if phase_id >= 31:
        _, path = materialize_phase_resume_log(phase_id, ROOT)
        return str(path.relative_to(ROOT))
    log = build_phase_processing_log(phase_id, ROOT)
    return str(save_phase_processing_log(log, ROOT).relative_to(ROOT))


def run_pending(
    target_phase: int = 30,
    with_dependencies: bool = False,
    audit_only: bool = False,
    dry_run: bool = False,
    recover_environment: bool = False,
) -> dict:
    preview = recovery_preview(target_phase)
    result = {
        "target_phase": target_phase,
        "with_dependencies": with_dependencies,
        "audit_only": audit_only,
        "dry_run": dry_run,
        "recover_environment": recover_environment,
        "preview": preview,
        "recovered_phases": [],
        "completed_conditions": [],
        "refreshed_logs": [],
        "failed": [],
        "blocked": [],
    }
    if audit_only or dry_run:
        return result
    if target_phase <= 30 and with_dependencies:
        phase_22 = inspect_phase_22_recovery(ROOT)
        if not phase_22["scientifically_reusable"]:
            invalid_baselines = [
                item for item in phase_22["upstream_baselines"] if not item["scientifically_reusable"]
            ]
            if invalid_baselines:
                result["blocked"].extend(
                    {
                        "phase_id": item["phase_id"],
                        "state": item["state"],
                        "reasons": item["issues"],
                    }
                    for item in invalid_baselines
                )
                return result
            try:
                recover_phase_22(ROOT)
            except Exception as error:
                result["failed"].append(
                    {
                        "phase_id": 22,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                return result
            verified_phase_22 = inspect_phase_22_recovery(ROOT)
            if not verified_phase_22["scientifically_reusable"]:
                result["blocked"].append({"phase_id": 22, "reasons": verified_phase_22["issues"]})
                return result
            result["recovered_phases"].append(22)
            result["refreshed_logs"].append(_refresh_log(22))
    environment = preview.get("environment")
    if isinstance(environment, dict) and not environment.get("ready_for_training"):
        if not recover_environment:
            result["blocked"].append({"phase_id": 1, "reasons": environment.get("reasons", [])})
            return result
        try:
            recover_environment_revision(ROOT)
        except Exception as error:
            result["failed"].append(
                {"phase_id": 1, "error_type": type(error).__name__, "error": str(error)}
            )
            return result
        preview = recovery_preview(target_phase)
        result["preview_after_environment_recovery"] = preview
        if not preview["environment"]["ready_for_training"]:
            result["blocked"].append(
                {"phase_id": 1, "reasons": preview["environment"].get("reasons", [])}
            )
            return result
        result["recovered_phases"].append(1)
    for phase_id in selected_phases(target_phase, with_dependencies):
        decision = plan_phase_resume(phase_id, ROOT, allow_execution=True)
        if decision["effective_action"] == "BLOCK" and decision["state"] == "UPSTREAM_INVALID":
            result["blocked"].append(
                {"phase_id": phase_id, "state": decision["state"], "reasons": decision["reasons"]}
            )
            return result
        if decision["state"] in {"CONDITION_INCOMPLETE", "FAILED"}:
            for condition_id in decision["inspection"]["conditions"]["missing_conditions"]:
                try:
                    record = run_condition(PHASE_SWEEP_IDS[phase_id], condition_id)
                    result["completed_conditions"].append(
                        {"phase_id": phase_id, "condition_id": condition_id, "run_id": record["run_id"]}
                    )
                except Exception as error:
                    result["failed"].append(
                        {
                            "phase_id": phase_id,
                            "condition_id": condition_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    return result
                condition_check = inspect_phase_state(phase_id, ROOT)["conditions"]
                if condition_id not in {item["condition_id"] for item in condition_check["verified_conditions"]}:
                    result["blocked"].append(
                        {"phase_id": phase_id, "condition_id": condition_id, "reasons": condition_check}
                    )
                    return result
        current = inspect_phase_state(phase_id, ROOT)
        if not current["conditions"]["complete"]:
            result["blocked"].append(
                {"phase_id": phase_id, "state": current["state"], "reasons": current["conditions"]}
            )
            return result
        if current["state"] != "VALID_REUSABLE":
            try:
                finalize_verified_sweep(phase_id, ROOT, replace_stale=True)
            except Exception as error:
                result["failed"].append(
                    {"phase_id": phase_id, "error_type": type(error).__name__, "error": str(error)}
                )
                return result
            result["recovered_phases"].append(phase_id)
        result["refreshed_logs"].append(_refresh_log(phase_id))
        verification = inspect_phase_state(phase_id, ROOT)
        if verification["state"] != "VALID_REUSABLE":
            result["blocked"].append(
                {"phase_id": phase_id, "state": verification["state"], "reasons": verification}
            )
            return result
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=int, choices=tuple(PHASE_SWEEP_IDS))
    parser.add_argument("--target-phase", type=int, choices=tuple(PHASE_SWEEP_IDS))
    parser.add_argument("--with-dependencies", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover-environment", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase_id is not None and args.target_phase is not None and args.phase_id != args.target_phase:
        raise SystemExit("--phase-id and --target-phase must identify the same Phase")
    target_phase = args.target_phase or args.phase_id or 30
    result = run_pending(
        target_phase,
        args.with_dependencies,
        args.audit_only,
        args.dry_run,
        args.recover_environment,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.audit_only or args.dry_run:
        return 0
    return 1 if result["failed"] or result["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
