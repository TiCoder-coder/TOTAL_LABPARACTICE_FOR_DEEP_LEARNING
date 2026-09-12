"""V2 E10 causal target-delta feature ablation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.data.feature_sets import compute_feature_fingerprint, get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)
from course_work.model_improvement_v2.e10_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    DELTA_FEATURES,
    build_e10_pretest_datasets,
    challenger_feature_order,
)

EXPERIMENT_ID = "E10"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E10"
REGISTRY_NAMESPACE = "V2_E10"
SEED = 42
CONTROL_ID = "TR_C2_ALT_LOOKBACK_E10_CONTROL"
CHALLENGER_ID = "TR_C2_ALT_LOOKBACK_E10_DELTA3"
CONFIG_PATH = Path("artifacts/model_improvement_v2/experiments/E10/e10_config_snapshot.json")
EXPECTED_INCUMBENT = {"rmse_wh": 59.85291570400546, "mae_wh": 26.650501720144604, "r2": 0.5789433617557903}


class E10PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E10PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E10PreflightError(f"Expected JSON object: {path}")
    return value


def load_e10_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_PATH)


def _audit_incumbent(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    incumbent = document.get("accepted_incumbent", {})
    if incumbent.get("source_experiment") != "E01" or incumbent.get("pooled_metrics") != EXPECTED_INCUMBENT or incumbent.get("test_status") != "NOT_ACCESSED":
        raise E10PreflightError("E10 accepted incumbent lock mismatch")
    e01 = _read_json(root / "artifacts/model_improvement_v2/experiments/E01/e01_reproduction_comparison.json")
    if ({"rmse_wh": e01.get("observed_rmse_wh"), "mae_wh": e01.get("observed_mae_wh"), "r2": e01.get("observed_r2")} != EXPECTED_INCUMBENT or e01.get("test_status") != "NOT_ACCESSED"):
        raise E10PreflightError("E01 evidence no longer matches accepted incumbent")
    checks = {
        "E02": ("e02_feature_ablation_comparison.json", "promotion_status", "NO_MEANINGFUL_IMPROVEMENT / TIE"),
        "E03": ("e03_prediction_ablation_comparison.json", "promotion_status", "NO_MEANINGFUL_IMPROVEMENT / TIE"),
        "E06": ("e06_lr_audit_comparison.json", "decision.action", "RETAIN_E01_LR_3E4"),
        "E07": ("e07_scheduler_ablation_comparison.json", "decision.action", "RETAIN_E01_SCHEDULER_OFF"),
        "E08": ("e08_sgd_lr_ablation_comparison.json", "decision.action", "RETAIN_E01_ADAMW"),
        "E09": ("e09_sgdm_ablation_comparison.json", "decision.action", "RETAIN_E01_ADAMW"),
    }
    for experiment, (name, field, expected) in checks.items():
        evidence = _read_json(root / f"artifacts/model_improvement_v2/experiments/{experiment}/{name}")
        actual: Any = evidence
        for part in field.split("."):
            actual = actual.get(part, {}) if isinstance(actual, dict) else None
        if actual != expected or evidence.get("test_status") != "NOT_ACCESSED":
            raise E10PreflightError(f"{experiment} predecessor decision/Test status mismatch")
    for experiment, name in (("E04", "e04_head_ablation_comparison.json"), ("E05", "e05_gate_ablation_comparison.json")):
        evidence = _read_json(root / f"artifacts/model_improvement_v2/experiments/{experiment}/{name}")
        if not str(evidence.get("global_comparison", {}).get("promotion_status", "")).startswith("DO_NOT_PROMOTE") or evidence.get("test_status") != "NOT_ACCESSED":
            raise E10PreflightError(f"{experiment} predecessor decision/Test status mismatch")
    return {"status": "PASS", "accepted_incumbent": "E01", "experiments_audited": list(checks) + ["E04", "E05"]}


def _expected_configs(root: Path) -> tuple[dict, dict]:
    base = deepcopy(_read_json(root / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json")["v1_config"])
    challenger = deepcopy(base)
    order = challenger_feature_order()
    challenger["data"]["feature_variant_id"] = CHALLENGER_VARIANT
    challenger["data"]["feature_count"] = len(order)
    challenger["model"]["input_size"] = len(order)
    challenger["lineage"]["feature_fingerprint"] = compute_feature_fingerprint(CHALLENGER_VARIANT, list(order))
    # These are provenance identifiers for the fresh per-fold scaling policy;
    # fitted scaler checksums are produced by each Stage A/B run.
    challenger["lineage"]["scaler_bundle_id"] = "FOLD_LOCAL__E10__FS2_TF1_DELTA3"
    challenger["lineage"]["scaler_bundle_checksum"] = (
        "776e80ad79edbfbb1d93b9d49cbc920ea0acc048e57d962e9cb47d973137bcdf"
    )
    return base, challenger


def validate_e10_document(document: Mapping[str, Any], root: Path, *, materialize_data: bool = True) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E10PreflightError("Wrong E10 identity/track")
    incumbent = _audit_incumbent(root, document)
    primary = document.get("primary_change", {})
    definitions = primary.get("definitions", {})
    expected_definitions = {"appliances_delta_1": "y[s] - y[s-1]", "appliances_abs_delta_1": "abs(y[s] - y[s-1])", "appliances_delta_2": "y[s] - y[s-2]"}
    if primary.get("delta_features") != list(DELTA_FEATURES) or definitions != expected_definitions or primary.get("past_target_only") is not True or primary.get("imputation") != "NONE":
        raise E10PreflightError("E10 causal delta definitions/order mismatch")
    control, challenger = document.get("control", {}), document.get("challenger", {})
    expected_control, expected_challenger = _expected_configs(root)
    if control.get("candidate_id") != CONTROL_ID or challenger.get("candidate_id") != CHALLENGER_ID:
        raise E10PreflightError("E10 candidate identity mismatch")
    if control.get("config") != expected_control or control.get("training_required") is not True or control.get("reuse_e01_metrics") is not False:
        raise E10PreflightError("E10 control must retrain E01-equivalent config on common population")
    if challenger.get("config") != expected_challenger or challenger.get("training_required") is not True:
        raise E10PreflightError("E10 challenger config mismatch")
    for item, variant, order in ((control, CONTROL_VARIANT, tuple(get_feature_list(CONTROL_VARIANT))), (challenger, CHALLENGER_VARIANT, challenger_feature_order())):
        config = item["config"]
        assert_no_test_access(split_id=config["data"]["target_access_mode"])
        if tuple(item.get("feature_order", ())) != order or item.get("feature_count") != len(order) or item.get("config_fingerprint") != compute_config_fingerprint(config):
            raise E10PreflightError(f"E10 {variant} feature/config fingerprint mismatch")
        if item.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD":
            raise E10PreflightError("E10 candidates require fresh initialization")
    changed = assert_one_primary_change(EXPERIMENT_ID, control["config"], challenger["config"])
    fixed_control = deepcopy(control["config"]); fixed_challenger = deepcopy(challenger["config"])
    for cfg in (fixed_control, fixed_challenger):
        for group, keys in (("data", ("feature_variant_id", "feature_count")), ("model", ("input_size",)), ("lineage", ("feature_fingerprint", "scaler_bundle_id", "scaler_bundle_checksum"))):
            for key in keys: cfg[group].pop(key, None)
    if fixed_control != fixed_challenger:
        raise E10PreflightError("E10 changes model/training outside feature variant")
    selection = document.get("selection_policy", {})
    expected_selection = {"primary_metric": "pooled_rmse_wh", "rmse_min_improvement_wh": 0.10, "mae_max_degradation_wh": 0.25, "worst_fold_rmse_max_degradation_wh": 0.50, "fold_rmse_std_max_degradation_wh": 0.50, "all_guardrails_required": True, "weighted_composite": False, "test_metrics_allowed": False, "decision_authority": "HUMAN"}
    if selection != expected_selection:
        raise E10PreflightError("E10 promotion guardrails are not human-locked")
    orchestration = document.get("v2_orchestration", {})
    expected = {"registry_namespace": REGISTRY_NAMESPACE, "execution_track": EXECUTION_TRACK, "training_candidate_count": 2, "expected_stage_a_runs": 6, "expected_stage_b_runs": 6, "expected_total_training_runs": 12, "preflight_consumes_run_id": False, "training_authorized": False, "test_access_authorized": False}
    for key, value in expected.items():
        if orchestration.get(key) != value: raise E10PreflightError(f"E10 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E10" not in value: raise E10PreflightError(f"E10 path mismatch: {key}")
        assert_v2_artifact_path(value)
    population = None
    if materialize_data:
        datasets, evidence, audit = build_e10_pretest_datasets(root)
        ids0 = datasets[CONTROL_VARIANT].window_records["target_id"].tolist()
        ids1 = datasets[CHALLENGER_VARIANT].window_records["target_id"].tolist()
        if ids0 != ids1 or any(v != "PASS" for v in evidence.fingerprint_status.values()) or audit.control.test_rows_read or audit.challenger.test_rows_read:
            raise E10PreflightError("E10 common population/fingerprint/Test firewall failed")
        population = {"status": "PASS", "target_count": len(ids0), "fold_fingerprints": evidence.fingerprint_status, "test_rows_read": 0, "test_target_ids_seen": 0}
    return {"status": "PASS", "experiment_id": EXPERIMENT_ID, "incumbent_audit": incumbent, "one_primary_change": "PASS", "changed_paths": list(changed), "training_candidates": [CONTROL_ID, CHALLENGER_ID], "expected_total_training_runs": 12, "population": population, "fold_local_x_scaling": True, "fold_local_y_scaling": True, "test_access": "NO", "training_executed": False, "inference_executed": False}


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e10_document(load_e10_config(resolved), resolved)


def build_e10_candidates(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec
    specs = []
    for item in (document["control"], document["challenger"]):
        specs.append(CandidateSpec(candidate_id=item["candidate_id"], model_family="TRANSFORMER_ENCODER", shortlist_position=0, config=deepcopy(item["config"]), config_fingerprint=item["config_fingerprint"], feature_variant_id=item["feature_variant_id"], target_scaling_option="YS1", lookback_steps=72, boundary_protocol="WB0_CONTEXT_CARRY_OVER", source_phase="V2_E10_IMMUTABLE_SNAPSHOT", candidate_role=item["role"]))
    return tuple(specs)


def build_e10_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.rolling_origin.real_run import RunContext
    validate_e10_document(document, root, materialize_data=False)
    candidates = build_e10_candidates(document)
    cache: dict[str, Any] = {}
    def dataset_factory(candidate, _fold):
        if candidate.candidate_id not in {CONTROL_ID, CHALLENGER_ID}: raise E10PreflightError("E10 dataset factory rejects candidate")
        if "bundle" not in cache: cache["bundle"] = build_e10_pretest_datasets(root)
        datasets, _evidence, audit = cache["bundle"]
        if audit.control.test_rows_read or audit.challenger.test_rows_read: raise PermissionError("E10 Test firewall failed")
        return datasets[candidate.feature_variant_id]
    _, evidence, _ = build_e10_pretest_datasets(root)
    out = root / "artifacts/model_improvement_v2/experiments/E10"
    feature_sets = {item["feature_variant_id"]: item for item in (document["control"], document["challenger"])}
    upstream = {"lineage": {}, "feature_sets": {"variant_feature_counts": {k: v["feature_count"] for k,v in feature_sets.items()}, "variant_fingerprints": {k: v["config"]["lineage"]["feature_fingerprint"] for k,v in feature_sets.items()}}, "window_fingerprints": {}}
    return RunContext(project_root=root, transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json", lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json", phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json", phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json", artifact_dir=out, registry_root=out/"registry", run_root=out/"runs", scientific_max_epochs=50, scientific_patience=10, seed=SEED, device=str(candidates[0].config["runtime"]["device_type"]), is_rehearsal=False, rehearsal_synthetic=False, reuse_completed_runs=False, dataset_factory=dataset_factory, candidate_specs=candidates, registry_namespace=REGISTRY_NAMESPACE, seed_before_model_construction=True, validated_registry_lifecycle=True, execution_track=EXECUTION_TRACK, registry_upstream_context=upstream, robase_train_ids=evidence.train_ids, robase_val_ids=evidence.validation_ids, apply_fold_x_scaling=True)


def evaluate_promotion(control: Mapping[str, float], challenger: Mapping[str, float], control_fold_rmse: list[float], challenger_fold_rmse: list[float]) -> dict[str, Any]:
    gates = {"rmse_improvement": float(control["rmse_wh"])-float(challenger["rmse_wh"]) >= .10, "mae_guardrail": float(challenger["mae_wh"])-float(control["mae_wh"]) <= .25, "worst_fold_rmse_guardrail": max(challenger_fold_rmse)-max(control_fold_rmse) <= .50, "fold_rmse_std_guardrail": statistics.pstdev(challenger_fold_rmse)-statistics.pstdev(control_fold_rmse) <= .50}
    return {"gates": gates, "eligible_for_human_promotion": all(gates.values()), "automatic_promotion": False}


def _write_success(root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes
    out = root / "artifacts/model_improvement_v2/experiments/E10"
    fold_rows = list(csv.DictReader((out/"rolling_origin_results.csv").open(encoding="utf-8")))
    folds = {cid: [float(r["rmse_wh"]) for r in fold_rows if r["candidate_id"] == cid] for cid in (CONTROL_ID, CHALLENGER_ID)}
    metrics = result.pooled_metrics_by_cid
    decision = evaluate_promotion(metrics[CONTROL_ID], metrics[CHALLENGER_ID], folds[CONTROL_ID], folds[CHALLENGER_ID])
    comparison = {"schema":"MODEL_IMPROVEMENT_V2_E10_DELTA_ABLATION-v1","experiment_id":"E10","primary_change":"CAUSAL_TARGET_DELTA_FEATURE_VARIANT","control_metrics":metrics[CONTROL_ID],"challenger_metrics":metrics[CHALLENGER_ID],"control_fold_rmse_wh":folds[CONTROL_ID],"challenger_fold_rmse_wh":folds[CHALLENGER_ID],"promotion_evaluation":decision,"test_status":"NOT_ACCESSED"}
    manifest = {"schema":"MODEL_IMPROVEMENT_V2_E10_EXECUTION-v1","status":"SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED","run_ids":{"stage_a":{f"{c}:{f}":r for (c,f),r in result.stage_a_run_ids.items()},"stage_b":{f"{c}:{f}":r for (c,f),r in result.stage_b_run_ids.items()}},"preflight":dict(preflight),"test_status":"NOT_ACCESSED"}
    atomic_write_bytes(out/"e10_execution_manifest.json",canonical_json_bytes(manifest)); atomic_write_bytes(out/"e10_target_delta_ablation_comparison.json",canonical_json_bytes(comparison))


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root(); document=load_e10_config(resolved); preflight=validate_e10_document(document,resolved); context=build_e10_run_context(resolved,document)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(context)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(resolved,document,result,preflight); print(result.summary); return 0


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--experiment",choices=["E10"],required=True); parser.add_argument("--mode",choices=["preflight","official"],default="preflight"); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--authorize-training",action="store_true"); args=parser.parse_args(argv)
    if args.seed != 42: print("ERROR: E10 screening seed must be 42",file=sys.stderr); return 2
    if args.mode == "official" and not args.authorize_training: print("REFUSED: E10 official mode requires --authorize-training.",file=sys.stderr); return 3
    if args.mode != "official" and args.authorize_training: print("ERROR: --authorize-training is valid only in official mode.",file=sys.stderr); return 2
    try:
        if args.mode == "official": return run_official()
        print(json.dumps(run_preflight(),indent=2,sort_keys=True)); return 0
    except (E10PreflightError,PermissionError,ValueError) as exc: print(f"E10 {args.mode.upper()} FAIL: {exc}",file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
