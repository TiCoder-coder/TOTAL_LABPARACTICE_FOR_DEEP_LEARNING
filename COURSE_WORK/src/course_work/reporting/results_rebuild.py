"""Compact, artifact-backed notebook results for Phase 43+ and V2.

This module is presentation-only.  It reads retained JSON evidence and never
loads datasets, checkpoints, models, scalers, or prediction sources.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML


def _read(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required presentation evidence missing: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {relative}")
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0])
    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_fmt(row.get(c)))}</td>" for c in columns) + "</tr>"
        for row in rows
    )
    return f"<div class='cw-r-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


_STYLE = """
<style>
.cw-results{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;border:1px solid #dbe3ee;border-radius:13px;background:#fff;margin:12px 0 22px;overflow:hidden;text-align:left!important}
.cw-results *{box-sizing:border-box}.cw-results header{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:15px 19px;background:#f4f7fb;border-bottom:1px solid #dbe3ee}
.cw-results h3{font-size:18px;margin:0}.cw-results .status{font-size:11px;font-weight:750;padding:5px 10px;border-radius:999px;background:#e8f7ef;color:#11613d;border:1px solid #a9dec1}
.cw-results .body{padding:14px 19px 18px}.cw-results h4{font-size:13px;margin:14px 0 7px;color:#334155}.cw-results h4:first-child{margin-top:0}
.cw-r-wrap{overflow:auto;border:1px solid #e2e8f0;border-radius:8px}.cw-results table{border-collapse:collapse;width:100%;font-size:12.5px;text-align:left!important}.cw-results th{background:#f7f9fc;color:#475569;text-align:left!important;padding:8px 10px;white-space:nowrap}.cw-results td{text-align:left!important;padding:8px 10px;border-top:1px solid #edf1f5;vertical-align:top}.cw-results tr:nth-child(even){background:#fbfcfe}
</style>
"""


def _render(
    title: str,
    status: Any,
    rows: list[dict[str, Any]],
    sources: list[str],
    subtitle: str = "Results / decision",
) -> HTML:
    if not rows:
        raise RuntimeError(f"Empty important table: {title}")
    # ``sources`` remains part of the internal contract so every rendered table
    # is still backed by artifacts read above. Source paths/checksums are audit
    # metadata and are intentionally omitted from the student-facing notebook.
    if not sources:
        raise RuntimeError(f"Missing internal result source: {title}")

    identity_column = next(iter(rows[0]))
    identities = list(dict.fromkeys(_fmt(row.get(identity_column)) for row in rows))
    status_rows = [
        {"Field": "Section", "Value": title},
        {"Field": "Status", "Value": status},
    ]
    context_rows = [
        {"Field": "Result scope", "Value": subtitle},
        {"Field": "Compared items", "Value": identities},
        {"Field": "Rows displayed", "Value": len(rows)},
    ]
    return HTML(
        f"{_STYLE}<article class='cw-results'><header><h3>{escape(title)}</h3>"
        f"<span class='status'>{escape(_fmt(status))}</span></header><div class='body'>"
        f"<h4>Phase status</h4>{_table(status_rows)}"
        f"<h4>Configuration / analysis</h4>{_table(context_rows)}"
        f"<h4>{escape(subtitle)}</h4>{_table(rows)}"
        "</div></article>"
    )


def render_verified_phase_result(phase_id: int, project_root: Path | str) -> HTML:
    """Render one compact Phase 43-59 table from current canonical evidence."""
    root = Path(project_root).resolve()
    if phase_id == 43:
        p = "artifacts/lstm_tuning/phase_43_signoff.json"; o = _read(root, p)
        rows = [{"Model": "LSTM tuned", "RMSE Wh": o["tuned_validation_rmse_wh"], "MAE Wh": o["tuned_validation_mae_wh"], "R²": o["tuned_validation_r2"], "Decision": o["status"]}]
    elif phase_id == 44:
        p = "artifacts/rolling_origin/phase_44_signoff.json"; o = _read(root, p)
        rows = [{"Winner": o["recommended_transformer_candidate_id"], "Pooled RMSE Wh": o["recommended_pooled_rmse_wh"], "Folds": o["fold_count"], "Test": o["test_status"], "Decision": o["overall_status"]}]
    elif phase_id == 45:
        p = "artifacts/final_model_lock/final_model_lock_summary.json"; o = _read(root, p)
        rows = [{"Locked model": o["locked_model_id"], "Feature set": o["locked_feature_variant_id"], "Lookback": o["locked_lookback_steps"], "Seeds": o["seeds"], "Decision": o["overall_status"]}]
    elif phase_id == 46:
        p = "artifacts/three_seed_final_runs/phase_46_signoff.json"; o = _read(root, p)
        rows = [{"Candidate": o["candidate_id"], "Seeds": o["seed_list"], "Completed": f"{o['completed_run_count']}/{o['planned_run_count']}", "Dev RMSE Wh": o["average_rmse_wh"], "Decision": o["overall_status"]}]
    elif phase_id == 47:
        p = "artifacts/final_test/phase_47_signoff.json"; o = _read(root, p)
        rows = []
        for seed in (42, 123, 2026):
            rows.append({"Model / seed": f"Transformer {seed}", "RMSE Wh": o[f"seed{seed}_rmse_wh"], "MAE Wh": o[f"seed{seed}_mae_wh"], "R²": o[f"seed{seed}_r2"], "Decision": "FROZEN_RESULT"})
        rows += [
            {"Model / seed": "Transformer mean", "RMSE Wh": o["transformer_mean_rmse_wh"], "MAE Wh": o["transformer_mean_mae_wh"], "R²": o["transformer_mean_r2"], "Decision": o["overall_status"]},
            {"Model / seed": "Persistence", "RMSE Wh": o["persistence_rmse_wh"], "MAE Wh": o["persistence_mae_wh"], "R²": o["persistence_r2"], "Decision": "BASELINE"},
        ]
    elif phase_id == 48:
        p = "artifacts/prediction_analysis/phase_48_signoff.json"; o = _read(root, p)
        rows = [{"Check": "Frozen prediction analysis", "Samples": o["n_test"], "Seeds": o["seed_list"], "Findings": o["findings_count"], "Figures": o["figures_count"], "Decision": o["overall_status"]}]
    elif phase_id == 49:
        p = "artifacts/residual_analysis/phase49_summary.json"; o = _read(root, p)
        rows = [{"Seed": seed, "RMSE Wh": m["rmse_wh"], "MAE Wh": m["mae_wh"], "R²": m["r2"], "Mean residual": m["mean_residual"]} for seed, m in o["seed_performance"].items()]
    elif phase_id == 50:
        p = "artifacts/error_by_regime/phase_50_signoff.json"; o = _read(root, p)
        rows = [{"Check": "Error by regime", "Samples": o["n_test"], "Best seed selected": o["best_seed_selected"], "New inference": o["new_test_inference"], "Test-derived threshold": o["test_derived_threshold_used"], "Decision": o["status"]}]
    elif phase_id == 51:
        p = "artifacts/worst_error_analysis/phase51_summary.json"; o = _read(root, p)
        rows = [{"Check": "Worst-error analysis", "Samples": o["candidate_lineage"]["n_test"], "Worst/seed": o["w1_per_seed_count"], "3-seed intersection": o["w1_three_seed_intersection"]["intersection"], "Exact inputs": o["exact_input_verified_count"], "Decision": o["phase51_status"]}]
    elif phase_id == 52:
        p = "artifacts/attention_extraction/attention_extraction_summary.json"; o = _read(root, p)
        rows = [{"Seed": seed, "Max prediction diff": rec["max_abs_difference"], "Equivalence": rec["status"], "Model mutation": o["model_mutation_per_seed"][seed]["status"], "Reproducibility": o["reproducibility_per_seed"][seed]["status"]} for seed, rec in o["prediction_equivalence_per_seed"].items()]
    elif phase_id == 53:
        p = "artifacts/attention_heatmaps/phase_53_signoff.json"; o = _read(root, p)
        rows = [{"Check": "Attention heatmaps", "Seeds": o["seed_list"], "Dense cases": o["dense_case_count"], "Lookback": o["lookback_steps"], "New inference": o["new_test_inference"], "Decision": o["overall_status"]}]
    elif phase_id == 54:
        p = "artifacts/last_query_attention/last_query_attention_summary.json"; o = _read(root, p)
        ent = o["normalized_entropy_summary"]["v2_range"]
        rows = [{"Seed": seed, "Mean normalized entropy": ent[f"mean_seed{seed}"], "Metrics rows": o["n_metrics_rows"], "Lookback": o["lookback"], "Decision": o["overall_status"]} for seed in (42, 123, 2026)]
    elif phase_id == 55:
        p = "artifacts/head_comparison/head_comparison_summary.json"; o = _read(root, p)
        rows = [{"Check": "Head comparison", "Pairs/layer": o["pair_count_per_layer"], "Total pairs": o["total_pair_count"], "Behavior cards": o["head_behavior_card_count"], "Tests": f"{o['tests_pass']}/{o['tests_total']}", "Decision": o["overall_status"]}]
    elif phase_id == 56:
        p = "artifacts/error_conditioned_attention/error_conditioned_attention_summary.json"; o = _read(root, p)
        rows = [{"Check": "Error-conditioned attention", "Seeds": o["seed_list"], "Findings": o["findings_count"], "High/low": o["high_low_metric_status"], "Deciles": o["decile_analysis_status"], "Decision": o["overall_status"]}]
    elif phase_id == 57:
        p = "artifacts/seed_stability_attention/seed_stability_attention_summary.json"; o = _read(root, p)
        rows = [{"Check": "Seed-stability attention", "Seeds": o["seed_list"], "Anchor seed": o["anchor_seed"], "Layer stability": o["layer_stability_status"], "Head matching": o["canonical_matching_status"], "Decision": o["overall_status"]}]
    elif phase_id == 58:
        p = "artifacts/final_tables/phase_58_signoff.json"; o = _read(root, p)
        rows = [{"Package": "Final V1 tables", "Version": o["version"], "CSV": o["csv_package_ready"], "Markdown": o["markdown_package_ready"], "LaTeX": o["latex_package_ready"], "Decision": o["overall_status"]}]
    elif phase_id == 59:
        p = "artifacts/final_conclusions/phase_59_signoff.json"; o = _read(root, p)
        rows = [{"Package": "Final V1 conclusions", "Version": o["version"], "Findings": o["findings_count"], "Tests": f"{o['tests_pass_count']}/{o['tests_count']}", "Post-Test retuning": o["post_test_retuning"], "Decision": o["overall_status"]}]
    else:
        raise ValueError(f"Unsupported rebuilt results phase: {phase_id}")
    return _render(f"Phase {phase_id} — Results", o.get("overall_status", o.get("status", o.get("phase51_status", "PASS"))), rows, [p])


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    return {"RMSE Wh": record.get("rmse_wh"), "MAE Wh": record.get("mae_wh"), "R²": record.get("r2")}


def render_verified_v2_results(project_root: Path | str) -> HTML:
    """Render the concise E01-E20 and final V2 evidence chain."""
    root = Path(project_root).resolve(); base = "artifacts/model_improvement_v2"
    rows: list[dict[str, Any]] = []
    sources: list[str] = []

    def add(exp: str, candidate: str, metrics: dict[str, Any], decision: str, source: str) -> None:
        rows.append({"Step": exp, "Candidate / result": candidate, **metrics, "Decision": decision})
        sources.append(source)

    p = f"{base}/experiments/E01/e01_reproduction_comparison.json"; o = _read(root, p)
    add("E01", o["candidate_id"], {"RMSE Wh": o["observed_rmse_wh"], "MAE Wh": o["observed_mae_wh"], "R²": o["observed_r2"]}, "ACCEPTED BASELINE", p)
    simple = {
        2: ("e02_feature_ablation_comparison.json", "challenger_feature_set", "challenger_metrics", "promotion_status"),
        3: ("e03_prediction_ablation_comparison.json", "challenger_formulation", "challenger_metrics", "promotion_status"),
    }
    for e, (name, candidate_key, metric_key, decision_key) in simple.items():
        p = f"{base}/experiments/E{e:02d}/{name}"; o = _read(root, p)
        add(f"E{e:02d}", o[candidate_key], _metrics(o[metric_key]), o[decision_key], p)
    for e, name in [(4, "e04_head_ablation_comparison.json"), (5, "e05_gate_ablation_comparison.json")]:
        p = f"{base}/experiments/E{e:02d}/{name}"; o = _read(root, p); g = o["global_comparison"]
        add(f"E{e:02d}", "MLP_HEAD" if e == 4 else "RESIDUAL_GATE_ON", _metrics(g["challenger_metrics"]), g["promotion_status"], p)
    for e, name in [(6,"e06_lr_audit_comparison.json"),(7,"e07_scheduler_ablation_comparison.json"),(8,"e08_sgd_lr_ablation_comparison.json"),(9,"e09_sgdm_ablation_comparison.json")]:
        p=f"{base}/experiments/E{e:02d}/{name}"; o=_read(root,p); d=o["decision"]
        add(f"E{e:02d}", d["selected_candidate_id"], {"RMSE Wh": d["best_observed_rmse_wh"], "MAE Wh": None, "R²": None}, d["action"], p)
    for e, name in [(10,"e10_target_delta_ablation_comparison.json"),(11,"e11_rolling_target_ablation_comparison.json"),(12,"e12_lag144_ablation_comparison.json")]:
        p=f"{base}/experiments/E{e:02d}/{name}"; o=_read(root,p)
        add(f"E{e:02d}", "CHALLENGER", _metrics(o["challenger_metrics"]), "NOT_ELIGIBLE (no Human-decision artifact)", p)
    for e, comparison, decision in [
        (13,"e13_hybrid_loss_comparison.json","e13_human_decision.json"),
        (14,"e14_bundle_comparison.json","e14_human_decision.json"),
    ]:
        dp=f"{base}/experiments/E{e:02d}/{decision}"; d=_read(root,dp)
        add(f"E{e:02d}", d["accepted_incumbent"], _metrics(d["accepted_metrics"]), d["decision"], dp)
    p=f"{base}/experiments/E15/e15_capacity_comparison.json"; comparison=_read(root,p)
    dp=f"{base}/experiments/E15/e15_human_decision.json"; decision=_read(root,dp)
    for challenger in comparison["challengers"]:
        add("E15", challenger["candidate_id"], _metrics(challenger["metrics"]), decision["decision"], p)
    sources.append(dp)
    p=f"{base}/experiments/E16/e16_depth_comparison.json"; comparison=_read(root,p)
    dp=f"{base}/experiments/E16/e16_human_decision.json"; decision=_read(root,dp)
    add("E16", comparison["challenger"]["candidate_id"], _metrics(comparison["challenger"]["metrics"]), decision["decision"], p)
    sources.append(dp)
    p=f"{base}/experiments/E16/e16_human_decision.json"; d=_read(root,p)
    add("E17", "Pre-LN", {"RMSE Wh": None,"MAE Wh":None,"R²":None}, d["e17_eligibility"], p)
    p=f"{base}/experiments/E18/e18_human_decision.json"; d=_read(root,p)
    add("E18", d["checkpoint_reuse_classification"], {"RMSE Wh":None,"MAE Wh":None,"R²":None}, f"{d['eligibility_status']}; FT {d['ft_a_b_c_status']}", p)
    p=f"{base}/experiments/E20/e20_config_snapshot.json"; d=_read(root,p)
    add("E19", "Alternative architecture", {"RMSE Wh":None,"MAE Wh":None,"R²":None}, d["finalist_selection"]["e19_status"], p)
    p=f"{base}/experiments/E20/e20_matched_seed_comparison.json"; d=_read(root,p)
    add("E20", "E14-M1 three-seed finalist", {"RMSE Wh":d["across_seed_summary"]["FINALIST"]["mean_rmse_wh"],"MAE Wh":None,"R²":None}, "HUMAN_REVIEW_REQUIRED; final lock records REJECT", p)
    lockp=f"{base}/final_model_lock/v2_final_model_lock.json"; lock=_read(root,lockp)
    dev=lock["development_metrics"]
    add("Step 14A", lock["final_prediction_policy"]["policy_id"], {"RMSE Wh":dev["pooled_rmse_wh"],"MAE Wh":dev["pooled_mae_wh"],"R²":dev["pooled_r2"]}, "SELECT_FINALIST_ENSEMBLE", lockp)
    add("Step 14B", "Persistence-neural blend", {"RMSE Wh":None,"MAE Wh":None,"R²":None}, "KEEP_NEURAL_ENSEMBLE", lockp)
    add("Step 16", lock["final_prediction_policy"]["candidate_id"], {"RMSE Wh":dev["pooled_rmse_wh"],"MAE Wh":dev["pooled_mae_wh"],"R²":dev["pooled_r2"]}, lock["lock_status"], lockp)
    closep=f"{base}/model_improvement_v2_final_closure.json"; close=_read(root,closep); bench=close["post_hoc_benchmark"]["ensemble_metrics"]
    add("Step 17", close["post_hoc_benchmark"]["label"], _metrics(bench), close["post_hoc_benchmark"]["test_status"], closep)
    final_dev=close["final_development_metrics"]
    add("Closure", close["final_policy"]["policy_id"], {"RMSE Wh":final_dev["pooled_rmse_wh"],"MAE Wh":final_dev["pooled_mae_wh"],"R²":final_dev["pooled_r2"]}, close["model_improvement_v2_status"], closep)
    return _render("MODEL_IMPROVEMENT_V2 — Results Summary", close["model_improvement_v2_status"], rows, list(dict.fromkeys(sources)), "Results / decision")
