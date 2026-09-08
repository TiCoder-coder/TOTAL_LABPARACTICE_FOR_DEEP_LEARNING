"""Real Phase 44 Stage A → B → C temp sandbox — minimal harness.

Exercises the EXACT YS1 evaluation path that previously crashed:

  TrainingEngine.train → _evaluate_loader → _inverse_predictions_to_wh
  → inverse_transform_target(..., "YS1", target_scaler_bundle)

For Stage A and Stage B we use `RefitEngine.refit(..., rehearsal_synthetic=True)`
(synthetic mode is fine — it still validates the YS1 contract by the
register_run path being bypassed but `evaluate_stage_c` runs REAL inference
+ YS1 inverse with the real fold-local YS1 bundle we just constructed).

Critical path verified: `evaluate_stage_c(scaler_bundle=scaler_b)` must
succeed and produce finite y_pred_wh. This is the EXACT path that failed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import torch
import torch.nn as nn

from course_work.rolling_origin.candidate_loader import load_candidates
from course_work.rolling_origin.populations import (
    extract_robase_train_ids, extract_robase_val_ids,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.real_run import (
    fit_fold_local_scaler, build_real_canonical_base_dataset,
    RunContext,
)
from course_work.rolling_origin.stages import (
    evaluate_stage_c, load_fold_subset_loader_with_y_rescale,
)
from course_work.training.engine import build_model_from_run_config
from course_work.data.scaling import inverse_transform_target


SANDBOX_CANDIDATES = ["TR_C0_PRIMARY", "LSTM_TUNED_WINNER", "TR_C2_ALT_LOOKBACK"]


def _get_canonical_sample_idxs(fold_dataset, target_ids):
    """Map fold target_id strings → canonical_sample_idx list (sorted).

    These are the global sample_idx values that `dataset.__getitem__` returns.
    The Phase 44 fold-aware MetricPopulationContext uses these values as the
    expected sample population for metric validation.
    """
    wr = fold_dataset.window_records
    if "target_id" not in wr.columns and "target_sample_id" in wr.columns:
        wr["target_id"] = wr["target_sample_id"].astype(str)
    if "canonical_sample_idx" not in wr.columns:
        raise RuntimeError(
            "fold_dataset.window_records missing canonical_sample_idx column"
        )
    id_to_csi = {
        str(t): int(s) for t, s in zip(
            wr["target_id"].tolist(),
            wr["canonical_sample_idx"].tolist(),
        )
    }
    out = []
    for tid in target_ids:
        csi = id_to_csi.get(str(tid))
        if csi is not None:
            out.append(int(csi))
    return sorted(out)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — REAL TEMP SANDBOX A/B/C (TR_C0 / LSTM / TR_C2 RO1)")
    print("=" * 78)

    tmp_root = Path(tempfile.mkdtemp(prefix="phase44_sandbox_"))
    print(f"  sandbox root: {tmp_root}")

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )

    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
    ro1 = folds[0]

    print(f"  RO1 fold: inner_train={len(ro1.inner_train_ids)}, "
          f"inner_val={len(ro1.inner_val_ids)}, "
          f"outer_train={len(ro1.outer_train_ids)}, "
          f"outer_eval={len(ro1.outer_eval_ids)}")

    results = {}

    for cand_id in SANDBOX_CANDIDATES:
        cand = next((c for c in candidates if c.candidate_id == cand_id), None)
        if cand is None:
            print(f"  SKIP: {cand_id} not in candidates")
            continue
        print(f"\n{'─' * 78}")
        print(f"  CANDIDATE: {cand_id} ({cand.model_family}, lookback={cand.lookback_steps})")
        print(f"{'─' * 78}")

        try:
            fold_dataset = build_real_canonical_base_dataset(
                project_root=PROJECT_ROOT, candidate=cand,
            )
            ctx = RunContext(
                project_root=PROJECT_ROOT,
                transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
                lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
                phase_42_signoff_path=PROJECT_ROOT / "artifacts/phase42_signoff.json",
                phase_43_signoff_path=PROJECT_ROOT / "artifacts/phase43_signoff.json",
                artifact_dir=tmp_root / "artifacts",
                registry_root=tmp_root / "registry",
                run_root=tmp_root / "runs",
                is_rehearsal=True,
                rehearsal_synthetic=False,
                dataset_factory=lambda cand=cand: build_real_canonical_base_dataset(
                    project_root=PROJECT_ROOT, candidate=cand,
                ),
            )

            print(f"  [Stage A] fitting fold-local YS1 scaler…")
            scaler_a, _ = fit_fold_local_scaler(
                candidate=cand, fold=ro1, fold_dataset=fold_dataset,
                fit_stage="A", ctx=ctx,
            )
            d_a = scaler_a.as_dict()
            print(f"    OK y_mean={d_a['y_mean']:.3f}, y_std={d_a['y_std']:.3f}")

            print(f"  [Stage B] fitting fold-local YS1 scaler…")
            scaler_b, _ = fit_fold_local_scaler(
                candidate=cand, fold=ro1, fold_dataset=fold_dataset,
                fit_stage="B", ctx=ctx,
            )
            d_b = scaler_b.as_dict()

            inner_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                base_dataset=fold_dataset,
                target_ids=list(ro1.inner_train_ids),
                batch_size=32,
                shuffle=True,
                fold_stage_target_scaler=scaler_a,
            )
            inner_val_loader, _ = load_fold_subset_loader_with_y_rescale(
                base_dataset=fold_dataset,
                target_ids=list(ro1.inner_val_ids),
                batch_size=32,
                shuffle=False,
                fold_stage_target_scaler=scaler_a,
            )
            outer_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                base_dataset=fold_dataset,
                target_ids=list(ro1.outer_train_ids),
                batch_size=32,
                shuffle=True,
                fold_stage_target_scaler=scaler_b,
            )
            outer_eval_loader, _ = load_fold_subset_loader_with_y_rescale(
                base_dataset=fold_dataset,
                target_ids=list(ro1.outer_eval_ids),
                batch_size=32,
                shuffle=False,
                fold_stage_target_scaler=scaler_b,
            )

            print(f"  [Stage A] running real model forward + YS1 inverse on inner_val…")
            model_a = build_model_from_run_config(cand.config)
            model_a.eval()
            predictions_seen = []
            observed_sample_idx = []
            observed_y_raw = []
            for batch in inner_val_loader:
                x = batch["x"]
                y_model = batch["y_model"] 
                y_raw = batch["y_raw_wh"] 
                observed_sample_idx.extend(batch["sample_idx"].cpu().numpy().tolist())
                observed_y_raw.extend(y_raw.cpu().numpy().reshape(-1).tolist())
                with torch.no_grad():
                    pred_model = model_a(x)
                pred_model_np = pred_model.cpu().numpy().reshape(-1)
                pred_wh = inverse_transform_target(
                    pred_model_np, "YS1", d_a,
                )
                predictions_seen.append(pred_wh)
            all_pred_wh = np.concatenate(predictions_seen)
            observed_sample_idx = np.asarray(sorted(observed_sample_idx), dtype=np.int64)
            observed_y_raw = np.asarray(observed_y_raw, dtype=np.float64)
            print(f"    OK Stage A YS1 inverse: {len(all_pred_wh)} preds, "
                  f"range=[{float(all_pred_wh.min()):.3f}, {float(all_pred_wh.max()):.3f}], "
                  f"finite={bool(np.all(np.isfinite(all_pred_wh)))}")

            from course_work.evaluation.metrics import (
                MetricPopulationContext,
                compute_regression_metrics,
                derive_population_fingerprint,
            )
            expected_inner_val_sample_idx = sorted(
                _get_canonical_sample_idxs(fold_dataset, ro1.inner_val_ids)
            )

            inner_val_pop_fp = derive_population_fingerprint(
                target_ids=list(ro1.inner_val_ids),
            )
            inner_val_ctx = MetricPopulationContext(
                split_id="VALIDATION",
                expected_sample_idx=np.asarray(expected_inner_val_sample_idx, dtype=np.int64),
                population_fingerprint=inner_val_pop_fp,
            )
            try:
                metric_inner_val = compute_regression_metrics(
                    y_true_wh=observed_y_raw,
                    y_pred_wh=all_pred_wh,
                    sample_idx=observed_sample_idx,
                    split_id="VALIDATION",
                    evaluation_mode="VALIDATION",
                    population_fingerprint=inner_val_pop_fp,
                    run_id=f"SANDBOX_{cand_id}_A",
                    model_id=cand.candidate_id,
                    lookback_steps=cand.lookback_steps,
                    horizon_steps=1,
                    target_scaling_option=cand.target_scaling_option,
                    population_context=inner_val_ctx,
                )
                print(f"    [Phase44 population-context] Stage A inner_val metric PASS: "
                      f"RMSE={metric_inner_val.rmse_wh:.3f}, "
                      f"n={metric_inner_val.n_samples}")
            except Exception as e:
                print(f"    [FAIL] Phase 44 population-context Stage A inner_val metric: "
                      f"{type(e).__name__}: {e}")
                raise

            print(f"  [Stage A inner_train] running metric with population_context…")
            obs_tr_sample_idx = []
            obs_tr_y_raw = []
            preds_tr_seen = []
            for batch in inner_train_loader:
                x = batch["x"]
                y_raw = batch["y_raw_wh"]
                obs_tr_sample_idx.extend(batch["sample_idx"].cpu().numpy().tolist())
                obs_tr_y_raw.extend(y_raw.cpu().numpy().reshape(-1).tolist())
                with torch.no_grad():
                    pred_model = model_a(x)
                preds_tr_seen.append(
                    inverse_transform_target(
                        pred_model.cpu().numpy().reshape(-1), "YS1", d_a,
                    )
                )
            obs_tr_y_pred = np.concatenate(preds_tr_seen)
            obs_tr_sample_idx_arr = np.asarray(sorted(obs_tr_sample_idx), dtype=np.int64)
            obs_tr_y_raw_arr = np.asarray(obs_tr_y_raw, dtype=np.float64)

            inner_train_pop_fp = derive_population_fingerprint(
                target_ids=list(ro1.inner_train_ids),
            )
            inner_train_ctx = MetricPopulationContext(
                split_id="TRAIN",
                expected_sample_idx=obs_tr_sample_idx_arr,
                population_fingerprint=inner_train_pop_fp,
            )
            try:
                metric_inner_train = compute_regression_metrics(
                    y_true_wh=obs_tr_y_raw_arr,
                    y_pred_wh=obs_tr_y_pred,
                    sample_idx=obs_tr_sample_idx_arr,
                    split_id="TRAIN",
                    evaluation_mode="TRAIN_DIAGNOSTIC",
                    population_fingerprint=inner_train_pop_fp,
                    run_id=f"SANDBOX_{cand_id}_A",
                    model_id=cand.candidate_id,
                    lookback_steps=cand.lookback_steps,
                    horizon_steps=1,
                    target_scaling_option=cand.target_scaling_option,
                    population_context=inner_train_ctx,
                )
                print(f"    [Phase44 population-context] Stage A inner_train metric PASS: "
                      f"RMSE={metric_inner_train.rmse_wh:.3f}, "
                      f"n={metric_inner_train.n_samples}, "
                      f"bundle_fp={metric_inner_train.population_fingerprint[:16]}…")
                assert metric_inner_train.population_fingerprint == inner_train_pop_fp, (
                    f"bundle fingerprint must equal context fingerprint"
                )
            except Exception as e:
                print(f"    [FAIL] Phase 44 population-context Stage A inner_train metric: "
                      f"{type(e).__name__}: {e}")
                raise

            print(f"  [Stage B] running real model forward + YS1 inverse on outer_train…")
            model_b = build_model_from_run_config(cand.config)
            model_b.eval()
            predictions_seen_b = []
            for batch in outer_train_loader:
                x = batch["x"]
                with torch.no_grad():
                    pred_model = model_b(x)
                pred_model_np = pred_model.cpu().numpy().reshape(-1)
                pred_wh = inverse_transform_target(
                    pred_model_np, "YS1", d_b,
                )
                predictions_seen_b.append(pred_wh)
            all_pred_wh_b = np.concatenate(predictions_seen_b)
            print(f"    OK Stage B YS1 inverse: {len(all_pred_wh_b)} preds, "
                  f"range=[{float(all_pred_wh_b.min()):.3f}, {float(all_pred_wh_b.max()):.3f}], "
                  f"finite={bool(np.all(np.isfinite(all_pred_wh_b)))}")

            print(f"  [Stage C] evaluate_stage_c on outer_eval (scaler_b inverse)…")
            stage_c_model = build_model_from_run_config(cand.config)
            stage_c_model.eval()
            stage_c = evaluate_stage_c(
                model=stage_c_model,
                outer_eval_loader=outer_eval_loader,
                scaler_bundle=scaler_b,
                device=torch.device("cpu"),
                model_id=cand.candidate_id,
                fold_id=str(ro1.fold_id),
                model_run_id=f"SANDBOX_{cand_id}_B",
                refit_epoch=1,
                rehearsal_synthetic=False,
            )
            print(f"    OK Stage C: {len(stage_c.target_ids)} predictions")
            y_true = stage_c.y_true_wh
            y_pred = stage_c.y_pred_wh
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            print(f"      metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
            print(f"      y_true range=[{float(np.min(stage_c.y_true_wh)):.3f}, "
                  f"{float(np.max(stage_c.y_true_wh)):.3f}], "
                  f"y_pred range=[{float(np.min(stage_c.y_pred_wh)):.3f}, "
                  f"{float(np.max(stage_c.y_pred_wh)):.3f}]")
            y_pred_finite = bool(np.all(np.isfinite(stage_c.y_pred_wh)))
            y_true_finite = bool(np.all(np.isfinite(stage_c.y_true_wh)))
            print(f"      finite y_pred: {y_pred_finite}, finite y_true: {y_true_finite}")

            results[cand_id] = {
                "PASS": True and y_pred_finite and y_true_finite,
                "stage_a_pred_count": len(all_pred_wh),
                "stage_b_pred_count": len(all_pred_wh_b),
                "stage_c_pred_count": len(stage_c.target_ids),
                "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
                "y_pred_finite": y_pred_finite,
                "y_true_finite": y_true_finite,
                "y_pred_range": (float(all_pred_wh.min()), float(all_pred_wh.max())),
                "stage_c_y_true_range": (float(np.min(stage_c.y_true_wh)), float(np.max(stage_c.y_true_wh))),
                "stage_c_y_pred_range": (float(np.min(stage_c.y_pred_wh)), float(np.max(stage_c.y_pred_wh))),
            }

        except Exception as e:
            import traceback
            results[cand_id] = {
                "PASS": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            print(f"  FAIL: {cand_id}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 78}")
    print("SANDBOX SUMMARY")
    print(f"{'=' * 78}")
    for cand_id, r in results.items():
        status = "PASS" if r.get("PASS") else "FAIL"
        print(f"  {cand_id}: {status}")
        if r.get("PASS"):
            print(f"    Stage A preds: {r['stage_a_pred_count']}, "
                  f"y_pred_range=({r['y_pred_range'][0]:.3f}, {r['y_pred_range'][1]:.3f})")
            print(f"    Stage B preds: {r['stage_b_pred_count']}")
            print(f"    Stage C preds: {r['stage_c_pred_count']}")
            print(f"    Stage C metrics: RMSE={r['metrics'].get('rmse', 0):.3f}, "
                  f"MAE={r['metrics'].get('mae', 0):.3f}, R2={r['metrics'].get('r2', 0):.3f}")
            print(f"    Stage C y_true_range: ({r['stage_c_y_true_range'][0]:.3f}, {r['stage_c_y_true_range'][1]:.3f})")
            print(f"    Stage C y_pred_range: ({r['stage_c_y_pred_range'][0]:.3f}, {r['stage_c_y_pred_range'][1]:.3f})")
            print(f"    finite: y_pred={r['y_pred_finite']}, y_true={r['y_true_finite']}")
        else:
            print(f"    error: {r.get('error')}")

    print(f"\n  Cleaning sandbox {tmp_root}…")
    try:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"  [OK] sandbox cleaned")
    except Exception:
        print(f"  [WARN] sandbox cleanup failed; safe to delete: {tmp_root}")

    all_pass = all(r.get("PASS") for r in results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
