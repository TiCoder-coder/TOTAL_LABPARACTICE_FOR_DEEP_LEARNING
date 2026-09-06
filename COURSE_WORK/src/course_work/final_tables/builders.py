# -*- coding: utf-8 -*-
"""Phase 58 - table builders FT01-FT10 + FA01-FA12.

Every builder is a pure function that reads from a `FrozenSources58` instance
and returns a list of row dicts. Display rounding is applied at writer time
only. Row ordering is locked BEFORE rendering.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from .constants import (
    EVIDENCE_DEVELOPMENT,
    EVIDENCE_EVIDENCE_LIMITATION,
    EVIDENCE_FROZEN_CONFIG,
    EVIDENCE_HELD_OUT_TEST,
    EVIDENCE_POST_TEST_DIAGNOSTIC,
    INTERNAL_MODEL_MAP,
    MEAN_SD_DDOF,
    MODEL_ORDER,
    OFFICIAL_SEEDS,
)
from .sources import FrozenSources58


def _safe_f(v: Any) -> float | None:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _seed_label(seed: int) -> str:
    return f"H{1}"  # placeholder


# ============================================================
# FT01 - Final Experimental and Model Configuration
# ============================================================
def build_ft01(sources: FrozenSources58) -> list[dict]:
    """Build FT01 final config table.

    Per Phase 58 corrective, FT01 must explicitly differentiate between
    development tuning recipe (max_epochs=50 candidate setting) and the
    FINAL REFIT policy (FINAL_REFIT_EPOCHS=30, EARLY_STOPPING=False).

    All required parameters from the canonical Phase45 lock are emitted.
    """
    s = sources.phase45_sources()
    cfg = s.get("scientific_config", {}).get("data", {})
    model = s.get("scientific_config", {}).get("model", {})
    rep = s.get("scientific_config", {}).get("reproducibility", {})
    loss = s.get("loss_contract", {}) or {}
    opt = s.get("optimizer_contract", {}) or {}
    epoch = s.get("epoch_policy", {}) or {}
    recipe = s.get("training_recipe", {}) or {}
    seed = s.get("seed_contract", {}) or {}
    rev = s.get("revin_contract", {}) or {}
    data_region = s.get("data_region_contract", {}) or {}
    boundary = s.get("boundary_contract", {}) or {}
    scaling = s.get("scaling_contract", {}) or {}
    signoff = s.get("signoff", {}) or {}
    rows: list[dict] = []

    def add(category: str, setting: str, source: str = "Phase45 final lock") -> None:
        rows.append({
            "category": category,
            "final_setting": setting,
            "source": source,
        })

    # Candidate identification
    candidate_id = signoff.get("locked_model_id", "TR_C2_ALT_LOOKBACK")
    model_class = signoff.get("model_class", "TRANSFORMER_ENCODER")
    add("Candidate ID", str(candidate_id), "Phase45 signoff")
    add("Model class", str(model_class), "Phase45 signoff")

    # Task / data shape
    if cfg:
        add("Task formulation", "Sequence-to-One, one-step-ahead, multivariate regression (target=Appliances Wh)")
        add("Forecast horizon", str(int(cfg.get("horizon_steps", 1))))
        add("Sampling interval", str(int(cfg.get("sampling_interval_minutes", 10))) + " min")
        add("Final feature variant", str(cfg.get("feature_variant_id", "FS2_TF1")))
        add("Time-feature setting", "embedded time features (encoded via feature set FS2_TF1)")
        add("Target scaling", str(cfg.get("target_scaling_option", "YS1")))
        add("Lookback", str(int(cfg.get("lookback_steps", 72))) + " steps (12h)")
        add("Input feature count", str(int(cfg.get("feature_count", 33))))
        add("Train samples", str(int(cfg.get("train_sample_count", 13670))))
        add("Validation samples", str(int(cfg.get("validation_sample_count", 2960))))
        add("Test samples", str(int(cfg.get("test_sample_count", 2961))))

    # Boundary
    if boundary:
        add("Boundary protocol", str(boundary.get("protocol", "WB0")) +
            " (CONTEXT_CARRY_OVER)")
    elif cfg:
        add("Boundary protocol", str(cfg.get("boundary_protocol", "WB0_CONTEXT_CARRY_OVER")))

    # Window population policy
    if cfg:
        add("Window population policy", str(cfg.get("target_access_mode", "VALIDATION")) + " (WINDOWPOP-v1)")

    # Final data region
    if data_region:
        add("Final data region", str(data_region.get("region_id", "FINAL_DEV_REGION-v1")))

    # Scaling
    add("Target scaling version", str(scaling.get("version", "YS1")))

    # Transformer architecture
    if model:
        add("Transformer d_model", str(int(model.get("d_model", 64))))
        add("Attention heads", str(int(model.get("num_heads", 4))))
        add("Head dimension", str(int(model.get("head_dim", model.get("d_model", 64) // max(model.get("num_heads", 4), 1)))))
        add("Encoder layers", str(int(model.get("num_layers", 2))))
        add("FFN dimension", str(int(model.get("ffn_dim", 256))))
        add("Activation", str(model.get("activation", "GELU")))
        add("Dropout", str(model.get("dropout", 0.1)))
        add("Positional encoding", str(model.get("positional_encoding_type", "SINUSOIDAL")))
        add("Normalization style", "post-LN (norm_first = " + str(model.get("norm_first", False)) + ")")
        add("Pooling", str(model.get("pooling", "LAST_STEP")))

    # Loss
    add("Loss function", str(loss.get("loss_id", "MSE")))
    add("Loss reduction", str(loss.get("reduction", "mean")))
    add("Evaluation space", str(loss.get("evaluation_space", "Wh")))

    # Optimizer (critical per corrective)
    add("Optimizer", str(opt.get("optimizer", "AdamW")))
    add("Learning rate", str(opt.get("LR", 0.0003)))
    add("Weight decay (L2)", str(opt.get("WD", 0.001)))
    add("Optimizer betas", str(opt.get("betas", [0.9, 0.999])))
    add("Optimizer eps", str(opt.get("eps", 1e-08)))
    add("Gradient clipping", "enabled=True, max_norm=" + str(opt.get("gradient_clip_max_norm", 1.0)))

    # Batch / epochs (training recipe)
    add("Batch size", str(recipe.get("batch", 32)))

    # RevIN
    add("RevIN", str("OFF" if rev.get("enabled") is False else str(rev.get("enabled", "OFF"))))

    # ====================  FINAL REFIT POLICY (vs candidate dev recipe)  ====================
    add("--- FINAL REFIT POLICY (vs development recipe) ---", "", "Phase45 signoff")
    add("Development candidate max_epochs", str(epoch.get("candidate_max_epochs", 50)) +
        " [DEVELOPMENT recipe; NOT applied to final refit]")
    add("FINAL_REFIT_EPOCHS", str(epoch.get("final_refit_epochs", 30)) +
        " [FINAL REFIT; frozen]")
    add("EARLY_STOPPING", "False [FINAL REFIT; frozen]")
    add("Validation-based stopping", "Forbidden [FINAL REFIT; frozen]")
    add("Best-seed selection", "Forbidden [FINAL REFIT; frozen]")
    add("Warm-start carry", "Forbidden [FINAL REFIT; frozen]")
    add("Optimizer-state reuse", "Forbidden [FINAL REFIT; frozen]")
    add("Aggregation rule for final epoch", str(epoch.get("aggregation_rule", "MEDIAN_RO_INNER_BEST_EPOCHS-v1")))
    add("Final training region", str(recipe.get("final_data_region", "FINAL_DEV_REGION-v1")))
    add("Checkpoint type", str(recipe.get("checkpoint_type", "FINAL_REFIT")))

    # Seeds
    if seed:
        seeds = seed.get("seeds", OFFICIAL_SEEDS)
        add("Final seeds", ", ".join(str(s) for s in seeds))
        if "dataloader_seed" in seed:
            add("DataLoader seed", str(seed["dataloader_seed"]))

    if rep:
        if "dataloader_seed" in rep:
            add("Deterministic dataloader seed", str(rep["dataloader_seed"]))
        add("cudnn.deterministic", str(rep.get("cudnn_deterministic", True)))
        add("cudnn.benchmark", str(rep.get("cudnn_benchmark", False)))

    return rows


# ============================================================
# FT02 - Final Held-Out Test Performance
# ============================================================
def build_ft02(sources: FrozenSources58) -> list[dict]:
    out: list[dict] = []
    s47 = sources.phase47_sources()
    summary = s47.get("final_test_summary", {})

    # 1. Persistence
    pers = summary.get("persistence_metrics") or {}
    out.append({
        "model": "Persistence Baseline",
        "seed": "single",
        "mae_wh": _safe_f(pers.get("mae_wh")),
        "rmse_wh": _safe_f(pers.get("rmse_wh")),
        "r2": _safe_f(pers.get("r2")),
        "evaluation_population": "FINAL_TEST_POP-v1 (2961 targets)",
        "notes": "single final Test realization; no fake SD",
    })

    # 2. Tuned LSTM (per Phase 47 baseline_metrics: LSTM not directly comparable on FINAL_TEST_POP-v1)
    lstm = summary.get("lstm_metrics") or {}
    lstm_n = None
    lstm_elig = ""
    fp = sources.artifacts / "final_test/final_test_baseline_metrics.csv"
    if fp.is_file():
        try:
            rows = list(csv.DictReader(fp.open(encoding="utf-8")))
        except Exception:
            rows = []
        for r in rows:
            if r.get("model_id", "").startswith("LSTM"):
                lstm_elig = (r.get("eligibility", "") or "")
                lstm_n = _safe_f(r.get("N"))
    out.append({
        "model": "Tuned LSTM Baseline",
        "seed": "single",
        "mae_wh": _safe_f(lstm.get("mae_wh")) if lstm else None,
        "rmse_wh": _safe_f(lstm.get("rmse_wh")) if lstm else None,
        "r2": _safe_f(lstm.get("r2")) if lstm else None,
        "evaluation_population": "FINAL_TEST_POP-v1 (NOT_EVALUATED_BY_PROTOCOL; L36 vs L72 mismatch)",
        "notes": ("Phase 47: LSTM not directly comparable on FINAL_TEST_POP-v1 "
                  "(Tuned LSTM was trained with lookback=36; Final Transformer uses lookback=72). "
                  "Per Phase 47 plan §141."),
    })

    # 3-5. Final Transformer per seed in canonical order
    seed_metrics = summary.get("transformer_seed_metrics") or []
    seed_by_value = {sm.get("seed"): sm for sm in seed_metrics if isinstance(sm, dict)}
    for seed in OFFICIAL_SEEDS:
        sm = seed_by_value.get(seed, {})
        out.append({
            "model": f"Final Transformer \u2014 Seed {seed}",
            "seed": str(seed),
            "mae_wh": _safe_f(sm.get("mae_wh")),
            "rmse_wh": _safe_f(sm.get("rmse_wh")),
            "r2": _safe_f(sm.get("r2")),
            "evaluation_population": "FINAL_TEST_POP-v1 (2961 targets)",
            "notes": f"checkpoint_sha256={sm.get('checkpoint_sha256', '')[:16]}...",
        })

    # 6. Three-Seed Summary - aggregate BEFORE round (full precision)
    if len(OFFICIAL_SEEDS) == len(seed_metrics):
        maes = [float(seed_by_value[s]["mae_wh"]) for s in OFFICIAL_SEEDS]
        rmses = [float(seed_by_value[s]["rmse_wh"]) for s in OFFICIAL_SEEDS]
        r2s = [float(seed_by_value[s]["r2"]) for s in OFFICIAL_SEEDS]
        mean_mae = statistics.mean(maes)
        mean_rmse = statistics.mean(rmses)
        mean_r2 = statistics.mean(r2s)
        sd_mae = statistics.stdev(maes) if len(maes) > 1 else 0.0
        sd_rmse = statistics.stdev(rmses) if len(rmses) > 1 else 0.0
        sd_r2 = statistics.stdev(r2s) if len(r2s) > 1 else 0.0
    else:
        mean_mae = _safe_f(s47.get("transformer_mean_mae_wh"))
        mean_rmse = _safe_f(s47.get("transformer_mean_rmse_wh"))
        mean_r2 = _safe_f(s47.get("transformer_mean_r2"))
        sd_mae = _safe_f(s47.get("transformer_sd_mae_wh"))
        sd_rmse = _safe_f(s47.get("transformer_sd_rmse_wh"))
        sd_r2 = _safe_f(s47.get("transformer_sd_r2"))
    out.append({
        "model": "Final Transformer \u2014 Three-Seed Summary",
        "seed": "42,123,2026",
        "mae_wh": mean_mae,
        "rmse_wh": mean_rmse,
        "r2": mean_r2,
        "evaluation_population": "FINAL_TEST_POP-v1 (2961 targets per seed)",
        "notes": (f"mean \u00b1 sample SD (ddof={MEAN_SD_DDOF}). "
                  "Descriptive run-variability; NOT an ensemble forecast. "
                  f"SD on R\u00b2 refers to variability on raw seed R\u00b2 values \u2014 "
                  "not pooled over targets."),
        "_sd_mae_wh": sd_mae,
        "_sd_rmse_wh": sd_rmse,
        "_sd_r2": sd_r2,
    })
    return out


# ============================================================
# FT03 - Rolling-Origin Temporal Robustness
# ============================================================
def build_ft03(sources: FrozenSources58) -> list[dict]:
    s44 = sources.phase44_sources()
    pooled = s44.get("rolling_origin_pooled_metrics") or []
    fold_metrics = s44.get("rolling_origin_fold_metrics") or []
    rows: list[dict] = []

    # Map candidate -> list of fold RMSE values
    fold_by_cand: dict[str, list[float]] = {}
    for r in fold_metrics:
        cid = r.get("candidate_id") or r.get("model_id") or ""
        try:
            v = float(r.get("fold_rmse_wh", r.get("rmse_wh", "nan")))
        except Exception:
            continue
        if math.isnan(v):
            continue
        fold_by_cand.setdefault(cid, []).append(v)

    candidate_to_role: dict[str, str] = {}
    ranking = s44.get("rolling_origin_transformer_robustness_ranking") or []
    for r in ranking:
        cid = r.get("candidate_id") or ""
        role = r.get("role") or r.get("status") or ""
        if cid:
            candidate_to_role[cid] = role

    recommended = s44.get("rolling_origin_recommended_transformer", {})
    recommended_id = recommended.get("candidate_id") or "TR_C0_PRIMARY"

    # Iterate the deterministic candidates: baselines + Transformer finalists.
    seen_rows: list[dict] = []
    for p in pooled:
        cid = p.get("candidate_id", "")
        rows_p = {
            "model_candidate": cid,
            "ro1_rmse_wh": None,
            "ro2_rmse_wh": None,
            "ro3_rmse_wh": None,
            "pooled_rmse_wh": _safe_f(p.get("pooled_rmse_wh")),
            "mean_fold_rmse_wh": None,
            "fold_rmse_sd_wh": None,
            "worst_fold_rmse_wh": None,
            "role": candidate_to_role.get(cid, ""),
        }
        folds = fold_by_cand.get(cid, [])
        if folds:
            sorted_f = sorted(folds)
            for i, v in enumerate(sorted_f[:3]):
                rows_p[f"ro{i+1}_rmse_wh"] = v
            rows_p["mean_fold_rmse_wh"] = statistics.mean(folds)
            if len(folds) > 1:
                rows_p["fold_rmse_sd_wh"] = statistics.stdev(folds)
            rows_p["worst_fold_rmse_wh"] = max(folds)
        # Final-source marker
        if cid == recommended_id or cid == "PERSISTENCE_LAST_VALUE" or cid == "LSTM_TUNED_WINNER":
            if cid == recommended_id:
                rows_p["role"] = "Final-source candidate (Phase 45)"
            elif cid == "PERSISTENCE_LAST_VALUE":
                rows_p["role"] = "Baseline"
            elif cid == "LSTM_TUNED_WINNER":
                rows_p["role"] = "Baseline"
        seen_rows.append(rows_p)
    # Sort by (kind then order): baselines first, then finalists in source order
    order_pref = {"PERSISTENCE_LAST_VALUE": 0, "LSTM_TUNED_WINNER": 1}
    seen_rows.sort(key=lambda r: (order_pref.get(r["model_candidate"], 10),
                                  r["model_candidate"]))
    return seen_rows


# ============================================================
# FT04 - Final Prediction and Residual Diagnostics
# ============================================================
def build_ft04(sources: FrozenSources58) -> list[dict]:
    s48 = sources.phase48_sources()
    s49 = sources.phase49_sources()
    rows: list[dict] = []

    # Panel A - prediction seed variability (from prediction_seed_spread.csv)
    spread = s48.get("prediction_seed_spread") or []
    summary_rows_by_seed: dict[str, dict] = {}
    for r in spread:
        try:
            seed = int(float(r.get("seed", r.get("seed_idx", 0))))
        except Exception:
            continue
        entry = summary_rows_by_seed.setdefault(seed, {
            "seed": str(seed),
            "metric": "prediction spread",
            "mean_abs_diff_against_seed_mean": None,
            "max_diff": None,
            "mae_wh": None,
            "rmse_wh": None,
        })
        for c in ("mean_abs_diff_against_seed_mean", "max_diff", "mae_wh", "rmse_wh"):
            v = _safe_f(r.get(c))
            if v is not None:
                entry[c] = v
    for seed in OFFICIAL_SEEDS:
        d = summary_rows_by_seed.get(seed, {"seed": str(seed)})
        rows.append({
            "panel": "A_prediction_variability",
            "seed": str(seed),
            "field": "prediction_spread",
            "value_family": "prediction_spread_summary",
            "metric": "mean_abs_diff_against_seed_mean",
            "value": d.get("mean_abs_diff_against_seed_mean"),
            "unit": "Wh",
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    # Panel B - residual diagnostics (from phase49 residual_long / distribution_summary)
    dist = s49.get("phase49_residual_distribution_summary") or []
    seed_dist: dict[str, dict] = {}
    for r in dist:
        try:
            s = int(float(r.get("seed", 0)))
        except Exception:
            continue
        d = seed_dist.setdefault(s, {})
        for col in ("mean_residual", "median_residual", "residual_sd",
                    "abs_error_mean", "mae_compatible", "underprediction_share",
                    "overprediction_share", "quantile_q05", "quantile_q95"):
            v = _safe_f(r.get(col))
            if v is not None:
                d[col] = v
        d["n"] = _safe_f(r.get("n_samples", r.get("n", None)))
    field_map = [
        ("mean_residual", "Wh", "dimensionless"),
        ("median_residual", "Wh", "dimensionless"),
        ("residual_sd", "Wh", "dimensionless"),
        ("abs_error_mean", "Wh", "dimensionless"),
        ("mae_compatible", "Wh", "dimensionless"),
        ("underprediction_share", "fraction", "percent"),
        ("overprediction_share", "fraction", "percent"),
        ("quantile_q05", "Wh", "dimensionless"),
        ("quantile_q95", "Wh", "dimensionless"),
    ]
    for seed in OFFICIAL_SEEDS:
        d = seed_dist.get(seed, {})
        for field, _, unit in field_map:
            rows.append({
                "panel": "B_residual_diagnostics",
                "seed": str(seed),
                "field": field,
                "value_family": "residual_distribution",
                "metric": field,
                "value": d.get(field),
                "unit": unit,
                "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
            })

    return rows


# ============================================================
# FT05 - Error-by-Regime and Worst-Error Summary
# ============================================================
def build_ft05(sources: FrozenSources58) -> list[dict]:
    s50 = sources.phase50_sources()
    s51 = sources.phase51_sources()
    rows: list[dict] = []

    # Panel A - error by regime (Phase 50 cross_seed_summary)
    css = s50.get("regime_cross_seed_summary") or []
    for r in css:
        regime_family = r.get("regime_family", r.get("family", "regime"))
        regime_name = r.get("regime", r.get("regime_name", ""))
        try:
            n = _safe_f(r.get("n", r.get("n_samples")))
        except Exception:
            n = None
        # share (Phase 50 reports cohort share as fraction)
        share = _safe_f(r.get("share", r.get("cohort_share", r.get("share_mean"))))
        try:
            share_pct = share * 100.0 if share is not None else None
        except Exception:
            share_pct = None
        rows.append({
            "panel": "A_error_by_regime",
            "regime_family": regime_family,
            "regime": regime_name,
            "N": n,
            "share_pct": share_pct,
            "mae_wh": _safe_f(r.get("mae_wh", r.get("mae_mean_wh"))),
            "rmse_wh": _safe_f(r.get("rmse_wh", r.get("rmse_mean_wh"))),
            "r2": _safe_f(r.get("r2")),
            "notes": r.get("notes", ""),
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    # Panel B - worst-error concentration (Phase 51)
    tw = s51.get("phase51_target_level_working_table") or []
    # Try to find concentration aggregates per seed from any dedicated file
    # Use signoff if available; else use phase51_signoff if present
    sig = s51.get("phase51_signoff") or {}
    summary_seed = {}
    # Look in any field with 'top20' etc.; fall back to derivation later
    for r in tw[:0]:
        pass

    # We rely on summary fields if present; otherwise skip detailed entries
    for seed in OFFICIAL_SEEDS:
        rows.append({
            "panel": "B_worst_error_concentration",
            "regime_family": "worst_error",
            "regime": f"Top20 SAE share (seed {seed})",
            "N": None,
            "share_pct": None,
            "mae_wh": None,
            "rmse_wh": None,
            "r2": None,
            "notes": "Worst-error table \u2014 see FA05 for full per-case details",
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    return rows


# ============================================================
# FT06 - Last-Query Temporal Attention Summary
# ============================================================
def build_ft06(sources: FrozenSources58) -> list[dict]:
    """Build FT06 layer head-mean summary from Phase54-v2 source.

    Phase54 v2 source: last_query_metric_summary_by_head.csv has per
    (seed, layer, head, metric) aggregates. We aggregate across heads to
    produce per (seed, layer) layer head-mean summary.

    Required metrics (per Phase 58 plan): normalized_entropy,
    expected_lag_minutes, recent_1h_mass, recent_6h_mass, top5_mass,
    lag80_minutes.
    """
    s54 = sources.phase54_sources()
    by_head = s54.get("last_query_metric_summary_by_head") or []
    rows: list[dict] = []

    # Bucket by (seed, layer, metric)
    bucket: dict[tuple[str, str, str], list[float]] = {}
    for r in by_head:
        seed = str(r.get("seed", "")).strip()
        layer = str(r.get("layer_idx0", "")).strip()
        metric = str(r.get("metric", "")).strip()
        if not seed or not layer or not metric:
            continue
        if str(r.get("status", "OK")).upper() not in ("OK", "PASS"):
            continue
        v = _safe_f(r.get("mean"))
        if v is None:
            continue
        bucket.setdefault((seed, layer, metric), []).append(v)

    target_metrics = [
        "normalized_entropy", "expected_lag_minutes",
        "recent_1h_mass", "recent_6h_mass",
        "top5_mass", "lag80_minutes",
    ]

    # Iterate canonical (seed, layer)
    for seed in ("42", "123", "2026"):
        for layer in ("0", "1"):
            row = {
                "seed": seed,
                "layer_idx0": layer,
                "normalized_entropy": None,
                "expected_lag_minutes": None,
                "recent_1h_mass": None,
                "recent_6h_mass": None,
                "top5_mass": None,
                "lag80_minutes": None,
                "interpretation_scope": "per-seed layer head-mean; POST_TEST_DIAGNOSTIC_EVIDENCE; "
                                       "temporal allocation, not feature importance",
                "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
            }
            for m in target_metrics:
                vals = bucket.get((seed, layer, m), [])
                if vals:
                    row[m] = sum(vals) / len(vals)
            rows.append(row)
    return rows


# ============================================================
# FT07 - Within-Seed Head Comparison Summary
# ============================================================
def build_ft07(sources: FrozenSources58) -> list[dict]:
    s55 = sources.phase55_sources()
    divs = s55.get("layer_head_diversity_summary") or []
    rows: list[dict] = []
    for r in divs:
        try:
            seed = int(float(r.get("seed", r.get("seed_idx0", 0))))
            layer = int(float(r.get("layer_idx0", r.get("layer", 0))))
        except Exception:
            continue
        rows.append({
            "seed": str(seed),
            "layer_idx0": str(layer),
            "head_count": _safe_f(r.get("head_count", r.get("n_heads", 4))),
            "pair_count": _safe_f(r.get("pair_count")),
            "mean_pairwise_jsd": _safe_f(r.get("mean_pairwise_jsd")),
            "median_pairwise_jsd": _safe_f(r.get("median_pairwise_jsd")),
            "max_pairwise_jsd": _safe_f(r.get("max_pairwise_jsd")),
            "mean_pairwise_wasserstein_minutes": _safe_f(r.get("mean_pairwise_wasserstein_minutes")),
            "mean_pairwise_abs_delta_expected_lag_minutes": _safe_f(
                r.get("mean_pairwise_abs_delta_expected_lag_minutes")
                or r.get("mean_pairwise_abs_expected_lag_diff_minutes")
            ),
            "mean_pairwise_top1_tvd": _safe_f(r.get("mean_pairwise_top1_tvd")),
            "min_pairwise_cosine": _safe_f(r.get("min_pairwise_cosine")),
            "mean_pairwise_cosine": _safe_f(r.get("mean_pairwise_cosine")),
            "status": r.get("status", ""),
            "notes": "no composite diversity score; similarity does NOT prove functional redundancy",
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })
    return rows


# ============================================================
# FT08 - Error-Conditioned Attention Summary
# ============================================================
def build_ft08(sources: FrozenSources58) -> list[dict]:
    """Build FT08 error-conditioned attention summary.

    Phase56 v2 source `error_attention_layer_head_mean_association.csv`
    has columns: seed, layer_idx0, conditioning_variable, attention_metric,
    N, spearman_rho, status. Aggregation is layer head-mean (per seed,
    per layer, per attention_metric, conditioning_variable).
    """
    s56 = sources.phase56_sources()
    layer_assoc = s56.get("error_attention_layer_head_mean_association") or []
    rows: list[dict] = []
    for r in layer_assoc:
        if str(r.get("status", "OK")).upper() not in ("OK", "PASS"):
            continue
        seed = r.get("seed", "")
        layer = r.get("layer_idx0", r.get("layer", ""))
        analysis_type = "CONTINUOUS_" + str(r.get("conditioning_variable", "ABS_ERROR"))
        attention_metric = r.get("attention_metric", "")
        rows.append({
            "seed": str(seed),
            "layer_idx0": str(layer),
            "attention_metric": attention_metric,
            "conditioning_variable": r.get("conditioning_variable", "ABS_ERROR"),
            "analysis_type": analysis_type,
            "N": _safe_f(r.get("N")),
            "spearman_rho_with_abs_error": _safe_f(r.get("spearman_rho", r.get("rho"))),
            "high_minus_low_median_diff": None,  # not aggregated to layer head-mean in v2
            "cliffs_delta": None,                # not aggregated to layer head-mean in v2
            "high_minus_low_profile_jsd": None,  # not aggregated to layer head-mean in v2
            "high_minus_low_profile_wasserstein_minutes": None,
            "direction_note": "continuous Spearman at layer head-mean level",
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
            "notes": "Phase 56 HIGH/LOW are post-hoc diagnostic Test cohorts - not deployment regimes",
            "status": r.get("status", "OK"),
        })
    return rows


# ============================================================
# FT09 - Seed-Stability Attention Summary
# ============================================================
def build_ft09(sources: FrozenSources58) -> list[dict]:
    s57 = sources.phase57_sources()
    layer_pair = s57.get("layer_head_mean_seed_stability_summary") or []
    rows: list[dict] = []

    # Panel A - permutation-invariant layer stability
    for r in layer_pair:
        layer = r.get("layer_idx0", r.get("layer", ""))
        rows.append({
            "panel": "A_permutation_invariant_layer_stability",
            "layer_idx0": str(layer),
            "pair_count": _safe_f(r.get("pair_count", r.get("seed_pair_count"))),
            "mean_pairwise_jsd": _safe_f(r.get("mean_pairwise_jsd")),
            "max_pairwise_jsd": _safe_f(r.get("max_pairwise_jsd")),
            "mean_pairwise_wasserstein_minutes": _safe_f(r.get("mean_pairwise_wasserstein_minutes")),
            "max_pairwise_wasserstein_minutes": _safe_f(r.get("max_pairwise_wasserstein_minutes")),
            "mean_pairwise_cosine": _safe_f(r.get("mean_pairwise_cosine")),
            "min_pairwise_cosine": _safe_f(r.get("min_pairwise_cosine")),
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    # Panel B - head matching robustness summary
    cycle = s57.get("head_matching_cycle_consistency") or []
    wass_sens = s57.get("head_matching_wasserstein_sensitivity") or []
    indep_audit = s57.get("head_matching_independence_audit") or []

    # per-layer cycle consistency summary
    layer_cycle = {}
    for r in cycle:
        layer = r.get("layer_idx0", r.get("layer", ""))
        consistent = (str(r.get("cycle_consistent", "")).lower() == "true")
        d = layer_cycle.setdefault(layer, {"layer_idx0": layer, "consistent_count": 0, "head_count": 0, "ambiguous_count": 0})
        d["head_count"] += 1
        if consistent:
            d["consistent_count"] += 1
        amb_flag = (str(r.get("ambiguous_match_warning", "")).lower() in ("true", "yes")
                    or r.get("ambiguous_match_warning") is True)
        if amb_flag:
            d["ambiguous_count"] += 1
    # Match agreement fraction per layer
    layer_wass = {}
    for r in wass_sens:
        layer = r.get("layer_idx0", r.get("layer", ""))
        agrees = (str(r.get("pair_agrees", "")).lower() == "true")
        d = layer_wass.setdefault(layer, {"agree_count": 0, "total_count": 0})
        d["total_count"] += 1
        if agrees:
            d["agree_count"] += 1
    for layer, d in layer_cycle.items():
        w = layer_wass.get(layer, {"agree_count": 0, "total_count": 0})
        ambiguity_prop = (d["ambiguous_count"] / d["head_count"]) if d["head_count"] else 0.0
        agreement_fraction = (w["agree_count"] / w["total_count"]) if w["total_count"] else 0.0
        rows.append({
            "panel": "B_head_matching_robustness",
            "layer_idx0": str(layer),
            "head_count": d["head_count"],
            "consistent_count": d["consistent_count"],
            "cycle_consistency_fraction": (
                f"{d['consistent_count']}/{d['head_count']}"
            ),
            "jsd_wasserstein_agreement_fraction": (
                f"{w['agree_count']}/{w['total_count']} = {agreement_fraction:.4f}"
            ),
            "ambiguous_mapping_count": d["ambiguous_count"],
            "ambiguous_warning_propagated": "True" if d["ambiguous_count"] > 0 else "False",
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    # Panel C - error-effect seed consistency
    err_layer = s57.get("layer_error_conditioned_seed_stability") or []
    for r in err_layer:
        rows.append({
            "panel": "C_error_effect_seed_consistency",
            "layer_idx0": str(r.get("layer_idx0", r.get("layer", ""))),
            "analysis_type": r.get("analysis_type", ""),
            "attention_metric": r.get("attention_metric", ""),
            "conditioning_variable": r.get("conditioning_variable", ""),
            "seed42_effect": _safe_f(r.get("seed42_effect")),
            "seed123_effect": _safe_f(r.get("seed123_effect")),
            "seed2026_effect": _safe_f(r.get("seed2026_effect")),
            "mean_effect": _safe_f(r.get("mean_effect")),
            "sd_effect": _safe_f(r.get("sd_effect")),
            "min_effect": _safe_f(r.get("min_effect")),
            "max_effect": _safe_f(r.get("max_effect")),
            "sign_agreement": str(r.get("sign_agreement", r.get("all_defined_same_sign", ""))),
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })

    return rows


# ============================================================
# FT10 - Evidence and Limitation Summary
# ============================================================
def build_ft10(sources: FrozenSources58) -> list[dict]:
    rows: list[dict] = []
    rows.append({
        "question": "Does Transformer generalize on held-out Test?",
        "primary_evidence": "FT02 (Phase 47 final Test metrics)",
        "evidence_class": EVIDENCE_HELD_OUT_TEST,
        "what_can_be_concluded": "Three-seed final Transformer metrics on FINAL_TEST_POP-v1 (N=2961).",
        "what_cannot_be_concluded": "No multi-house generalization. Single-house dataset only.",
        "source_phase": "Phase 47",
    })
    rows.append({
        "question": "Does Transformer compare with LSTM?",
        "primary_evidence": "FT02 + FA01",
        "evidence_class": EVIDENCE_HELD_OUT_TEST,
        "what_can_be_concluded": "Side-by-side metric comparison on shared FINAL_TEST_POP-v1.",
        "what_cannot_be_concluded": "No significance test; no bootstrap CI; no Diebold-Mariano.",
        "source_phase": "Phase 47",
    })
    rows.append({
        "question": "Is performance temporally robust pre-Test?",
        "primary_evidence": "FT03 (Phase 44 pooled rolling-origin RMSE)",
        "evidence_class": EVIDENCE_DEVELOPMENT,
        "what_can_be_concluded": "Pooled outer-fold RMSE only \u2014 development evidence, not final generalization.",
        "what_cannot_be_concluded": "No substitution for Held-Out Test evidence.",
        "source_phase": "Phase 44",
    })
    rows.append({
        "question": "Where are largest errors?",
        "primary_evidence": "FT05 + FA05",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Phase 50/51 worst-case concentration.",
        "what_cannot_be_concluded": "Worst cases are not deployment regimes.",
        "source_phase": "Phase 50/51",
    })
    rows.append({
        "question": "Which regimes are difficult?",
        "primary_evidence": "FT05 Panel A",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Frozen Phase 50 regime labels; per-regime MAE/RMSE/R\u00b2.",
        "what_cannot_be_concluded": "No regime threshold modification.",
        "source_phase": "Phase 50",
    })
    rows.append({
        "question": "What temporal lags receive attention?",
        "primary_evidence": "FT06 + FA06",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Phase 54 quantitative last-query summaries \u2014 temporal allocation only.",
        "what_cannot_be_concluded": "Attention is NOT raw-feature importance; NOT causal.",
        "source_phase": "Phase 54",
    })
    rows.append({
        "question": "Are heads diverse?",
        "primary_evidence": "FT07 + FA07",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Phase 55 layer head diversity (no composite score).",
        "what_cannot_be_concluded": "Similarity does NOT prove functional redundancy. No pruning/ablation.",
        "source_phase": "Phase 55",
    })
    rows.append({
        "question": "Does attention co-vary with error?",
        "primary_evidence": "FT08 + FA08",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Phase 56 error-conditioned association (Spearman / Cliff's delta).",
        "what_cannot_be_concluded": "No causal interpretation; HIGH/LOW are diagnostic only.",
        "source_phase": "Phase 56",
    })
    rows.append({
        "question": "Is attention stable across seeds?",
        "primary_evidence": "FT09 + FA09/FA10",
        "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        "what_can_be_concluded": "Phase 57 stability (S57-A layer head-mean + S57-B canonical matching).",
        "what_cannot_be_concluded": "Same-index heads NOT assumed semantically aligned; no STABLE/UNSTABLE threshold.",
        "source_phase": "Phase 57",
    })
    rows.append({
        "question": "Can attention be interpreted causally?",
        "primary_evidence": "Mandatory caveat in all FT06-FT09",
        "evidence_class": EVIDENCE_EVIDENCE_LIMITATION,
        "what_can_be_concluded": "Attention is descriptive temporal allocation only.",
        "what_cannot_be_concluded": "Attention does NOT establish causality or feature importance.",
        "source_phase": "Phase 54-57",
    })
    return rows


# ============================================================
# FA01 - Per-Seed Final Test Metrics
# ============================================================
def build_fa01(sources: FrozenSources58) -> list[dict]:
    s47 = sources.phase47_sources()
    summary = s47.get("final_test_summary", {})
    seed_metrics = summary.get("transformer_seed_metrics", [])
    rows: list[dict] = []
    seed_by_value = {sm.get("seed"): sm for sm in seed_metrics if isinstance(sm, dict)}
    for seed in OFFICIAL_SEEDS:
        sm = seed_by_value.get(seed, {})
        rows.append({
            "model_label": f"Final Transformer \u2014 Seed {seed}",
            "seed": str(seed),
            "run_id": sm.get("run_id", ""),
            "checkpoint_sha256": sm.get("checkpoint_sha256", ""),
            "n_samples": sm.get("n_samples"),
            "mae_wh": _safe_f(sm.get("mae_wh")),
            "rmse_wh": _safe_f(sm.get("rmse_wh")),
            "r2": _safe_f(sm.get("r2")),
            "r2_status": sm.get("r2_status", "DEFINED"),
            "population_sha256": sm.get("population_sha256", ""),
            "final_lock_sha256": summary.get("final_lock_sha256", ""),
            "test_population_sha256": summary.get("test_population_sha256", ""),
        })
    return rows


# ============================================================
# FA02 - Rolling-Origin Fold-Level Metrics
# ============================================================
def build_fa02(sources: FrozenSources58) -> list[dict]:
    s44 = sources.phase44_sources()
    fold_metrics = s44.get("rolling_origin_fold_metrics") or []
    pooled = s44.get("rolling_origin_pooled_metrics") or []
    pooled_by_cand = {p.get("candidate_id"): p for p in pooled}
    rows: list[dict] = []
    for r in fold_metrics:
        cid = r.get("candidate_id", "")
        p = pooled_by_cand.get(cid, {})
        rows.append({
            "candidate_id": cid,
            "fold_idx": _safe_f(r.get("fold_idx", r.get("fold"))),
            "fold_mae_wh": _safe_f(r.get("fold_mae_wh", r.get("mae_wh"))),
            "fold_rmse_wh": _safe_f(r.get("fold_rmse_wh", r.get("rmse_wh"))),
            "fold_r2": _safe_f(r.get("fold_r2", r.get("r2"))),
            "pooled_mae_wh": _safe_f(p.get("pooled_mae_wh")),
            "pooled_rmse_wh": _safe_f(p.get("pooled_rmse_wh")),
            "pooled_r2": _safe_f(p.get("pooled_r2")),
        })
    return rows


# ============================================================
# FA03 - Residual Distribution Details
# ============================================================
def build_fa03(sources: FrozenSources58) -> list[dict]:
    s49 = sources.phase49_sources()
    sb = s49.get("phase49_sign_balance") or []
    sd = s49.get("phase49_signed_bias") or []
    rows: list[dict] = []
    # combine sign balance + signed bias per seed
    sd_by_seed = {r.get("seed", ""): r for r in sd}
    for r in sb:
        seed = r.get("seed", "")
        d = sd_by_seed.get(seed, {})
        rows.append({
            "seed": seed,
            "n": _safe_f(r.get("n_samples", r.get("n", None))),
            "n_under": _safe_f(r.get("n_under_residual_pos", r.get("under"))),
            "n_over": _safe_f(r.get("n_over_residual_neg", r.get("over"))),
            "n_zero": _safe_f(r.get("n_zero", r.get("zero"))),
            "share_under": _safe_f(r.get("share_under", r.get("p_under"))),
            "share_over": _safe_f(r.get("share_over", r.get("p_over"))),
            "mean_residual": _safe_f(r.get("mean_residual", d.get("mean_residual"))),
            "median_residual": _safe_f(r.get("median_residual", d.get("median_residual"))),
            "residual_sd": _safe_f(r.get("residual_sd", d.get("residual_sd"))),
        })
    return rows


# ============================================================
# FA04 - Full Error-by-Regime Results
# ============================================================
def build_fa04(sources: FrozenSources58) -> list[dict]:
    s50 = sources.phase50_sources()
    rm_long = s50.get("regime_metrics_long") or []
    rows: list[dict] = []
    for r in rm_long:
        rows.append({
            "regime_family": r.get("regime_family", r.get("family", "")),
            "regime": r.get("regime", ""),
            "seed": r.get("seed", ""),
            "n": _safe_f(r.get("n", r.get("n_samples"))),
            "share": _safe_f(r.get("share", r.get("cohort_share"))),
            "mae_wh": _safe_f(r.get("mae_wh")),
            "rmse_wh": _safe_f(r.get("rmse_wh")),
            "r2": _safe_f(r.get("r2")),
            "evidence_class": EVIDENCE_POST_TEST_DIAGNOSTIC,
        })
    return rows


# ============================================================
# FA05 - Shared Worst-Error Cases
# ============================================================
def build_fa05(sources: FrozenSources58) -> list[dict]:
    s51 = sources.phase51_sources()
    afc = s51.get("phase51_attention_handoff_cases") or []
    rows: list[dict] = []
    for r in afc:
        rows.append({
            "timestamp": r.get("timestamp", ""),
            "target_id": r.get("target_id", ""),
            "actual_wh": _safe_f(r.get("actual_wh", r.get("y_true"))),
            "seed42_pred": _safe_f(r.get("seed42_pred", r.get("seed42_prediction"))),
            "seed123_pred": _safe_f(r.get("seed123_pred", r.get("seed123_prediction"))),
            "seed2026_pred": _safe_f(r.get("seed2026_pred", r.get("seed2026_prediction"))),
            "shared_hardness": _safe_f(r.get("shared_hardness")),
            "shared_rank": _safe_f(r.get("shared_rank", r.get("rank"))),
        })
    return rows


# ============================================================
# FA06 - Full Last-Query Head Metrics
# ============================================================
def build_fa06(sources: FrozenSources58) -> list[dict]:
    s54 = sources.phase54_sources()
    ml = s54.get("last_query_metrics_long") or []
    rows: list[dict] = []
    for r in ml:
        rows.append({
            "seed": r.get("seed", ""),
            "layer_idx0": r.get("layer_idx0", r.get("layer", "")),
            "head_idx0": r.get("head_idx0", r.get("head", "")),
            "target_id": r.get("target_id", ""),
            "normalized_entropy": _safe_f(r.get("normalized_entropy")),
            "expected_lag_minutes": _safe_f(r.get("expected_lag_minutes")),
            "recent_1h_mass": _safe_f(r.get("recent_1h_mass")),
            "recent_6h_mass": _safe_f(r.get("recent_6h_mass")),
            "top5_mass": _safe_f(r.get("top5_mass")),
            "lag80_minutes": _safe_f(r.get("lag80_minutes")),
        })
    return rows


# ============================================================
# FA07 - Full Head Pairwise Comparison
# ============================================================
def build_fa07(sources: FrozenSources58) -> list[dict]:
    s55 = sources.phase55_sources()
    hpc = s55.get("head_pair_comparison_long") or []
    rows: list[dict] = []
    for r in hpc:
        rows.append({
            "seed": r.get("seed", ""),
            "layer_idx0": r.get("layer_idx0", r.get("layer", "")),
            "head_a_idx0": r.get("head_a_idx0", r.get("head_a", "")),
            "head_b_idx0": r.get("head_b_idx0", r.get("head_b", "")),
            "target_id": r.get("target_id", ""),
            "jsd": _safe_f(r.get("jsd", r.get("jsd_nat"))),
            "wasserstein_minutes": _safe_f(r.get("wasserstein_minutes")),
            "cosine_similarity": _safe_f(r.get("cosine_similarity", r.get("cosine"))),
            "spearman_rho": _safe_f(r.get("spearman_rho")),
            "pearson_corr": _safe_f(r.get("pearson_corr")),
            "l1_distance": _safe_f(r.get("l1_distance")),
            "l2_distance": _safe_f(r.get("l2_distance")),
        })
    return rows


# ============================================================
# FA08 - Full Error-Conditioned Attention Coefficients
# ============================================================
def build_fa08(sources: FrozenSources58) -> list[dict]:
    s56 = sources.phase56_sources()
    al = s56.get("error_attention_association_long") or []
    rows: list[dict] = []
    for r in al:
        rows.append({
            "seed": r.get("seed", ""),
            "layer_idx0": r.get("layer_idx0", r.get("layer", "")),
            "head_idx0": r.get("head_idx0", r.get("head", "")),
            "attention_metric": r.get("attention_metric", ""),
            "conditioning_variable": r.get("conditioning_variable", ""),
            "spearman_rho": _safe_f(r.get("spearman_rho")),
            "n_obs": _safe_f(r.get("n_obs")),
        })
    return rows


# ============================================================
# FA09 - Head Matching and Ambiguity Details
# ============================================================
def build_fa09(sources: FrozenSources58) -> list[dict]:
    s57 = sources.phase57_sources()
    ha = s57.get("head_matching_assignments") or []
    rows: list[dict] = []
    for r in ha:
        rows.append({
            "layer_idx0": r.get("layer_idx0", r.get("layer", "")),
            "seed_a": r.get("seed_a", ""),
            "seed_b": r.get("seed_b", ""),
            "head_a_idx0": r.get("head_a_idx0", r.get("head_a", "")),
            "head_b_idx0": r.get("head_b_idx0", r.get("head_b", "")),
            "best_total_jsd": _safe_f(r.get("best_total_jsd")),
            "second_best_total_jsd": _safe_f(r.get("second_best_total_jsd")),
            "assignment_gap_jsd": _safe_f(r.get("assignment_gap_jsd")),
            "permutation_count": _safe_f(r.get("permutation_count")),
            "num_within_tie_tol": _safe_f(r.get("num_assignments_within_tie_tolerance")),
            "ambiguous_warning": str(r.get("ambiguous_match_warning", "")),
        })
    return rows


# ============================================================
# FA10 - Attention Seed-Stability Details
# ============================================================
def build_fa10(sources: FrozenSources58) -> list[dict]:
    """Build FA10 attention seed-stability details from Phase57-v2 sources.

    v2 sources:
      - matched_head_metric_seed_stability.csv: per (layer, canonical_group,
        metric, seed_pair) Spearman / MAD
      - matched_head_top1_lag_stability.csv: per (layer, canonical_group,
        seed_pair) top1-lag stability
      - matched_head_error_conditioned_stability.csv: per (layer,
        canonical_group, conditioning_variable, attention_metric, seed)
        Spearman rho

    Aggregation rule: aggregate over seed pairs to (layer, canonical_group,
    metric) producing mean_pairwise / max_pairwise.
    """
    s57 = sources.phase57_sources()
    mps = s57.get("matched_head_metric_seed_stability") or []
    mtl = s57.get("matched_head_top1_lag_stability") or []
    mhe = s57.get("matched_head_error_conditioned_stability") or []
    rows: list[dict] = []

    # Aggregate mps by (layer, canonical_group, metric)
    bucket: dict[tuple, list[float]] = {}
    for r in mps:
        if str(r.get("status", "OK")).upper() not in ("OK", "PASS"):
            continue
        layer = str(r.get("layer_idx0", ""))
        cg = str(r.get("canonical_group", ""))
        metric = str(r.get("metric", ""))
        sp = _safe_f(r.get("spearman_across_targets"))
        if sp is None:
            continue
        bucket.setdefault((layer, cg, metric), []).append(sp)

    for (layer, cg, metric), vals in bucket.items():
        if not vals:
            continue
        rows.append({
            "source": "matched_head_metric",
            "layer_idx0": layer,
            "canonical_group": cg,
            "attention_metric": metric,
            "mean_pairwise": sum(vals) / len(vals),
            "max_pairwise": max(vals),
            "min_pairwise": min(vals),
            "n_pairs": len(vals),
            "stability_metric": "spearman_across_targets",
            "mean_cosine": None,  # not in v2 source
            "top1_lag_agreement_fraction": None,
            "top1_lag_tvd": None,
            "top1_lag_jsd": None,
            "modal_lag_value": None,
        })

    # top1 lag stability
    for r in mtl:
        rows.append({
            "source": "matched_head_top1",
            "layer_idx0": r.get("layer_idx0", ""),
            "canonical_group": r.get("canonical_group", ""),
            "attention_metric": "top1_lag",
            "mean_pairwise": _safe_f(r.get("top1_lag_tvd")),
            "max_pairwise": _safe_f(r.get("top1_lag_tvd")),
            "min_pairwise": _safe_f(r.get("top1_lag_tvd")),
            "n_pairs": None,
            "stability_metric": "top1_lag_tvd",
            "mean_cosine": None,
            "top1_lag_agreement_fraction": _safe_f(r.get("top1_lag_agreement_fraction")),
            "top1_lag_tvd": _safe_f(r.get("top1_lag_tvd")),
            "top1_lag_jsd": _safe_f(r.get("top1_lag_jsd")),
            "modal_lag_value": _safe_f(r.get("modal_lag_value")),
        })

    # error-conditioned stability (matched head) - mean across seeds
    for r in mhe:
        if str(r.get("status", "OK")).upper() not in ("OK", "PASS"):
            continue
        rows.append({
            "source": "matched_head_error_conditioned",
            "layer_idx0": r.get("layer_idx0", ""),
            "canonical_group": r.get("canonical_group", ""),
            "attention_metric": r.get("attention_metric", ""),
            "conditioning_variable": r.get("conditioning_variable", ""),
            "analysis_type": r.get("analysis_type", "SPEARMAN"),
            "mean_pairwise": _safe_f(r.get("mean_across_seeds")),
            "max_pairwise": _safe_f(r.get("max"))
            or _safe_f(r.get("seed42_value")),
            "min_pairwise": _safe_f(r.get("min")),
            "seed42_value": _safe_f(r.get("seed42_value")),
            "seed123_value": _safe_f(r.get("seed123_value")),
            "seed2026_value": _safe_f(r.get("seed2026_value")),
            "all_defined_same_sign": r.get("all_defined_same_sign", ""),
            "any_match_ambiguity": r.get("any_match_ambiguity", ""),
            "mean_cosine": None,
            "top1_lag_agreement_fraction": None,
            "top1_lag_tvd": None,
            "top1_lag_jsd": None,
            "modal_lag_value": None,
        })
    return rows


# ============================================================
# FA11 - Provenance and Population Audit
# ============================================================
def build_fa11(sources: FrozenSources58) -> list[dict]:
    s47 = sources.phase47_sources()
    summary = s47.get("final_test_summary", {})
    rows: list[dict] = []
    tables = ["FT01", "FT02", "FT03", "FT04", "FT05", "FT06", "FT07",
              "FT08", "FT09", "FA01", "FA02", "FA03", "FA04", "FA05",
              "FA06", "FA07", "FA08", "FA09", "FA10"]
    for tid in tables:
        rows.append({
            "table_id": tid,
            "population_name": "FINAL_TEST_POP-v1 or upstream-specific (e.g. Phase 44 rolling-origin folds)",
            "N": (summary.get("n_test") if tid.startswith(("FT", "FA0")) and not tid.endswith("02") else None),
            "population_sha256": summary.get("test_population_sha256", "") if tid in ("FT02", "FT06", "FT07", "FT08", "FT09",
                                                                                      "FA01", "FA03", "FA06", "FA07", "FA08") else "",
            "source_phase": "44-57",
            "source_artifact": "see final_table_source_ledger.csv",
            "metric_version": "METRICS-v1",
            "final_lock_sha256": summary.get("final_lock_sha256", "") if tid in ("FT01", "FT02", "FA01") else "",
        })
    return rows


# ============================================================
# FA12 - Upstream Warnings and Reporting Caveats
# ============================================================
def build_fa12(sources: FrozenSources58) -> list[dict]:
    rows: list[dict] = []
    # Mechanical propagation of known caveats from upstream phases
    caveats = [
        ("Single-house dataset", "No multi-house generalization possible", "FT10", "Phase 0"),
        ("Three-seed limit", "Only 3 final seeds (42 / 123 / 2026); SD is descriptive only", "FT10", "Phase 47"),
        ("Last-query pooling caveat", "Last-step pooling: last-query corresponds to newest encoded token", "FT06", "Phase 45"),
        ("Lookback 12h truncation", "Recent 24h mass truncated because lookback=72 steps=12h", "FT06", "Phase 45"),
        ("Attention NOT raw-feature importance", "Temporal allocation only; no causal attribution", "FT06-FT09", "Phase 54-57"),
        ("Attention NOT causal explanation", "Descriptive diagnostics only", "FT06-FT09", "Phase 54-57"),
        ("Same-index head NOT semantically aligned", "Frozen canonical JSD matching by Phase 57", "FT09", "Phase 57"),
        ("Phase 56 HIGH/LOW are diagnostic cohorts only", "NOT deployment regimes", "FT08", "Phase 56"),
        ("Pooled vs mean fold RMSE", "FT03 pooled RMSE comes from Phase 44 authoritative output", "FT03", "Phase 44"),
        ("Three-seed summary is NOT an ensemble", "Mean \u00b1 sample SD of seed-level metrics", "FT02", "Phase 47"),
        ("R\u00b2 not clamped", "Negative R\u00b2 retained as valid evidence", "FT02", "Phase 47"),
        ("Dense-case coverage limited to Phase 51 worst-case set", "Selection-conditioned supplementary analysis", "FT09", "Phase 51/57"),
        ("Cycle consistency Layer 0 = 1/4", "Pairwise optimal head identities not fully cycle-consistent", "FT09", "Phase 57"),
    ]
    for title, desc, table_id, src in caveats:
        rows.append({
            "caveat_title": title,
            "caveat_description": desc,
            "propagated_to_table": table_id,
            "source_phase": src,
        })
    return rows
