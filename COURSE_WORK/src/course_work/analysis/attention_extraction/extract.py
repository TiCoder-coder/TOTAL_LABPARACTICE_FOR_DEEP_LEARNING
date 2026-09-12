"""Phase 52 — main extraction orchestrator.

Reads:
  - frozen Phase 47 prediction bundles
  - frozen Phase 51 attention handoff case set
  - three frozen final Transformer checkpoints (42/123/2026)
  - exact FINAL_TEST_POP-v1 inputs (reconstructed via Phase 51-F logic)

Writes:
  - dense_case_attention_seed{42,123,2026}.npz [K, L, H, L, L] float32
  - last_query_attention_seed{42,123,2026}.npz [N_test, L, H, L] float32
  - raw_attention_checksums.json
  - all audit / ordering / summary / handoff artifacts

STRICT:
  - no training, no fitting, no backward, no optimizer
  - strict load each checkpoint
  - model.eval() + torch.inference_mode
  - no averaging before storage
  - same dense case set + same dense order across all seeds
  - same all-Test target order across all seeds
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audits import (
    model_mutation_check,
    prediction_equivalence_check,
    probability_audit,
    reproducibility_check,
    tensor_integrity_summary,
)
from .inputs import build_all_test_inputs
from .model_io import (
    build_model_from_final_lock,
    extract_batch,
    model_state_fingerprint,
    strict_load_checkpoint,
)
from .sources import (
    DEFAULT_EXTRACTION_BATCH,
    EXTRACTION_BATCH_FALLBACK,
    FEATURE_COUNT,
    LAST_QUERY_STORAGE_AXES,
    LOOKBACK,
    NUM_HEADS,
    NUM_LAYERS,
    PHASE_DIR_REL,
    RAW_DTYPE,
    RAW_DIR_REL,
    REPRODUCIBILITY_PROBE_CASES,
    SEEDS,
    FrozenSources,
    load_frozen_sources,
)
from .summaries import (
    consistency_check,
    full_matrix_summary,
    last_query_summary,
    top_source_summary,
)
from .writers import (
    _sha256_file,
    reload_verify_npz,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)


def _load_phase51_case_set(project_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load canonical Phase51 dense case set.

    The dense case set is the deduplicated target_id set from Phase51.
    """
    fp = project_root / "artifacts/worst_error_analysis/phase51_attention_handoff_cases.csv"
    rows: list[dict[str, Any]] = []
    with fp.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    seen: set[str] = set()
    ordered: list[str] = []
    for r in rows:
        tid = str(r["target_id"])
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered, rows


def _load_phase47_predictions(
    project_root: Path,
    seed: int,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Load frozen Phase 47 prediction bundle for a seed.

    Returns:
      - y_pred_wh map by target_id
      - tuple of target_ids in canonical order
    """
    fp = project_root / f"artifacts/final_test/predictions/final_test_predictions_seed{seed}.csv"
    target_ids: list[str] = []
    y_pred: dict[str, float] = {}
    with fp.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            tid = r["target_id"]
            target_ids.append(tid)
            y_pred[tid] = float(r["y_pred_wh"])
    return y_pred, tuple(target_ids)


def _load_y_inverse_transform(
    project_root: Path,
) -> tuple[float, float, str]:
    """Load frozen FINAL_SCALING-v1 Y inverse transform.

    Returns (mean, scale, option) such that Wh = model_output * scale + mean.

    For YS1 (the locked option), scale and mean come from the registry.
    For YS0 (identity), returns (0.0, 1.0).
    """
    import json

    reg = json.loads((project_root / "artifacts/scaling/final_dev/final_scaler_registry.json").read_text())
    ys0 = reg["target_bundles"]["YS0"]
    ys1 = reg["target_bundles"]["YS1"]

    # Phase 47 stored Wh via the YS1 transform (per final_lock)
    # ys1 is the active bundle; mean/scale come from the registry (deterministic).
    return (
        float(ys1["mean"]),
        float(ys1["scale"]),
        str(ys1["option"]),
    )


def _build_dense_indices(
    all_target_ids: tuple[str, ...],
    dense_target_ids: list[str],
) -> list[int]:
    """Return list of dense target indices into all_target_ids (in dense order)."""
    id_to_idx = {tid: i for i, tid in enumerate(all_target_ids)}
    out: list[int] = []
    for tid in dense_target_ids:
        if tid not in id_to_idx:
            raise RuntimeError(f"Dense target {tid} not in FINAL_TEST_POP-v1")
        out.append(id_to_idx[tid])
    return out


def _select_batch_size(
    model: torch.nn.Module,
    sample_x: torch.Tensor,
    candidates: tuple[int, ...] = (DEFAULT_EXTRACTION_BATCH,) + EXTRACTION_BATCH_FALLBACK,
) -> dict[str, Any]:
    """Try batch sizes in order; return the first that runs."""
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None
    for bs in candidates:
        try:
            if sample_x.shape[0] >= bs:
                x = sample_x[:bs]
            else:
                x = sample_x[:1].repeat(bs, 1, 1, 1)
            with torch.inference_mode():
                _ = extract_batch(model, x)
            return {
                "selected_batch_size": int(bs),
                "attempted_batch_sizes": [int(b) for b in candidates[: candidates.index(bs) + 1]],
                "failure_history": attempts,
                "fallback_used": bool(bs != DEFAULT_EXTRACTION_BATCH),
            }
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            attempts.append({"batch_size": int(bs), "error": str(e)[:200]})
            last_exc = e
            continue
    raise RuntimeError(f"No batch size fits: last error {last_exc}")


def _extract_one_seed(
    seed: int,
    sources: FrozenSources,
    bundle_all: Any,
    bundle_dense: Any,
    dense_target_ids: list[str],
    frozen_y_pred_wh: dict[str, float],
    output_raw_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract attention for one seed and write raw NPZs.

    Returns dict with extraction manifest details, batch audit, prediction
    equivalence stats, model mutation, and reproducibility.
    """
    # Build + strict-load model
    model = build_model_from_final_lock(sources.project_root)
    strict_info = strict_load_checkpoint(
        seed,
        sources.final_checkpoint_per_seed_path[seed],
        model,
    )
    fp_before = model_state_fingerprint(model)

    # Build input batches
    x_all = torch.from_numpy(bundle_all.x).to(device)  # [N, L, F]
    x_dense = torch.from_numpy(bundle_dense.x).to(device)  # [K, L, F]
    N_test = x_all.shape[0]
    K = x_dense.shape[0]

    # All-Test last-query accumulation
    last_query = np.zeros((N_test, NUM_LAYERS, NUM_HEADS, LOOKBACK), dtype=RAW_DTYPE)

    # Per-layer tensor integrity checks (after first batch)
    integrity_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []

    # Prediction equivalence accumulation (computed in Wh coordinate)
    y_mean, y_scale, y_option = _load_y_inverse_transform(sources.project_root)
    pred_equiv_max = 0.0
    pred_equiv_min = float("inf")
    pred_equiv_total = 0
    pred_equiv_fail_count = 0

    # ── Select batch size on first dense batch ────────────────────────────
    probe_x = x_dense[: min(DEFAULT_EXTRACTION_BATCH, x_dense.shape[0])]
    batch_audit = _select_batch_size(model, probe_x)

    # ── Stream over all-Test and dense cases ──────────────────────────────
    # For dense cases, we extract full [B, L, H, L, L] tensors and persist them.
    # For non-dense test cases, we extract only last-query.
    dense_indices = _build_dense_indices(bundle_all.target_ids, dense_target_ids)
    dense_set = set(dense_indices)
    # Pre-allocate dense storage: [K, L, H, L, L] float32
    dense_storage = np.zeros((K, NUM_LAYERS, NUM_HEADS, LOOKBACK, LOOKBACK), dtype=RAW_DTYPE)

    # Map K (dense order) → dense_storage row index
    # dense_indices already in dense_target_ids order; position = list index.
    dense_k_idx = list(range(K))

    # Reproducibility probe (first 2 dense cases)
    repro_probe_tids: list[str] = []
    repro_pred_run1: np.ndarray | None = None
    repro_pred_run2: np.ndarray | None = None
    repro_attn_run1: np.ndarray | None = None
    repro_attn_run2: np.ndarray | None = None
    if K >= REPRODUCIBILITY_PROBE_CASES:
        probe = x_dense[:REPRODUCIBILITY_PROBE_CASES]
        with torch.inference_mode():
            pred1, attn1 = extract_batch(model, probe)
            pred2, attn2 = extract_batch(model, probe)
        repro_pred_run1 = pred1.detach().cpu().numpy().astype(np.float32)
        repro_pred_run2 = pred2.detach().cpu().numpy().astype(np.float32)
        repro_attn_run1 = attn1[-1].detach().cpu().numpy().astype(np.float32)  # last layer
        repro_attn_run2 = attn2[-1].detach().cpu().numpy().astype(np.float32)
        repro_probe_tids = list(dense_target_ids[:REPRODUCIBILITY_PROBE_CASES])

    selected_bs = int(batch_audit["selected_batch_size"])

    # Process all-Test in batches
    for start in range(0, N_test, selected_bs):
        end = min(start + selected_bs, N_test)
        xb = x_all[start:end]
        with torch.inference_mode():
            pred, attn_layers = extract_batch(model, xb)

        # Last-query
        lq = np.stack(
            [al.detach().cpu().numpy().astype(RAW_DTYPE)[:, :, -1, :] for al in attn_layers],
            axis=1,
        )  # [B, L, H, L]
        last_query[start:end] = lq

        # Prediction equivalence check (in Wh coordinate, allclose-style tolerance)
        for j in range(end - start):
            tid = bundle_all.target_ids[start + j]
            fpred = frozen_y_pred_wh.get(tid)
            if fpred is None:
                continue
            ipred_model = float(pred[j, 0].item())
            ipred_wh = ipred_model * y_scale + y_mean
            diff = abs(ipred_wh - fpred)
            tol = 1e-5 + 1e-5 * abs(fpred)
            pred_equiv_max = max(pred_equiv_max, diff)
            pred_equiv_min = min(pred_equiv_min, diff)
            pred_equiv_total += 1
            if diff > tol:
                pred_equiv_fail_count += 1

        # Integrity + probability audit on this batch
        attn_layers_np = [al.detach().cpu().numpy().astype(RAW_DTYPE) for al in attn_layers]
        for layer_idx0, layer_attn in enumerate(attn_layers_np):
            for row in tensor_integrity_summary(seed, [layer_attn]):
                row["batch_start"] = int(start)
                row["batch_end"] = int(end)
                integrity_rows.append(row)
            for row in probability_audit(seed, [layer_attn]):
                row["batch_start"] = int(start)
                row["batch_end"] = int(end)
                probability_rows.append(row)

        # Dense storage
        for j in range(end - start):
            abs_idx = start + j
            if abs_idx in dense_set:
                kpos = dense_indices.index(abs_idx)
                # attn_layers_np[l] has shape [B, H, L, L]
                # dense_storage[kpos, l, h, q, s]
                for layer_idx0 in range(NUM_LAYERS):
                    dense_storage[kpos, layer_idx0, :, :, :] = attn_layers_np[layer_idx0][j]

    # ── Write raw NPZs ──────────────────────────────────────────────────
    raw_dir = output_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    dense_fp = raw_dir / f"dense_case_attention_seed{seed}.npz"
    lq_fp = raw_dir / f"last_query_attention_seed{seed}.npz"

    write_npz_atomic(
        {
            "attention": dense_storage,  # [K, L, H, L, L]
            "target_ids": np.array(bundle_dense.target_ids),
            "layer_count": np.array([NUM_LAYERS], dtype=np.int32),
            "head_count": np.array([NUM_HEADS], dtype=np.int32),
            "lookback_steps": np.array([LOOKBACK], dtype=np.int32),
            "axis_order": np.array(DENSE_STORAGE_AXES := "case_layer_head_query_source"),
            "dtype": np.array(RAW_DTYPE),
            "checkpoint_sha256": np.array(sources.final_checkpoint_per_seed_sha256[seed]),
            "final_lock_sha256": np.array(sources.final_lock_sha256),
            "case_order_sha256": np.array(_sha256_file(
                sources.project_root / "artifacts/worst_error_analysis/phase51_attention_handoff_cases.csv"
            )),
            "position_map_sha256": np.array(sources.attention_verify_serialization_sha256),
        },
        dense_fp,
    )

    write_npz_atomic(
        {
            "last_query_attention": last_query,  # [N, L, H, L]
            "target_ids": np.array(bundle_all.target_ids),
            "layer_count": np.array([NUM_LAYERS], dtype=np.int32),
            "head_count": np.array([NUM_HEADS], dtype=np.int32),
            "lookback_steps": np.array([LOOKBACK], dtype=np.int32),
            "axis_order": np.array("target_layer_head_source"),
            "dtype": np.array(RAW_DTYPE),
            "checkpoint_sha256": np.array(sources.final_checkpoint_per_seed_sha256[seed]),
            "final_lock_sha256": np.array(sources.final_lock_sha256),
            "target_order_sha256": np.array(sources.test_population_sha256),
            "position_map_sha256": np.array(sources.attention_verify_serialization_sha256),
        },
        lq_fp,
    )

    # Reload verify
    dense_reload = reload_verify_npz(
        dense_fp,
        {
            "attention": ((K, NUM_LAYERS, NUM_HEADS, LOOKBACK, LOOKBACK), RAW_DTYPE),
            "target_ids": ((K,), "<U32"),
        },
    )
    lq_reload = reload_verify_npz(
        lq_fp,
        {
            "last_query_attention": ((N_test, NUM_LAYERS, NUM_HEADS, LOOKBACK), RAW_DTYPE),
            "target_ids": ((N_test,), "<U32"),
        },
    )

    # Model mutation
    fp_after = model_state_fingerprint(model)
    mut = model_mutation_check(seed, fp_before, fp_after)

    # Reproducibility
    repro = None
    if repro_pred_run1 is not None:
        repro = reproducibility_check(
            seed,
            repro_pred_run1,
            repro_pred_run2,
            repro_attn_run1,
            repro_attn_run2,
        )

    return {
        "seed": seed,
        "strict_load_info": strict_info,
        "model_mutation": mut,
        "prediction_equivalence": {
            "max_abs_difference": pred_equiv_max,
            "min_abs_difference": pred_equiv_min,
            "N_checked": pred_equiv_total,
            "fail_count": pred_equiv_fail_count,
            "all_pass": pred_equiv_fail_count == 0 and pred_equiv_total == N_test,
        },
        "batch_audit": batch_audit,
        "n_test": N_test,
        "k_dense": K,
        "integrity_rows": integrity_rows,
        "probability_rows": probability_rows,
        "reproducibility": repro,
        "reproducibility_probe_tids": repro_probe_tids,
        "dense_npz_reload": dense_reload,
        "last_query_npz_reload": lq_reload,
        "dense_target_ids": list(bundle_dense.target_ids),
        "all_target_ids": list(bundle_all.target_ids),
    }


def run_phase52_extraction(project_root: Path | None = None) -> dict[str, Any]:
    """Run the full Phase 52 extraction pipeline."""
    root = project_root or Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    sources = load_frozen_sources(root)

    # Dense case set
    dense_target_ids, dense_case_rows = _load_phase51_case_set(root)
    K = len(dense_target_ids)

    # Build inputs (all + dense)
    bundle_all = build_all_test_inputs(project_root=root)
    dense_indices = _build_dense_indices(bundle_all.target_ids, dense_target_ids)
    dense_x = bundle_all.x[dense_indices]
    dense_input_start = [bundle_all.input_start_indices[i] for i in dense_indices]
    dense_input_end = [bundle_all.input_end_indices[i] for i in dense_indices]
    dense_target_indices = [bundle_all.target_indices[i] for i in dense_indices]
    dense_target_timestamps = [bundle_all.target_timestamps[i] for i in dense_indices]
    dense_target_ids_in_bundle = [bundle_all.target_ids[i] for i in dense_indices]
    # Construct bundle_dense-like object via dataclasses
    from .inputs import TestInputsBundle

    bundle_dense = TestInputsBundle(
        x=dense_x,
        target_ids=tuple(dense_target_ids_in_bundle),
        target_timestamps=tuple(dense_target_timestamps),
        target_indices=tuple(dense_target_indices),
        input_start_indices=tuple(dense_input_start),
        input_end_indices=tuple(dense_input_end),
        feature_names=bundle_all.feature_names,
        feature_set_id=bundle_all.feature_set_id,
        feature_fingerprint=bundle_all.feature_fingerprint,
        boundary_protocol=bundle_all.boundary_protocol,
        lookback_steps=bundle_all.lookback_steps,
        horizon_steps=bundle_all.horizon_steps,
        cadence_minutes=bundle_all.cadence_minutes,
        population_id=bundle_all.population_id,
        population_sha256=bundle_all.population_sha256,
        phase10_signoff_sha256=bundle_all.phase10_signoff_sha256,
        final_scaling_x_sha256=bundle_all.final_scaling_x_sha256,
    )

    # Output dirs
    raw_dir = root / "artifacts/attention_extraction/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Run for each seed
    per_seed_results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        # Frozen predictions
        frozen_y_pred, frozen_tids = _load_phase47_predictions(root, seed)
        per_seed_results[seed] = _extract_one_seed(
            seed,
            sources,
            bundle_all,
            bundle_dense,
            dense_target_ids,
            frozen_y_pred,
            raw_dir,
            device="cpu",
        )

    return {
        "sources": sources,
        "K": K,
        "dense_target_ids": dense_target_ids,
        "dense_case_rows": dense_case_rows,
        "bundle_all_target_count": len(bundle_all.target_ids),
        "per_seed": per_seed_results,
        "bundle_all": bundle_all,
        "bundle_dense": bundle_dense,
    }
