"""PHASE 44 — Fold-population metric contract tests.

Verifies:

  1. canonical full TRAIN still rejects incomplete subset
  2. canonical VALIDATION still rejects incomplete subset
  3. Phase 44 explicit inner_train subset passes
  4. Phase 44 explicit inner_val subset passes
  5. outer_train subset passes
  6. outer_eval subset passes
  7. missing ID fails
  8. extra ID fails
  9. duplicate ID fails
 10. wrong fingerprint fails
 11. MetricPopulationContext rejects bad construction
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.evaluation.metrics import (
    MetricPopulationContext,
    align_prediction_bundle_to_expected_population,
    compute_regression_metrics,
    derive_population_fingerprint,
    expected_sample_indices,
    validate_population_coverage,
)


# --- 1. Canonical full TRAIN still rejects incomplete subset ----------

def test_canonical_full_train_rejects_incomplete_subset():
    """Without an explicit MetricPopulationContext, the canonical full
    TRAIN split must still reject any subset.
    """
    expected_full = expected_sample_indices(
        "TRAIN", lookback_steps=144, project_root=PROJECT_ROOT,
    )
    observed = expected_full[:100]  # subset of first 100
    with pytest.raises(ValueError, match="Observed sample population"):
        validate_population_coverage(observed, expected_full)


# --- 2. Canonical VALIDATION still rejects incomplete subset ----------

def test_canonical_validation_rejects_incomplete_subset():
    expected_full = expected_sample_indices(
        "VALIDATION", lookback_steps=144, project_root=PROJECT_ROOT,
    )
    observed = expected_full[:50]
    with pytest.raises(ValueError, match="Observed sample population"):
        validate_population_coverage(observed, expected_full)


# --- 3. Phase 44 inner_train subset passes (with explicit context) -----

def test_phase44_inner_train_subset_passes():
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    # simulate y_true/y_pred for these samples
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
        population_fingerprint=fp, run_id="X", model_id="Y",
        population_context=pop_ctx,
    )
    assert metric.n_samples == len(expected)
    assert np.isfinite(metric.rmse_wh)


# --- 4. Phase 44 inner_val subset passes (with explicit context) -------

def test_phase44_inner_val_subset_passes():
    expected = np.array([100, 200, 300], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="VALIDATION")
    pop_ctx = MetricPopulationContext(
        split_id="VALIDATION", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="VALIDATION", evaluation_mode="VALIDATION",
        population_fingerprint=fp, run_id="X", model_id="Y",
        population_context=pop_ctx,
    )
    assert metric.n_samples == len(expected)


# --- 5. outer_train subset passes (with explicit context) --------------

def test_phase44_outer_train_subset_passes():
    """Stage B refits on outer_train; the metric call uses split_id='TRAIN'
    and an explicit Phase 44 fold-aware MetricPopulationContext to override
    the canonical full-TRAIN default. This is the EXACT path that crashed
    in RUN_TR_ROB_0004.
    """
    expected = np.arange(1000, 1010, dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
        population_fingerprint=fp, run_id="X", model_id="Y",
        population_context=pop_ctx,
    )
    assert metric.n_samples == len(expected)


# --- 6. outer_eval subset passes (with explicit context) ---------------

def test_phase44_outer_eval_subset_passes():
    """Stage C evaluates on outer_eval. MetricPopulationContext.split_id
    must equal the metric call split_id. Outer eval is reported via the
    VALIDATION evaluation mode with explicit fold-aware expected
    population. This was the second crash mode in RUN_TR_ROB_0004.
    """
    expected = np.array([5000, 5001, 5002, 5003], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="VALIDATION")
    pop_ctx = MetricPopulationContext(
        split_id="VALIDATION", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="VALIDATION", evaluation_mode="VALIDATION",
        population_fingerprint=fp, run_id="X", model_id="Y",
        population_context=pop_ctx,
    )
    assert metric.n_samples == len(expected)


# --- 7. Missing ID fails -----------------------------------------------

def test_missing_id_fails():
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=4)
    y_pred = y_true + rng.normal(0, 5, size=4)
    # observed has only 4 IDs (missing one)
    observed = expected[:4]
    with pytest.raises(ValueError, match="Observed sample population"):
        compute_regression_metrics(
            y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=observed,
            split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
            population_fingerprint=fp, run_id="X", model_id="Y",
            population_context=pop_ctx,
        )


# --- 8. Extra ID fails -------------------------------------------------

def test_extra_id_fails():
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=6)
    y_pred = y_true + rng.normal(0, 5, size=6)
    # observed has 6 IDs (one extra)
    observed = np.array([10, 20, 30, 40, 50, 999], dtype=np.int64)
    with pytest.raises(ValueError, match="Observed sample population"):
        compute_regression_metrics(
            y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=observed,
            split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
            population_fingerprint=fp, run_id="X", model_id="Y",
            population_context=pop_ctx,
        )


# --- 9. Duplicate ID fails ---------------------------------------------

def test_duplicate_id_fails():
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=5)
    y_pred = y_true + rng.normal(0, 5, size=5)
    observed = np.array([10, 20, 30, 30, 40], dtype=np.int64)  # duplicate 30
    with pytest.raises(ValueError, match="duplicates"):
        compute_regression_metrics(
            y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=observed,
            split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
            population_fingerprint=fp, run_id="X", model_id="Y",
            population_context=pop_ctx,
        )


# --- 10. Wrong fingerprint fails ---------------------------------------

def test_wrong_fingerprint_fails():
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    real_fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    wrong_fp = "POP-TRAIN-deadbeefdeadbeef"
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=real_fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=5)
    y_pred = y_true + rng.normal(0, 5, size=5)
    observed = expected.copy()
    with pytest.raises(ValueError, match="Population fingerprint mismatch"):
        compute_regression_metrics(
            y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=observed,
            split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
            population_fingerprint=wrong_fp, run_id="X", model_id="Y",
            population_context=pop_ctx,
        )


# --- 11. MetricPopulationContext rejects bad construction ---------------

def test_pop_context_rejects_duplicates_in_expected():
    expected = np.array([10, 20, 30, 30], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    with pytest.raises(ValueError, match="duplicates"):
        MetricPopulationContext(
            split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
        )


def test_pop_context_rejects_empty_expected():
    expected = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="non-empty"):
        MetricPopulationContext(
            split_id="TRAIN", expected_sample_idx=expected, population_fingerprint="X",
        )


def test_pop_context_rejects_missing_fingerprint():
    expected = np.array([10], dtype=np.int64)
    with pytest.raises(ValueError, match="fingerprint is required"):
        MetricPopulationContext(
            split_id="TRAIN", expected_sample_idx=expected, population_fingerprint="",
        )


def test_pop_context_rejects_split_id_mismatch_in_metric_call():
    expected = np.array([10, 20, 30], dtype=np.int64)
    fp = derive_population_fingerprint(sample_idx=expected, split_id="TRAIN")
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN", expected_sample_idx=expected, population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=3)
    y_pred = y_true + rng.normal(0, 5, size=3)
    with pytest.raises(ValueError, match="split_id"):
        compute_regression_metrics(
            y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
            split_id="VALIDATION", evaluation_mode="VALIDATION",
            population_fingerprint=fp, run_id="X", model_id="Y",
            population_context=pop_ctx,
        )


# --- 12. derive_population_fingerprint is deterministic ----------------

def test_pop_fingerprint_deterministic():
    expected = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    fp1 = derive_population_fingerprint(sample_idx=expected)
    fp2 = derive_population_fingerprint(sample_idx=expected)
    assert fp1 == fp2
    # different IDs → different fingerprint
    fp3 = derive_population_fingerprint(sample_idx=np.array([1, 2, 3, 4, 6], dtype=np.int64))
    assert fp1 != fp3
    # split_id is no longer part of the hash — same IDs produce same
    # fingerprint regardless of split_id metadata.
    fp4 = derive_population_fingerprint(sample_idx=expected, split_id="VALIDATION")
    assert fp1 == fp4
    # different ordering → different fingerprint (input is sorted internally,
    # but different representations are normalized)
    fp5 = derive_population_fingerprint(sample_idx=np.array([5, 4, 3, 2, 1], dtype=np.int64))
    assert fp1 == fp5  # both sorts to [1,2,3,4,5] → identical fingerprint


def test_pop_fingerprint_target_ids_form_matches_canonical_fold_fingerprint():
    """The target_ids form must produce a fingerprint byte-equal to the
    one precomputed by FoldDefinition (which uses
    populations.compute_population_fingerprint). This is the SINGLE
    CANONICAL FINGERPRINT contract.
    """
    ids = ["TGT_00000001", "TGT_00000002", "TGT_00000003"]
    from course_work.rolling_origin.populations import compute_population_fingerprint
    fp_fold = compute_population_fingerprint(ids)
    fp_derive = derive_population_fingerprint(target_ids=ids)
    assert fp_derive == fp_fold, (
        f"derive_population_fingerprint must match FoldDefinition's "
        f"per-role fingerprint byte-for-byte. Got {fp_derive!r} vs {fp_fold!r}"
    )


def test_bundle_fingerprint_equals_context_fingerprint_after_metric_call():
    """When compute_regression_metrics is called with a population_context,
    the returned MetricResult.population_fingerprint MUST equal the
    context fingerprint. This is the SINGLE-CANONICAL contract that
    would have prevented RUN_TR_ROB_0005's
    `Population fingerprint mismatch` error.
    """
    expected = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    fp = derive_population_fingerprint(target_ids=["TGT_A", "TGT_B", "TGT_C", "TGT_D", "TGT_E"])
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN",
        expected_sample_idx=expected,
        population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
        population_fingerprint=fp, run_id="X", model_id="Y",
        population_context=pop_ctx,
    )
    assert metric.population_fingerprint == pop_ctx.population_fingerprint, (
        "MetricResult.population_fingerprint MUST equal the "
        "MetricPopulationContext.population_fingerprint when a context is "
        "provided. This is the SINGLE-CANONICAL-FINGERPRINT contract."
    )


def test_inner_train_and_inner_val_have_distinct_fingerprints():
    """Different fold-role populations must produce different fingerprints
    so that a wrong-context-fingerprint (e.g. accidentally passing inner_val
    where inner_train is required) is detected.
    """
    inner_train_ids = ["TGT_1", "TGT_2", "TGT_3", "TGT_4", "TGT_5"]
    inner_val_ids = ["TGT_100", "TGT_101", "TGT_102"]
    fp_train = derive_population_fingerprint(target_ids=inner_train_ids)
    fp_val = derive_population_fingerprint(target_ids=inner_val_ids)
    assert fp_train != fp_val


def test_outer_train_and_outer_eval_have_distinct_fingerprints():
    outer_train_ids = ["TGT_200", "TGT_201", "TGT_202", "TGT_203"]
    outer_eval_ids = ["TGT_300", "TGT_301", "TGT_302"]
    fp_train = derive_population_fingerprint(target_ids=outer_train_ids)
    fp_eval = derive_population_fingerprint(target_ids=outer_eval_ids)
    assert fp_train != fp_eval


def test_target_id_form_normalizes_input_order():
    """Sorting happens internally so order of input IDs does not affect
    fingerprint — different input orders of the SAME target_ids yield
    identical fingerprints.
    """
    ids_a = ["TGT_5", "TGT_3", "TGT_1", "TGT_4", "TGT_2"]
    ids_b = ["TGT_1", "TGT_2", "TGT_3", "TGT_4", "TGT_5"]
    fp_a = derive_population_fingerprint(target_ids=ids_a)
    fp_b = derive_population_fingerprint(target_ids=ids_b)
    assert fp_a == fp_b


def test_target_id_form_string_normalization():
    """String normalization: int-looking IDs are coerced to strings."""
    ids_str = ["TGT_1", "TGT_2", "TGT_3"]
    ids_int = [1, 2, 3]
    fp_str = derive_population_fingerprint(target_ids=ids_str)
    fp_int = derive_population_fingerprint(target_ids=ids_int)
    # Both produce a fingerprint — but they are NOT byte-equal because
    # the serialization differs (TGT_1 vs 1). This is the documented
    # contract — callers must use a consistent representation.
    assert fp_str != fp_int
    # Same form same fp:
    fp_str_2 = derive_population_fingerprint(target_ids=["TGT_1", "TGT_2", "TGT_3"])
    assert fp_str == fp_str_2
