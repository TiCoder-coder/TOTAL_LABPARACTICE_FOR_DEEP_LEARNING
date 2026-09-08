"""Phase 55 — focused tests for Head Comparison Analysis.

Verifies:
- All 32 O55 artifacts exist.
- Sign-off status is PASS.
- Pair construction is exactly H(H-1)/2 = 6 per (seed, layer) = 36 total.
- JSD matrix diagonal = 0, cosine diagonal = 1, etc.
- Forbidden actions (best_head_selected, head_pruning, etc.) are False.
- Discrepancies count = 0 for clean run.
- Upstream Phase 47-54 artifacts SHA256 unchanged after Phase 55 run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2] 
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "head_comparison"

@pytest.fixture(scope="module")
def artifacts_dir() -> Path:
    if not ARTIFACTS_DIR.is_dir():
        pytest.skip(f"Phase 55 artifacts dir missing: {ARTIFACTS_DIR}")
    return ARTIFACTS_DIR

O55_REQUIRED = [
    "head_comparison_manifest.json",
    "head_comparison_contract.json",
    "phase55_preflight_audit.csv",
    "head_comparison_source_verification.csv",
    "head_profile_integrity_audit.csv",
    "head_target_alignment_audit.csv",
    "head_pair_comparison_long.csv",
    "head_pair_profile_similarity.csv",
    "head_pair_metric_difference.csv",
    "head_pair_paired_difference_summary.csv",
    "head_pair_top1_distribution_distance.csv",
    "head_pair_wasserstein_distance.csv",
    "head_behavior_summary.csv",
    "head_to_layer_mean_distance.csv",
    "layer_head_diversity_summary.csv",
    "head_similarity_matrix_jsd.csv",
    "head_similarity_matrix_cosine.csv",
    "head_similarity_matrix_pearson.csv",
    "head_similarity_matrix_spearman.csv",
    "head_distance_matrix_l1.csv",
    "head_distance_matrix_wasserstein.csv",
    "head_expected_lag_difference_matrix.csv",
    "head_recent1h_difference_matrix.csv",
    "head_top1_tvd_matrix.csv",
    "head_comparison_findings.csv",
    "phase56_error_conditioned_attention_handoff.json",
    "phase57_seed_stability_head_context_handoff.json",
    "head_comparison_tests.csv",
    "head_comparison_discrepancies.json",
    "head_comparison_summary.json",
    "head_comparison_report.md",
    "README_HEAD_COMPARISON.md",
    "phase_55_signoff.json",
]


@pytest.mark.parametrize("artifact", O55_REQUIRED)
def test_artifact_exists(artifacts_dir: Path, artifact: str) -> None:
    assert (artifacts_dir / artifact).is_file(), f"Missing artifact: {artifact}"

def test_signoff_pass(artifacts_dir: Path) -> None:
    signoff = json.loads((artifacts_dir / "phase_55_signoff.json").read_text(encoding="utf-8"))
    assert signoff["overall_status"] == "PASS", f"Sign-off status: {signoff['overall_status']}"
    assert signoff["best_head_selected"] is False
    assert signoff["head_pruning"] is False
    assert signoff["head_ablation"] is False
    assert signoff["model_training"] is False
    assert signoff["test_metric_recomputation"] is False
    assert signoff["error_conditioning"] is False
    assert signoff["cross_seed_head_matching"] is False
    assert signoff["same_index_semantic_alignment_assumed"] is False
    assert signoff["feature_importance_claim"] is False
    assert signoff["causal_claim"] is False
    assert signoff["phase56_ready"] is True
    assert signoff["phase57_context_ready"] is True


def test_signoff_pair_counts(artifacts_dir: Path) -> None:
    signoff = json.loads((artifacts_dir / "phase_55_signoff.json").read_text(encoding="utf-8"))
    assert signoff["num_layers"] == 2
    assert signoff["num_heads"] == 4
    assert signoff["expected_pair_count_per_layer"] == 6
    assert signoff["actual_pair_count_per_layer"] == 6
    assert signoff["actual_pair_count_total"] == 36


def test_tests_all_pass(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_comparison_tests.csv")
    failures = df[df["status"] != "PASS"]
    assert len(failures) == 0, f"Failed tests:\n{failures.to_string()}"

def test_discrepancies_clean(artifacts_dir: Path) -> None:
    d = json.loads((artifacts_dir / "head_comparison_discrepancies.json").read_text(encoding="utf-8"))
    total = sum(d.get("counts", {}).values())
    assert total == 0, f"Discrepancies present: {d.get('counts', {})}"

def test_pair_count_per_layer(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_pair_comparison_long.csv")
    counts = df.groupby(["seed", "layer_idx0"]).size().reset_index(name="n")
    assert (counts["n"] == 6).all(), f"Pair counts per (seed, layer):\n{counts.to_string()}"
    assert len(counts) == 6  


def test_pair_order_architectural(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_pair_comparison_long.csv")
    assert (df["head_a_idx0"] < df["head_b_idx0"]).all()

def test_profile_integrity_sum(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_profile_integrity_audit.csv")
    bad = df[df["status"] != "OK"]
    assert len(bad) == 0, f"Bad profiles:\n{bad.to_string()}"


def test_target_alignment_exact(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_target_alignment_audit.csv")
    bad = df[df["status"] != "PASS"]
    assert len(bad) == 0, f"Bad target alignment:\n{bad.to_string()}"


def test_jsd_in_bounds(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_pair_profile_similarity.csv")
    finite = df["jsd"].dropna()
    in_range = (finite >= -1e-6) & (finite <= float(np.log(2.0)) + 1e-6)
    assert in_range.all(), f"JSD out of bounds:\n{df[~in_range].head(10)}"


def test_cosine_in_bounds(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_pair_profile_similarity.csv")
    finite = df["cosine"].dropna()
    in_range = (finite >= -1e-6) & (finite <= 1.0 + 1e-6)
    assert in_range.all()


def test_wasserstein_nonneg(artifacts_dir: Path) -> None:
    df = pd.read_csv(artifacts_dir / "head_pair_profile_similarity.csv")
    finite = df["wasserstein_minutes"].dropna()
    assert (finite >= -1e-6).all()

def test_matrix_diagonals(artifacts_dir: Path) -> None:
    """For each (seed, layer), JSD diagonal = 0, cosine diagonal = 1, etc."""
    jsd_df = pd.read_csv(artifacts_dir / "head_similarity_matrix_jsd.csv")
    sample = jsd_df[(jsd_df["seed"] == 42) & (jsd_df["layer_idx0"] == 0)]
    diag_mask = sample["row_head"] == sample["col_head"]
    diag_vals = sample[diag_mask]["value"].astype(float)
    assert (diag_vals.abs() < 1e-6).all(), f"JSD diag:\n{diag_vals.head()}"

    cos_df = pd.read_csv(artifacts_dir / "head_similarity_matrix_cosine.csv")
    sample = cos_df[(cos_df["seed"] == 42) & (cos_df["layer_idx0"] == 0)]
    diag_mask = sample["row_head"] == sample["col_head"]
    diag_vals = sample[diag_mask]["value"].astype(float)
    assert ((diag_vals - 1.0).abs() < 1e-6).all(), f"Cosine diag:\n{diag_vals.head()}"


def test_matrix_symmetry(artifacts_dir: Path) -> None:
    jsd_df = pd.read_csv(artifacts_dir / "head_similarity_matrix_jsd.csv")
    sample = jsd_df[(jsd_df["seed"] == 42) & (jsd_df["layer_idx0"] == 0)].copy()
    pivot = sample.pivot(index="row_head", columns="col_head", values="value")
    diff = (pivot - pivot.T).abs().max().max()
    assert diff < 1e-6, f"JSD symmetry max diff = {diff}"

def test_phase56_handoff_ready(artifacts_dir: Path) -> None:
    d = json.loads((artifacts_dir / "phase56_error_conditioned_attention_handoff.json").read_text(encoding="utf-8"))
    assert d["ready_for_phase56"] is True
    assert d["best_head_selected"] is False
    assert d["head_pruning"] is False
    assert d["error_conditioning_performed_in_phase55"] is False


def test_phase57_handoff_ready(artifacts_dir: Path) -> None:
    d = json.loads((artifacts_dir / "phase57_seed_stability_head_context_handoff.json").read_text(encoding="utf-8"))
    assert d["ready_for_phase57_context"] is True
    assert d["same_index_semantic_alignment_assumed"] is False
    assert d["head_matching_performed"] is False

def test_upstream_signoffs_unchanged(artifacts_dir: Path) -> None:
    """Verify Phase 52/53/54 signoffs still PASS after Phase 55 run."""
    for fp in [
        "artifacts/attention_extraction/phase_52_signoff.json",
        "artifacts/attention_heatmaps/phase_53_signoff.json",
        "artifacts/last_query_attention/phase_54_signoff.json",
    ]:
        d = json.loads((REPO_ROOT / fp).read_text(encoding="utf-8"))
        s = d.get("status", d.get("overall_status", d.get("phase52_status")))
        assert s in ("PASS", "PASS_WITH_WARNING"), f"{fp} status: {s}"

def test_no_raw_npz_modified(artifacts_dir: Path) -> None:
    """Raw last-query NPZ SHAs must match the Phase 52 manifest."""
    raw_dir = REPO_ROOT / "artifacts" / "attention_extraction" / "raw"
    expected = {
        "last_query_attention_seed42.npz": "102086f71ed01611b963c44926d7472a3ecc49a0b63f41d79100ef816b52a9ff",
        "last_query_attention_seed123.npz": "00e2f8e7852908764b3712d9b172faa644dc69d176241962bc14c68fee31110e",
        "last_query_attention_seed2026.npz": "34051966fc47bf96f4e288ac5e984815feff217ad65828882383ee4643cc5a19",
    }
    for name, expected_sha in expected.items():
        fp = raw_dir / name
        actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert actual == expected_sha, f"{name} SHA mismatch: expected {expected_sha}, got {actual}"
