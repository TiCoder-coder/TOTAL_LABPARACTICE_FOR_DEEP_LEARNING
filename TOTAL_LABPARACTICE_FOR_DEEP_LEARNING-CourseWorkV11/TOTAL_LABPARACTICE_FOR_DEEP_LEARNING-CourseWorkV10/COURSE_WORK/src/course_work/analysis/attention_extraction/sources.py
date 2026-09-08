"""Phase 52 — environment, paths, constants, and frozen sources.

This module is the SINGLE SOURCE OF TRUTH for Phase 52 paths and locked
contracts. It NEVER mutates any upstream artifact; it only declares
constants and reads frozen Phase 51 / 47 / 48 / 49 / 50 evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ATTENTION_EXTRACTION_VERSION = "ATTENTION_EXTRACTION-v1"
PHASE = 52
PHASE_DIR_REL = Path("artifacts/attention_extraction")
RAW_DIR_REL = Path("artifacts/attention_extraction/raw")

SEEDS: tuple[int, ...] = (42, 123, 2026)

LOOKBACK = 72
FEATURE_COUNT = 33
NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 64
FFN_DIM = 256
POOLING = "LAST_STEP"
REVIN_ENABLED = False
FEATURE_SET = "FS2_TF1"
BOUNDARY_PROTOCOL = "WB0_CONTEXT_CARRY_OVER"

DENSE_STORAGE_AXES = "case_layer_head_query_source"
LAST_QUERY_STORAGE_AXES = "target_layer_head_source"

RAW_DTYPE = "float32"

DEFAULT_EXTRACTION_BATCH = 8
EXTRACTION_BATCH_FALLBACK: tuple[int, ...] = (4, 2, 1)

EPS_H = 1e-12  
ROW_SUM_ATOL = 1e-5
ROW_SUM_RTOL = 1e-5
PRED_EQUIV_ATOL = 1e-5
PRED_EQUIV_RTOL = 1e-5
ATTN_NONNEGATIVE_EPS = -1e-7
ATTN_MAX_TOLERANCE = 1.0 + 1e-6
MIN_LAST_QUERY_MASS_BOUND = 1.0 - ROW_SUM_ATOL
MAX_LAST_QUERY_MASS_BOUND = 1.0 + ROW_SUM_ATOL

REPRODUCIBILITY_PROBE_CASES = 2

TOP_K_SOURCES = 5

RECENT_WINDOWS_STEPS: dict[str, int] = {
    "recent_1h": 6,    # 60 min / 10 min cadence
    "recent_6h": 36,   # 6 h / 10 min cadence
    "recent_12h": 72,  # 12 h / 10 min cadence
    "recent_24h": 144, # 24 h / 10 min cadence
}

CADENCE_MINUTES = 10
HORIZON = 1


def _sha256_file(path: Path) -> str:
    if path.is_dir():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenSources:
    """All Phase 52 frozen sources, with SHA-256 verification."""

    project_root: Path
    final_lock_sha256: str
    final_config_sha256: str
    final_lineage_sha256: str
    final_recipe_sha256: str
    final_scaler_x_sha256: str
    final_scaler_y_sha256: str
    final_feature_fingerprint: str
    test_population_sha256: str
    test_target_count: int
    phase47_signoff_sha256: str
    phase48_signoff_sha256: str
    phase49_signoff_sha256: str
    phase50_signoff_sha256: str
    phase51_signoff_sha256: str
    phase51_handoff_sha256: str
    phase51_selection_contract_sha256: str
    phase51_attention_handoff_cases_sha256: str
    phase47_predictions_per_seed_sha256: dict[int, str]
    attention_verify_provenance_sha256: str
    attention_verify_serialization_sha256: str
    final_checkpoint_per_seed_sha256: dict[int, str]
    final_checkpoint_per_seed_metadata_sha256: dict[int, str]
    final_checkpoint_per_seed_path: dict[int, Path]
    extra: dict[str, Any] = field(default_factory=dict)


def load_frozen_sources(project_root: Path) -> FrozenSources:
    """Read every canonical Phase 52 source and SHA-256-verify it.

    Pure read-only operation. No mutation of any source file.
    """
    root = project_root

    p47_so_fp = root / "artifacts/final_test/phase_47_signoff.json"
    p47_so = json.loads(p47_so_fp.read_text())
    pred_checksums_fp = root / "artifacts/final_test/prediction_checksums.json"
    pred_checksums = json.loads(pred_checksums_fp.read_text())

    pred_per_seed: dict[int, str] = {}
    for seed in SEEDS:
        key = f"seed_{seed}"
        entry = pred_checksums["predictions"][key]
        rel = entry["path"]
        observed = _sha256_file(root / rel)
        if observed != entry["sha256"]:
            raise RuntimeError(
                f"Phase47 prediction bundle SHA drift for {key}: "
                f"expected {entry['sha256']}, got {observed}"
            )
        pred_per_seed[seed] = observed

    pop_manifest = json.loads((root / "artifacts/final_test/final_test_population_manifest.json").read_text())
    if pop_manifest["target_count"] != 2961:
        raise RuntimeError("Test target count != 2961")
    if pop_manifest["population_id"] != "FINAL_TEST_POP-v1":
        raise RuntimeError("Population id != FINAL_TEST_POP-v1")

    lock = json.loads((root / "artifacts/final_model_lock/final_model_lock_fingerprint.json").read_text())
    config = json.loads((root / "artifacts/final_model_lock/final_model_scientific_config.json").read_text())

    final_lock_sha = lock["combined_lock_sha256"]
    final_lineage_sha = lock["final_lineage_sha256"]
    final_config_sha = lock["final_model_config_sha256"]
    final_recipe_sha = lock["final_training_recipe_sha256"]

    if config["lineage"]["dataloader_fingerprint"] is None:
        raise RuntimeError("Final config lineage fingerprint missing")

    feature_contract = json.loads((root / "artifacts/final_model_lock/final_feature_contract.json").read_text())
    if feature_contract["feature_count_expected_from_runtime"] != FEATURE_COUNT:
        raise RuntimeError("Final feature count != 33")

    scaler_registry = json.loads((root / "artifacts/scaling/final_dev/final_scaler_registry.json").read_text())
    x_artifact_rel = scaler_registry["x_bundles"][FEATURE_SET]["artifact_path"]
    x_sha = _sha256_file(root / x_artifact_rel)
    y_artifact_rel = scaler_registry["target_bundles"]["YS0"]["artifact_path"]
    if y_artifact_rel is None:
        y_artifact_rel = scaler_registry["target_bundles"]["YS1"]["artifact_path"]
    y_sha = _sha256_file(root / y_artifact_rel) if y_artifact_rel else ""

    p48_so_fp = root / "artifacts/prediction_analysis/phase_48_signoff.json"
    p48_so = json.loads(p48_so_fp.read_text()) if p48_so_fp.exists() else {"status": "MISSING"}

    p49_so_fp = root / "artifacts/residual_analysis/phase_49_signoff.json"
    p49_so = json.loads(p49_so_fp.read_text()) if p49_so_fp.exists() else {"status": "MISSING"}

    p50_so_fp = root / "artifacts/error_by_regime/phase_50_signoff.json"
    p50_so = json.loads(p50_so_fp.read_text()) if p50_so_fp.exists() else {"status": "MISSING"}

    p51_so = json.loads((root / "artifacts/worst_error_analysis/phase51_signoff.json").read_text())

    p51_handoff = json.loads((root / "artifacts/worst_error_analysis/phase52_attention_extraction_handoff.json").read_text())
    p51_handoff_sha = _sha256_file(root / "artifacts/worst_error_analysis/phase52_attention_extraction_handoff.json")

    p51_sel_contract_fp = root / "artifacts/worst_error_analysis/worst_error_selection_contract.json"
    p51_sel_contract_sha = _sha256_file(p51_sel_contract_fp)

    p51_handoff_cases_fp = root / "artifacts/worst_error_analysis/phase51_attention_handoff_cases.csv"
    p51_handoff_cases_sha = _sha256_file(p51_handoff_cases_fp)

    av_prov_fp = root / "artifacts/attention_verification/attention_provenance_schema.json"
    av_prov_sha = _sha256_file(av_prov_fp)
    av_ser_fp = root / "artifacts/attention_verification/attention_serialization_schema.json"
    av_ser_sha = _sha256_file(av_ser_fp)

    reuse_cache = json.loads((root / "artifacts/three_seed_final_runs/phase46_reuse_cache.json").read_text())
    ckpt_sha: dict[int, str] = {}
    ckpt_meta_sha: dict[int, str] = {}
    ckpt_path: dict[int, Path] = {}
    for seed in SEEDS:
        run_info = reuse_cache["runs"][str(seed)]
        ckpt_path[seed] = root / run_info["checkpoint_path"]
        ckpt_sha[seed] = _sha256_file(ckpt_path[seed])
        if ckpt_sha[seed] != run_info["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint SHA drift for seed {seed}")
        meta_fp = root / run_info["metadata_path"]
        ckpt_meta_sha[seed] = _sha256_file(meta_fp)

    return FrozenSources(
        project_root=root,
        final_lock_sha256=final_lock_sha,
        final_config_sha256=final_config_sha,
        final_lineage_sha256=final_lineage_sha,
        final_recipe_sha256=final_recipe_sha,
        final_scaler_x_sha256=x_sha,
        final_scaler_y_sha256=y_sha,
        final_feature_fingerprint=feature_contract["feature_fingerprint"],
        test_population_sha256=pop_manifest["target_ids_sha256"],
        test_target_count=pop_manifest["target_count"],
        phase47_signoff_sha256=_sha256_file(p47_so_fp),
        phase48_signoff_sha256=_sha256_file(p48_so_fp) if p48_so_fp.exists() else "",
        phase49_signoff_sha256=_sha256_file(p49_so_fp) if p49_so_fp.exists() else "",
        phase50_signoff_sha256=_sha256_file(p50_so_fp) if p50_so_fp.exists() else "",
        phase51_signoff_sha256=_sha256_file(root / "artifacts/worst_error_analysis/phase51_signoff.json"),
        phase51_handoff_sha256=p51_handoff_sha,
        phase51_selection_contract_sha256=p51_sel_contract_sha,
        phase51_attention_handoff_cases_sha256=p51_handoff_cases_sha,
        phase47_predictions_per_seed_sha256=pred_per_seed,
        attention_verify_provenance_sha256=av_prov_sha,
        attention_verify_serialization_sha256=av_ser_sha,
        final_checkpoint_per_seed_sha256=ckpt_sha,
        final_checkpoint_per_seed_metadata_sha256=ckpt_meta_sha,
        final_checkpoint_per_seed_path=ckpt_path,
        extra={
            "phase47_status": p47_so.get("phase47_status"),
            "phase48_status": p48_so.get("status"),
            "phase49_status": p49_so.get("status"),
            "phase50_status": p50_so.get("status"),
            "phase51_status": p51_so.get("phase51_status"),
            "phase51_ready_for_phase52": p51_so.get("ready_for_phase52"),
            "phase51_phase52_authorized_at_signoff": p51_so.get("phase52_authorized"),
            "phase51_case_count_unique": p51_handoff.get("final_candidate_lineage", {}).get("case_count_unique"),
            "phase51_memberships_total": p51_handoff.get("final_candidate_lineage", {}).get("memberships_total"),
        },
    )
