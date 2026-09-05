"""Phase 56 - frozen sources and constants.

Phase 56 reads frozen Phase 49 residuals, Phase 50 regime labels, Phase 52
raw last-query NPZ (verification fallback only), Phase 54 per-vector metrics
+ mean temporal profiles + layer head-mean profiles + recent-mass summary,
and Phase 55 head behavior + head-pair comparison + layer diversity (handoff
reference only).

All values locked before any numerical computation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Canonical constants (frozen)
# ---------------------------------------------------------------------------

SEEDS: tuple[int, ...] = (42, 123, 2026)
NUM_LAYERS: int = 2
NUM_HEADS: int = 4
N_TEST: int = 2961
LOOKBACK: int = 72
LAG_STEP_MINUTES: int = 10

# Lag support in minutes (recency-ordered: pos L-1 -> 10 min, pos 0 -> 720 min)
LAG_MINUTES: np.ndarray = (LOOKBACK - np.arange(LOOKBACK)) * LAG_STEP_MINUTES

# Tolerances
PROFILE_SUM_TOL: float = 1e-5
HIGH_LOW_DIFF_TOL: float = 1e-5
CLIFFS_TOL: float = 1e-9

# Cohort rule (locked)
COHORT_RULE: str = "RANK_BASED_20_60_20"
COHORT_EDGE_FRACTION: float = 0.20
DECILE_COUNT: int = 10

# Core attention metric set (CORE_ATTENTION_METRICS-v1)
CORE_ATTENTION_METRICS_V1: tuple[str, ...] = (
    "normalized_entropy",
    "expected_lag_minutes",
    "recent_1h_mass",
    "recent_6h_mass",
    "top5_mass",
    "lag80_minutes",
)

# Conditioning variables
CONDITIONING_VARIABLES: tuple[str, ...] = (
    "ABS_ERROR",
    "SIGNED_RESIDUAL",
    "SHARED_HARDNESS",
)

# Secondary full-matrix metrics
FULL_MATRIX_METRICS: tuple[str, ...] = (
    "mean_query_entropy",
    "mean_self_attention_weight",
    "mean_absolute_query_source_distance_steps",
    "forward_within_input_mass",
)

# Metric bounds
JSD_BOUNDS: tuple[float, float] = (0.0, float(np.log(2.0)))
CLIFFS_BOUNDS: tuple[float, float] = (-1.0, 1.0)
SPEARMAN_BOUNDS: tuple[float, float] = (-1.0, 1.0)


@dataclass
class FrozenSources56:
    """Frozen Phase 49/50/52/54/55 sources for Phase 56."""
    project_root: Path
    p49_dir: Path
    p50_dir: Path
    p52_dir: Path
    p54_dir: Path
    p55_dir: Path
    # Phase 49 residuals
    p49_signoff: dict
    p49_signoff_sha: str
    residual_long: list[dict[str, str]]
    residual_long_sha: str
    # Phase 50 regime
    p50_signoff: dict
    p50_signoff_sha: str
    test_regime_assignment: list[dict[str, str]]
    test_regime_assignment_sha: str
    # Phase 52 raw + summaries
    raw_last_query_files: dict[int, Path]
    raw_last_query_sha: dict[int, str]
    attention_full_matrix_summary: list[dict[str, str]]
    attention_full_matrix_summary_sha: str
    raw_attention_checksums: dict
    # Phase 54 metrics
    p54_signoff: dict
    p54_signoff_sha: str
    last_query_metrics_long: list[dict[str, str]]
    last_query_metrics_long_sha: str
    last_query_profile_by_lag: list[dict[str, str]]
    last_query_profile_by_lag_sha: str
    # Phase 55 handoff ref
    p55_signoff: dict
    p55_signoff_sha: str
    p55_handoff: dict
    p55_handoff_sha: str
    # Phase 51 shared hardness (derived + frozen subset)
    p51_hardness_disagreement: list[dict[str, str]]
    p51_hardness_disagreement_sha: str
    # Phase 55 signoff sha (for immutability)
    final_lock_sha256: str = ""
    test_population_sha256: str = ""


def _sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(fp: Path) -> list[dict[str, str]]:
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def load_frozen_sources56(project_root: Path | str) -> FrozenSources56:
    root = Path(project_root)

    p49_dir = root / "artifacts" / "residual_analysis"
    p50_dir = root / "artifacts" / "error_by_regime"
    p52_dir = root / "artifacts" / "attention_extraction"
    p54_dir = root / "artifacts" / "last_query_attention"
    p55_dir = root / "artifacts" / "head_comparison"

    # Phase 49
    p49_signoff_fp = p49_dir / "phase_49_signoff.json"
    p49_signoff = _read_json(p49_signoff_fp)
    p49_signoff_sha = _sha256_file(p49_signoff_fp)
    res_fp = p49_dir / "residual_long_table.csv"
    residual_long = _read_csv(res_fp)
    residual_long_sha = _sha256_file(res_fp)

    # Phase 50
    p50_signoff_fp = p50_dir / "phase_50_signoff.json"
    p50_signoff = _read_json(p50_signoff_fp)
    p50_signoff_sha = _sha256_file(p50_signoff_fp)
    reg_fp = p50_dir / "test_regime_assignment.csv"
    test_regime_assignment = _read_csv(reg_fp)
    test_regime_assignment_sha = _sha256_file(reg_fp)

    # Phase 52
    raw_dir = p52_dir / "raw"
    raw_files: dict[int, Path] = {}
    raw_shas: dict[int, str] = {}
    for seed in SEEDS:
        fp = raw_dir / f"last_query_attention_seed{seed}.npz"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing raw last-query NPZ: {fp}")
        raw_files[seed] = fp
        raw_shas[seed] = _sha256_file(fp)
    full_mat_fp = p52_dir / "attention_full_matrix_summary.csv"
    full_mat = _read_csv(full_mat_fp)
    full_mat_sha = _sha256_file(full_mat_fp)
    chk_fp = p52_dir / "raw_attention_checksums.json"
    raw_attention_checksums = _read_json(chk_fp)

    # Phase 54
    p54_signoff_fp = p54_dir / "phase_54_signoff.json"
    p54_signoff = _read_json(p54_signoff_fp)
    p54_signoff_sha = _sha256_file(p54_signoff_fp)
    metrics_long_fp = p54_dir / "last_query_metrics_long.csv"
    metrics_long = _read_csv(metrics_long_fp)
    metrics_long_sha = _sha256_file(metrics_long_fp)
    profile_fp = p54_dir / "last_query_profile_by_lag.csv"
    profile_by_lag = _read_csv(profile_fp)
    profile_sha = _sha256_file(profile_fp)

    # Phase 55
    p55_signoff_fp = p55_dir / "phase_55_signoff.json"
    p55_signoff = _read_json(p55_signoff_fp)
    p55_signoff_sha = _sha256_file(p55_signoff_fp)
    p55_handoff_fp = p55_dir / "phase56_error_conditioned_attention_handoff.json"
    p55_handoff = _read_json(p55_handoff_fp)
    p55_handoff_sha = _sha256_file(p55_handoff_fp)

    # Phase 51 shared hardness disagreement
    p51_fp = root / "artifacts" / "worst_error_analysis" / "hardness_vs_seed_disagreement.csv"
    p51_hardness = _read_csv(p51_fp)
    p51_hardness_sha = _sha256_file(p51_fp)

    # Final lock + test population sha (from Phase 55 handoff)
    final_lock = str(p55_handoff.get("final_lock_sha256", ""))
    test_pop = str(p55_handoff.get("test_population_sha256", ""))

    return FrozenSources56(
        project_root=root,
        p49_dir=p49_dir,
        p50_dir=p50_dir,
        p52_dir=p52_dir,
        p54_dir=p54_dir,
        p55_dir=p55_dir,
        p49_signoff=p49_signoff,
        p49_signoff_sha=p49_signoff_sha,
        residual_long=residual_long,
        residual_long_sha=residual_long_sha,
        p50_signoff=p50_signoff,
        p50_signoff_sha=p50_signoff_sha,
        test_regime_assignment=test_regime_assignment,
        test_regime_assignment_sha=test_regime_assignment_sha,
        raw_last_query_files=raw_files,
        raw_last_query_sha=raw_shas,
        attention_full_matrix_summary=full_mat,
        attention_full_matrix_summary_sha=full_mat_sha,
        raw_attention_checksums=raw_attention_checksums,
        p54_signoff=p54_signoff,
        p54_signoff_sha=p54_signoff_sha,
        last_query_metrics_long=metrics_long,
        last_query_metrics_long_sha=metrics_long_sha,
        last_query_profile_by_lag=profile_by_lag,
        last_query_profile_by_lag_sha=profile_sha,
        p55_signoff=p55_signoff,
        p55_signoff_sha=p55_signoff_sha,
        p55_handoff=p55_handoff,
        p55_handoff_sha=p55_handoff_sha,
        p51_hardness_disagreement=p51_hardness,
        p51_hardness_disagreement_sha=p51_hardness_sha,
        final_lock_sha256=final_lock,
        test_population_sha256=test_pop,
    )
