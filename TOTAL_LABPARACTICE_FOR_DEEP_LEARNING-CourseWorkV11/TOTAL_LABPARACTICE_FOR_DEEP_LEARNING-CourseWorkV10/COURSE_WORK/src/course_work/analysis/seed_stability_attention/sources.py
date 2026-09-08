"""Phase 57 - frozen sources and constants.

Phase 57 reads:
- Phase 52 raw last_query_attention NPZ + dense_case_attention NPZ + checksums
- Phase 54 per-vector metrics + mean temporal profiles + layer head-mean profiles +
  top1 lag frequency + lag-bin mass + recent-mass summary + seed overall profile
- Phase 55 head behavior + head-pair comparison + layer diversity (handoff ref only)
- Phase 56 error_attention_* artifacts + error_conditioning_assignment + handoff
- Phase 48 prediction_seed_spread.csv (secondary diagnostic only)

All values locked before any scientific computation. No model loading.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Canonical constants (frozen)
# ---------------------------------------------------------------------------

# Correct upstream-active revisions — read by orchestrator and findings
PHASE52_VERSION: str = "ATTENTION_EXTRACTION-v1"
PHASE54_VERSION: str = "LAST_QUERY_ATTENTION-v2"
PHASE55_VERSION: str = "HEAD_COMPARISON-v2"
PHASE56_VERSION: str = "ERROR_CONDITIONED_ATTENTION-v2"

# Phase57 active revision (corrective v2)
PHASE57_VERSION: str = "SEED_STABILITY_ATTENTION-v2"

SEEDS: tuple[int, ...] = (42, 123, 2026)
NUM_LAYERS: int = 2
NUM_HEADS: int = 4  # canonical Final Transformer architecture (locked via H2/H4 protocol)
H2_LAYER_HEAD_COUNTS: tuple[int, ...] = (NUM_HEADS, NUM_HEADS)  # uniform across layers (per-layer count = 4)
N_TEST: int = 2961
LOOKBACK: int = 72
LAG_STEP_MINUTES: int = 10
LAG_MINUTES: np.ndarray = (LOOKBACK - np.arange(LOOKBACK)) * LAG_STEP_MINUTES

# Tolerances
PROFILE_SUM_TOL: float = 1e-5
MATCH_TIE_TOL: float = 1e-12
JSD_BOUNDS: tuple[float, float] = (0.0, float(np.log(2.0)))

# Anchor (deterministic first-predeclared seed)
ANCHOR_SEED: int = 42
ANCHOR_REASON: str = "FIRST_PREDECLARED_FINAL_SEED"

# Required seed pairs (canonical order)
SEED_PAIRS: tuple[tuple[int, int], ...] = (
    (42, 123),
    (42, 2026),
    (123, 2026),
)

# Core attention metric set (CORE_ATTENTION_METRICS-v1) — reused from Phase 56
CORE_ATTENTION_METRICS_V1: tuple[str, ...] = (
    "normalized_entropy",
    "expected_lag_minutes",
    "recent_1h_mass",
    "recent_6h_mass",
    "top5_mass",
    "lag80_minutes",
)

# Conditioning variables (Phase 56 → Phase 57 reindexing)
CONDITIONING_VARIABLES: tuple[str, ...] = (
    "ABS_ERROR",
    "SIGNED_RESIDUAL",
    "SHARED_HARDNESS",
)


@dataclass
class FrozenSources57:
    """Frozen Phase 52/54/55/56/48 sources for Phase 57."""
    project_root: Path
    p48_dir: Path
    p52_dir: Path
    p54_dir: Path
    p55_dir: Path
    p56_dir: Path

    # Phase 52 raw
    raw_last_query_files: dict[int, Path]
    raw_last_query_sha: dict[int, str]
    raw_dense_files: dict[int, Path]
    raw_dense_sha: dict[int, str]
    raw_attention_checksums: dict
    raw_attention_checksums_sha: str
    target_order_path: Path
    target_order_sha: str
    dense_case_order_path: Path
    dense_case_order_sha: str

    # Phase 54 metrics
    p54_signoff: dict
    p54_signoff_sha: str
    last_query_metrics_long: list[dict[str, str]]
    last_query_metrics_long_sha: str
    last_query_profile_by_lag: list[dict[str, str]]
    last_query_profile_by_lag_sha: str
    last_query_layer_head_mean_profile: list[dict[str, str]]
    last_query_layer_head_mean_profile_sha: str
    last_query_top1_lag_frequency: list[dict[str, str]]
    last_query_top1_lag_frequency_sha: str
    last_query_lag_bin_mass: list[dict[str, str]]
    last_query_lag_bin_mass_sha: str
    last_query_seed_overall_profile: list[dict[str, str]]
    last_query_seed_overall_profile_sha: str

    # Phase 55 handoff reference
    p55_signoff: dict
    p55_signoff_sha: str
    p55_handoff: dict
    p55_handoff_sha: str

    # Phase 56 error-conditioned effects
    p56_signoff: dict
    p56_signoff_sha: str
    p56_handoff: dict
    p56_handoff_sha: str
    error_attention_association_long: list[dict[str, str]]
    error_attention_association_long_sha: str
    error_attention_high_low_metric: list[dict[str, str]]
    error_attention_high_low_metric_sha: str
    error_attention_high_low_profile: list[dict[str, str]]
    error_attention_high_low_profile_sha: str
    error_attention_signed_metric: list[dict[str, str]]
    error_attention_signed_metric_sha: str
    error_attention_layer_head_mean: list[dict[str, str]]
    error_attention_layer_head_mean_sha: str
    error_attention_shared_cohort: list[dict[str, str]]
    error_attention_shared_cohort_sha: str
    error_conditioning_assignment: list[dict[str, str]]
    error_conditioning_assignment_sha: str

    # Phase 48 prediction spread (secondary)
    prediction_seed_spread: list[dict[str, str]]
    prediction_seed_spread_sha: str

    # Inherited provenance (from upstream handoffs) — corrected in v2 to separate fields
    # final_lock_sha256  = canonical Phase45 combined lock SHA
    # phase47_canonical_test_population_sha256 = Phase47 test population SHA
    # phase52_attention_target_order_sha256 = Phase52 attention target-order SHA
    # raw_last_query_seed42_sha256 = seed 42 raw last-query NPZ SHA
    final_lock_sha256: str = ""
    test_population_sha256: str = ""
    phase47_canonical_test_population_sha256: str = ""
    phase52_attention_target_order_sha256: str = ""
    raw_last_query_seed42_sha256: str = ""
    raw_last_query_seed123_sha256: str = ""
    raw_last_query_seed2026_sha256: str = ""


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


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 57 missing required {label}: {path}")
    return path


def load_frozen_sources57(project_root: Path | str) -> FrozenSources57:
    root = Path(project_root)

    p48_dir = root / "artifacts" / "prediction_analysis"
    p52_dir = root / "artifacts" / "attention_extraction"
    p54_dir = root / "artifacts" / "last_query_attention"
    p55_dir = root / "artifacts" / "head_comparison"
    p56_dir = root / "artifacts" / "error_conditioned_attention"

    # Phase 52 raw
    raw_dir = p52_dir / "raw"
    last_query_files = {}
    last_query_shas = {}
    for s in SEEDS:
        fp = _require(raw_dir / f"last_query_attention_seed{s}.npz", f"Phase 52 last-query NPZ seed{s}")
        last_query_files[s] = fp
        last_query_shas[s] = _sha256_file(fp)
    dense_files = {}
    dense_shas = {}
    for s in SEEDS:
        fp = _require(raw_dir / f"dense_case_attention_seed{s}.npz", f"Phase 52 dense-case NPZ seed{s}")
        dense_files[s] = fp
        dense_shas[s] = _sha256_file(fp)
    raw_checksums_fp = _require(p52_dir / "raw_attention_checksums.json", "raw_attention_checksums.json")
    raw_attention_checksums = _read_json(raw_checksums_fp)
    raw_attention_checksums_sha = _sha256_file(raw_checksums_fp)
    target_order_fp = _require(p52_dir / "attention_test_target_order.csv", "attention_test_target_order.csv")
    target_order_sha = _sha256_file(target_order_fp)
    dense_case_order_fp = _require(p52_dir / "attention_dense_case_order.csv", "attention_dense_case_order.csv")
    dense_case_order_sha = _sha256_file(dense_case_order_fp)

    # Phase 54 metrics
    p54_signoff_fp = _require(p54_dir / "phase_54_signoff.json", "phase_54_signoff.json")
    p54_signoff = _read_json(p54_signoff_fp)
    p54_signoff_sha = _sha256_file(p54_signoff_fp)

    def _load_csv_in(dir_path: Path, name: str):
        fp = _require(dir_path / name, name)
        return _read_csv(fp), _sha256_file(fp)

    last_query_metrics_long, last_query_metrics_long_sha = _load_csv_in(p54_dir, "last_query_metrics_long.csv")
    last_query_profile_by_lag, last_query_profile_by_lag_sha = _load_csv_in(p54_dir, "last_query_profile_by_lag.csv")
    last_query_layer_head_mean_profile, last_query_layer_head_mean_profile_sha = _load_csv_in(
        p54_dir, "last_query_layer_head_mean_profile.csv"
    )
    last_query_top1_lag_frequency, last_query_top1_lag_frequency_sha = _load_csv_in(
        p54_dir, "last_query_top1_lag_frequency.csv"
    )
    last_query_lag_bin_mass, last_query_lag_bin_mass_sha = _load_csv_in(
        p54_dir, "last_query_lag_bin_mass.csv"
    )
    last_query_seed_overall_profile, last_query_seed_overall_profile_sha = _load_csv_in(
        p54_dir, "last_query_seed_overall_profile.csv"
    )

    # Phase 55 handoff reference
    p55_signoff_fp = _require(p55_dir / "phase_55_signoff.json", "phase_55_signoff.json")
    p55_signoff = _read_json(p55_signoff_fp)
    p55_signoff_sha = _sha256_file(p55_signoff_fp)
    p55_handoff_fp = _require(p55_dir / "phase57_seed_stability_head_context_handoff.json", "phase57_seed_stability_head_context_handoff.json")
    p55_handoff = _read_json(p55_handoff_fp)
    p55_handoff_sha = _sha256_file(p55_handoff_fp)

    # Phase 56 error-conditioned effects
    p56_signoff_fp = _require(p56_dir / "phase_56_signoff.json", "phase_56_signoff.json")
    p56_signoff = _read_json(p56_signoff_fp)
    p56_signoff_sha = _sha256_file(p56_signoff_fp)
    p56_handoff_fp = _require(p56_dir / "phase57_seed_stability_attention_handoff.json", "phase57_seed_stability_attention_handoff.json")
    p56_handoff = _read_json(p56_handoff_fp)
    p56_handoff_sha = _sha256_file(p56_handoff_fp)

    error_attention_association_long, error_attention_association_long_sha = _load_csv_in(
        p56_dir, "error_attention_association_long.csv"
    )
    error_attention_high_low_metric, error_attention_high_low_metric_sha = _load_csv_in(
        p56_dir, "error_attention_high_low_metric_comparison.csv"
    )
    error_attention_high_low_profile, error_attention_high_low_profile_sha = _load_csv_in(
        p56_dir, "error_attention_high_low_profile_comparison.csv"
    )
    error_attention_signed_metric, error_attention_signed_metric_sha = _load_csv_in(
        p56_dir, "error_attention_signed_metric_comparison.csv"
    )
    error_attention_layer_head_mean, error_attention_layer_head_mean_sha = _load_csv_in(
        p56_dir, "error_attention_layer_head_mean_association.csv"
    )
    error_attention_shared_cohort, error_attention_shared_cohort_sha = _load_csv_in(
        p56_dir, "error_attention_shared_cohort_layer_summary.csv"
    )
    error_conditioning_assignment, error_conditioning_assignment_sha = _load_csv_in(
        p56_dir, "error_conditioning_assignment.csv"
    )

    # Phase 48 prediction spread
    prediction_seed_spread, prediction_seed_spread_sha = _load_csv_in(
        p48_dir, "prediction_seed_spread.csv"
    )

    # Inherited provenance — corrected in v2 to separate fields with non-conflated identities.
    # 1) final_lock_sha256 = canonical Phase45 combined lock SHA
    # 2) phase47_canonical_test_population_sha256 = Phase47 test population SHA
    # 3) phase52_attention_target_order_sha256 = Phase52 attention target-order SHA
    # 4) raw_last_query_seedNN_sha256 per seed (raw last-query NPZ SHAs)
    final_lock: str = ""
    # prefer fresh Phase47 line in Phase56 v2 handoff
    cand_p47 = str(p56_handoff.get("phase47_canonical_test_population_sha256", ""))
    if not cand_p47:
        cand_p47 = str(p55_handoff.get("phase47_canonical_test_population_sha256", ""))
    # read final_lock from p56 then p55 then p54 explicit fields (no fall-through to raw NPZ)
    cand_lock = str(p56_handoff.get("final_lock_sha256", ""))
    if not cand_lock:
        cand_lock = str(p55_handoff.get("final_lock_sha256", ""))
    if not cand_lock:
        cand_lock = str(p54_signoff.get("final_lock_sha256", ""))
    if not cand_lock and p54_signoff.get("active_revision"):
        cand_lock = str(p54_signoff.get("active_lock_fingerprint", ""))

    # legacy compat: some early v1 handoffs stored raw_attention_sha into final_lock_sha256
    # explicitly avoid that fallback — must come from canonical lock chain

    # Phase52 attention target-order SHA comes from raw_attention_checksums.json (preferred)
    raw_checksum_obj = raw_attention_checksums if isinstance(raw_attention_checksums, dict) else {}
    cand_p52_target = (
        str(raw_checksum_obj.get("phase52_attention_target_order_sha256", ""))
        or str(raw_checksum_obj.get("attention_test_target_order_sha256", ""))
        or target_order_sha
    )

    # raw_last_query seed SHAs from raw_attention_checksums.json (preferred) or computed
    def _raw_seed(s: int, alias_a: str, alias_b: str) -> str:
        return (
            str(raw_checksum_obj.get(alias_a, ""))
            or str(raw_checksum_obj.get(alias_b, ""))
            or last_query_shas.get(s, "")
        )

    raw_seed42_sha = _raw_seed(42, "last_query_seed42_sha256", "phase52_last_query_seed42_sha256")
    raw_seed123_sha = _raw_seed(123, "last_query_seed123_sha256", "phase52_last_query_seed123_sha256")
    raw_seed2026_sha = _raw_seed(2026, "last_query_seed2026_sha256", "phase52_last_query_seed2026_sha256")

    # test_population_sha256 keeps its v1 meaning (= Phase47 test population, derived upstream)
    # but we now also expose the Phase52 attention target-order SHA separately.
    test_pop: str = cand_p47

    # In v1 Phase57 the final_lock field stored the seed42 raw NPZ SHA by mistake.
    # In v2 we set it strictly to the canonical Phase45 combined lock.
    # If upstream chain is missing the canonical lock, force-empty string + raise in preflight.

    return FrozenSources57(
        project_root=root,
        p48_dir=p48_dir,
        p52_dir=p52_dir,
        p54_dir=p54_dir,
        p55_dir=p55_dir,
        p56_dir=p56_dir,
        raw_last_query_files=last_query_files,
        raw_last_query_sha=last_query_shas,
        raw_dense_files=dense_files,
        raw_dense_sha=dense_shas,
        raw_attention_checksums=raw_attention_checksums,
        raw_attention_checksums_sha=raw_attention_checksums_sha,
        target_order_path=target_order_fp,
        target_order_sha=target_order_sha,
        dense_case_order_path=dense_case_order_fp,
        dense_case_order_sha=dense_case_order_sha,
        p54_signoff=p54_signoff,
        p54_signoff_sha=p54_signoff_sha,
        last_query_metrics_long=last_query_metrics_long,
        last_query_metrics_long_sha=last_query_metrics_long_sha,
        last_query_profile_by_lag=last_query_profile_by_lag,
        last_query_profile_by_lag_sha=last_query_profile_by_lag_sha,
        last_query_layer_head_mean_profile=last_query_layer_head_mean_profile,
        last_query_layer_head_mean_profile_sha=last_query_layer_head_mean_profile_sha,
        last_query_top1_lag_frequency=last_query_top1_lag_frequency,
        last_query_top1_lag_frequency_sha=last_query_top1_lag_frequency_sha,
        last_query_lag_bin_mass=last_query_lag_bin_mass,
        last_query_lag_bin_mass_sha=last_query_lag_bin_mass_sha,
        last_query_seed_overall_profile=last_query_seed_overall_profile,
        last_query_seed_overall_profile_sha=last_query_seed_overall_profile_sha,
        p55_signoff=p55_signoff,
        p55_signoff_sha=p55_signoff_sha,
        p55_handoff=p55_handoff,
        p55_handoff_sha=p55_handoff_sha,
        p56_signoff=p56_signoff,
        p56_signoff_sha=p56_signoff_sha,
        p56_handoff=p56_handoff,
        p56_handoff_sha=p56_handoff_sha,
        error_attention_association_long=error_attention_association_long,
        error_attention_association_long_sha=error_attention_association_long_sha,
        error_attention_high_low_metric=error_attention_high_low_metric,
        error_attention_high_low_metric_sha=error_attention_high_low_metric_sha,
        error_attention_high_low_profile=error_attention_high_low_profile,
        error_attention_high_low_profile_sha=error_attention_high_low_profile_sha,
        error_attention_signed_metric=error_attention_signed_metric,
        error_attention_signed_metric_sha=error_attention_signed_metric_sha,
        error_attention_layer_head_mean=error_attention_layer_head_mean,
        error_attention_layer_head_mean_sha=error_attention_layer_head_mean_sha,
        error_attention_shared_cohort=error_attention_shared_cohort,
        error_attention_shared_cohort_sha=error_attention_shared_cohort_sha,
        error_conditioning_assignment=error_conditioning_assignment,
        error_conditioning_assignment_sha=error_conditioning_assignment_sha,
        prediction_seed_spread=prediction_seed_spread,
        prediction_seed_spread_sha=prediction_seed_spread_sha,
        final_lock_sha256=cand_lock,
        test_population_sha256=test_pop,
        phase47_canonical_test_population_sha256=cand_p47,
        phase52_attention_target_order_sha256=cand_p52_target,
        raw_last_query_seed42_sha256=raw_seed42_sha,
        raw_last_query_seed123_sha256=raw_seed123_sha,
        raw_last_query_seed2026_sha256=raw_seed2026_sha,
    )
