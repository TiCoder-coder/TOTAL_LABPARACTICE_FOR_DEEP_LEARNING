"""Phase 55 - frozen sources and constants.

Reads Phase 54 canonical artifacts only. No raw NPZ loaded for analysis
(numpy verification fallbacks use Phase 52 raw NPZ only for head profile
integrity check, NOT for scientific recomputation).

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

# Lag support in minutes (recency-ordered: pos L-1 -> 10 min, pos 0 -> 720 min)
LAG_MINUTES: np.ndarray = (LOOKBACK - np.arange(LOOKBACK)) * 10

# Tolerances
PROFILE_SUM_TOL: float = 1e-5  # probability profile sum tolerance
TOP1_SUM_TOL: float = 1e-6     # top1 lag frequency sum tolerance

# Pair construction (Phase 55 §9)
HEAD_ORDER: str = "ARCHITECTURAL"  # no reordering by similarity
MAX_PAIR_COUNT_PER_LAYER: int = NUM_HEADS * (NUM_HEADS - 1) // 2  # = 6
EXPECTED_PAIR_COUNT_PER_LAYER: int = MAX_PAIR_COUNT_PER_LAYER

# Delta convention (Phase 55 §56)
PAIR_DELTA_CONVENTION: str = "A_minus_B"  # Delta = metric_A - metric_B, head_a < head_b

# Metric bounds
JS_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "jsd": (0.0, float(np.log(2.0))),       # natural log; [0, ln(2)] approx [0, 0.693]
    "cosine": (0.0, 1.0),
    "pearson": (-1.0, 1.0),
    "spearman": (-1.0, 1.0),
    "tvd": (0.0, 1.0),
}
COSINE_BOUNDS: tuple[float, float] = (0.0, 1.0)
WASSERSTEIN_BOUNDS_MINUTES: tuple[float, float] = (0.0, float(LOOKBACK * 10))  # [0, 720]


# ---------------------------------------------------------------------------
# Frozen sources container
# ---------------------------------------------------------------------------

@dataclass
class FrozenSources55:
    """Frozen Phase 54 + Phase 52 sources for Phase 55."""
    project_root: Path
    p54_dir: Path
    p54_signoff_sha: str
    p54_signoff: dict
    p54_contract_sha: str
    handoff_p55: dict
    handoff_p55_sha: str
    last_query_metrics_long: list[dict[str, str]]
    last_query_metrics_long_sha: str
    metric_summary_by_head: list[dict[str, str]]
    metric_summary_by_head_sha: str
    profile_by_lag: list[dict[str, str]]
    profile_by_lag_sha: str
    layer_head_mean_profile: list[dict[str, str]]
    layer_head_mean_profile_sha: str
    lag_bin_mass: list[dict[str, str]]
    lag_bin_mass_sha: str
    recent_mass_summary: list[dict[str, str]]
    recent_mass_summary_sha: str
    coverage_radius_summary: list[dict[str, str]]
    coverage_radius_summary_sha: str
    top1_lag_frequency: list[dict[str, str]]
    top1_lag_frequency_sha: str
    top1_tie_summary: list[dict[str, str]]
    top1_tie_summary_sha: str
    raw_last_query_files: dict[int, Path]
    raw_last_query_sha: dict[int, str]
    raw_checksums_frozen_sha: dict[str, str]


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


def load_frozen_sources55(project_root: Path | str) -> FrozenSources55:
    """Load all frozen Phase 54 + Phase 52 sources required by Phase 55.

    No raw NPZ is loaded into numpy for analysis here; raw NPZ SHA verification
    only. Numerical Phase 55 analysis uses Phase 54 derived CSV tables only.
    """
    root = Path(project_root)
    p54_dir = root / "artifacts" / "last_query_attention"

    # Required Phase 54 artifacts
    p54_signoff_fp = p54_dir / "phase_54_signoff.json"
    p54_signoff = _read_json(p54_signoff_fp)
    p54_signoff_sha = _sha256_file(p54_signoff_fp)

    p54_contract_fp = p54_dir / "last_query_attention_contract.json"
    p54_contract = _read_json(p54_contract_fp)
    p54_contract_sha = _sha256_file(p54_contract_fp)

    handoff_fp = p54_dir / "phase55_head_comparison_handoff.json"
    handoff = _read_json(handoff_fp)
    handoff_sha = _sha256_file(handoff_fp)

    # Required Phase 54 tables
    paths_shas: dict[str, tuple[Path, str, list[dict[str, str]]]] = {}
    for name in [
        "last_query_metrics_long",
        "last_query_metric_summary_by_head",
        "last_query_profile_by_lag",
        "last_query_layer_head_mean_profile",
        "last_query_lag_bin_mass",
        "last_query_recent_mass_summary",
        "last_query_coverage_radius_summary",
        "last_query_top1_lag_frequency",
        "last_query_top1_tie_summary",
    ]:
        fp = p54_dir / f"{name}.csv"
        if not fp.is_file():
            raise FileNotFoundError(f"Required Phase 54 artifact missing: {fp}")
        sha = _sha256_file(fp)
        rows = _read_csv(fp)
        paths_shas[name] = (fp, sha, rows)

    # Raw last-query NPZ files (verification SHA only)
    raw_dir = root / "artifacts" / "attention_extraction" / "raw"
    raw_files: dict[int, Path] = {}
    raw_shas: dict[int, str] = {}
    for seed in SEEDS:
        fp = raw_dir / f"last_query_attention_seed{seed}.npz"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing raw last-query NPZ: {fp}")
        raw_files[seed] = fp
        raw_shas[seed] = _sha256_file(fp)

    # Frozen Phase 52 checksums reference
    chk_fp = root / "artifacts" / "attention_extraction" / "raw_attention_checksums.json"
    chk = _read_json(chk_fp)
    raw_checksums_frozen_sha = {}
    files_section = chk.get("files", {})
    for seed in SEEDS:
        key = f"last_query_attention_seed{seed}.npz"
        # raw checksums file may store under different nesting
        entry = files_section.get(key)
        if entry is None:
            entry = files_section.get("last_query", {}).get(key, {})
        if isinstance(entry, dict):
            for ek, ev in entry.items():
                if "sha" in ek.lower():
                    raw_checksums_frozen_sha[key] = str(ev)
                    break

    return FrozenSources55(
        project_root=root,
        p54_dir=p54_dir,
        p54_signoff_sha=p54_signoff_sha,
        p54_signoff=p54_signoff,
        p54_contract_sha=p54_contract_sha,
        handoff_p55=handoff,
        handoff_p55_sha=handoff_sha,
        last_query_metrics_long=paths_shas["last_query_metrics_long"][2],
        last_query_metrics_long_sha=paths_shas["last_query_metrics_long"][1],
        metric_summary_by_head=paths_shas["last_query_metric_summary_by_head"][2],
        metric_summary_by_head_sha=paths_shas["last_query_metric_summary_by_head"][1],
        profile_by_lag=paths_shas["last_query_profile_by_lag"][2],
        profile_by_lag_sha=paths_shas["last_query_profile_by_lag"][1],
        layer_head_mean_profile=paths_shas["last_query_layer_head_mean_profile"][2],
        layer_head_mean_profile_sha=paths_shas["last_query_layer_head_mean_profile"][1],
        lag_bin_mass=paths_shas["last_query_lag_bin_mass"][2],
        lag_bin_mass_sha=paths_shas["last_query_lag_bin_mass"][1],
        recent_mass_summary=paths_shas["last_query_recent_mass_summary"][2],
        recent_mass_summary_sha=paths_shas["last_query_recent_mass_summary"][1],
        coverage_radius_summary=paths_shas["last_query_coverage_radius_summary"][2],
        coverage_radius_summary_sha=paths_shas["last_query_coverage_radius_summary"][1],
        top1_lag_frequency=paths_shas["last_query_top1_lag_frequency"][2],
        top1_lag_frequency_sha=paths_shas["last_query_top1_lag_frequency"][1],
        top1_tie_summary=paths_shas["last_query_top1_tie_summary"][2],
        top1_tie_summary_sha=paths_shas["last_query_top1_tie_summary"][1],
        raw_last_query_files=raw_files,
        raw_last_query_sha=raw_shas,
        raw_checksums_frozen_sha=raw_checksums_frozen_sha,
    )
