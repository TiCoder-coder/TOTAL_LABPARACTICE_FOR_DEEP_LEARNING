"""Phase 54 — frozen sources and constants.

Defines all frozen paths, seeds, lookback, layers, heads, lag mapping,
recent windows, and the loader for upstream Phase 52/53 artifacts.

All SHA256 + manifest values are loaded from canonical Phase 52 artifacts;
no constants are hardcoded in a way that contradicts Phase 52.
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
LOOKBACK: int = 72
NUM_LAYERS: int = 2
NUM_HEADS: int = 4
N_TEST: int = 2961
EPSILON_H: float = 1e-12  # entropy epsilon (matches Phase 52 contract)
EPSILON_NEG: float = 1e-7  # negative tolerance for probability recheck
SUM_TOL: float = 1e-5  # row sum tolerance
PROB_MAX_TOL: float = 1e-6  # max weight tolerance (1 + tol)

# Last-query definition
LAST_QUERY_DEFINITION: str = "A[:, :, L-1, :]"

# Raw axis order
RAW_AXIS_ORDER: tuple[str, ...] = ("target", "layer", "head", "source")

# Recent windows in steps (Phase 52 canonical)
RECENT_WINDOWS_STEPS: dict[str, int] = {
    "1h": 6,
    "6h": 36,
    "12h": 72,
    "24h": 144,
}
RECENT_WINDOWS_HOURS: dict[str, int] = {"1h": 1, "6h": 6, "12h": 12, "24h": 24}

# Coverage levels for Lag50/Lag80/Lag90
COVERAGE_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90)

# Non-overlapping lag bins (canonical from Phase 54 detail)
# For L=72, lags >72 are unreachable. Bin (145, LOOKBACK) would invert to
# (145, 72) — invalid interval. Following Phase 54 plan §1100-1160:
# unreachable bins are kept as declared conceptual bins but explicitly
# marked NOT_APPLICABLE for L=72; current contract documents this via
# non_overlap_bin_applicability_under_current_lookback.
NON_OVERLAP_BINS: tuple[tuple[int, int], ...] = (
    (1, 6),     # BIN_10M_TO_1H    — applicable for L=72 (covers lags 1..6)
    (7, 36),    # BIN_GT1H_TO_6H   — applicable for L=72 (covers lags 7..36)
    (37, 72),   # BIN_GT6H_TO_12H  — applicable for L=72 (covers lags 37..72)
    (73, 144),  # BIN_GT12H_TO_24H — NOT_APPLICABLE for L=72 (would require L>=144)
    (145, 240), # BIN_BEYOND_24H   — NOT_APPLICABLE for L=72 (would require L>=240)
)
NON_OVERLAP_BINS_RAW_PLAN: tuple[tuple[int, int], ...] = (
    (1, 6),
    (7, 36),
    (37, 72),
    (73, 144),
    (145, LOOKBACK),   # legacy Phase 54 plan interval declaration (kept for traceability)
)
NON_OVERLAP_BIN_LABELS: tuple[str, ...] = (
    "BIN_10M_TO_1H",
    "BIN_GT1H_TO_6H",
    "BIN_GT6H_TO_12H",
    "BIN_GT12H_TO_24H",
    "BIN_BEYOND_24H",
)
# Per-bin applicability under the current L=72 lookback.
# An interval [a, b] is applicable iff a <= b AND a <= L AND b >= 1.
# Non-applicable bins receive zero mass and are not assigned attention mass.
NON_OVERLAP_BIN_APPLICABILITY_UNDER_LOOKBACK: dict[int, bool] = {
    0: True,   # BIN_10M_TO_1H    (1..6)
    1: True,   # BIN_GT1H_TO_6H   (7..36)
    2: True,   # BIN_GT6H_TO_12H  (37..72)
    3: False,  # BIN_GT12H_TO_24H (73..144) — unreachable for L=72
    4: False,  # BIN_BEYOND_24H   (145..240) — unreachable for L=72
}

# Cadence
CADENCE_MINUTES: int = 10  # 10-min cadence

# Architectural display strings (zero-indexed display)
DISPLAY_OFFSET: int = 1  # layer_idx0 + DISPLAY_OFFSET = display label (1-based)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrozenSources54:
    """Container of frozen Phase 54 source artifacts and verification metadata."""
    project_root: Path
    phase52_signoff: dict = field(default_factory=dict)
    phase53_signoff: dict = field(default_factory=dict)
    phase52_handoff: dict = field(default_factory=dict)
    phase53_context_handoff: dict = field(default_factory=dict)
    raw_files: dict[int, Path] = field(default_factory=dict)
    raw_observed_sha256: dict[str, str] = field(default_factory=dict)
    raw_expected_sha256: dict[str, str] = field(default_factory=dict)
    target_order_sha: str = ""
    observed_target_order_sha: str = ""
    lag_map_sha: str = ""
    observed_lag_map_sha: str = ""
    target_order: list[dict[str, str]] = field(default_factory=list)
    lag_map: list[dict[str, str]] = field(default_factory=list)
    pooling: str = ""
    last_query_directly_corresponds_to_pooled_token: bool = False
    phase52_last_query_summary_path: Path | None = None
    phase52_recent_mass_summary_path: Path | None = None
    phase52_top_source_summary_path: Path | None = None
    phase53_report_cases_path: Path | None = None
    phase52_dense_case_order_path: Path | None = None
    phase52_dense_case_position_map_path: Path | None = None
    phase53_render_config_fingerprint: dict = field(default_factory=dict)

    @property
    def n_test(self) -> int:
        return len(self.target_order) if self.target_order else N_TEST


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _sha256_file(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()


def _read_csv(fp: Path) -> list[dict[str, str]]:
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def load_frozen_sources_54(project_root: Path) -> FrozenSources54:
    """Load all frozen Phase 52/53 sources required by Phase 54."""
    root = Path(project_root).resolve()
    p52_dir = root / "artifacts" / "attention_extraction"
    p52_raw = p52_dir / "raw"
    p53_dir = root / "artifacts" / "attention_heatmaps"

    src = FrozenSources54(project_root=root)

    # Phase 52 signoff
    src.phase52_signoff = _read_json(p52_dir / "phase_52_signoff.json")
    # Phase 53 signoff
    src.phase53_signoff = _read_json(p53_dir / "phase_53_signoff.json")
    # Phase 52 numerical handoff (Phase 54 specific)
    src.phase52_handoff = _read_json(p52_dir / "phase54_last_query_attention_handoff.json")
    # Phase 53 context handoff
    src.phase53_context_handoff = _read_json(p53_dir / "phase54_last_query_attention_context_handoff.json")

    # Raw files
    src.raw_files = {
        seed: p52_raw / f"last_query_attention_seed{seed}.npz"
        for seed in SEEDS
    }

    # Shas
    expected = src.phase52_handoff.get("raw_files", {})
    src.raw_expected_sha256 = {fname: expected.get(fname, "") for fname in expected}
    for fname in src.raw_expected_sha256:
        for seed in SEEDS:
            if f"seed{seed}" in fname:
                fp = src.raw_files[seed]
                if fp.is_file():
                    src.raw_observed_sha256[fname] = _sha256_file(fp)
                break

    # Target order
    target_order_fp = p52_dir / "attention_test_target_order.csv"
    src.target_order = _read_csv(target_order_fp)
    src.target_order_sha = src.phase52_handoff.get("target_order_sha", "")
    src.observed_target_order_sha = _sha256_file(target_order_fp)

    # Lag map
    lag_map_fp = p52_dir / "attention_relative_position_map.csv"
    src.lag_map = _read_csv(lag_map_fp)
    src.lag_map_sha = src.phase52_handoff.get("lag_map_sha", "")
    src.observed_lag_map_sha = _sha256_file(lag_map_fp)

    # Definitions from handoff
    defs = src.phase52_handoff.get("definitions", {})
    src.pooling = defs.get("pooling", "")
    src.last_query_directly_corresponds_to_pooled_token = defs.get(
        "last_query_directly_corresponds_to_pooled_token", False
    )

    # Reference CSVs for reconstruction audit
    src.phase52_last_query_summary_path = p52_dir / "attention_last_query_summary.csv"
    src.phase52_recent_mass_summary_path = p52_dir / "attention_recent_mass_summary.csv"
    src.phase52_top_source_summary_path = p52_dir / "attention_top_source_summary.csv"
    src.phase53_report_cases_path = p53_dir / "attention_heatmap_report_cases.csv"
    src.phase52_dense_case_order_path = p52_dir / "attention_dense_case_order.csv"
    src.phase52_dense_case_position_map_path = p52_dir / "attention_case_position_map.csv"

    # Phase 53 render config fingerprint (context only)
    fp = p53_dir / "attention_heatmap_render_config_fingerprint.json"
    if fp.is_file():
        src.phase53_render_config_fingerprint = _read_json(fp)

    return src


# ---------------------------------------------------------------------------
# Last-query loading (Phase 52 NPZ)
# ---------------------------------------------------------------------------

def load_last_query_per_seed(src: FrozenSources54) -> dict[int, np.ndarray]:
    """Load raw last-query arrays keyed by seed.

    Returns: dict seed -> np.ndarray of shape [N_test, L, H, L] float32.
    """
    out: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        fp = src.raw_files[seed]
        npz = np.load(fp)
        key = [k for k in npz.files if "last_query" in k][0]
        arr = npz[key]
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.shape != (N_TEST, NUM_LAYERS, NUM_HEADS, LOOKBACK):
            raise ValueError(
                f"seed{seed} shape {arr.shape} != expected ({N_TEST}, {NUM_LAYERS}, {NUM_HEADS}, {LOOKBACK})"
            )
        out[seed] = arr
    return out


def lag_steps_array() -> np.ndarray:
    """Return lag_steps for raw positions: lag_steps_p = L - p (H=1). Shape (L,)."""
    return np.array([LOOKBACK - p for p in range(LOOKBACK)], dtype=np.int32)


def lag_minutes_array() -> np.ndarray:
    """Return lag_minutes for raw positions. Shape (L,)."""
    return lag_steps_array() * CADENCE_MINUTES
