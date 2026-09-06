"""Phase 53 — frozen sources, constants, paths.

This module is the SINGLE SOURCE OF TRUTH for Phase 53 paths and locked
contracts. It NEVER mutates any upstream artifact; it only declares
constants and reads frozen Phase 51 / 52 evidence.

Phase 53 is a strictly visualization phase:

* no model checkpoint loading;
* no model.forward / forward_with_attention;
* no new attention extraction;
* no new Test inference;
* no training, no scaler fitting, no optimizer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Canonical Phase 53 contract ──────────────────────────────────────────────

ATTENTION_HEATMAPS_VERSION = "ATTENTION_HEATMAPS-v1"
PHASE = 53
PHASE_DIR_REL = Path("artifacts/attention_heatmaps")
IMAGES_DIR_REL = Path("artifacts/attention_heatmaps/images")
CASE_GRIDS_DIR_REL = Path("artifacts/attention_heatmaps/images/case_grids")
CROSS_SEED_DIR_REL = Path("artifacts/attention_heatmaps/images/cross_seed_report")
FIXED_PROB_DIR_REL = Path("artifacts/attention_heatmaps/images/fixed_probability_report")
INDIVIDUAL_DIR_REL = Path("artifacts/attention_heatmaps/images/individual_maps")

# Read from Phase 52 frozen outputs
PHASE52_DIR_REL = Path("artifacts/attention_extraction")
PHASE51_DIR_REL = Path("artifacts/worst_error_analysis")

SEEDS: tuple[int, ...] = (42, 123, 2026)
LOOKBACK = 72
NUM_LAYERS = 2
NUM_HEADS = 4
RAW_DTYPE = "float32"

# Lag tick policy
LAG_TICK_STEPS_REQUESTED = (1, 6, 36, 72, 144)
INCLUDE_OLDEST_TICK = True
CADENCE_MINUTES = 10

# Mode A — FIXED_PROBABILITY
MODE_A_NAME = "FIXED_PROBABILITY"
MODE_A_VMIN = 0.0
MODE_A_VMAX = 1.0

# Mode B — CASE_SHARED_SCALE
MODE_B_NAME = "CASE_SHARED_SCALE"
MODE_B_VMIN = 0.0  # derived; per-case vmax
MODE_B_VMAX_FORMULA = "MAX_OVER_ALL_SEEDS_LAYERS_HEADS_FOR_CASE"

# Render config contract
MATRIX_ORIENTATION = "QUERY_ROWS_SOURCE_COLUMNS"
TRANSPOSE = False
ORIGIN = "ROW0_TOP"
X_ORDER = "OLDEST_TO_NEWEST"
Y_ORDER = "OLDEST_TO_NEWEST_TOP_TO_BOTTOM"
X_LABEL = "Source / key historical position"
Y_LABEL = "Query historical position"
COLORBAR_LABEL = "Attention weight"
COLORMAP_POLICY = "SEQUENTIAL_PERCEPTUALLY_UNIFORM"
INTERPOLATION = "NONE_OR_NEAREST"
ASPECT = "SQUARE_PER_MATRIX"

# Report case rule
REPORT_CASE_RULE = "PHASE51_W2_SHARED_RANK_1_TO_5"
REPORT_CASE_RANK_RANGE = (1, 5)  # inclusive
MAX_REPORT_CASES = 5

# Render quality
DPI = 300
COLORMAP = "viridis"  # perceptually uniform sequential

# Numerical tolerance
NONNEGATIVE_EPS = -1e-7
MAX_ATTENTION_TOLERANCE = 1.0 + 1e-6
ROW_SUM_ATOL = 1e-5
ROW_SUM_RTOL = 1e-5


def _sha256_file(path: Path) -> str:
    if path.is_dir() or not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenSources53:
    """All Phase 53 frozen sources (READ-ONLY)."""

    project_root: Path
    phase52_signoff_sha256: str
    phase52_handoff_sha256: str
    raw_attention_checksums_sha256: str
    dense_case_order_sha256: str
    relative_position_map_sha256: str
    case_position_map_sha256: str
    case_metadata_sha256: str
    dense_case_count: int
    case_order: list[str]
    raw_file_path_per_seed: dict[int, Path]
    raw_file_sha256_per_seed: dict[int, str]
    phase51_handoff_sha256: str
    worst_shared_top20_sha256: str
    casebook_index_sha256: str
    extra: dict[str, Any]


def load_frozen_sources_53(project_root: Path) -> FrozenSources53:
    """Read every canonical Phase 53 source and SHA-256 verify it.

    Pure read-only. No mutation of any source file.
    """
    root = project_root

    # Phase 52 signoff + handoff
    p52_so_fp = root / "artifacts/attention_extraction/phase_52_signoff.json"
    p52_so = json.loads(p52_so_fp.read_text()) if p52_so_fp.exists() else {}
    p52_handoff_fp = root / "artifacts/attention_extraction/phase53_attention_heatmaps_handoff.json"
    p52_handoff = json.loads(p52_handoff_fp.read_text()) if p52_handoff_fp.exists() else {}

    raw_chk_fp = root / "artifacts/attention_extraction/raw_attention_checksums.json"
    raw_chk = json.loads(raw_chk_fp.read_text()) if raw_chk_fp.exists() else {}

    # Case-order, position map, case metadata
    co_fp = root / "artifacts/attention_extraction/attention_dense_case_order.csv"
    rpm_fp = root / "artifacts/attention_extraction/attention_relative_position_map.csv"
    cpm_fp = root / "artifacts/attention_extraction/attention_case_position_map.csv"
    cm_fp = root / "artifacts/attention_extraction/attention_case_metadata.csv"

    case_order: list[str] = []
    if co_fp.exists():
        for line in co_fp.read_text().splitlines()[1:]:
            if line.strip():
                fields = line.split(",")
                # case_row_idx0 is first column; target_id is the second
                if len(fields) >= 2:
                    case_order.append(fields[1])

    # Raw dense NPZ per seed
    raw_path: dict[int, Path] = {}
    raw_sha: dict[int, str] = {}
    for seed in SEEDS:
        fn = f"dense_case_attention_seed{seed}.npz"
        fp = root / "artifacts/attention_extraction/raw" / fn
        raw_path[seed] = fp
        raw_sha[seed] = _sha256_file(fp)

    # Phase 51 frozen handoff + casebook + shared top20
    p51_handoff_fp = root / "artifacts/worst_error_analysis/phase52_attention_extraction_handoff.json"
    ws_top20_fp = root / "artifacts/worst_error_analysis/worst_shared_top20.csv"
    cb_idx_fp = root / "artifacts/worst_error_analysis/casebook_index.csv"

    return FrozenSources53(
        project_root=root,
        phase52_signoff_sha256=_sha256_file(p52_so_fp),
        phase52_handoff_sha256=_sha256_file(p52_handoff_fp),
        raw_attention_checksums_sha256=_sha256_file(raw_chk_fp),
        dense_case_order_sha256=_sha256_file(co_fp),
        relative_position_map_sha256=_sha256_file(rpm_fp),
        case_position_map_sha256=_sha256_file(cpm_fp),
        case_metadata_sha256=_sha256_file(cm_fp),
        dense_case_count=p52_handoff.get("dense_case_count", len(case_order)),
        case_order=case_order,
        raw_file_path_per_seed=raw_path,
        raw_file_sha256_per_seed=raw_sha,
        phase51_handoff_sha256=_sha256_file(p51_handoff_fp),
        worst_shared_top20_sha256=_sha256_file(ws_top20_fp),
        casebook_index_sha256=_sha256_file(cb_idx_fp),
        extra={
            "phase52_status": p52_so.get("phase52_status"),
            "ready_for_phase53": p52_handoff.get("ready_for_phase53"),
            "phase53_authorized_in_p52": p52_handoff.get("phase53_authorized"),
            "raw_file_shas_from_handoff": p52_handoff.get("raw_files", {}),
        },
    )
