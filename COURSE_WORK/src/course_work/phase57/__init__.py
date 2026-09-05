"""Phase 57 - seed-stability attention check (SEED_STABILITY_ATTENTION-v1)."""

from . import analyses, figures, findings
from .sources import (
    ANCHOR_REASON,
    ANCHOR_SEED,
    CORE_ATTENTION_METRICS_V1,
    H2_LAYER_HEAD_COUNTS,
    LAG_MINUTES,
    LOOKBACK,
    MATCH_TIE_TOL,
    NUM_HEADS,
    NUM_LAYERS,
    N_TEST,
    SEED_PAIRS,
    SEEDS,
    FrozenSources57,
    load_frozen_sources57,
)
from .orchestrator import run_phase57

__all__ = [
    "analyses",
    "figures",
    "findings",
    "ANCHOR_REASON",
    "ANCHOR_SEED",
    "CORE_ATTENTION_METRICS_V1",
    "H2_LAYER_HEAD_COUNTS",
    "LAG_MINUTES",
    "LOOKBACK",
    "MATCH_TIE_TOL",
    "NUM_HEADS",
    "NUM_LAYERS",
    "N_TEST",
    "SEED_PAIRS",
    "SEEDS",
    "FrozenSources57",
    "load_frozen_sources57",
    "run_phase57",
]
