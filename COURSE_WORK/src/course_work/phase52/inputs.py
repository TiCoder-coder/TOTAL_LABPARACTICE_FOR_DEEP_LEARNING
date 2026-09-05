"""Phase 52 — deterministic input reconstruction for ALL Test targets.

This module produces the exact ordered [N_test, L, F] model-visible input
batch for the entire FINAL_TEST_POP-v1 in canonical target_sample_id order.

It is a DETERMINISTIC, READ-ONLY operation:

  - Loads frozen Phase 47 input pipeline (FEATURES-v1 derived CSV, WINDOWPOP-v1
    window index, frozen FINAL_SCALING-v1 X scaler in transform-only mode).
  - Reuses the exact Phase 51-F reconstruction logic.
  - Does NOT load checkpoints, train, fit scaler, or modify any source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from course_work.phase51.exact_input_reconstruction import (
    CONTINUOUS_FEATURES,
    FEATURE_COUNT,
    FS2_TF1_VARIANT_ID,
    LOOKBACK,
    PASS_THROUGH_FEATURES,
    _load_feature_view,
    _load_final_x_scaler,
    _load_window_index,
    _transform_x_block,
    _verify_final_feature_contract,
    _verify_final_scaling_v1_x_checksum,
    _verify_phase10_signoff,
)
from course_work.utils.artifacts import get_project_root


@dataclass(frozen=True)
class TestInputsBundle:
    """Frozen bundle of exact ordered model-visible input for all Test targets."""

    x: np.ndarray  # [N_test, L, F] float32
    target_ids: tuple[str, ...]
    target_timestamps: tuple[str, ...]
    target_indices: tuple[int, ...]  # raw row indices in FEATURES-v1 timeline
    input_start_indices: tuple[int, ...]  # raw row indices
    input_end_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_set_id: str
    feature_fingerprint: str
    boundary_protocol: str
    lookback_steps: int
    horizon_steps: int
    cadence_minutes: int
    population_id: str
    population_sha256: str
    phase10_signoff_sha256: str
    final_scaling_x_sha256: str


def build_all_test_inputs(
    project_root: Path | None = None,
    selected_target_ids: list[str] | None = None,
) -> TestInputsBundle:
    """Build the exact ordered model-visible input batch for ALL Test targets.

    Returns a TestInputsBundle with x in canonical target_sample_id order.

    STRICT SAFETY:
      - No checkpoint loading
      - No model inference
      - scaler.transform() only (fit is forbidden)
      - All sources are read-only
    """
    root = project_root if project_root is not None else get_project_root()

    # Verification gates
    phase10_so = _verify_phase10_signoff(root)
    phase10_so_fp = root / "artifacts/windows/phase_10_signoff.json"
    phase10_so_sha = hashlib.sha256(phase10_so_fp.read_bytes()).hexdigest()
    final_scaling_sha = _verify_final_scaling_v1_x_checksum(root)
    final_contract = _verify_final_feature_contract(root)

    # Sources
    fv = _load_feature_view(root)
    wi = _load_window_index(root)
    scaler = _load_final_x_scaler(root)

    canonical_features = list(final_contract["feature_names"])
    assert len(canonical_features) == FEATURE_COUNT

    raw_full = np.ascontiguousarray(
        fv[canonical_features].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    cont_count = len(CONTINUOUS_FEATURES)
    pt_count = len(PASS_THROUGH_FEATURES)
    assert cont_count + pt_count == FEATURE_COUNT

    scaled_cont = _transform_x_block(scaler, raw_full[:, :cont_count])
    model_full = np.concatenate(
        [scaled_cont, raw_full[:, cont_count:].astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    # Filter to TEST L=72 WB0-valid records (canonical 2961)
    mask = (
        (wi["target_split_id"] == "TEST")
        & (wi["lookback_steps"] == LOOKBACK)
        & (wi["WB0_valid"] == True)
        & (wi["included_common_population"] == True)
    )
    candidate = wi[mask].sort_values("target_sample_id")
    if len(candidate) != 2961:
        raise RuntimeError(f"Expected 2961 TEST L=72 WB0 records; got {len(candidate)}")

    target_ids = [str(x) for x in candidate["target_sample_id"].tolist()]
    target_timestamps = [str(x) for x in candidate["target_timestamp"].tolist()]
    input_starts = [int(x) for x in candidate["input_start_raw_row_index"].tolist()]
    input_ends = [int(x) for x in candidate["input_end_raw_row_index"].tolist()]
    target_idxs = [int(x) for x in candidate["target_raw_row_index"].tolist()]

    if selected_target_ids is not None:
        wanted = set(selected_target_ids)
        keep = [i for i, t in enumerate(target_ids) if t in wanted]
        target_ids = [target_ids[i] for i in keep]
        target_timestamps = [target_timestamps[i] for i in keep]
        input_starts = [input_starts[i] for i in keep]
        input_ends = [input_ends[i] for i in keep]
        target_idxs = [target_idxs[i] for i in keep]

    n = len(target_ids)
    x = np.zeros((n, LOOKBACK, FEATURE_COUNT), dtype=np.float32)
    for i in range(n):
        x[i] = model_full[input_starts[i]:input_ends[i] + 1, :]

    if n > 0:
        assert x.shape == (n, LOOKBACK, FEATURE_COUNT), f"Shape {x.shape} != expected"
        # No target row leakage
        for i in range(n):
            assert input_ends[i] < target_idxs[i], f"Window leaks target row for {target_ids[i]}"

    return TestInputsBundle(
        x=x,
        target_ids=tuple(target_ids),
        target_timestamps=tuple(target_timestamps),
        target_indices=tuple(target_idxs),
        input_start_indices=tuple(input_starts),
        input_end_indices=tuple(input_ends),
        feature_names=tuple(canonical_features),
        feature_set_id=FS2_TF1_VARIANT_ID,
        feature_fingerprint=final_contract["feature_fingerprint"],
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        lookback_steps=LOOKBACK,
        horizon_steps=1,
        cadence_minutes=10,
        population_id="FINAL_TEST_POP-v1",
        population_sha256=hashlib.sha256(
            ",".join(sorted(target_ids)).encode("utf-8")
        ).hexdigest(),
        phase10_signoff_sha256=phase10_so_sha,
        final_scaling_x_sha256=final_scaling_sha,
    )
