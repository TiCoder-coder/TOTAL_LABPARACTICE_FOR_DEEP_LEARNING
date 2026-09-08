"""Phase 54 — coverage radii (Lag50/80/90) and non-overlap lag bins.

Coverage radii use cumulative recency mass in newest->oldest order:
    recency[k] = a[L-k] for k = 1..L
    C(k) = sum_{j=1..k} recency[j]
    LagX = min{k: C(k) >= X}

Non-overlap bins are defined by lag-step ranges; only bins where the upper
bound is <= L are emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources import COVERAGE_LEVELS, NON_OVERLAP_BIN_LABELS, NON_OVERLAP_BINS


@dataclass
class CoverageRadii:
    """Coverage radii for one last-query vector.

    All values are stored in steps. If a level cannot be reached within L
    (e.g., cumulative mass never reaches 0.90), the corresponding lag
    is set to L (full history) and the `reached` flag is False.
    """
    lag50_steps: int
    lag50_reached: bool
    lag80_steps: int
    lag80_reached: bool
    lag90_steps: int
    lag90_reached: bool

    @property
    def as_dict(self) -> dict[str, int | bool]:
        return {
            "lag50_steps": self.lag50_steps,
            "lag50_reached": self.lag50_reached,
            "lag80_steps": self.lag80_steps,
            "lag80_reached": self.lag80_reached,
            "lag90_steps": self.lag90_steps,
            "lag90_reached": self.lag90_reached,
        }


def compute_coverage_radii(a: np.ndarray) -> CoverageRadii:
    """Compute Lag50 / Lag80 / Lag90 coverage radii in steps."""
    arr = np.asarray(a, dtype=np.float64)
    L = arr.shape[0]
    s = float(arr.sum())
    if s > 0:
        arr = arr / s
    # Recency order: newest->oldest => reverse raw array
    recency = arr[::-1]
    cum = np.cumsum(recency)
    result: dict[float, tuple[int, bool]] = {}
    for lvl in COVERAGE_LEVELS:
        idx = int(np.searchsorted(cum, lvl, side="left"))
        if idx >= L:
            # Did not reach this level within L
            result[lvl] = (L, False)
        else:
            result[lvl] = (idx + 1, True)  # +1 because searchsorted gives 0-indexed position
    return CoverageRadii(
        lag50_steps=result[0.50][0],
        lag50_reached=result[0.50][1],
        lag80_steps=result[0.80][0],
        lag80_reached=result[0.80][1],
        lag90_steps=result[0.90][0],
        lag90_reached=result[0.90][1],
    )


@dataclass
class LagBinMass:
    """Non-overlapping lag-bin mass for one last-query vector."""
    bin_label: str
    lag_start_steps: int
    lag_end_steps: int
    effective_start: int  # inclusive source position index0
    effective_end: int  # inclusive source position index0
    mass: float
    applicable: bool


def compute_lag_bin_masses(a: np.ndarray) -> list[LagBinMass]:
    """Compute non-overlapping lag-bin masses.

    Each bin (start, end) corresponds to lag_steps in [start, end]. The
    corresponding source positions are p in [L-end, L-start] (inclusive).
    Only bins where end <= L are applicable.

    Returns a list of LagBinMass, one per canonical bin (in order).
    """
    arr = np.asarray(a, dtype=np.float64)
    L = arr.shape[0]
    s = float(arr.sum())
    if s > 0:
        arr = arr / s
    out: list[LagBinMass] = []
    for (start, end), label in zip(NON_OVERLAP_BINS, NON_OVERLAP_BIN_LABELS):
        applicable = end <= L
        if not applicable:
            out.append(LagBinMass(
                bin_label=label,
                lag_start_steps=start,
                lag_end_steps=end,
                effective_start=-1,
                effective_end=-1,
                mass=0.0,
                applicable=False,
            ))
            continue
        # Source positions p where lag_steps_p in [start, end]
        # lag_steps_p = L - p, so start <= L-p <= end => L-end <= p <= L-start
        eff_start = L - end
        eff_end = L - start
        mass = float(arr[eff_start:eff_end + 1].sum())
        out.append(LagBinMass(
            bin_label=label,
            lag_start_steps=start,
            lag_end_steps=end,
            effective_start=eff_start,
            effective_end=eff_end,
            mass=mass,
            applicable=True,
        ))
    return out
