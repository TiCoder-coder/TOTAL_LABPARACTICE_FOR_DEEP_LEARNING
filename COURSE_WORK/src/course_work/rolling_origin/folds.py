"""Phase 44 — Fold definitions and temporal assertions.

Defines the canonical K=3 expanding-origin fold protocol:

  RO1: outer_train = original Train
        inner_val   = last |V1| of Train
        inner_train = Train \\ inner_val
        outer_eval  = V1

  RO2: outer_train = Train + V1
        inner_val   = V1
        inner_train = Train
        outer_eval  = V2

  RO3: outer_train = Train + V1 + V2
        inner_val   = V2
        inner_train = Train + V1
        outer_eval  = V3

Temporal ordering is enforced STRICTLY. Violations raise AssertionError.

All assertions are pure functions on ordered lists of integers (target_ids),
not on windows or features — this keeps the population-level invariant
independent of any loader / scaler / window decision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from course_work.rolling_origin.populations import compute_population_fingerprint


@dataclass(frozen=True)
class FoldId:
    """Stable identifier for a fold."""

    index: int  # 1-based fold number

    def __str__(self) -> str:
        return f"RO{self.index}"


@dataclass(frozen=True)
class FoldDefinition:
    """Immutable fold definition. All IDs are sorted string target_ids."""

    fold_id: FoldId
    inner_train_ids: tuple[str, ...]
    inner_val_ids: tuple[str, ...]
    outer_train_ids: tuple[str, ...]
    outer_eval_ids: tuple[str, ...]

    # Fingerprints (populated by build_rolling_folds).
    inner_train_fingerprint: str = ""
    inner_val_fingerprint: str = ""
    outer_train_fingerprint: str = ""
    outer_eval_fingerprint: str = ""
    fold_population_fingerprint: str = ""

    def as_dict(self) -> dict:
        return {
            "fold_id": str(self.fold_id),
            "inner_train_ids": list(self.inner_train_ids),
            "inner_val_ids": list(self.inner_val_ids),
            "outer_train_ids": list(self.outer_train_ids),
            "outer_eval_ids": list(self.outer_eval_ids),
            "inner_train_fingerprint": self.inner_train_fingerprint,
            "inner_val_fingerprint": self.inner_val_fingerprint,
            "outer_train_fingerprint": self.outer_train_fingerprint,
            "outer_eval_fingerprint": self.outer_eval_fingerprint,
            "fold_population_fingerprint": self.fold_population_fingerprint,
        }


def _split_val_into_k(rval_ids: list[int], k: int) -> list[list[int]]:
    if len(rval_ids) == 0:
        raise ValueError("Cannot split empty RVAL_IDS into folds")
    if k < 2:
        raise ValueError(f"K must be >= 2 (got {k})")
    splits = [list(s) for s in np.array_split(rval_ids, k)]
    for i, s in enumerate(splits):
        if len(s) == 0:
            raise ValueError(f"Fold {i+1} of K={k} is empty")
    return splits


def _strict_temporal_ordering(a: Iterable, b: Iterable) -> bool:
    """Strict temporal ordering.

    For chronological integer IDs, this is `max(a) < min(b)`.
    For string IDs (e.g. "TGT_00000144"), we recover the integer part and
    compare chronologically.
    """
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list:
        return True
    a_max_chrono = max(_chronological_key(x) for x in a_list)
    b_min_chrono = min(_chronological_key(x) for x in b_list)
    return a_max_chrono < b_min_chrono


def _chronological_key(x) -> int:
    """Convert a target_id (int or string like 'TGT_00000144') to int."""
    s = str(x)
    # Extract the trailing digits if any
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return 0
    return 0


def _disjoint(a: Iterable, b: Iterable) -> bool:
    return set(str(x) for x in a).isdisjoint(set(str(y) for y in b))


def build_rolling_folds(
    rtrn_ids,
    rval_ids,
    k: int = 3,
) -> list[FoldDefinition]:
    """Build the canonical K-fold expanding-origin definitions.

    Pre-conditions:
      - rtrn_ids and rval_ids are sorted ascending (chronologically), disjoint,
        no Test rows.
    """
    rtrn = sorted((str(x) for x in rtrn_ids), key=_chronological_key)
    rval = sorted((str(x) for x in rval_ids), key=_chronological_key)
    assert _disjoint(rtrn, rval), "RTRN and RVAL must be disjoint"
    assert len(rtrn) > 0, "RTRN must be non-empty"

    V = _split_val_into_k(rval, k)
    n_v1 = len(V[0])

    folds: list[FoldDefinition] = []
    for k_idx in range(k):
        outer_eval = V[k_idx]
        prior_V = sum((V[:k_idx]), []) if k_idx > 0 else []
        outer_train = list(rtrn) + list(prior_V)

        if k_idx == 0:
            inner_val = list(rtrn[-n_v1:])
            inner_train = list(rtrn[:-n_v1])
        else:
            inner_val = list(V[k_idx - 1])
            inner_train = list(rtrn) + sum((V[: k_idx - 1]), [])

        fold = FoldDefinition(
            fold_id=FoldId(index=k_idx + 1),
            inner_train_ids=tuple(inner_train),
            inner_val_ids=tuple(inner_val),
            outer_train_ids=tuple(outer_train),
            outer_eval_ids=tuple(outer_eval),
        )

        # --- Temporal ordering assertions ---
        if len(fold.inner_train_ids) == 0 or len(fold.inner_val_ids) == 0:
            raise AssertionError(
                f"Fold {fold.fold_id}: empty inner_train or inner_val "
                f"({len(fold.inner_train_ids)}, {len(fold.inner_val_ids)})"
            )
        if len(fold.outer_eval_ids) == 0:
            raise AssertionError(f"Fold {fold.fold_id}: empty outer_eval")
        if not _strict_temporal_ordering(fold.inner_train_ids, fold.inner_val_ids):
            raise AssertionError(
                f"Fold {fold.fold_id}: inner_train not strictly before inner_val "
                f"(max(inner_train)={max(fold.inner_train_ids)}, "
                f"min(inner_val)={min(fold.inner_val_ids)})"
            )
        if not _strict_temporal_ordering(fold.inner_val_ids, fold.outer_eval_ids):
            raise AssertionError(
                f"Fold {fold.fold_id}: inner_val not strictly before outer_eval "
                f"(max(inner_val)={max(fold.inner_val_ids)}, "
                f"min(outer_eval)={min(fold.outer_eval_ids)})"
            )
        if not _disjoint(fold.outer_train_ids, fold.outer_eval_ids):
            raise AssertionError(
                f"Fold {fold.fold_id}: outer_train and outer_eval not disjoint"
            )

        # --- Augment with fingerprints ---
        fp_inner_train = compute_population_fingerprint(fold.inner_train_ids)
        fp_inner_val = compute_population_fingerprint(fold.inner_val_ids)
        fp_outer_train = compute_population_fingerprint(fold.outer_train_ids)
        fp_outer_eval = compute_population_fingerprint(fold.outer_eval_ids)
        fp_fold = compute_population_fingerprint(
            list(fold.inner_train_ids)
            + list(fold.inner_val_ids)
            + list(fold.outer_eval_ids)
        )
        fold = FoldDefinition(
            fold_id=fold.fold_id,
            inner_train_ids=fold.inner_train_ids,
            inner_val_ids=fold.inner_val_ids,
            outer_train_ids=fold.outer_train_ids,
            outer_eval_ids=fold.outer_eval_ids,
            inner_train_fingerprint=fp_inner_train,
            inner_val_fingerprint=fp_inner_val,
            outer_train_fingerprint=fp_outer_train,
            outer_eval_fingerprint=fp_outer_eval,
            fold_population_fingerprint=fp_fold,
        )
        folds.append(fold)

    # --- Cross-fold assertions ---
    # V_k disjoint, union == rval
    for i in range(k):
        for j in range(i + 1, k):
            assert _disjoint(folds[i].outer_eval_ids, folds[j].outer_eval_ids), (
                f"Folds RO{i+1} and RO{j+1} outer_eval not disjoint"
            )
    union_outer_eval: set[int] = set()
    for f in folds:
        union_outer_eval.update(f.outer_eval_ids)
    assert union_outer_eval == set(rval), (
        f"Union of outer_eval across folds != RVAL_IDS "
        f"(missing {len(set(rval) - union_outer_eval)} ids, "
        f"extra {len(union_outer_eval - set(rval))} ids)"
    )

    return folds


def validate_fold_temporal_ordering(folds: list[FoldDefinition]) -> list[dict]:
    """Return a per-fold validation table (used by O44.11 temporal leakage tests).

    Each row has fields:
      fold_id
      max_inner_train_id
      min_inner_val_id
      max_inner_val_id
      min_outer_eval_id
      temporal_ok
      disjoint_ok
      status
    """
    from course_work.rolling_origin.folds import _chronological_key

    rows: list[dict] = []
    for f in folds:
        max_it = max(f.inner_train_ids, key=_chronological_key) if f.inner_train_ids else ""
        min_iv = min(f.inner_val_ids, key=_chronological_key) if f.inner_val_ids else ""
        max_iv = max(f.inner_val_ids, key=_chronological_key) if f.inner_val_ids else ""
        min_oe = min(f.outer_eval_ids, key=_chronological_key) if f.outer_eval_ids else ""
        temporal_ok = (max_it < min_iv) and (max_iv < min_oe)
        disjoint_ok = _disjoint(f.outer_train_ids, f.outer_eval_ids)
        rows.append(
            {
                "fold_id": str(f.fold_id),
                "max_inner_train_id": str(max_it),
                "min_inner_val_id": str(min_iv),
                "max_inner_val_id": str(max_iv),
                "min_outer_eval_id": str(min_oe),
                "temporal_ok": bool(temporal_ok),
                "disjoint_ok": bool(disjoint_ok),
                "status": "PASS" if (temporal_ok and disjoint_ok) else "FAIL",
            }
        )
    return rows


def serialize_fold_manifest(folds: list[FoldDefinition]) -> dict:
    """Serialize the frozen fold manifest (O44.4)."""
    return {
        "version": "ROLLING_ORIGIN_FOLD_MANIFEST-v1",
        "phase": 44,
        "K": len(folds),
        "fold_protocol": "RO3_EXPANDING_PRETEST-v1",
        "fold_local_scaling": True,
        "nested_epoch_selection": True,
        "full_history_refit": True,
        "outer_eval_used_for_selection": False,
        "outer_eval_region": "original Validation",
        "test_region_used": False,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "folds": [f.as_dict() for f in folds],
    }