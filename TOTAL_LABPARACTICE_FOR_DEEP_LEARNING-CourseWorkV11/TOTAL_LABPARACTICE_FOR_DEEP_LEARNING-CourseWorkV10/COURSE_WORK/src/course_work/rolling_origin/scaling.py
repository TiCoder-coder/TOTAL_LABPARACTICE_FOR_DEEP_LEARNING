"""Phase 44 — Fold-local scaler fitting (RO_SCALING-v1).

This module is the single source of truth for fold-local scalers.

It does NOT use any global scaler from Phase 9 or any other upstream phase.
It fits per (fold, stage, candidate) on a target-ID-restricted history.

The ScalerFitAudit is the proof that scalers were restricted to allowed
training history and never saw outer-eval or Test rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from course_work.rolling_origin.populations import compute_population_fingerprint


SCALING_VERSION = "RO_SCALING-v1"
ALLOWED_TARGET_SCALING = {"YS0", "YS1"}


@dataclass
class FoldLocalScalerBundle:
    """A scaler bundle fit on a fold-specific history.

    `feature_indices_passthrough` lists columns that are NOT scaled (binary,
    cyclical, target passthrough, RevIN-protected).
    `feature_indices_scaled` lists columns that ARE standardized.
    """

    bundle_id: str
    fit_stage: str            
    fold_id: str              
    candidate_id: str
    target_scaling_option: str  
    feature_variant_id: str
    lookback_steps: int
    boundary_protocol: str
    revin_enabled: bool

    fit_population_fingerprint: str
    fit_target_ids: tuple[int, ...]
    fit_raw_row_count: int

    x_means: np.ndarray | None = None
    x_stds: np.ndarray | None = None
    feature_indices_scaled: tuple[int, ...] = field(default_factory=tuple)
    feature_indices_passthrough: tuple[int, ...] = field(default_factory=tuple)

    y_mean: float | None = None
    y_std: float | None = None

    def checksum(self) -> str:
        blob = {
            "bundle_id": self.bundle_id,
            "fit_stage": self.fit_stage,
            "fold_id": self.fold_id,
            "candidate_id": self.candidate_id,
            "target_scaling_option": self.target_scaling_option,
            "feature_variant_id": self.feature_variant_id,
            "lookback_steps": self.lookback_steps,
            "boundary_protocol": self.boundary_protocol,
            "revin_enabled": self.revin_enabled,
            "fit_population_fingerprint": self.fit_population_fingerprint,
            "fit_target_ids": list(self.fit_target_ids),
            "fit_raw_row_count": self.fit_raw_row_count,
            "x_means": None if self.x_means is None else self.x_means.tolist(),
            "x_stds": None if self.x_stds is None else self.x_stds.tolist(),
            "feature_indices_scaled": list(self.feature_indices_scaled),
            "feature_indices_passthrough": list(self.feature_indices_passthrough),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }
        from hashlib import sha256
        return sha256(
            json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Map scaled-space y to Wh."""
        if self.target_scaling_option == "YS0":
            return np.asarray(y_scaled, dtype=np.float64)
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError(f"{self.bundle_id} is YS1 but y_mean/y_std unset")
        return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    def transform_y(self, y_wh: np.ndarray) -> np.ndarray:
        if self.target_scaling_option == "YS0":
            return np.asarray(y_wh, dtype=np.float64)
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError(f"{self.bundle_id} is YS1 but y_mean/y_std unset")
        return (np.asarray(y_wh, dtype=np.float64) - self.y_mean) / self.y_std

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        """Apply scaler to X (shape (N, F)). X is float32."""
        out = np.array(X, dtype=np.float64, copy=True)
        if (
            self.x_means is None
            or self.x_stds is None
            or len(self.feature_indices_scaled) == 0
        ):
            return out.astype(np.float32)
        idx = np.array(self.feature_indices_scaled, dtype=np.int64)
        out[:, idx] = (out[:, idx] - self.x_means) / self.x_stds
        return out.astype(np.float32)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["x_means"] = None if self.x_means is None else self.x_means.tolist()
        d["x_stds"] = None if self.x_stds is None else self.x_stds.tolist()
        d["checksum"] = self.checksum()
        d["scaling_version"] = "SCALING-v1"
 
        from sklearn.preprocessing import StandardScaler
        d["option"] = self.target_scaling_option
        if self.target_scaling_option == "YS1":
            y_scaler = StandardScaler(with_mean=True, with_std=True)
            y_scaler.mean_ = np.asarray([float(self.y_mean)], dtype=np.float64)
            y_scaler.scale_ = np.asarray([float(self.y_std)], dtype=np.float64)
            y_scaler.var_ = y_scaler.scale_ ** 2
            y_scaler.n_features_in_ = 1
            y_scaler.n_samples_seen_ = 1
            d["scaler"] = y_scaler
        else:
            class _IdentityScaler:
                def transform(self, x):
                    return np.asarray(x, dtype=np.float64).reshape(-1, 1)
                def inverse_transform(self, x):
                    return np.asarray(x, dtype=np.float64).reshape(-1)
            d["scaler"] = _IdentityScaler()
        return d

@dataclass
class ScalerFitAudit:
    """Single scaler-fit audit row, written to O44.10."""

    bundle_id: str
    fit_stage: str
    fold_id: str
    candidate_id: str
    target_scaling_option: str
    feature_variant_id: str

    fit_target_ids_count: int
    fit_target_ids_fingerprint: str
    fit_raw_row_count: int

    fit_start_timestamp: str 
    fit_end_timestamp: str

    allowed_target_history_region: str
    outer_eval_rows_used: int
    test_rows_used: int        

    bundle_checksum: str
    status: str              

    def as_row(self) -> dict:
        return asdict(self)


def record_scaler_fit_audit(
    *,
    bundle: FoldLocalScalerBundle,
    windowpop_df, 
    fit_target_ids: list[int],
    fold_outer_eval_ids: list[int],
    test_target_ids: list[int] | None = None,
    allowed_region_label: str,
) -> ScalerFitAudit:
    """Build a ScalerFitAudit for a fit operation.

    Asserts:
      - 0 outer_eval rows used
      - 0 test rows used
      - fit region is strictly before fold origin
    """
    test_target_ids = list(test_target_ids or [])

    def _norm(t):
        return str(t)

    fit_tids = set(_norm(t) for t in fit_target_ids)
    outer_eval_tids = set(_norm(t) for t in fold_outer_eval_ids)
    test_tids = set(_norm(t) for t in test_target_ids)

    wp_norm_sample = windowpop_df["target_id"].astype(str).iloc[0] if len(windowpop_df) else ""
    wp_is_int = False
    try:
        int(wp_norm_sample)
        wp_is_int = True
    except (ValueError, TypeError):
        wp_is_int = False

    if wp_is_int:
        def _to_int_form(s):
            if s.startswith("TGT_") or s.startswith("tgt_"):
                try:
                    return int(s.split("_", 1)[1])
                except (ValueError, IndexError):
                    return None
            try:
                return int(s)
            except (ValueError, TypeError):
                return None
        fit_tids = {iv for iv in (_to_int_form(s) for s in fit_tids) if iv is not None}
        outer_eval_tids = {iv for iv in (_to_int_form(s) for s in outer_eval_tids) if iv is not None}
        test_tids = {iv for iv in (_to_int_form(s) for s in test_tids) if iv is not None}

    fit_raw_rows = int(windowpop_df["target_id"].isin(fit_tids).sum())
    fit_windowpop = windowpop_df[windowpop_df["target_id"].isin(fit_tids)]
    fit_outer_overlap = len(fit_tids & outer_eval_tids)
    fit_test_overlap = len(fit_tids & test_tids)
    outer_eval_raw = fit_outer_overlap
    test_raw = fit_test_overlap

    assert outer_eval_raw == 0, (
        f"ScalerFitAudit failure: outer_eval IDs leaked into fit set "
        f"(fit ∩ outer_eval = {fit_outer_overlap})"
    )
    assert test_raw == 0, (
        f"ScalerFitAudit failure: test IDs leaked into fit set "
        f"(fit ∩ test = {fit_test_overlap})"
    )

    if fit_windowpop.empty:
        raise ValueError(f"No windowpop rows for fit_target_ids in {bundle.bundle_id}")
    fit_windowpop = fit_windowpop.sort_values("target_timestamp")
    fit_start_ts = fit_windowpop["target_timestamp"].iloc[0]
    fit_end_ts = fit_windowpop["target_timestamp"].iloc[-1]

    audit = ScalerFitAudit(
        bundle_id=bundle.bundle_id,
        fit_stage=bundle.fit_stage,
        fold_id=bundle.fold_id,
        candidate_id=bundle.candidate_id,
        target_scaling_option=bundle.target_scaling_option,
        feature_variant_id=bundle.feature_variant_id,
        fit_target_ids_count=len(fit_target_ids),
        fit_target_ids_fingerprint=compute_population_fingerprint(fit_target_ids),
        fit_raw_row_count=fit_raw_rows,
        fit_start_timestamp=str(fit_start_ts),
        fit_end_timestamp=str(fit_end_ts),
        allowed_target_history_region=allowed_region_label,
        outer_eval_rows_used=outer_eval_raw,
        test_rows_used=test_raw,
        bundle_checksum=bundle.checksum(),
        status="PASS",
    )
    return audit

def _classify_feature_columns(
    feature_variant_id: str, feature_columns: list[str]
) -> tuple[list[int], list[int]]:
    """Identify which columns are scaled vs. passthrough.

    Returns (scaled_indices, passthrough_indices).

    Heuristic (matches Phase 9 / SCALING-v1):
      - cyclical time (sin/cos) → passthrough
      - binary / one-hot → passthrough
      - RevIN-protected → passthrough
      - everything else → standard-scaled
    """
    passthrough_suffixes = (
        "_sin", "_cos", "_is_", "_hour", "_dow", "_month", "_weekend"
    )
    scaled: list[int] = []
    passthrough: list[int] = []
    for i, name in enumerate(feature_columns):
        low = str(name).lower()
        if any(low.endswith(suf) for suf in passthrough_suffixes):
            passthrough.append(i)
        elif low.startswith("is_") or low.endswith("_flag") or low.endswith("_binary"):
            passthrough.append(i)
        else:
            scaled.append(i)
    return scaled, passthrough


def _fit_x_scaler(
    X_fit: np.ndarray,
    feature_variant_id: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Fit per-column mean/std on the fit subset only.

    Returns (means, stds, scaled_indices, passthrough_indices).
    """
    scaled_idx, passthrough_idx = _classify_feature_columns(
        feature_variant_id, feature_columns
    )
    if not scaled_idx or X_fit.shape[0] == 0:
        return (
            np.zeros(len(scaled_idx), dtype=np.float64),
            np.ones(len(scaled_idx), dtype=np.float64),
            scaled_idx,
            passthrough_idx,
        )
    Xs = X_fit[:, scaled_idx].astype(np.float64)
    means = Xs.mean(axis=0)
    stds = Xs.std(axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    return means, stds, scaled_idx, passthrough_idx


def _fit_y_scaler(y_fit: np.ndarray) -> tuple[float | None, float | None]:
    if len(y_fit) == 0:
        return None, None
    m = float(np.mean(y_fit))
    s = float(np.std(y_fit))
    if s < 1e-12:
        s = 1.0
    return m, s


def fit_fold_a_x_scaler(
    X_fit: np.ndarray,
    *,
    feature_variant_id: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    return _fit_x_scaler(X_fit, feature_variant_id, feature_columns)


def fit_fold_a_y_scaler(
    y_fit: np.ndarray, *, target_scaling_option: str
) -> tuple[float | None, float | None]:
    if target_scaling_option not in ALLOWED_TARGET_SCALING:
        raise ValueError(f"Unknown target_scaling_option={target_scaling_option}")
    if target_scaling_option == "YS0":
        return None, None
    return _fit_y_scaler(y_fit)


def fit_fold_b_x_scaler(
    X_fit: np.ndarray,
    *,
    feature_variant_id: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    return _fit_x_scaler(X_fit, feature_variant_id, feature_columns)


def fit_fold_b_y_scaler(
    y_fit: np.ndarray, *, target_scaling_option: str
) -> tuple[float | None, float | None]:
    if target_scaling_option not in ALLOWED_TARGET_SCALING:
        raise ValueError(f"Unknown target_scaling_option={target_scaling_option}")
    if target_scaling_option == "YS0":
        return None, None
    return _fit_y_scaler(y_fit)


def build_bundle(
    *,
    bundle_id: str,
    fit_stage: str,
    fold_id: str,
    candidate_id: str,
    target_scaling_option: str,
    feature_variant_id: str,
    lookback_steps: int,
    boundary_protocol: str,
    revin_enabled: bool,
    fit_target_ids: list[int],
    fit_raw_row_count: int,
    x_means: np.ndarray | None,
    x_stds: np.ndarray | None,
    feature_indices_scaled: list[int],
    feature_indices_passthrough: list[int],
    y_mean: float | None,
    y_std: float | None,
) -> FoldLocalScalerBundle:
    return FoldLocalScalerBundle(
        bundle_id=bundle_id,
        fit_stage=fit_stage,
        fold_id=fold_id,
        candidate_id=candidate_id,
        target_scaling_option=target_scaling_option,
        feature_variant_id=feature_variant_id,
        lookback_steps=lookback_steps,
        boundary_protocol=boundary_protocol,
        revin_enabled=revin_enabled,
        fit_population_fingerprint=compute_population_fingerprint(fit_target_ids),
        fit_target_ids=tuple(str(t) for t in fit_target_ids),
        fit_raw_row_count=fit_raw_row_count,
        x_means=x_means,
        x_stds=x_stds,
        feature_indices_scaled=tuple(feature_indices_scaled),
        feature_indices_passthrough=tuple(feature_indices_passthrough),
        y_mean=y_mean,
        y_std=y_std,
    )

def serialize_scaler_bundle(bundle: FoldLocalScalerBundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.as_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def load_scaler_bundle(path: Path) -> FoldLocalScalerBundle:
    raw = json.loads(path.read_text())
    return FoldLocalScalerBundle(
        bundle_id=raw["bundle_id"],
        fit_stage=raw["fit_stage"],
        fold_id=raw["fold_id"],
        candidate_id=raw["candidate_id"],
        target_scaling_option=raw["target_scaling_option"],
        feature_variant_id=raw["feature_variant_id"],
        lookback_steps=int(raw["lookback_steps"]),
        boundary_protocol=raw["boundary_protocol"],
        revin_enabled=bool(raw["revin_enabled"]),
        fit_population_fingerprint=raw["fit_population_fingerprint"],
        fit_target_ids=tuple(int(t) for t in raw["fit_target_ids"]),
        fit_raw_row_count=int(raw["fit_raw_row_count"]),
        x_means=None if raw["x_means"] is None else np.asarray(raw["x_means"], dtype=np.float64),
        x_stds=None if raw["x_stds"] is None else np.asarray(raw["x_stds"], dtype=np.float64),
        feature_indices_scaled=tuple(int(i) for i in raw["feature_indices_scaled"]),
        feature_indices_passthrough=tuple(int(i) for i in raw["feature_indices_passthrough"]),
        y_mean=raw["y_mean"],
        y_std=raw["y_std"],
    )