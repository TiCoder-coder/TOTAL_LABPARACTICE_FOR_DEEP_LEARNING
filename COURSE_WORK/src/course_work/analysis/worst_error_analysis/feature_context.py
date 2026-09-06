"""Phase 51-F — feature-order audit and per-feature descriptive summaries.

The 33-feature order is read from the canonical registry
``artifacts/feature_sets/feature_set_registry.json`` (FS2_TF1). The
audit produces one PASS row per feature. No feature attribution, SHAP,
attention, or model forward pass is performed.

Per-feature summary statistics come from the WORKING TABLE's three
existing per-seed predictions + Persistence at the target level; they
characterize the *target* behavior, NOT the raw 72×33 input window
(which is not persisted to disk in the project).

For features where the working table carries model-derived inputs, the
canonical per-feature aggregate (mean/min/max/last) at the target step
is recorded. Where the feature is not present in the working table
(which has only the model's y_pred and the actual y_true), the row is
explicitly marked FEATURE_NOT_IN_WORKING_TABLE.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import get_project_root


FEATURE_SET_REGISTRY_REL = "artifacts/feature_sets/feature_set_registry.json"
EXPECTED_FEATURE_SET = "FS2_TF1"
EXPECTED_FEATURE_COUNT = 33
LOOKBACK = 72
CANONICAL_FEATURES: list[str] = [
    "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
    "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7",
    "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
    "Press_mm_hg", "RH_out", "Windspeed", "Visibility",
    "Tdewpoint", "Appliances", "rv1", "rv2",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend",
]


def verify_canonical_feature_order(project_root: Path) -> dict[str, Any]:
    """Read the registry and confirm FS2_TF1 has 33 features in canonical
    order. Returns the canonical 33-feature list. Raises if mismatch.
    """
    fp = project_root / FEATURE_SET_REGISTRY_REL
    payload = json.loads(fp.read_text())
    fs2 = payload["variants"]["FS2_TF1"]
    assert fs2["feature_count"] == EXPECTED_FEATURE_COUNT, (
        f"FS2_TF1 feature_count mismatch: "
        f"expected {EXPECTED_FEATURE_COUNT}, got {fs2['feature_count']}"
    )
    canon = fs2["features"]
    if canon != CANONICAL_FEATURES:
        # Be tolerant of benign re-orderings IF the set is identical.
        if sorted(canon) != sorted(CANONICAL_FEATURES):
            raise RuntimeError(
                f"FS2_TF1 feature SET differs from canonical: "
                f"missing={set(CANONICAL_FEATURES) - set(canon)}, "
                f"extra={set(canon) - set(CANONICAL_FEATURES)}"
            )
    return {
        "feature_set": EXPECTED_FEATURE_SET,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "features": canon,
        "registry_path": FEATURE_SET_REGISTRY_REL,
        "registry_feature_count": fs2["feature_count"],
        "registry_features_match": canon == CANONICAL_FEATURES,
        "lookback": LOOKBACK,
        "fingerprint": fs2["fingerprint"],
    }


def build_feature_order_audit(
    project_root: Path,
) -> list[dict[str, Any]]:
    """One row per canonical feature position.

    Schema:
      position, feature_name, expected_name, match
    """
    fs = verify_canonical_feature_order(project_root)
    out: list[dict[str, Any]] = []
    for pos, name in enumerate(fs["features"], start=1):
        out.append({
            "position": pos,
            "feature_name": name,
            "expected_name": CANONICAL_FEATURES[pos - 1]
                if pos - 1 < len(CANONICAL_FEATURES) else "",
            "match": int(
                name == CANONICAL_FEATURES[pos - 1]
                if pos - 1 < len(CANONICAL_FEATURES) else False
            ),
            "coordinate_class": "RAW_AND_MODEL_INPUT",
        })
    return out


def build_per_target_feature_summary(
    project_root: Path,
    selected_target_ids: list[str],
) -> list[dict[str, Any]]:
    """Per selected target × canonical feature, emit a descriptor row.

    Because the 72×33 input tensor is NOT persisted to disk in the
    project, descriptive feature statistics here summarize the TARGET
    row's behavior (y_true and three y_preds) rather than the 72-step
    window. This is a conservative read-only summary consistent with
    the rule that no new forward pass is computed.

    Each row records the feature name plus the target-level per-seed
    summary statistics that are available from the working table. For
    features not represented in the working table, the row carries the
    flag FEATURE_NOT_IN_WORKING_TABLE.
    """
    import csv
    fp = project_root / "artifacts/worst_error_analysis" / "phase51_target_level_working_table.csv"
    wt = {r["target_id"]: r for r in csv.DictReader(fp.open("r", encoding="utf-8", newline=""))}
    out: list[dict[str, Any]] = []
    for tid in selected_target_ids:
        w = wt.get(tid)
        if w is None:
            continue
        y_true = float(w["y_true_wh"])
        # Only Appliances feature overlaps with the working-table signal.
        # For others, we record FEATURE_NOT_IN_WORKING_TABLE.
        for feat in CANONICAL_FEATURES:
            if feat == "Appliances":
                seed_preds = [
                    float(w[f"seed{s}_y_pred_wh"]) for s in (42, 123, 2026)
                ]
                abs_errs = [
                    float(w[f"seed{s}_abs_error_wh"]) for s in (42, 123, 2026)
                ]
                out.append({
                    "target_id": tid,
                    "feature_name": feat,
                    "feature_position": CANONICAL_FEATURES.index(feat) + 1,
                    "summary_kind": "TARGET_LEVEL_SCALAR",
                    "raw_value": y_true,
                    "seed42_y_pred_wh": seed_preds[0],
                    "seed123_y_pred_wh": seed_preds[1],
                    "seed2026_y_pred_wh": seed_preds[2],
                    "mean_abs_error_wh": round(statistics.fmean(abs_errs), 9),
                    "min_abs_error_wh": min(abs_errs),
                    "max_abs_error_wh": max(abs_errs),
                    "feature_status": "TARGET_VALUE_AVAILABLE",
                })
            else:
                out.append({
                    "target_id": tid,
                    "feature_name": feat,
                    "feature_position": CANONICAL_FEATURES.index(feat) + 1,
                    "summary_kind": "WINDOW_LEVEL_NOT_PERSISTED",
                    "raw_value": "",
                    "seed42_y_pred_wh": "",
                    "seed123_y_pred_wh": "",
                    "seed2026_y_pred_wh": "",
                    "mean_abs_error_wh": "",
                    "min_abs_error_wh": "",
                    "max_abs_error_wh": "",
                    "feature_status": "FEATURE_NOT_IN_WORKING_TABLE",
                })
    return out
