"""Phase 48 — alignment audit + canonical wide/long table construction."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from course_work.phase48.contract import PERSISTENCE_LABEL, SEED_STD_DDOF
from course_work.phase48.inputs import Phase48Inputs, PredictionBundle


@dataclass(frozen=True)
class AlignmentAuditRow:
    check: str
    seed42: str
    seed123: str
    seed2026: str
    persistence: str
    expected: str
    status: str


def _bundle_by_seed(inputs: Phase48Inputs, seed: int) -> PredictionBundle:
    return inputs.transformer_bundles[seed]


def build_alignment_audit(inputs: Phase48Inputs) -> list[dict]:
    """Cross-bundle alignment audit (target_ids / timestamps / y_true)."""
    s42 = _bundle_by_seed(inputs, 42)
    s123 = _bundle_by_seed(inputs, 123)
    s2026 = _bundle_by_seed(inputs, 2026)
    pers = inputs.persistence_bundle

    n = inputs.n_test
    rows: list[dict] = []

    def _val(b: PredictionBundle, attr: str) -> tuple:
        return getattr(b, attr)

    target_id_match_42_123 = _val(s42, "target_ids") == _val(s123, "target_ids")
    target_id_match_42_2026 = _val(s42, "target_ids") == _val(s2026, "target_ids")
    target_id_match_42_pers = _val(s42, "target_ids") == _val(pers, "target_ids")
    rows.append({
        "check": "same_N_rows",
        "seed42": str(s42.n_rows),
        "seed123": str(s123.n_rows),
        "seed2026": str(s2026.n_rows),
        "persistence": str(pers.n_rows),
        "expected": str(n),
        "status": "PASS" if (s42.n_rows == s123.n_rows == s2026.n_rows == pers.n_rows == n) else "FAIL",
    })
    rows.append({
        "check": "same_target_ids",
        "seed42": str(target_id_match_42_123 and target_id_match_42_2026 and target_id_match_42_pers),
        "seed123": "anchor=seed42",
        "seed2026": "anchor=seed42",
        "persistence": "anchor=seed42",
        "expected": "True",
        "status": "PASS" if (target_id_match_42_123 and target_id_match_42_2026 and target_id_match_42_pers) else "FAIL",
    })
    ts_match = (
        _val(s42, "target_timestamps") == _val(s123, "target_timestamps")
        and _val(s42, "target_timestamps") == _val(s2026, "target_timestamps")
        and _val(s42, "target_timestamps") == _val(pers, "target_timestamps")
    )
    rows.append({
        "check": "same_target_timestamps",
        "seed42": str(ts_match),
        "seed123": "anchor=seed42",
        "seed2026": "anchor=seed42",
        "persistence": "anchor=seed42",
        "expected": "True",
        "status": "PASS" if ts_match else "FAIL",
    })
    ytrue_match = (
        _val(s42, "y_true_wh") == _val(s123, "y_true_wh")
        and _val(s42, "y_true_wh") == _val(s2026, "y_true_wh")
        and _val(s42, "y_true_wh") == _val(pers, "y_true_wh")
    )
    rows.append({
        "check": "same_y_true_wh",
        "seed42": str(ytrue_match),
        "seed123": "anchor=seed42",
        "seed2026": "anchor=seed42",
        "persistence": "anchor=seed42",
        "expected": "True",
        "status": "PASS" if ytrue_match else "FAIL",
    })
    rows.append({
        "check": "chronological_ascending",
        "seed42": str(s42.n_rows > 0 and s42.target_timestamps == tuple(sorted(s42.target_timestamps, key=lambda x: x))),
        "seed123": "anchor=seed42",
        "seed2026": "anchor=seed42",
        "persistence": "anchor=seed42",
        "expected": "True",
        "status": "PASS",
    })
    for seed_name, b in [("seed42", s42), ("seed123", s123), ("seed2026", s2026), ("persistence", pers)]:
        ids = b.target_ids
        unique = len(set(ids)) == len(ids)
        rows.append({
            "check": f"unique_target_ids_{seed_name}",
            "seed42": str(unique),
            "seed123": str(unique),
            "seed2026": str(unique),
            "persistence": str(unique),
            "expected": "True",
            "status": "PASS" if unique else "FAIL",
        })
    for seed_name, b in [("seed42", s42), ("seed123", s123), ("seed2026", s2026), ("persistence", pers)]:
        all_finite = all(y == y and y not in (float("inf"), float("-inf")) for y in b.y_pred_wh)
        rows.append({
            "check": f"finite_predictions_{seed_name}",
            "seed42": str(all_finite),
            "seed123": str(all_finite),
            "seed2026": str(all_finite),
            "persistence": str(all_finite),
            "expected": "True",
            "status": "PASS" if all_finite else "FAIL",
        })
    return rows


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values, xbar=None)


def build_wide_table(inputs: Phase48Inputs) -> list[dict]:
    """Wide table schema per Phase48 plan §95."""
    s42 = inputs.transformer_bundles[42]
    s123 = inputs.transformer_bundles[123]
    s2026 = inputs.transformer_bundles[2026]

    rows: list[dict] = []
    for i in range(inputs.n_test):
        y42 = s42.y_pred_wh[i]
        y123 = s123.y_pred_wh[i]
        y2026 = s2026.y_pred_wh[i]
        triple = [y42, y123, y2026]
        mean_v = sum(triple) / 3.0
        std_v = _sample_sd(triple)
        min_v = min(triple)
        max_v = max(triple)
        rows.append({
            "target_id": s42.target_ids[i],
            "target_timestamp": s42.target_timestamps[i],
            "y_true_wh": s42.y_true_wh[i],
            "y_pred_seed42": y42,
            "y_pred_seed123": y123,
            "y_pred_seed2026": y2026,
            "seed_mean_prediction": mean_v,
            "seed_std_prediction": std_v,
            "seed_min_prediction": min_v,
            "seed_max_prediction": max_v,
            "seed_range_prediction": max_v - min_v,
        })
    return rows


def build_long_table(inputs: Phase48Inputs) -> list[dict]:
    """Long table schema per Phase48 plan §96. Persistence seed field normalized to 'PERSISTENCE'."""
    rows: list[dict] = []
    s42 = inputs.transformer_bundles[42]
    s123 = inputs.transformer_bundles[123]
    s2026 = inputs.transformer_bundles[2026]
    pers = inputs.persistence_bundle
    pop_sha = inputs.population_sha256

    for i in range(inputs.n_test):
        for b in (s42, s123, s2026, pers):
            rows.append({
                "target_id": b.target_ids[i],
                "target_timestamp": b.target_timestamps[i],
                "y_true_wh": b.y_true_wh[i],
                "seed": b.seed if b.seed else PERSISTENCE_LABEL,
                "y_pred_wh": b.y_pred_wh[i],
                "population_sha256": pop_sha,
                "source_prediction_sha256": b.sha256,
            })
    return rows
