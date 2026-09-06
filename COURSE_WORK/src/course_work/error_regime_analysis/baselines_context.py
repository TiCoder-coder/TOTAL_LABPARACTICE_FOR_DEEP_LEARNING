"""Phase 50-F — baselines + upstream context by regime.

- Persistence: compute regime metrics from Phase 47 frozen persistence predictions
- LSTM: read canonical eligibility; emit NOT_APPLICABLE
- Phase 48 seed-spread: join target-level seed-spread to regime assignment
- Phase 49 sign-consensus: derive deterministic per-target consensus from wide table
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from . import contract
from . import sources
from . import error_join


def _classify_consensus_3seeds(s42: float, s123: float, s2026: float) -> str:
    signs = []
    for v in (s42, s123, s2026):
        if v > 0:
            signs.append("+")
        elif v < 0:
            signs.append("-")
        else:
            signs.append("0")
    pos = signs.count("+")
    neg = signs.count("-")
    zero = signs.count("0")
    if zero == 3:
        return "ALL_ZERO"
    if zero > 0 and pos == 0 and neg == 0:
        # some zeros but no sign — handled by zero==3 case
        return "ALL_ZERO"
    if zero > 0:
        # mixed zeros — Phase 49 had 0 ALL_EXACT; remaining zeros mixed with signs → MIXED
        return "MIXED"
    if pos == 3:
        return "ALL_UNDER"
    if neg == 3:
        return "ALL_OVER"
    if pos == 2 and neg == 1:
        return "TWO_UNDER_ONE_OVER"
    if pos == 1 and neg == 2:
        return "TWO_OVER_ONE_UNDER"
    return "MIXED"


def write_persistence_regime_metrics(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else sources.project_root()
    rows, _ = sources.load_phase47_persistence_predictions(root)
    # Build y_true/y_pred maps per target_id (Persistence has only one row per target_id)
    y_true = {r["target_id"]: float(r["y_true_wh"]) for r in rows}
    y_pred = {r["target_id"]: float(r["y_pred_wh"]) for r in rows}

    # Load test regime assignment
    assign = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/test_regime_assignment.csv").open()
        )
    )
    fam_col = {
        "R1_TARGET_LEVEL": "target_level_regime",
        "R2_EXTREME_HIGH": "extreme_high_regime",
        "R3_CHANGE_MAGNITUDE": "change_magnitude_regime",
        "R4_CHANGE_DIRECTION": "change_direction_regime",
        "R5_TIME_OF_DAY": "time_of_day_regime",
        "R6_DAY_TYPE": "day_type_regime",
    }

    # Global Persistence
    gt = [y_true[a["target_id"]] for a in assign]
    gp = [y_pred[a["target_id"]] for a in assign]
    g_metrics = error_join._compute_seed_metrics(gt, gp)

    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_metrics_persistence.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "baseline",
            "regime_family",
            "regime_label",
            "N",
            "mae_wh",
            "rmse_wh",
            "r2",
            "r2_status",
            "mbe_wh",
            "sae",
            "sse",
            "underprediction_fraction",
            "overprediction_fraction",
            "exact_fraction",
            "global_mae_wh",
            "global_rmse_wh",
            "mae_lift_pct",
            "rmse_lift_pct",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        # Global row
        w.writerow(
            {
                "baseline": "PERSISTENCE",
                "regime_family": "GLOBAL",
                "regime_label": "ALL_TEST",
                "N": g_metrics["N"],
                "mae_wh": _fmt(g_metrics["mae_wh"]),
                "rmse_wh": _fmt(g_metrics["rmse_wh"]),
                "r2": _fmt(g_metrics["r2"]) if g_metrics["r2_status"] == "DEFINED" else "",
                "r2_status": g_metrics["r2_status"],
                "mbe_wh": _fmt(g_metrics["mbe_wh"]),
                "sae": _fmt(g_metrics["sae"]),
                "sse": _fmt(g_metrics["sse"]),
                "underprediction_fraction": _fmt(g_metrics["underprediction_fraction"]),
                "overprediction_fraction": _fmt(g_metrics["overprediction_fraction"]),
                "exact_fraction": _fmt(g_metrics["exact_fraction"]),
                "global_mae_wh": _fmt(g_metrics["mae_wh"]),
                "global_rmse_wh": _fmt(g_metrics["rmse_wh"]),
                "mae_lift_pct": "0.0",
                "rmse_lift_pct": "0.0",
                "status": "PASS",
            }
        )
        for fam, col in fam_col.items():
            labels = contract.REGIME_LABELS[fam]
            for label in labels:
                bucket_t = []
                bucket_p = []
                for a in assign:
                    if a[col] == label:
                        bucket_t.append(y_true[a["target_id"]])
                        bucket_p.append(y_pred[a["target_id"]])
                m = error_join._compute_seed_metrics(bucket_t, bucket_p)
                mae_lift = 100 * (m["mae_wh"] - g_metrics["mae_wh"]) / g_metrics["mae_wh"] if g_metrics["mae_wh"] else float("nan")
                rmse_lift = 100 * (m["rmse_wh"] - g_metrics["rmse_wh"]) / g_metrics["rmse_wh"] if g_metrics["rmse_wh"] else float("nan")
                w.writerow(
                    {
                        "baseline": "PERSISTENCE",
                        "regime_family": fam,
                        "regime_label": label,
                        "N": m["N"],
                        "mae_wh": _fmt(m["mae_wh"]),
                        "rmse_wh": _fmt(m["rmse_wh"]),
                        "r2": _fmt(m["r2"]) if m["r2_status"] == "DEFINED" else "",
                        "r2_status": m["r2_status"],
                        "mbe_wh": _fmt(m["mbe_wh"]),
                        "sae": _fmt(m["sae"]),
                        "sse": _fmt(m["sse"]),
                        "underprediction_fraction": _fmt(m["underprediction_fraction"]),
                        "overprediction_fraction": _fmt(m["overprediction_fraction"]),
                        "exact_fraction": _fmt(m["exact_fraction"]),
                        "global_mae_wh": _fmt(g_metrics["mae_wh"]),
                        "global_rmse_wh": _fmt(g_metrics["rmse_wh"]),
                        "mae_lift_pct": _fmt(mae_lift),
                        "rmse_lift_pct": _fmt(rmse_lift),
                        "status": "PASS",
                    }
                )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_lstm_regime_metrics(project_root: Path | None = None) -> str:
    """LSTM is NOT_ELIGIBLE_CONFIG_MISMATCH per Phase 47 canonical status.

    Emit regime_metrics_lstm.csv with one row per regime family showing status.
    """
    root = project_root if project_root is not None else sources.project_root()
    elig = sources.load_phase47_lstm_eligibility(root)
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_metrics_lstm.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "baseline",
                "regime_family",
                "regime_label",
                "N",
                "mae_wh",
                "rmse_wh",
                "r2",
                "mbe_wh",
                "eligibility_status",
                "reason",
                "status",
            ],
        )
        w.writeheader()
        for fam in contract.REGIME_FAMILIES:
            for label in contract.REGIME_LABELS[fam]:
                w.writerow(
                    {
                        "baseline": "LSTM_TUNED_DEV",
                        "regime_family": fam,
                        "regime_label": label,
                        "N": 0,
                        "mae_wh": "",
                        "rmse_wh": "",
                        "r2": "",
                        "mbe_wh": "",
                        "eligibility_status": elig["eligibility_status"],
                        "reason": elig["reason"],
                        "status": "NOT_APPLICABLE",
                    }
                )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_seed_spread_by_regime(project_root: Path | None = None) -> str:
    """regime_seed_spread.csv: cross-seed prediction spread (SD) per regime per target.

    Semantics: CROSS_SEED_PREDICTION_SPREAD, NOT confidence interval.
    """
    root = project_root if project_root is not None else sources.project_root()
    spread_rows, _ = sources.load_phase48_seed_spread(root)
    assign = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/test_regime_assignment.csv").open()
        )
    )
    assign_by_id = {a["target_id"]: a for a in assign}

    fam_col = {
        "R1_TARGET_LEVEL": "target_level_regime",
        "R2_EXTREME_HIGH": "extreme_high_regime",
        "R3_CHANGE_MAGNITUDE": "change_magnitude_regime",
        "R4_CHANGE_DIRECTION": "change_direction_regime",
        "R5_TIME_OF_DAY": "time_of_day_regime",
        "R6_DAY_TYPE": "day_type_regime",
    }

    # Per-regime per-target spread (we report per-target then aggregate)
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_seed_spread.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "N",
            "spread_mean_wh",
            "spread_std_wh",
            "spread_min_wh",
            "spread_max_wh",
            "spread_median_wh",
            "spread_p90_wh",
            "spread_semantics",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for fam, col in fam_col.items():
            for label in contract.REGIME_LABELS[fam]:
                vals = []
                for r in spread_rows:
                    a = assign_by_id.get(r["target_id"])
                    if a is None or a[col] != label:
                        continue
                    vals.append(float(r["seed_std_prediction"]))
                if vals:
                    vals_sorted = sorted(vals)
                    n = len(vals)
                    mean = sum(vals) / n
                    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n >= 2 else 0
                    sd = math.sqrt(var)
                    median = vals_sorted[n // 2] if n % 2 == 1 else (
                        vals_sorted[n // 2 - 1] + vals_sorted[n // 2]
                    ) / 2
                    p90_idx = max(0, min(n - 1, int(0.9 * n)))
                    w.writerow(
                        {
                            "regime_family": fam,
                            "regime_label": label,
                            "N": n,
                            "spread_mean_wh": _fmt(mean),
                            "spread_std_wh": _fmt(sd),
                            "spread_min_wh": _fmt(min(vals)),
                            "spread_max_wh": _fmt(max(vals)),
                            "spread_median_wh": _fmt(median),
                            "spread_p90_wh": _fmt(vals_sorted[p90_idx]),
                            "spread_semantics": "CROSS_SEED_PREDICTION_SPREAD",
                            "status": "PASS",
                        }
                    )
                else:
                    w.writerow(
                        {
                            "regime_family": fam,
                            "regime_label": label,
                            "N": 0,
                            "spread_mean_wh": "",
                            "spread_std_wh": "",
                            "spread_min_wh": "",
                            "spread_max_wh": "",
                            "spread_median_wh": "",
                            "spread_p90_wh": "",
                            "spread_semantics": "CROSS_SEED_PREDICTION_SPREAD",
                            "status": "EMPTY_REGIME",
                        }
                    )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_sign_consensus_by_regime(project_root: Path | None = None) -> str:
    """regime_sign_consensus.csv: per-target deterministic consensus_class joined to regime.

    Mapping frozen BEFORE aggregation:
      ALL_UNDER                -> ALL_UNDER
      ALL_OVER                 -> ALL_OVER
      ALL_EXACT                -> ALL_ZERO
      TWO_UNDER_ONE_OVER       -> MIXED
      TWO_OVER_ONE_UNDER       -> MIXED
      MIXED                    -> MIXED
    """
    root = project_root if project_root is not None else sources.project_root()
    wide, _ = sources.load_phase49_residual_wide(root)
    assign = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/test_regime_assignment.csv").open()
        )
    )
    assign_by_id = {a["target_id"]: a for a in assign}

    # Compute per-target consensus_class deterministically
    consensus: dict[str, str] = {}
    for r in wide:
        tid = r["target_id"]
        s42 = float(r["seed42_residual_wh"])
        s123 = float(r["seed123_residual_wh"])
        s2026 = float(r["seed2026_residual_wh"])
        consensus[tid] = _classify_consensus_3seeds(s42, s123, s2026)

    fam_col = {
        "R1_TARGET_LEVEL": "target_level_regime",
        "R2_EXTREME_HIGH": "extreme_high_regime",
        "R3_CHANGE_MAGNITUDE": "change_magnitude_regime",
        "R4_CHANGE_DIRECTION": "change_direction_regime",
        "R5_TIME_OF_DAY": "time_of_day_regime",
        "R6_DAY_TYPE": "day_type_regime",
    }

    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_sign_consensus.csv"
    # Verify mapping counts against Phase 49 aggregated counts
    canonical_counts = {"ALL_UNDER": 1003, "ALL_OVER": 1085, "TWO_UNDER_ONE_OVER": 423, "TWO_OVER_ONE_UNDER": 450}
    from collections import Counter
    actual = Counter(consensus.values())
    mapping = {
        "ALL_UNDER": "ALL_UNDER",
        "ALL_OVER": "ALL_OVER",
        "TWO_UNDER_ONE_OVER": "MIXED",
        "TWO_OVER_ONE_UNDER": "MIXED",
        "MIXED": "MIXED",
        "ALL_ZERO": "ALL_ZERO",
    }
    # Document the mapping
    mapping_doc_path = out_dir / "regime_sign_consensus_mapping.json"
    mapping_doc = {
        "source_phase": 49,
        "source_artifact": "artifacts/residual_analysis/residual_wide_table.csv",
        "canonical_aggregated_artifact": "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv",
        "per_target_method": "deterministic from wide_table seed42/123/2026_residual_wh signs",
        "mapping_to_phase50_target_categories": mapping,
        "source_class_to_phase50_target_category": {
            "ALL_UNDER": "ALL_UNDER",
            "ALL_OVER": "ALL_OVER",
            "TWO_UNDER_ONE_OVER": "MIXED",
            "TWO_OVER_ONE_UNDER": "MIXED",
            "MIXED": "MIXED",
            "ALL_EXACT": "ALL_ZERO",
            "ALL_ZERO": "ALL_ZERO",
        },
    }
    mapping_doc_path.write_text(
        json.dumps(mapping_doc, indent=2, sort_keys=True), encoding="utf-8"
    )
    mapping_sha = hashlib.sha256(mapping_doc_path.read_bytes()).hexdigest()

    # Sanity check: per-target counts should agree with canonical aggregated counts
    for k, exp in canonical_counts.items():
        assert actual.get(k, 0) == exp, f"per-target count for {k}: expected {exp}, got {actual.get(k, 0)}"

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "consensus_source_class",
            "consensus_phase50_category",
            "N",
            "fraction_within_regime",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for fam, col in fam_col.items():
            for label in contract.REGIME_LABELS[fam]:
                rows_for_label = [
                    consensus[a["target_id"]]
                    for a in assign
                    if a[col] == label
                ]
                n = len(rows_for_label)
                cnt = Counter(rows_for_label)
                for src_class, phase50_cat in mapping.items():
                    cn = cnt.get(src_class, 0)
                    w.writerow(
                        {
                            "regime_family": fam,
                            "regime_label": label,
                            "consensus_source_class": src_class,
                            "consensus_phase50_category": phase50_cat,
                            "N": cn,
                            "fraction_within_regime": _fmt(cn / n) if n else "0.0",
                            "status": "PASS",
                        }
                    )
    csv_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return csv_sha, mapping_sha


def write_phase48_seed_spread_status(project_root: Path | None = None) -> str:
    """regime_seed_spread_status.json: documents the semantic invariant."""
    root = project_root if project_root is not None else sources.project_root()
    doc = {
        "version": "REGIME_SEED_SPREAD_STATUS-v1",
        "spread_semantics": "CROSS_SEED_PREDICTION_SPREAD",
        "is_confidence_interval": False,
        "source_artifact": "artifacts/prediction_analysis/prediction_seed_spread.csv",
        "method": "deterministic join on target_id to frozen test_regime_assignment",
        "status": "PASS",
    }
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_seed_spread_status.json"
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_lstm_status(project_root: Path | None = None) -> str:
    """regime_lstm_status.json: documents LSTM NOT_APPLICABLE."""
    root = project_root if project_root is not None else sources.project_root()
    elig = sources.load_phase47_lstm_eligibility(root)
    doc = {
        "version": "REGIME_LSTM_STATUS-v1",
        "eligibility_status": elig["eligibility_status"],
        "reason": elig["reason"],
        "source_artifact": "artifacts/final_test/final_test_lstm_eligibility.json",
        "phase50_status": "NOT_APPLICABLE",
        "no_inference_performed": True,
        "no_retrain_performed": True,
        "no_fabricated_residuals": True,
    }
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_lstm_status.json"
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return repr(x)
    return str(x)


def materialize_phase50_f(project_root: Path | None = None) -> dict:
    root = project_root if project_root is not None else sources.project_root()
    persist_sha = write_persistence_regime_metrics(root)
    lstm_sha = write_lstm_regime_metrics(root)
    spread_sha = write_seed_spread_by_regime(root)
    sign_sha, mapping_sha = write_sign_consensus_by_regime(root)
    spread_status_sha = write_phase48_seed_spread_status(root)
    lstm_status_sha = write_lstm_status(root)
    return {
        "subphase": "50-F",
        "status": "PASS",
        "regime_metrics_persistence_sha256": persist_sha,
        "regime_metrics_lstm_sha256": lstm_sha,
        "regime_seed_spread_sha256": spread_sha,
        "regime_seed_spread_status_sha256": spread_status_sha,
        "regime_sign_consensus_sha256": sign_sha,
        "regime_sign_consensus_mapping_sha256": mapping_sha,
        "regime_lstm_status_sha256": lstm_status_sha,
    }
