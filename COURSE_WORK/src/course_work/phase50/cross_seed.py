"""Phase 50-E — cross-seed aggregation + predeclared contrasts + rank stability + Train-vs-Test prevalence.

All cross-seed statistics use sample SD with ddof=1.
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


SEEDS = contract.SEEDS


def _sample_std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def build_cross_seed_summary(project_root: Path | None = None) -> dict:
    """Aggregate regime metrics across 3 seeds using mean + sample SD (ddof=1).

    Returns dict:
        {
          "(family, label, metric)": {
              "mean": float,
              "sd": float,  # sample SD with ddof=1
              "per_seed": {"42": v, "123": v, "2026": v}
          },
          ...
        }
    """
    root = project_root if project_root is not None else sources.project_root()
    rows = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/regime_metrics_long.csv").open()
        )
    )
    by_key: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        key = (r["regime_family"], r["regime_label"])
        seed = r["seed"]
        by_key[key][seed]["N"] = int(r["N"])
        for m in (
            "mae_wh",
            "rmse_wh",
            "r2",
            "mbe_wh",
            "sae_share",
            "sse_share",
            "sae_disproportion",
            "sse_disproportion",
            "mae_lift_pct",
            "rmse_lift_pct",
            "rmse_lift_wh",
            "rmse_lift_ratio",
        ):
            v = r[m]
            by_key[key][seed][m] = float(v) if v != "" else float("nan")

    summary: dict = {}
    for key, seeds in by_key.items():
        per_metric = {}
        for metric in (
            "mae_wh",
            "rmse_wh",
            "r2",
            "mbe_wh",
            "sae_share",
            "sse_share",
            "rmse_lift_pct",
            "rmse_lift_wh",
            "rmse_lift_ratio",
        ):
            per_seed_vals = [seeds[s][metric] for s in SEEDS]
            per_metric[metric] = {
                "mean": sum(per_seed_vals) / len(per_seed_vals),
                "sd": _sample_std(per_seed_vals),
                "per_seed": dict(zip(SEEDS, per_seed_vals)),
            }
        summary[key] = per_metric
    return summary


def write_cross_seed_summary(
    summary: dict, project_root: Path | None = None
) -> str:
    root = project_root if project_root is not None else sources.project_root()
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_cross_seed_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "metric",
            "seed_42_value",
            "seed_123_value",
            "seed_2026_value",
            "mean",
            "sd_ddof1",
            "n_seeds",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (fam, label), metrics in summary.items():
            for metric, vals in metrics.items():
                w.writerow(
                    {
                        "regime_family": fam,
                        "regime_label": label,
                        "metric": metric,
                        "seed_42_value": _fmt(vals["per_seed"]["42"]),
                        "seed_123_value": _fmt(vals["per_seed"]["123"]),
                        "seed_2026_value": _fmt(vals["per_seed"]["2026"]),
                        "mean": _fmt(vals["mean"]),
                        "sd_ddof1": _fmt(vals["sd"]),
                        "n_seeds": len(SEEDS),
                        "status": "PASS",
                    }
                )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_rmse_lift_table(
    summary: dict, project_root: Path | None = None
) -> str:
    """regime_rmse_lift.csv: per (regime_family, regime_label, seed) with all 3 lift forms."""
    root = project_root if project_root is not None else sources.project_root()
    rows = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/regime_metrics_long.csv").open()
        )
    )
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_rmse_lift.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "seed",
            "regime_rmse_wh",
            "global_seed_rmse_wh",
            "rmse_lift_wh",
            "rmse_lift_ratio",
            "rmse_lift_pct",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rmse = r["rmse_wh"]
            g_rmse = r["global_rmse_wh"]
            if rmse == "" or g_rmse == "":
                continue
            w.writerow(
                {
                    "regime_family": r["regime_family"],
                    "regime_label": r["regime_label"],
                    "seed": r["seed"],
                    "regime_rmse_wh": rmse,
                    "global_seed_rmse_wh": g_rmse,
                    "rmse_lift_wh": r["rmse_lift_wh"],
                    "rmse_lift_ratio": r["rmse_lift_ratio"],
                    "rmse_lift_pct": r["rmse_lift_pct"],
                    "status": "PASS",
                }
            )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_predeclared_contrasts(
    summary: dict, project_root: Path | None = None
) -> str:
    """regime_pairwise_contrasts.csv: predeclared contrasts only."""
    root = project_root if project_root is not None else sources.project_root()
    contrasts = [
        ("R1_TARGET_LEVEL", "TL_HIGH", "TL_LOW", "TL_HIGH vs TL_LOW"),
        ("R1_TARGET_LEVEL", "TL_HIGH", "TL_MID", "TL_HIGH vs TL_MID"),
        ("R2_EXTREME_HIGH", "EXTREME_HIGH", "NON_EXTREME", "EXTREME_HIGH vs NON_EXTREME"),
        ("R3_CHANGE_MAGNITUDE", "CHANGE_RAPID", "CHANGE_NORMAL", "CHANGE_RAPID vs CHANGE_NORMAL"),
        ("R4_CHANGE_DIRECTION", "DIR_UP", "DIR_DOWN", "DIR_UP vs DIR_DOWN"),
        ("R6_DAY_TYPE", "DAY_WEEKEND", "DAY_WEEKDAY", "DAY_WEEKEND vs DAY_WEEKDAY"),
    ]
    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_pairwise_contrasts.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "contrast_id",
            "seed",
            "regime_a",
            "regime_b",
            "rmse_a",
            "rmse_b",
            "delta_rmse_a_minus_b",
            "mae_a",
            "mae_b",
            "delta_mae_a_minus_b",
            "mbe_a",
            "mbe_b",
            "delta_mbe_a_minus_b",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        rows_long = list(
            csv.DictReader(
                (root / "artifacts/error_by_regime/regime_metrics_long.csv").open()
            )
        )
        per_seed_lookup: dict = defaultdict(dict)
        for r in rows_long:
            key = (r["regime_family"], r["regime_label"], r["seed"])
            per_seed_lookup[key] = r

        for fam, la, lb, contrast_id in contrasts:
            for seed in SEEDS:
                a = per_seed_lookup.get((fam, la, seed))
                b = per_seed_lookup.get((fam, lb, seed))
                if a is None or b is None:
                    continue
                if a["rmse_wh"] == "" or b["rmse_wh"] == "":
                    continue
                w.writerow(
                    {
                        "regime_family": fam,
                        "contrast_id": contrast_id,
                        "seed": seed,
                        "regime_a": la,
                        "regime_b": lb,
                        "rmse_a": a["rmse_wh"],
                        "rmse_b": b["rmse_wh"],
                        "delta_rmse_a_minus_b": f"{float(a['rmse_wh']) - float(b['rmse_wh']):.6f}",
                        "mae_a": a["mae_wh"],
                        "mae_b": b["mae_wh"],
                        "delta_mae_a_minus_b": f"{float(a['mae_wh']) - float(b['mae_wh']):.6f}",
                        "mbe_a": a["mbe_wh"],
                        "mbe_b": b["mbe_wh"],
                        "delta_mbe_a_minus_b": f"{float(a['mbe_wh']) - float(b['mbe_wh']):.6f}",
                        "status": "PASS",
                    }
                )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_train_vs_test_prevalence(project_root: Path | None = None) -> str:
    """regime_train_vs_test_prevalence.csv: per (regime_family, regime_label) N and fraction."""
    root = project_root if project_root is not None else sources.project_root()
    train_assign = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/train_regime_assignment.csv").open()
        )
    )
    test_assign = list(
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

    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_train_vs_test_prevalence.csv"
    train_total = len(train_assign)
    test_total = len(test_assign)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "n_train",
            "fraction_train",
            "n_test",
            "fraction_test",
            "prevalence_delta_test_minus_train",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for fam, col in fam_col.items():
            labels = contract.REGIME_LABELS[fam]
            for label in labels:
                n_tr = sum(1 for r in train_assign if r[col] == label)
                n_te = sum(1 for r in test_assign if r[col] == label)
                frac_tr = n_tr / train_total if train_total else 0
                frac_te = n_te / test_total if test_total else 0
                w.writerow(
                    {
                        "regime_family": fam,
                        "regime_label": label,
                        "n_train": n_tr,
                        "fraction_train": _fmt(frac_tr),
                        "n_test": n_te,
                        "fraction_test": _fmt(frac_te),
                        "prevalence_delta_test_minus_train": _fmt(frac_te - frac_tr),
                        "status": "PASS",
                    }
                )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def write_rank_stability(project_root: Path | None = None) -> str:
    """regime_rank_stability.csv: rank of (family, label) by RMSE per seed + Spearman across seeds."""
    root = project_root if project_root is not None else sources.project_root()
    rows = list(
        csv.DictReader(
            (root / "artifacts/error_by_regime/regime_metrics_long.csv").open()
        )
    )
    # Build RMSE per (family, label, seed)
    rmse: dict = defaultdict(dict)
    for r in rows:
        if r["rmse_wh"] == "":
            continue
        key = (r["regime_family"], r["regime_label"])
        rmse[key][r["seed"]] = float(r["rmse_wh"])

    out_dir = root / "artifacts" / "error_by_regime"
    out_path = out_dir / "regime_rank_stability.csv"

    def _rank(d: dict) -> dict:
        items = sorted(d.items(), key=lambda kv: kv[1])
        return {k: i + 1 for i, (k, _) in enumerate(items)}

    def _spearman(r1: dict, r2: dict) -> float:
        keys = sorted(set(r1) | set(r2))
        if len(keys) < 2:
            return float("nan")
        d1 = [r1.get(k, float("nan")) for k in keys]
        d2 = [r2.get(k, float("nan")) for k in keys]
        # Spearman on ranks
        from itertools import groupby
        # rank with average ties
        def _avg_rank(xs):
            sorted_x = sorted(enumerate(xs), key=lambda t: (t[1], t[0]))
            ranks = [0.0] * len(xs)
            i = 0
            while i < len(sorted_x):
                j = i
                while j + 1 < len(sorted_x) and sorted_x[j + 1][1] == sorted_x[i][1]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    ranks[sorted_x[k][0]] = avg
                i = j + 1
            return ranks
        ra = _avg_rank(d1)
        rb = _avg_rank(d2)
        # Pearson on ranks
        ma = sum(ra) / len(ra)
        mb = sum(rb) / len(rb)
        num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
        den_a = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(len(ra))))
        den_b = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(len(rb))))
        if den_a == 0 or den_b == 0:
            return float("nan")
        return num / (den_a * den_b)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "regime_family",
            "regime_label",
            "rmse_seed_42",
            "rmse_seed_123",
            "rmse_seed_2026",
            "rank_seed_42",
            "rank_seed_123",
            "rank_seed_2026",
            "spearman_42_vs_123",
            "spearman_42_vs_2026",
            "spearman_123_vs_2026",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        ranks_42 = _rank({k: v.get("42", float("inf")) for k, v in rmse.items()})
        ranks_123 = _rank({k: v.get("123", float("inf")) for k, v in rmse.items()})
        ranks_2026 = _rank({k: v.get("2026", float("inf")) for k, v in rmse.items()})
        sp_42_123 = _spearman(ranks_42, ranks_123)
        sp_42_2026 = _spearman(ranks_42, ranks_2026)
        sp_123_2026 = _spearman(ranks_123, ranks_2026)
        for (fam, label), seed_vals in rmse.items():
            w.writerow(
                {
                    "regime_family": fam,
                    "regime_label": label,
                    "rmse_seed_42": _fmt(seed_vals.get("42")),
                    "rmse_seed_123": _fmt(seed_vals.get("123")),
                    "rmse_seed_2026": _fmt(seed_vals.get("2026")),
                    "rank_seed_42": ranks_42.get((fam, label)),
                    "rank_seed_123": ranks_123.get((fam, label)),
                    "rank_seed_2026": ranks_2026.get((fam, label)),
                    "spearman_42_vs_123": _fmt(sp_42_123),
                    "spearman_42_vs_2026": _fmt(sp_42_2026),
                    "spearman_123_vs_2026": _fmt(sp_123_2026),
                    "status": "PASS",
                }
            )
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return repr(x)  # full precision; CSV readers parse as float
    return str(x)


def materialize_phase50_e(project_root: Path | None = None) -> dict:
    root = project_root if project_root is not None else sources.project_root()
    summary = build_cross_seed_summary(root)
    cs_sha = write_cross_seed_summary(summary, root)
    lift_sha = write_rmse_lift_table(summary, root)
    con_sha = write_predeclared_contrasts(summary, root)
    prev_sha = write_train_vs_test_prevalence(root)
    rank_sha = write_rank_stability(root)
    return {
        "subphase": "50-E",
        "status": "PASS",
        "n_keys": len(summary),
        "regime_cross_seed_summary_sha256": cs_sha,
        "regime_rmse_lift_sha256": lift_sha,
        "regime_pairwise_contrasts_sha256": con_sha,
        "regime_train_vs_test_prevalence_sha256": prev_sha,
        "regime_rank_stability_sha256": rank_sha,
    }
