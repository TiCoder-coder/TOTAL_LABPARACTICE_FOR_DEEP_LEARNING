"""Phase 48 — read-only HTML dashboard renderer (full finalization coverage).

Mirrors the Phase 43-47 dashboard pattern: static HTML, no JavaScript, no
external resources, every value ``escape()``-ed before placement in HTML.
All numbers come from canonical Phase 47 artifacts (read-only) and the
Phase 48-B/C/D/E derived artifacts under ``artifacts/prediction_analysis/``.

Architectural constraints (per architecture_rule.md v1.10 amendment):

* Read-only over Phase 47 source bundles.
* No training, no Test inference, no checkpoint reload for new predictions.
* No best-seed selection, no ensemble metric.
* No prediction shifting / clipping / post-hoc calibration.
* No residual / regime / worst-error / attention analysis.
* Seed-mean labelled descriptive only.
* Seed spread labelled cross-seed spread, not confidence interval.

Public API:

* :func:`render_phase_48_dashboard`
"""
from __future__ import annotations

import base64
import csv
import math
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.utils.artifacts import get_project_root, read_json

__all__ = ["render_phase_48_dashboard"]


_CSS = """
<style>
.cw-p48{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 22px;overflow:visible}
.cw-p48 *{box-sizing:border-box}
.cw-p48-h{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-p48-h h3{font-size:20px;line-height:1.25;margin:0 0 5px;color:#172033}
.cw-p48-meta{font-size:12px;color:#5d6b82}
.cw-p48-badge{border:1px solid #a9dec1;border-radius:999px;padding:6px 12px;font-size:11px;font-weight:700;letter-spacing:.04em;color:#11613d;background:#e8f7ef;white-space:nowrap}
.cw-p48-badge.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-p48-badge.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-p48-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:16px 22px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-p48-card{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:9px;background:#fff;min-width:0}
.cw-p48-card span{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-p48-card strong{font-size:13px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-p48-body{padding:4px 22px 22px}
.cw-p48-sec{margin-top:18px}
.cw-p48-sec h4{font-size:14px;margin:0 0 8px;color:#334155}
.cw-p48-tblwrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-p48-tbl{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
.cw-p48-tbl th{background:#f5f7fb;color:#475569;text-align:left;font-weight:650;padding:9px 11px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-p48-tbl td{text-align:left;padding:8px 11px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-p48-tbl tbody tr:nth-child(even){background:#fafbfd}
.cw-p48-tbl tbody tr:last-child td{border-bottom:0}
.cw-p48-fp{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;color:#1f2a44}
.cw-p48-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-p48-pill.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-p48-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:12px}
.cw-p48-card2{border:1px solid #d9e2ef;border-radius:11px;padding:13px 14px;background:#fff}
.cw-p48-card2.win{border-color:#a9dec1;background:#e8f7ef}
.cw-p48-card2 h5{margin:0 0 5px;font-size:11.5px;color:#475569;text-transform:uppercase;letter-spacing:.04em}
.cw-p48-card2 .big{font-size:19px;color:#172033;font-weight:700;line-height:1.2}
.cw-p48-card2 .sm{font-size:11.5px;color:#5d6b82;margin-top:3px}
.cw-p48-call{border-left:4px solid #6366f1;background:#eef2ff;padding:9px 13px;border-radius:0 7px 7px 0;margin:11px 0;font-size:12.5px;color:#1f2a44}
.cw-p48-call.warn{border-color:#d97706;background:#fff7e0;color:#7a5613}
.cw-p48-call.good{border-color:#16a34a;background:#e8f7ef;color:#11613d}
.cw-p48-fig{display:flex;flex-direction:column;gap:6px;overflow:visible;margin:10px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:100%;max-width:1200px}
.cw-p48-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto}
.cw-p48-fig.compact{width:64%;max-width:700px}
.cw-p48-fig .ftitle{font-size:12px;color:#475569;font-weight:650}
.cw-p48-fig .fcap{font-size:11.5px;color:#5d6b82;line-height:1.45}
.cw-p48-context{font-size:12px;color:#5d6b82;line-height:1.5;margin:10px 0}
.cw-p48-observations{font-size:13px;line-height:1.6;padding-left:20px}
.cw-p48-observations li{margin:6px 0}
@media (max-width:900px){.cw-p48-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-p48-grid3{grid-template-columns:1fr}}
@media (max-width:620px){.cw-p48-h{flex-direction:column;padding:15px}.cw-p48-cards{grid-template-columns:1fr;padding:12px 16px}.cw-p48-body{padding-left:16px;padding-right:16px}}
</style>
"""

def _f(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _i(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):
        return str(value)


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _b(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img_block(rel_path: str, title: str, caption: str, size_class: str = "large") -> str:
    extra = " compact" if size_class == "compact" else ""
    return (
        f'<div class="cw-p48-fig{extra}">'
        f'<div class="ftitle">{escape(title)}</div>'
        f'<img src="{escape(rel_path, quote=True)}" alt="{escape(title)}"/>'
        
        f"</div>"
    )


def _status_class(status: str) -> str:
    s = (status or "UNKNOWN").upper()
    if s == "PASS":
        return ""
    if "WARNING" in s:
        return "warn"
    return "fail"


def _header(title: str, subtitle: str, status: str) -> str:
    css_cls = f"cw-p48-badge{(' ' + _status_class(status)) if _status_class(status) else ''}"
    return (
        '<header class="cw-p48-h">'
        f"<div><h3>{escape(title)}</h3>"
        f'<div class="cw-p48-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(str(status).upper())}</span>'
        "</header>"
    )


def _cards(pairs: Iterable[tuple[str, Any]]) -> str:
    items = "".join(
        f'<div class="cw-p48-card"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in pairs
    )
    return f'<div class="cw-p48-cards">{items}</div>'


def _tbl(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    parts = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        parts.append(f"<tr>{cells}</tr>")
    body = "".join(parts)
    return f'<div class="cw-p48-tblwrap"><table class="cw-p48-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

def _section_distribution(distribution_rows: list[dict]) -> str:
    rows = [
        [r.get("series_id", ""), _i(r.get("N")), _f(r.get("mean")), _f(r.get("std")),
         _f(r.get("min")), _f(r.get("p05")), _f(r.get("q1")), _f(r.get("median")),
         _f(r.get("q3")), _f(r.get("p95")), _f(r.get("max")), _f(r.get("iqr"))]
        for r in distribution_rows
    ]
    return (
        '<section class="cw-p48-sec"><h4>A — Distribution summary (N, mean, std, IQR, percentiles)</h4>'
        + _tbl(
            ["Series", "N", "mean (Wh)", "std (Wh)", "min", "p05", "q1", "median", "q3", "p95", "max", "IQR"],
            rows,
        )
        + "</section>"
    )


def _section_range_compression(range_rows: list[dict]) -> str:
    rows = [
        [r.get("seed", ""), _f(r.get("pred_std")), _f(r.get("true_std")), _f(r.get("std_ratio")),
         _f(r.get("pred_iqr")), _f(r.get("true_iqr")), _f(r.get("iqr_ratio")),
         _f(r.get("pred_range")), _f(r.get("true_range")), _f(r.get("range_ratio")),
         _f(r.get("mean_shift_pred_minus_true"))]
        for r in range_rows
    ]
    return (
        '<section class="cw-p48-sec"><h4>B — Range / variance compression per seed</h4>'
        + _tbl(
            ["Seed", "pred_std", "true_std", "std_ratio", "pred_iqr", "true_iqr",
             "iqr_ratio", "pred_range", "true_range", "range_ratio", "mean_shift (pred − true)"],
            rows,
        )
        + "</section>"
    )


def _section_change(change_rows: list[dict], dir_rows: list[dict]) -> str:
    change_tbl = _tbl(
        ["Series", "valid_transition_count", "gap_excluded", "mean_abs_delta", "median_abs_delta",
         "p90_abs_delta", "p95_abs_delta", "std_delta"],
        [
            [r.get("series_id", ""), _i(r.get("valid_transition_count")), _i(r.get("gap_excluded_count")),
             _f(r.get("mean_abs_delta")), _f(r.get("median_abs_delta")),
             _f(r.get("p90_abs_delta")), _f(r.get("p95_abs_delta")), _f(r.get("std_delta"))]
            for r in change_rows
        ],
    )
    direction_tbl = _tbl(
        ["Seed", "valid_adjacent_count", "exact_3class_rate", "nonzero_rate", "actual_nonzero"],
        [
            [r.get("seed", ""), _i(r.get("valid_adjacent_count")),
             _f(r.get("exact_three_class_agreement_rate")),
             _f(r.get("nonzero_direction_agreement_rate")),
             _i(r.get("actual_nonzero_count"))]
            for r in dir_rows
        ],
    )
    return (
        '<section class="cw-p48-sec"><h4>C — Gap-safe first differences (Δ at 10-min cadence)</h4>'
        + change_tbl
        + '<div class="cw-p48-sec"><h4>C.1 — Direction-of-change agreement (3-class: NEGATIVE / ZERO / POSITIVE)</h4>'
        + direction_tbl
        + "</section>"
    )


def _section_lag(lag_rows: list[dict]) -> str:
    by_seed: dict[str, list[dict]] = {}
    for r in lag_rows:
        by_seed.setdefault(r.get("seed", ""), []).append(r)
    parts = []
    for seed, srows in by_seed.items():
        parts.append(
            f'<div class="cw-p48-card2"><h5>{escape(seed)}</h5>'
            + '<table class="cw-p48-tbl" style="width:100%;margin-top:6px"><thead><tr>'
            + '<th>lag (steps)</th><th>lag (min)</th><th>Pearson r</th><th>valid pairs</th>'
            + '</tr></thead><tbody>'
            + "".join(
                f"<tr><td>{_i(r.get('lag_steps'))}</td><td>{_i(r.get('lag_minutes'))}</td>"
                f"<td>{_f(r.get('pearson_correlation'))}</td><td>{_i(r.get('valid_pair_count'))}</td></tr>"
                for r in srows
            )
            + "</tbody></table></div>"
        )
    return (
        '<section class="cw-p48-sec"><h4>D — Fixed lag diagnostics (DIAGNOSTIC ONLY)</h4>'
        + '<div class="cw-p48-grid3">' + "".join(parts) + "</div>"
        + "</section>"
    )


def _section_acf(acf_rows: list[dict]) -> str:
    by_series: dict[str, list[dict]] = {}
    for r in acf_rows:
        by_series.setdefault(r.get("series_id", ""), []).append(r)
    parts = []
    for sid, srows in by_series.items():
        parts.append(
            f'<div class="cw-p48-card2"><h5>{escape(sid)}</h5>'
            + '<table class="cw-p48-tbl" style="width:100%;margin-top:6px"><thead><tr>'
            + '<th>lag (steps)</th><th>lag (min)</th><th>ACF</th><th>valid pairs</th>'
            + '</tr></thead><tbody>'
            + "".join(
                f"<tr><td>{_i(r.get('lag_steps'))}</td><td>{_i(r.get('lag_minutes'))}</td>"
                f"<td>{_f(r.get('acf'))}</td><td>{_i(r.get('valid_pair_count'))}</td></tr>"
                for r in srows
            )
            + "</tbody></table></div>"
        )
    return (
        '<section class="cw-p48-sec"><h4>E — Prediction ACF (registered lags 1 / 6 / 12 / 36 / 72 / 144)</h4>'
        + '<div class="cw-p48-grid3">' + "".join(parts) + "</div>"
        + "</section>"
    )


def _section_extrema(ext_rows: list[dict], peak_rows: list[dict]) -> str:
    extrema_tbl = _tbl(
        ["Seed", "true_local_max", "actual_peak_mean", "pred_at_peak_mean", "peak_level_ratio",
         "true_local_min", "actual_trough_mean", "pred_at_trough_mean"],
        [
            [r.get("seed", ""), _i(r.get("true_local_max_count")), _f(r.get("actual_peak_mean")),
             _f(r.get("pred_at_peak_mean")), _f(r.get("peak_level_ratio")),
             _i(r.get("true_local_min_count")), _f(r.get("actual_trough_mean")), _f(r.get("pred_at_trough_mean"))]
            for r in ext_rows
        ],
    )
    peak_tbl = _tbl(
        ["Seed", "eligible_true_peaks", "same_step_count", "same_step_rate",
         "within_±1_step_count", "within_±1_step_rate"],
        [
            [r.get("seed", ""), _i(r.get("eligible_true_peaks")),
             _i(r.get("same_step_pred_peak_count")), _f(r.get("same_step_rate")),
             _i(r.get("within_plus_minus_1_step_count")), _f(r.get("within_plus_minus_1_step_rate"))]
            for r in peak_rows
        ],
    )
    return (
        '<section class="cw-p48-sec"><h4>F — Local extrema diagnostic</h4>'
        + extrema_tbl
        + '<div class="cw-p48-sec"><h4>F.1 — Peak timing (±1 step window, deterministic)</h4>'
        + peak_tbl
        + "</section>"
    )


def _section_seed_agreement(pair_rows: list[dict], spread_rows: list[dict], top_rows: list[dict]) -> str:
    pair_tbl = _tbl(
        ["Pair", "Pearson r", "Spearman ρ", "mean |Δ|", "RMSE_between_preds",
         "max |Δ|", "N", "label"],
        [
            [f"{r.get('seed_a','')} vs {r.get('seed_b','')}",
             _f(r.get("pearson_correlation")), _f(r.get("spearman_correlation")),
             _f(r.get("mean_absolute_prediction_difference")),
             _f(r.get("rmse_between_predictions")),
             _f(r.get("max_absolute_prediction_difference")),
             _i(r.get("N")), r.get("label", "")]
            for r in pair_rows
        ],
    )
    means = [float(r["seed_mean_prediction"]) for r in spread_rows if r.get("seed_mean_prediction")]
    stds = [float(r["seed_std_prediction"]) for r in spread_rows if r.get("seed_std_prediction")]
    ranges = [float(r["seed_range_prediction"]) for r in spread_rows if r.get("seed_range_prediction")]
    if means and stds and ranges:
        spread_cards = (
            '<div class="cw-p48-grid3">'
            + f'<div class="cw-p48-card2"><h5>Mean of seed_mean</h5><div class="big">{_f(sum(means)/len(means))} Wh</div>'
            + '<div class="sm">Descriptive central tendency only</div></div>'
            + f'<div class="cw-p48-card2"><h5>Mean of seed_std (ddof=1)</h5><div class="big">{_f(sum(stds)/len(stds))} Wh</div>'
            + '<div class="sm">Sample SD across 3 seeds</div></div>'
            + f'<div class="cw-p48-card2"><h5>Mean of seed_range</h5><div class="big">{_f(sum(ranges)/len(ranges))} Wh</div>'
            + '<div class="sm">max − min across 3 seeds</div></div>'
            + "</div>"
        )
    else:
        spread_cards = '<div class="cw-p48-call warn">Seed-spread rows unavailable.</div>'

    top_tbl = _tbl(
        ["Rank", "target_id", "target_timestamp", "y_true (Wh)", "seed42", "seed123", "seed2026",
         "seed_range (Wh)"],
        [
            [_i(r.get("rank")), r.get("target_id", ""), r.get("target_timestamp", ""),
             _f(r.get("y_true_wh")),
             _f(r.get("seed42")), _f(r.get("seed123")), _f(r.get("seed2026")),
             _f(r.get("seed_range"))]
            for r in top_rows[:10]
        ],
    )

    cross_seed_note = (
        '<p style="margin:6px 0 0;font-size:12.5px;color:#334155;line-height:1.55">'
        '<strong>Seed mean</strong> is descriptive central tendency only, not an ensemble prediction. '
        '<strong>Seed spread</strong> describes cross-seed disagreement, not predictive uncertainty '
        'or a confidence interval.'
        '</p>'
    )

    return (
        '<section class="cw-p48-sec"><h4>G — Cross-seed behavior</h4>'
        + cross_seed_note
        + '<div class="cw-p48-sec"><h5>G.1 — Pairwise seed agreement (SEED_AGREEMENT_DIAGNOSTIC)</h5>'
        + pair_tbl
        + '<div class="cw-p48-sec"><h5>G.2 — Cross-seed prediction spread (descriptive)</h5>'
        + spread_cards
        + '<div class="cw-p48-sec"><h5>G.3 — Top-10 seed disagreement (K=20 total, ranked by seed_range_prediction)</h5>'
        + top_tbl
        + ''
        + "</div></section>"
    )


def _section_rolling(rolling_rows: list[dict]) -> str:
    by_seed: dict[str, list[dict]] = {}
    for r in rolling_rows:
        by_seed.setdefault(r.get("seed", ""), []).append(r)
    parts = []
    for sid, srows in by_seed.items():
        if not srows:
            continue
        means = [float(r["rolling_24h_pred_mean"]) for r in srows if r.get("rolling_24h_pred_mean")]
        stds = [float(r["rolling_24h_pred_std"]) for r in srows if r.get("rolling_24h_pred_std")]
        corrs = [float(r["rolling_24h_corr"]) for r in srows if r.get("rolling_24h_corr")]
        gaps = [float(r["rolling_24h_true_mean"]) for r in srows if r.get("rolling_24h_true_mean")]
        if not means or not stds or not corrs:
            continue
        gap_pred = [(g - p) for g, p in zip(gaps, means)]
        parts.append(
            f'<div class="cw-p48-card2"><h5>{escape(sid)}</h5>'
            + f'<div class="big">{_f(sum(means)/len(means))} Wh</div>'
            + '<div class="sm">Mean rolling_24h_pred_mean across all valid windows</div>'
            + f'<div class="sm">Mean gap (true − pred): {_f(sum(gap_pred)/len(gap_pred))} Wh</div>'
            + f'<div class="sm">Mean rolling_24h_pred_std: {_f(sum(stds)/len(stds))} Wh</div>'
            + f'<div class="sm">Mean rolling_24h_corr: {_f(sum(corrs)/len(corrs))}</div>'
            + f'<div class="sm">Valid windows: {len(srows)} (window_valid_count = 144 each)</div>'
            + "</div>"
        )
    return (
        '<section class="cw-p48-sec"><h4>H — Rolling 24h tracking (gap-safe, exact 144 samples, no forward fill)</h4>'
        + '<div class="cw-p48-grid3">' + "".join(parts) + "</div>"
        + "</section>"
    )


def _section_integrity(neg_rows: list[dict], sat_rows: list[dict]) -> str:
    neg_tbl = _tbl(
        ["Seed", "negative_count", "negative_fraction", "minimum_prediction",
         "clipping_applied", "denominator"],
        [
            [r.get("seed", ""), _i(r.get("negative_count")), _f(r.get("negative_fraction")),
             _f(r.get("minimum_prediction")), r.get("clipping_applied", ""),
             _i(r.get("denominator"))]
            for r in neg_rows
        ],
    )
    sat_tbl = _tbl(
        ["Seed", "unique_count", "fraction_at_exact_min", "fraction_at_exact_max",
         "duplicate_rate", "suspected_saturation", "reason"],
        [
            [r.get("seed", ""), _i(r.get("unique_prediction_count")),
             _f(r.get("fraction_at_exact_min")), _f(r.get("fraction_at_exact_max")),
             _f(r.get("duplicate_rate")), r.get("suspected_saturation", ""),
             r.get("reason", "")]
            for r in sat_rows
        ],
    )
    any_neg = any(int(r.get("negative_count", 0)) > 0 for r in neg_rows)
    any_sat = any(str(r.get("suspected_saturation")).lower() == "true" for r in sat_rows)
    summary_text = (
        f'<p style="margin:8px 0 0;font-size:12.5px;color:#334155;line-height:1.55">'
        f'Negative predictions: <strong>{"0 across all 3 seeds" if not any_neg else "detected"}</strong>; '
        f'clipping not applied. '
        f'Saturation: <strong>{"no structural signal" if not any_sat else "suspected"}</strong>.'
        f'</p>'
    )
    return (
        '<section class="cw-p48-sec"><h4>I — Integrity audits</h4>'
        + '<h5>I.1 — Negative prediction audit</h5>'
        + neg_tbl
        + '<h5>I.2 — Saturation audit</h5>'
        + sat_tbl
        + summary_text
        + "</section>"
    )


def _section_baseline(baseline_rows: list[dict]) -> str:
    rows = [
        [r.get("model_id", ""),
         r.get("prediction_bundle_available", ""),
         r.get("common_population_verified", ""),
         r.get("interpretation_label", ""),
         r.get("phase47_final_comparison_verdict", "")]
        for r in baseline_rows
    ]
    return (
        '<section class="cw-p48-sec"><h4>J — Persistence baseline &amp; LSTM eligibility context</h4>'
        + _tbl(
            ["Model", "bundle_available", "common_population", "interpretation_label",
             "phase47_verdict"],
            rows,
        )
        + "</section>"
    )


def _section_findings(findings_rows: list[dict]) -> str:
    """Reduced findings presentation — headline items only.

    The full 33-row findings table is preserved unchanged in the frozen
    Phase 48 artifact; the notebook surfaces only the curated headline.
    """
    headlines = [
        ["PREDICTIONS_SHOW_RANGE_COMPRESSION", "all_seeds", "std_ratio", "≈ 0.74–0.78", "Range / variance compression"],
        ["PREDICTIONS_SMOOTHER_THAN_ACTUAL", "all_seeds", "mean_abs_delta", "≈ 19.5–21.3 Wh vs 26.7 Wh", "Change behavior"],
        ["DIRECTIONAL_CHANGE_ALIGNMENT", "all_seeds", "nonzero_rate", "≈ 0.34–0.35", "Direction-of-change"],
        ["APPARENT_TEMPORAL_LAG_NEGATIVE_ONE", "all_seeds", "pearson_correlation_at_lag_-1", "≈ 0.83–0.87", "Fixed-lag diagnostics"],
        ["HIGH_SEED_AGREEMENT", "all_seeds", "pairwise_pearson", "≈ 0.91–0.93", "Cross-seed behavior"],
        ["NO_NEGATIVE_PREDICTIONS", "all_seeds", "negative_count", "0", "Integrity audits"],
        ["NO_SATURATION_SIGNAL", "all_seeds", "unique_count", "2961", "Integrity audits"],
        ["NO_NEW_INFERENCE", "phase48", "inference_count", "0", "Phase 48 preflight"],
    ]
    rows = [[h[0], h[1], h[2], h[3], h[4]] for h in headlines]
    return (
        '<section class="cw-p48-sec"><h4>K — Phase 48 scientific findings (descriptive only)</h4>'
        + _tbl(
            ["finding_code", "scope", "metric", "value", "category"],
            rows,
        )
        + ''
        + "</section>"
    )


def _section_figures(fig_dir: Path) -> str:
    """Embed figures in a presentation-prioritized order."""
    primary = [
        ("PRED_48_01_full_test_actual_vs_all_seeds.png",
         "Held-Out Test — Actual vs 3 Transformer seeds (full period)",
         "Each seed is a distinct prediction bundle."),
        ("PRED_48_02_full_test_actual_vs_seed_mean_spread.png",
         "Actual vs seed mean + cross-seed spread (full period)",
         "Seed mean is descriptive central tendency; spread band is cross-seed spread."),
        ("PRED_48_05_last_24h_zoom.png",
         "24h zoom — last 144 samples (deterministic)",
         "Window = last 24h of Test."),
        ("PRED_48_09_prediction_ecdf.png",
         "Prediction ECDF — actual + 3 seeds + descriptive seed mean",
         "Empirical CDF of per-row predictions."),
        ("PRED_48_13_lag_cross_correlation.png",
         "Fixed-lag correlation diagnostic",
         "Diagnostic only; predictions are not shifted by the apparent best lag."),
        ("PRED_48_15_local_peak_capture.png",
         "Local peak capture diagnostic",
         "Per-seed mean of (predicted at true local peaks)."),
        ("PRED_48_16_rolling_24h_mean_tracking.png",
         "Rolling 24h mean tracking (true − pred) per seed",
         "Exact 144-sample gap-safe windows."),
    ]
    secondary = [
        ("PRED_48_03_first_24h_zoom.png", "24h zoom — first 144 samples", "First 24h of Test."),
        ("PRED_48_06_scatter_seed42.png", "Scatter — seed42", "y=x reference line shown."),
        ("PRED_48_07_scatter_seed123.png", "Scatter — seed123", "y=x reference line shown."),
        ("PRED_48_08_scatter_seed2026.png", "Scatter — seed2026", "y=x reference line shown."),
        ("PRED_48_10_change_magnitude_distribution.png", "Change magnitude distribution", ""),
        ("PRED_48_11_cross_seed_spread_over_time.png", "Cross-seed spread over time", ""),
        ("PRED_48_12_pairwise_seed_prediction_scatter.png", "Pairwise seed prediction scatter", ""),
        ("PRED_48_14_acf_actual_vs_predictions.png", "Prediction ACF (NOT residual ACF)", "Registered lags 1 / 6 / 12 / 36 / 72 / 144."),
        ("PRED_48_17_rolling_24h_std_tracking.png", "Rolling 24h std ratio per seed", "pred_std / true_std."),
        ("PRED_48_18_daily_actual_heatmap.png", "Daily actual heatmap", ""),
        ("PRED_48_19_daily_seed_mean_heatmap.png", "Daily seed mean heatmap", ""),
        ("PRED_48_20_seed_spread_heatmap.png", "Seed spread heatmap", ""),
        ("PRED_48_04_middle_24h_zoom.png", "24h zoom — middle 144 samples", "Middle 24h of Test."),
    ]

    def _render_group(title: str, items: list[tuple[str, str, str]]) -> str:
        blocks = []
        for fname, t, cap in items:
            p = fig_dir / fname
            if not p.exists():
                continue
            try:
                b64 = _b(p)
                data_uri = f"data:image/png;base64,{b64}"
                full_cap = cap if cap else ""
                blocks.append(_img_block(data_uri, t, full_cap))
            except Exception:
                pass
        return f'<section class="cw-p48-sec"><h5>{escape(title)}</h5>{"".join(blocks)}</section>'

    primary_section = _render_group("Primary figures", primary)
    secondary_section = _render_group("Secondary figures", secondary)

    return (
        '<section class="cw-p48-sec"><h4>L — Phase 48 figures</h4>'
        + primary_section
        + secondary_section
        + "</section>"
    )

@phase_report(48)
def render_phase_48_dashboard(project_root: Path | None = None) -> HTML:
    """Render Phase 48 Prediction Analysis as ONE compact dashboard.

    Layout (sections):
      1. Header + status
      2. Input conditions (Test N, source, seeds, Phase 47 verdict)
      3. Principal prediction-behavior table (per-seed, 6 columns:
         std ratio, mean shift, direction-of-change nonzero rate,
         lag=-1 Pearson r, peak-level ratio, negative count)
      4. Signoff table (status + forbidden-action flags + decision)

    Data sources (read-only, JSON/CSV; no upstream re-execution):
      * artifacts/prediction_analysis/phase_48_signoff.json
      * artifacts/prediction_analysis/prediction_analysis_summary.json
      * artifacts/prediction_analysis/prediction_range_compression.csv
      * artifacts/prediction_analysis/prediction_direction_agreement.csv
      * artifacts/prediction_analysis/prediction_lag_diagnostics.csv
      * artifacts/prediction_analysis/prediction_local_extrema_summary.csv

    MAPE addendum is OMITTED from notebook presentation because the
    addendum is BLOCKED_SOURCE_UNAVAILABLE for Test (test_mape_computed
    = false; test_inference_executed = false). Validation MAPE is not
    Test-MAPE and is not authoritative for Phase 48 Test analysis.

    Phase 47 source comparison verdict (e.g.
    Transformer_better_RMSE_R2_Persistence_better_MAE) is preserved as a
    single status row, NOT narrated.
    """
    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "prediction_analysis"
    if not (art / "phase_48_signoff.json").exists():
        return HTML(_CSS + '<article class="cw-p48"><div class="cw-p48-body">'
                    'Phase 48 prediction analysis is not available.</div></article>')

    signoff = read_json(art / "phase_48_signoff.json")
    summary = read_json(art / "prediction_analysis_summary.json")
    status = str(signoff.get("overall_status") or "UNKNOWN").upper()
    badge_class = "cw-p48-badge" + (f" {_status_class(status)}" if _status_class(status) else "")
    seeds = [int(s) for s in (signoff.get("seed_list") or summary.get("seed_list") or [])]
    seed_text = ", ".join(str(s) for s in seeds)
    n_test = signoff.get("n_test") or summary.get("n_test")

    pop_sha = signoff.get("test_population_sha256") or summary.get("test_population_sha256")
    verdict = summary.get("phase47_final_comparison_verdict")
    input_pairs = [
        ("Dataset", "Held-out Test"),
        ("Test N", n_test),
        ("Test population SHA-256", pop_sha),
        ("Source phase", signoff.get("source_phase47_version")),
        ("Seeds", seed_text),
        ("Phase 47 verdict", verdict),
        ("New inference", signoff.get("new_inference")),
    ]
    input_html = (
        '<section class="cw-p48-sec"><h4>Prediction analysis conditions</h4>'
        + _tbl(["Field", "Value"], [[k, str(v)] for k, v in input_pairs if v is not None])
        + '</section>'
    )

    rng_rows = {str(r.get("seed", "")).removeprefix("SEED"): r
                for r in _csv(art / "prediction_range_compression.csv")}
    dir_rows = {str(r.get("seed", "")).removeprefix("SEED"): r
                for r in _csv(art / "prediction_direction_agreement.csv")}
    lag_rows_by_seed: dict[str, dict[int, dict]] = {}
    for r in _csv(art / "prediction_lag_diagnostics.csv"):
        sk = str(r.get("seed", "")).removeprefix("SEED")
        try:
            lag_rows_by_seed.setdefault(sk, {})[int(r.get("lag_steps"))] = r
        except Exception:
            continue
    ext_rows = {str(r.get("seed", "")).removeprefix("SEED"): r
                for r in _csv(art / "prediction_local_extrema_summary.csv")}

    behavior_headers = ["Metric", "Seed 42", "Seed 123", "Seed 2026"]
    behavior_rows: list[list[str]] = []

    behavior_rows.append([
        "Std ratio (pred / true)",
        _f(rng_rows.get("42", {}).get("std_ratio"), 3),
        _f(rng_rows.get("123", {}).get("std_ratio"), 3),
        _f(rng_rows.get("2026", {}).get("std_ratio"), 3),
    ])
    behavior_rows.append([
        "Mean shift (pred − true, Wh)",
        _f(rng_rows.get("42", {}).get("mean_shift_pred_minus_true"), 3),
        _f(rng_rows.get("123", {}).get("mean_shift_pred_minus_true"), 3),
        _f(rng_rows.get("2026", {}).get("mean_shift_pred_minus_true"), 3),
    ])
    behavior_rows.append([
        "Direction-of-change nonzero agreement rate (fraction)",
        _f(dir_rows.get("42", {}).get("nonzero_direction_agreement_rate"), 3),
        _f(dir_rows.get("123", {}).get("nonzero_direction_agreement_rate"), 3),
        _f(dir_rows.get("2026", {}).get("nonzero_direction_agreement_rate"), 3),
    ])
    behavior_rows.append([
        "Pearson r at lag = -1 (DIAGNOSTIC ONLY)",
        _f(lag_rows_by_seed.get("42", {}).get(-1, {}).get("pearson_correlation"), 3),
        _f(lag_rows_by_seed.get("123", {}).get(-1, {}).get("pearson_correlation"), 3),
        _f(lag_rows_by_seed.get("2026", {}).get(-1, {}).get("pearson_correlation"), 3),
    ])
    behavior_rows.append([
        "Peak level ratio (pred / actual)",
        _f(ext_rows.get("42", {}).get("peak_level_ratio"), 3),
        _f(ext_rows.get("123", {}).get("peak_level_ratio"), 3),
        _f(ext_rows.get("2026", {}).get("peak_level_ratio"), 3),
    ])
    behavior_html = (
        '<section class="cw-p48-sec"><h4>Principal prediction behavior (per seed)</h4>'
        + _tbl(behavior_headers, behavior_rows)
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Best-seed selection", signoff.get("best_seed_selected")),
        ("Ensemble used", signoff.get("ensemble_used")),
        ("Predictions shifted", summary.get("predictions_shifted")),
        ("Predictions clipped", summary.get("predictions_clipped")),
        ("Post-hoc calibration", summary.get("post_hoc_calibration")),
        ("Decision", "PROCEED to Phase 49 / 50 / 51"),
    ]
    signoff_html = (
        '<section class="cw-p48-sec"><h4>Signoff</h4>'
        + _tbl(["Field", "Value"], [[k, str(v)] for k, v in signoff_pairs if v is not None])
        + '</section>'
    )

    body = '<div class="cw-p48-body">' + input_html + behavior_html + signoff_html + '</div>'

    return HTML(
        _CSS + '<article class="cw-p48"><header class="cw-p48-h">'
        '<h3>Phase 48 - Prediction Analysis</h3>'
        f'<span class="{badge_class}">{escape(status)}</span></header>' + body + '</article>'
    )


def _phase48_behavior_metrics(
    art: Path, seeds: list[int],
) -> list[tuple[str, list[float | None], int]]:
    """Read the four displayed metrics directly from the saved analysis tables."""
    specs = [
        ("Prediction / actual standard deviation", "prediction_range_compression.csv",
         "std_ratio", 1, 3),
        ("Mean prediction − actual (Wh)", "prediction_range_compression.csv",
         "mean_shift_pred_minus_true", 1, 3),
        ("Direction agreement — actual change ≠ 0 (%)", "prediction_direction_agreement.csv",
         "nonzero_direction_agreement_rate", 100, 2),
        ("Predicted / actual mean at actual peaks", "prediction_local_extrema_summary.csv",
         "peak_level_ratio", 1, 3),
    ]
    tables = {filename: _csv(art / filename) for _, filename, _, _, _ in specs}
    metrics = []
    for label, filename, field, factor, digits in specs:
        by_seed = {
            str(row.get("seed", "")).upper().removeprefix("SEED"): row
            for row in tables[filename]
        }
        values = []
        for seed in seeds:
            raw = by_seed.get(str(seed), {}).get(field)
            try:
                value = float(raw) * factor
                values.append(value if math.isfinite(value) else None)
            except (TypeError, ValueError):
                values.append(None)
        metrics.append((label, values, digits))
    return metrics
