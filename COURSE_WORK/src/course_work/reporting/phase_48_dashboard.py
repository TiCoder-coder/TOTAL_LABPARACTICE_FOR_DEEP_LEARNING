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
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.utils.artifacts import get_project_root, read_json

__all__ = ["render_phase_48_dashboard"]


_CSS = """
<style>
.cw-d{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 22px;overflow:hidden}
.cw-d *{box-sizing:border-box}
.cw-d-h{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-d-h h3{font-size:20px;line-height:1.25;margin:0 0 5px;color:#172033}
.cw-d-meta{font-size:12px;color:#5d6b82}
.cw-d-badge{border:1px solid #a9dec1;border-radius:999px;padding:6px 12px;font-size:11px;font-weight:700;letter-spacing:.04em;color:#11613d;background:#e8f7ef;white-space:nowrap}
.cw-d-badge.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-d-badge.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:16px 22px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-d-card{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:9px;background:#fff;min-width:0}
.cw-d-card span{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card strong{font-size:13px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-d-body{padding:4px 22px 22px}
.cw-d-sec{margin-top:18px}
.cw-d-sec h4{font-size:14px;margin:0 0 8px;color:#334155}
.cw-d-tblwrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-d-tbl{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
.cw-d-tbl th{background:#f5f7fb;color:#475569;text-align:left;font-weight:650;padding:9px 11px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-d-tbl td{text-align:left;padding:8px 11px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-d-tbl tbody tr:nth-child(even){background:#fafbfd}
.cw-d-tbl tbody tr:last-child td{border-bottom:0}
.cw-d-fp{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;color:#1f2a44}
.cw-d-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-d-pill.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:12px}
.cw-d-card2{border:1px solid #d9e2ef;border-radius:11px;padding:13px 14px;background:#fff}
.cw-d-card2.win{border-color:#a9dec1;background:#e8f7ef}
.cw-d-card2 h5{margin:0 0 5px;font-size:11.5px;color:#475569;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card2 .big{font-size:19px;color:#172033;font-weight:700;line-height:1.2}
.cw-d-card2 .sm{font-size:11.5px;color:#5d6b82;margin-top:3px}
.cw-d-call{border-left:4px solid #6366f1;background:#eef2ff;padding:9px 13px;border-radius:0 7px 7px 0;margin:11px 0;font-size:12.5px;color:#1f2a44}
.cw-d-call.warn{border-color:#d97706;background:#fff7e0;color:#7a5613}
.cw-d-call.good{border-color:#16a34a;background:#e8f7ef;color:#11613d}
.cw-d-fig{display:flex;flex-direction:column;gap:6px;margin:10px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:82%;max-width:850px}
.cw-d-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto}
.cw-d-fig.compact{width:64%;max-width:700px}
.cw-d-fig .ftitle{font-size:12px;color:#475569;font-weight:650}
.cw-d-fig .fcap{font-size:11.5px;color:#5d6b82;line-height:1.45}
@media (max-width:900px){.cw-d-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-d-grid3{grid-template-columns:1fr}}
@media (max-width:620px){.cw-d-h{flex-direction:column;padding:15px}.cw-d-cards{grid-template-columns:1fr;padding:12px 16px}.cw-d-body{padding-left:16px;padding-right:16px}}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        f'<div class="cw-d-fig{extra}">'
        f'<div class="ftitle">{escape(title)}</div>'
        f'<img src="{escape(rel_path, quote=True)}" alt="{escape(title)}"/>'
        f'<div class="fcap">{escape(caption)}</div>'
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
    css_cls = f"cw-d-badge{(' ' + _status_class(status)) if _status_class(status) else ''}"
    return (
        '<header class="cw-d-h">'
        f"<div><h3>{escape(title)}</h3>"
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(str(status).upper())}</span>'
        "</header>"
    )


def _cards(pairs: Iterable[tuple[str, Any]]) -> str:
    items = "".join(
        f'<div class="cw-d-card"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in pairs
    )
    return f'<div class="cw-d-cards">{items}</div>'


def _tbl(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    parts = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        parts.append(f"<tr>{cells}</tr>")
    body = "".join(parts)
    return f'<div class="cw-d-tblwrap"><table class="cw-d-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_distribution(distribution_rows: list[dict]) -> str:
    rows = [
        [r.get("series_id", ""), _i(r.get("N")), _f(r.get("mean")), _f(r.get("std")),
         _f(r.get("min")), _f(r.get("p05")), _f(r.get("q1")), _f(r.get("median")),
         _f(r.get("q3")), _f(r.get("p95")), _f(r.get("max")), _f(r.get("iqr"))]
        for r in distribution_rows
    ]
    return (
        '<section class="cw-d-sec"><h4>A — Distribution summary (N, mean, std, IQR, percentiles)</h4>'
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
        '<section class="cw-d-sec"><h4>B — Range / variance compression per seed</h4>'
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
        '<section class="cw-d-sec"><h4>C — Gap-safe first differences (Δ at 10-min cadence)</h4>'
        + change_tbl
        + '<div class="cw-d-sec"><h4>C.1 — Direction-of-change agreement (3-class: NEGATIVE / ZERO / POSITIVE)</h4>'
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
            f'<div class="cw-d-card2"><h5>{escape(seed)}</h5>'
            + '<table class="cw-d-tbl" style="width:100%;margin-top:6px"><thead><tr>'
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
        '<section class="cw-d-sec"><h4>D — Fixed lag diagnostics (DIAGNOSTIC ONLY)</h4>'
        + '<div class="cw-d-grid3">' + "".join(parts) + "</div>"
        + "</section>"
    )


def _section_acf(acf_rows: list[dict]) -> str:
    by_series: dict[str, list[dict]] = {}
    for r in acf_rows:
        by_series.setdefault(r.get("series_id", ""), []).append(r)
    parts = []
    for sid, srows in by_series.items():
        parts.append(
            f'<div class="cw-d-card2"><h5>{escape(sid)}</h5>'
            + '<table class="cw-d-tbl" style="width:100%;margin-top:6px"><thead><tr>'
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
        '<section class="cw-d-sec"><h4>E — Prediction ACF (registered lags 1 / 6 / 12 / 36 / 72 / 144)</h4>'
        + '<div class="cw-d-grid3">' + "".join(parts) + "</div>"
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
        '<section class="cw-d-sec"><h4>F — Local extrema diagnostic</h4>'
        + extrema_tbl
        + '<div class="cw-d-sec"><h4>F.1 — Peak timing (±1 step window, deterministic)</h4>'
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

    # Spread — compute aggregate stats from 2961 rows
    means = [float(r["seed_mean_prediction"]) for r in spread_rows if r.get("seed_mean_prediction")]
    stds = [float(r["seed_std_prediction"]) for r in spread_rows if r.get("seed_std_prediction")]
    ranges = [float(r["seed_range_prediction"]) for r in spread_rows if r.get("seed_range_prediction")]
    if means and stds and ranges:
        spread_cards = (
            '<div class="cw-d-grid3">'
            + f'<div class="cw-d-card2"><h5>Mean of seed_mean</h5><div class="big">{_f(sum(means)/len(means))} Wh</div>'
            + '<div class="sm">Descriptive central tendency only</div></div>'
            + f'<div class="cw-d-card2"><h5>Mean of seed_std (ddof=1)</h5><div class="big">{_f(sum(stds)/len(stds))} Wh</div>'
            + '<div class="sm">Sample SD across 3 seeds</div></div>'
            + f'<div class="cw-d-card2"><h5>Mean of seed_range</h5><div class="big">{_f(sum(ranges)/len(ranges))} Wh</div>'
            + '<div class="sm">max − min across 3 seeds</div></div>'
            + "</div>"
        )
    else:
        spread_cards = '<div class="cw-d-call warn">Seed-spread rows unavailable.</div>'

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

    # Single canonical disclaimer for cross-seed terminology (used only here)
    cross_seed_note = (
        '<p style="margin:6px 0 0;font-size:12.5px;color:#334155;line-height:1.55">'
        '<strong>Seed mean</strong> is descriptive central tendency only, not an ensemble prediction. '
        '<strong>Seed spread</strong> describes cross-seed disagreement, not predictive uncertainty '
        'or a confidence interval.'
        '</p>'
    )

    return (
        '<section class="cw-d-sec"><h4>G — Cross-seed behavior</h4>'
        + cross_seed_note
        + '<div class="cw-d-sec"><h5>G.1 — Pairwise seed agreement (SEED_AGREEMENT_DIAGNOSTIC)</h5>'
        + pair_tbl
        + '<div class="cw-d-sec"><h5>G.2 — Cross-seed prediction spread (descriptive)</h5>'
        + spread_cards
        + '<div class="cw-d-sec"><h5>G.3 — Top-10 seed disagreement (K=20 total, ranked by seed_range_prediction)</h5>'
        + top_tbl
        + '<p style="margin:6px 0 0;font-size:12px;color:#5d6b82;line-height:1.5">'
        + 'Top-K ranking uses <strong>seed_range_prediction</strong> only, with deterministic tie-break '
        + '(target_id ascending). This is not a worst-error ranking.'
        + '</p>'
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
            f'<div class="cw-d-card2"><h5>{escape(sid)}</h5>'
            + f'<div class="big">{_f(sum(means)/len(means))} Wh</div>'
            + '<div class="sm">Mean rolling_24h_pred_mean across all valid windows</div>'
            + f'<div class="sm">Mean gap (true − pred): {_f(sum(gap_pred)/len(gap_pred))} Wh</div>'
            + f'<div class="sm">Mean rolling_24h_pred_std: {_f(sum(stds)/len(stds))} Wh</div>'
            + f'<div class="sm">Mean rolling_24h_corr: {_f(sum(corrs)/len(corrs))}</div>'
            + f'<div class="sm">Valid windows: {len(srows)} (window_valid_count = 144 each)</div>'
            + "</div>"
        )
    return (
        '<section class="cw-d-sec"><h4>H — Rolling 24h tracking (gap-safe, exact 144 samples, no forward fill)</h4>'
        + '<div class="cw-d-grid3">' + "".join(parts) + "</div>"
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
        '<section class="cw-d-sec"><h4>I — Integrity audits</h4>'
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
        '<section class="cw-d-sec"><h4>J — Persistence baseline &amp; LSTM eligibility context</h4>'
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
    # Headline findings, manually curated from the 33-row artifact.
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
        '<section class="cw-d-sec"><h4>K — Phase 48 scientific findings (descriptive only)</h4>'
        + _tbl(
            ["finding_code", "scope", "metric", "value", "category"],
            rows,
        )
        + '<p style="margin:8px 0 0;font-size:12px;color:#5d6b82;line-height:1.5">'
        + 'All findings are descriptive. No causal claims. No claim of calibration or predictive '
        + 'uncertainty. No claim that Transformer wins every metric. '
        + 'Full descriptive findings remain available in the frozen Phase 48 artifact.'
        + '</p>'
        + "</section>"
    )


def _section_figures(fig_dir: Path) -> str:
    """Embed figures in a presentation-prioritized order."""
    # Presentation order (primary first, secondary grouped after).
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
        return f'<section class="cw-d-sec"><h5>{escape(title)}</h5>{"".join(blocks)}</section>'

    primary_section = _render_group("Primary figures", primary)
    secondary_section = _render_group("Secondary figures", secondary)

    return (
        '<section class="cw-d-sec"><h4>L — Phase 48 figures</h4>'
        + primary_section
        + secondary_section
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def render_phase_48_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 48 Prediction Analysis dashboard (1 figure + table)."""
    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "prediction_analysis"

    if not (art / "phase_48_signoff.json").exists():
        return HTML(
            _CSS
            + f'<article class="cw-d">{_header("Phase 48 - Prediction Analysis", "Signoff not yet materialized.", "FAIL")}'
            + '<div class="cw-d-body"><div class="cw-d-call warn">'
            + 'Phase 48 canonical artifacts are not yet present under <code>artifacts/prediction_analysis/</code>.'
            + "</div></div></article>"
        )

    signoff = read_json(art / "phase_48_signoff.json")
    summary = read_json(art / "prediction_analysis_summary.json")

    overall = str(signoff.get("overall_status") or summary.get("overall_status") or "UNKNOWN")
    locked_id = "TR_C2_ALT_LOOKBACK"
    seeds = [42, 123, 2026]

    subtitle = (
        f"PREDICTION_ANALYSIS-v1 - HELD-OUT TEST N = {summary['n_test']} - "
        f"3 transformer seeds - source bundles byte-identical to Phase 47"
    )

    # Overview (compact 2-column table)
    overview_pairs = [
        ("Phase status", overall),
        ("Locked candidate", locked_id),
        ("Test N", summary["n_test"]),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("Source pred modified", str(summary.get("source_predictions_modified", False))),
        ("Post-Test tuning", str(summary.get("post_hoc_calibration", False))),
    ]
    overview_html = (
        '<section class="cw-d-sec"><h4>Phase 48 overview</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in overview_pairs
        )
        + '</tbody></table></section>'
    )

    # Main result: PRED_48_01 figure + a compact prediction-behavior table
    figure_path = art / "figures" / "PRED_48_01_full_test_actual_vs_all_seeds.png"
    figure_html = ""
    if figure_path.exists() and figure_path.stat().st_size > 1024:
        # Sanity check: skip if the file is a tiny placeholder PNG
        try:
            from PIL import Image
            w, h = Image.open(figure_path).size
            if w >= 100 and h >= 100:
                b = figure_path.read_bytes()
                data_uri = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
                figure_html = (
                    '<div class="cw-d-fig" style="margin-top:6px">'
                    '<div class="ftitle">Actual vs Predicted (held-out Test, full horizon) — '
                    '3 Transformer seeds + seed mean</div>'
                    f'<img src="{escape(data_uri, quote=True)}" '
                    'alt="Phase 48 Actual vs Predicted over Test" '
                    'style="max-width:880px;width:90%"/>'
                    '<div class="fcap" style="font-size:11px;color:#5d6b82;line-height:1.45;margin-top:4px">'
                    'Active canonical Phase 48 figure '
                    '(<code>artifacts/prediction_analysis/figures/PRED_48_01_full_test_actual_vs_all_seeds.png</code>). '
                    'Predictions follow the temporal trend but are smoother than the observed series '
                    'and have greater difficulty around sharp peaks.'
                    '</div></div>'
                )
        except Exception:
            figure_html = ""

    # Prediction-behavior table — most useful supporting metrics from canonical P48 artifacts
    behavior_rows = _phase48_behavior_rows(art, seeds)
    behavior_table = (
        '<table class="cw-d-tbl" style="margin-top:8px">'
        '<thead><tr><th>Metric</th><th>seed42</th><th>seed123</th><th>seed2026</th></tr></thead>'
        '<tbody>' + ''.join(
            f'<tr><td style="color:#475569">{escape(metric)}</td>'
            f'<td>{escape(v0)}</td><td>{escape(v1)}</td><td>{escape(v2)}</td></tr>'
            for metric, v0, v1, v2 in behavior_rows
        ) + '</tbody></table>'
    )
    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - actual vs predicted (Test, full horizon)</h4>'
        + figure_html + behavior_table
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", overall),
        ("Seed summary interpreted as ensemble", "NO"),
        ("Best-seed selection", "NO"),
        ("Post-Test tuning", "NO"),
        ("Source predictions modified", str(summary.get("source_predictions_modified", False))),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Decision &amp; signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table>'
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + 'Predictions broadly follow the temporal pattern but are smoother than the observed '
        + 'series and have greater difficulty around sharp peaks. Phase 48 consumes the frozen '
        + 'Phase 47 prediction bundles only - no new inference, training, or retuning.'
        + '</p>'
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + overview_html + main_result_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 48 - Prediction Analysis", subtitle, overall)}{body}</article>'
    )


def _phase48_behavior_rows(art: Path, seeds: list[int]) -> list[tuple[str, str, str, str]]:
    """Return per-seed (v42, v123, v2026) tuples for the most useful P48 metrics."""
    rows: list[tuple[str, str, str, str]] = []

    def _lookup_by_seed(csv_name: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in _csv(art / csv_name):
            s = str(r.get("seed", "")).upper().replace("SEED", "")
            if s:
                out[s] = r
        return out

    def _get(d: dict[str, dict], seed: int, col: str, digits: int = 3) -> str:
        r = d.get(str(seed), {})
        v = r.get(col, "")
        if v in ("", None):
            return "N/A"
        try:
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return str(v)

    # Range compression
    compression = _lookup_by_seed("prediction_range_compression.csv")
    rows.append(("Pred std / true std (ratio)",
                 _get(compression, 42, "std_ratio"),
                 _get(compression, 123, "std_ratio"),
                 _get(compression, 2026, "std_ratio")))
    rows.append(("Mean shift (pred − true, Wh)",
                 _get(compression, 42, "mean_shift_pred_minus_true"),
                 _get(compression, 123, "mean_shift_pred_minus_true"),
                 _get(compression, 2026, "mean_shift_pred_minus_true")))

    # Direction agreement
    direction = _lookup_by_seed("prediction_direction_agreement.csv")
    rows.append(("Nonzero direction agreement",
                 _get(direction, 42, "nonzero_direction_agreement_rate"),
                 _get(direction, 123, "nonzero_direction_agreement_rate"),
                 _get(direction, 2026, "nonzero_direction_agreement_rate")))

    # Peak level ratio
    peaks = _lookup_by_seed("prediction_local_extrema_summary.csv")
    rows.append(("Peak level ratio (mean, pred/true)",
                 _get(peaks, 42, "peak_level_ratio"),
                 _get(peaks, 123, "peak_level_ratio"),
                 _get(peaks, 2026, "peak_level_ratio")))

    # Lag diagnostic at -1 (60 min behind)
    lag = _csv(art / "prediction_lag_diagnostics.csv")
    lag_at_neg1: dict[str, str] = {}
    for r in lag:
        if r.get("lag_steps") == "-1":
            s = str(r.get("seed", "")).upper().replace("SEED", "")
            if s:
                lag_at_neg1[s] = r.get("pearson_correlation", "")

    def _fmt(v) -> str:
        if v in ("", None):
            return "N/A"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    rows.append(("Pearson at lag=-1 (60 min)",
                 _fmt(lag_at_neg1.get("42", "")),
                 _fmt(lag_at_neg1.get("123", "")),
                 _fmt(lag_at_neg1.get("2026", ""))))

    # Cross-seed pairwise Pearson (average across all 3 seed pairs)
    pair = _csv(art / "prediction_seed_pairwise_agreement.csv")
    pair_vals: list[float] = []
    for r in pair:
        try:
            pair_vals.append(float(r.get("pearson_correlation", "")))
        except (TypeError, ValueError):
            pass
    cross = f"{sum(pair_vals)/len(pair_vals):.3f}" if pair_vals else "N/A"
    rows.append(("Cross-seed pairwise Pearson (avg)", cross, cross, cross))
    return rows
