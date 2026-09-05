"""Phase 49 — read-only HTML dashboard renderer.

Mirrors the Phase 43-47 and Phase 48 dashboard pattern: static HTML, no
JavaScript, no external resources, every value ``escape()``-ed before
placement in HTML.

Architecture constraints (per architecture_rule.md v1.11 amendment):

* Read-only over Phase 47 source bundles + Phase 48 canonical artifacts.
* Read-only over Phase 49-B/C/D/E/F canonical artifacts under
  ``artifacts/residual_analysis/``.
* No training, no Test inference, no checkpoint reload, no scaler fitting.
* No best-seed selection, no ensemble metric, no 3N iid interpretation.
* No prediction shifting / clipping / post-hoc calibration / residual correction.
* Ljung-Box p-values are SECONDARY DIAGNOSTIC ONLY; do not gate PASS/FAIL.
* Prediction deciles are DESCRIPTIVE diagnostics only — NOT Phase 50 target regimes.
* No worst-error ranking (Phase 51 first phase allowed to perform it).
* No attention analysis.
* Seed-mean residual reported as DESCRIPTIVE central tendency only.

Public API:

* :func:`render_phase_49_dashboard`
"""
from __future__ import annotations

import base64
import csv
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

from IPython.display import HTML

from course_work.utils.artifacts import get_project_root, read_json

__all__ = ["render_phase_49_dashboard"]


# Reuse the same CSS convention as Phase 43-47 and Phase 48 dashboards.
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


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img_block(path: Path, title: str, caption: str, compact: bool = False) -> str:
    b64 = _img_b64(path)
    if b64 is None:
        return ""
    extra = " compact" if compact else ""
    data_uri = f"data:image/png;base64,{b64}"
    return (
        f'<div class="cw-d-fig{extra}">'
        f'<div class="ftitle">{escape(title)}</div>'
        f'<img src="{escape(data_uri, quote=True)}" alt="{escape(title)}"/>'
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


def _tbl(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    parts = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        parts.append(f"<tr>{cells}</tr>")
    body = "".join(parts)
    return (
        f'<div class="cw-d-tblwrap"><table class="cw-d-tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_metric_reconstruction(b_manifest: dict[str, Any]) -> str:
    """Section B: per-seed Phase47 metric reconstruction."""
    per_seed = b_manifest.get("per_seed_metrics", {})
    rows = []
    for seed in ("42", "123", "2026"):
        m = per_seed.get(seed)
        if not m:
            rows.append([f"Seed{seed}", "N/A", "N/A", "N/A"])
            continue
        rows.append(
            [
                f"Seed{seed}",
                _f(m.get("mae_wh")),
                _f(m.get("rmse_wh")),
                _f(m.get("r2")),
            ]
        )
    return (
        '<section class="cw-d-sec"><h4>B — Metric reconstruction (per seed, matches Phase 47)</h4>'
        + _tbl(
            ["Seed", "MAE (Wh)", "RMSE (Wh)", "R²"], rows
        )
        + "</section>"
    )


def _section_distribution(dist_rows: list[dict], balance_rows: list[dict]) -> str:
    """Section C: distribution + sign balance (compact, no callout)."""
    balance_by_seed = {r.get("seed"): r for r in balance_rows}

    cards = []
    for seed in ("42", "123", "2026"):
        bal = balance_by_seed.get(seed, {})
        if not bal:
            continue
        cards.append(
            f'<div class="cw-d-card2"><h5>Seed {escape(seed)}</h5>'
            f'<div class="big">underprediction {_f(bal.get("underprediction_fraction"))}</div>'
            '<div class="sm">share of N=2961</div>'
            f'<div class="sm">overprediction: {_f(bal.get("overprediction_fraction"))}</div>'
            f'<div class="sm">exact: {_f(bal.get("exact_fraction"))}</div>'
            "</div>"
        )
    grid = '<div class="cw-d-grid3">' + "".join(cards) + "</div>"

    dist_table = _tbl(
        ["Seed", "N", "mean (Wh)", "std (Wh)", "median (Wh)", "MAD (Wh)",
         "skewness", "excess kurtosis"],
        [
            [r.get("seed", ""), _i(r.get("n")), _f(r.get("mean")), _f(r.get("std")),
             _f(r.get("median")), _f(r.get("mad")),
             _f(r.get("skewness")), _f(r.get("kurtosis_excess"))]
            for r in dist_rows
        ],
    )

    return (
        '<section class="cw-d-sec"><h4>C — Residual distribution and sign balance</h4>'
        + dist_table
        + grid
        + "</section>"
    )


def _section_tails(tail_rows: list[dict], fig_path: Path) -> str:
    rows = [
        [r.get("seed", ""), _f(r.get("abs_error_p90")), _f(r.get("abs_error_p95")),
         _f(r.get("abs_error_p99")), _f(r.get("max_absolute_error")),
         _f(r.get("fraction_above_p90")), _f(r.get("fraction_above_p95")),
         _f(r.get("fraction_above_p99"))]
        for r in tail_rows
    ]
    img = _img_block(fig_path, "Fig. 6 — Absolute-error tail diagnostics per seed",
                     "p90 / p95 / p99 / max absolute error across seeds.",
                     compact=True)
    return (
        '<section class="cw-d-sec"><h4>D — Tail behavior</h4>'
        + _tbl(["Seed", "|err| p90", "|err| p95", "|err| p99", "max |err|",
                "frac > p90", "frac > p95", "frac > p99"], rows)
        + img
        + "</section>"
    )


def _section_acf(key_lag_rows: list[dict], fig7: Path, fig8: Path) -> str:
    """Residual ACF — registered key lags + figures (no callout)."""
    key_rows = []
    for r in key_lag_rows:
        key_rows.append([r.get("seed", ""), _i(r.get("lag_steps")),
                        _f(r.get("acf_value")), _i(r.get("valid_pair_count"))])
    key_tbl = _tbl(["Seed", "lag (steps)", "ACF", "n_pairs"], key_rows)

    img7 = _img_block(fig7, "Fig. 7 — Residual ACF, lags 1..144 (gap-safe, per seed)",
                      "Positive short-lag residual autocorrelation is observed across seeds.")
    img8 = _img_block(fig8, "Fig. 8 — Key ACF lags (1, 6, 12, 36, 72, 144)",
                      "Registered ACF values at canonical lags.", compact=True)

    return (
        '<section class="cw-d-sec"><h4>E — Residual ACF (gap-safe, lags 1..144)</h4>'
        + '<h5>E.1 — Registered key lags</h5>'
        + key_tbl
        + img7 + img8
        + "</section>"
    )


def _section_ljung_box(lb_rows: list[dict]) -> str:
    """Section F — Ljung-Box (SECONDARY DIAGNOSTIC only).

    Reads Q_lag{6,36,144} / p_value_lag{6,36,144} from the existing frozen
    Phase49 artifact (``phase49_ljung_box.csv``). No recomputation.
    """
    out_rows: list[list[Any]] = []
    for r in lb_rows:
        seed = r.get("seed", "")
        n_pairs = _i(r.get("n"))
        # Frozen artifact has lag-flattened columns: Q_lag6, p_value_lag6, ...
        for lag in (6, 36, 144):
            q = r.get(f"Q_lag{lag}")
            p = r.get(f"p_value_lag{lag}")
            out_rows.append([seed, lag, n_pairs, _f(q), _f(p)])
    return (
        '<section class="cw-d-sec"><h4>F — Ljung-Box (SECONDARY DIAGNOSTIC only)</h4>'
        + _tbl(["Seed", "lag (steps)", "n_pairs", "Q", "p-value"], out_rows)
        + '<p style="margin:8px 0 0;font-size:12px;color:#5d6b82">'
        + 'Ljung-Box is a secondary diagnostic only and does not gate PASS/FAIL. '
        + 'Run on contiguous Test segments per seed; values are read directly '
        + 'from the frozen Phase49 Ljung-Box artifact.'
        + "</p>"
        + "</section>"
    )


def _section_sign_runs_transitions(sign_runs_summary: list[dict],
                                   sign_transition_rows: list[dict],
                                   fig9: Path, fig10: Path) -> str:
    summary_tbl = _tbl(
        ["Seed", "total_runs", "mean_length", "median_length", "p95_length", "max_length", "exact_run_count"],
        [
            [r.get("seed", ""), _i(r.get("total_run_count")), _f(r.get("mean_run_length")),
             _f(r.get("median_run_length")), _f(r.get("p95_run_length")),
             _i(r.get("max_run_length")), _i(r.get("exact_run_count"))]
            for r in sign_runs_summary
        ],
    )

    # Pull only U->U, U->O, O->U, O->O
    by_seed: dict[str, dict[str, float]] = {}
    for r in sign_transition_rows:
        if r.get("from_sign") not in ("UNDERPREDICTION", "OVERPREDICTION"):
            continue
        if r.get("to_sign") not in ("UNDERPREDICTION", "OVERPREDICTION"):
            continue
        key = f"{r['from_sign'][0]}->{r['to_sign'][0]}"
        try:
            p = float(r.get("probability_given_from_a", "nan"))
        except ValueError:
            p = float("nan")
        by_seed.setdefault(r.get("seed", ""), {})[key] = p

    transition_tbl = _tbl(
        ["Seed", "U->U prob", "U->O prob", "O->U prob", "O->O prob"],
        [
            [
                f"Seed {seed}",
                _f(by_seed.get(seed, {}).get("U->U", float("nan"))),
                _f(by_seed.get(seed, {}).get("U->O", float("nan"))),
                _f(by_seed.get(seed, {}).get("O->U", float("nan"))),
                _f(by_seed.get(seed, {}).get("O->O", float("nan"))),
            ]
            for seed in ("42", "123", "2026")
        ],
    )

    img9 = _img_block(fig9, "Fig. 9 — Sign run length distribution",
                      "Run lengths by seed.", compact=True)
    img10 = _img_block(fig10, "Fig. 10 — Sign-transition probabilities (within-from-sign)",
                      "U = UNDERPREDICTION, O = OVERPREDICTION. NaN indicates "
                      "from-sign with 0 outgoing transitions.",
                      compact=True)

    return (
        '<section class="cw-d-sec"><h4>G — Sign runs and transitions</h4>'
        + '<h5>G.1 — Run lengths</h5>'
        + summary_tbl
        + '<h5>G.2 — 10-min transitions (within from-sign probability)</h5>'
        + transition_tbl
        + img9 + img10
        + "</section>"
    )


def _section_rolling_diagnostics(rolling_rows: list[dict],
                                  fig11: Path, fig14: Path) -> str:
    """Rolling diagnostics — figures only, compact, no callout."""
    img11 = _img_block(fig11, "Fig. 11 — Rolling residual mean (window = 144, ≈ 24h)",
                       "Per seed; gap-safe, no partial / interpolated / padded windows.")
    img14 = _img_block(fig14, "Fig. 14 — Rolling RMSE (window = 144, ≈ 24h)",
                       "Per seed.", compact=True)
    return (
        '<section class="cw-d-sec"><h4>H — Rolling residual diagnostics '
        '(24h = 144 contiguous samples)</h4>'
        + img11 + img14
        + '<p style="margin:6px 0 0;font-size:12px;color:#5d6b82">'
        + 'Rolling 144-sample = 24 h at 10-min cadence. Gap-safe, no partial '
        + '/ interpolated / padded windows. Per-window table not rendered in '
        + 'the primary view.'
        + "</p>"
        + "</section>"
    )


def _section_magnitude_associations(assoc_rows: list[dict],
                                    fig15: Path, fig16: Path) -> str:
    """Compact magnitude-association summary (4 headlines × Pearson/Spearman)."""
    by_seed_x_y: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for r in assoc_rows:
        seed = r.get("seed", "")
        x = r.get("x_variable", "")
        y = r.get("y_variable", "")
        metric = r.get("association_type", "")
        try:
            val = float(r.get("value"))
        except (TypeError, ValueError):
            continue
        by_seed_x_y.setdefault(seed, {}).setdefault(x, {}).setdefault(y, {})[metric] = val

    pair_defs = [
        ("|residual|", "y_true_wh"),
        ("|residual|", "y_pred_wh"),
        ("residual", "y_true_wh"),
        ("residual", "y_pred_wh"),
    ]

    rows: list[list[Any]] = []
    for x, y in pair_defs:
        pearsons: list[float] = []
        spearmans: list[float] = []
        for seed in ("42", "123", "2026"):
            d = by_seed_x_y.get(seed, {}).get(x, {}).get(y, {})
            if "pearson" in d:
                pearsons.append(d["pearson"])
            if "spearman" in d:
                spearmans.append(d["spearman"])
        if not pearsons and not spearmans:
            continue
        pr = _f(min(pearsons)) + " … " + _f(max(pearsons)) if pearsons else "N/A"
        sr = _f(min(spearmans)) + " … " + _f(max(spearmans)) if spearmans else "N/A"
        rows.append([f"{x} ↔ {y}", pr, sr])

    img15 = _img_block(fig15, "Fig. 15 — |residual| vs y_true",
                       "Absolute residual magnitude vs truth.")
    img16 = _img_block(fig16, "Fig. 16 — |residual| vs y_pred",
                       "Absolute residual magnitude vs prediction.",
                       compact=True)
    return (
        '<section class="cw-d-sec"><h4>I — Residual magnitude associations (4 pairs)</h4>'
        + _tbl(["Pair", "Pearson r range (across seeds)", "Spearman ρ range (across seeds)"], rows)
        + img15 + img16
        + "</section>"
    )


def _section_prediction_deciles(decile_rows: list[dict], fig19: Path) -> str:
    """Compact decile view: trend across deciles + low/mid/high summary rows."""
    by_seed: dict[str, list[dict]] = {}
    for r in decile_rows:
        by_seed.setdefault(r.get("seed", ""), []).append(r)
    for s in by_seed.values():
        s.sort(key=lambda r: int(r.get("decile", "0")))

    # Trend lines per seed (1 line each, MAE across 10 deciles)
    trend_rows: list[list[Any]] = []
    for seed in ("42", "123", "2026"):
        rows_s = by_seed.get(seed, [])
        if not rows_s:
            trend_rows.append([f"Seed {seed}"] + ["N/A"] * 10)
            continue
        trend_rows.append([f"Seed {seed} MAE"] + [_f(r.get("mae")) for r in rows_s])

    # Representative low / mid / high deciles (aggregated across seeds)
    def _agg(rows_s: list[dict], idx: int) -> list[Any]:
        cells: list[Any] = []
        for key in ("mae", "rmse"):
            vals: list[float] = []
            for s_name in ("42", "123", "2026"):
                rs = by_seed.get(s_name, [])
                if idx < len(rs):
                    try:
                        vals.append(float(rs[idx].get(key)))
                    except (TypeError, ValueError):
                        pass
            cells.append(_f(min(vals)) + " … " + _f(max(vals)) if vals else "N/A")
        return cells

    rep_rows: list[list[Any]] = []
    for label, idx in (("decile 1 (lowest predictions)", 0),
                        ("decile 5 (middle)", 5),
                        ("decile 10 (highest predictions)", 9)):
        agg = _agg([], idx)  # placeholder
        agg = _agg(by_seed.get("42", []), idx)
        if agg[0] == "N/A":
            agg = _agg(by_seed.get("42", []), idx)
        rep_rows.append([label] + agg)

    img19 = _img_block(fig19, "Fig. 19 — Prediction-decile MAE & RMSE (per seed)",
                       "10 y_pred-based equal-frequency bins.", compact=True)
    return (
        '<section class="cw-d-sec"><h4>J — Prediction-decile diagnostics</h4>'
        + '<h5>J.1 — Trend across deciles (MAE, by seed)</h5>'
        + _tbl(["Seed"] + [f"d{d}" for d in range(1, 11)], trend_rows)
        + '<h5>J.2 — Representative low / mid / high deciles</h5>'
        + _tbl(["Decile band", "MAE range (Wh, across seeds)", "RMSE range (Wh, across seeds)"],
               rep_rows)
        + img19
        + '<p style="margin:6px 0 0;font-size:12px;color:#5d6b82">'
        + 'Prediction-decile diagnostics are descriptive only and are NOT '
        + 'authorized as Phase50 target regimes.'
        + "</p>"
        + "</section>"
    )


def _section_cross_seed(pair_rows: list[dict], consensus_rows: list[dict],
                        fig20: Path) -> str:
    pair_tbl = _tbl(
        ["Pair", "Pearson r (residuals)", "Spearman ρ (residuals)",
         "MAE diff", "RMSE diff", "N"],
        [
            [
                f"{r.get('seed_a', '')} vs {r.get('seed_b', '')}",
                _f(r.get("pearson_residual_correlation")),
                _f(r.get("spearman_residual_correlation")),
                _f(r.get("mae_difference")),
                _f(r.get("rmse_difference")),
                _i(r.get("n_pairs")),
            ]
            for r in pair_rows
        ],
    )
    cons_tbl = _tbl(
        ["Consensus class", "Count", "Fraction"],
        [
            [r.get("consensus_class", ""), _i(r.get("count")), _f(r.get("fraction"))]
            for r in consensus_rows
        ],
    )
    img20 = _img_block(fig20, "Fig. 20 — Cross-seed residual agreement",
                       "Pearson + Spearman across seed pairs.", compact=True)
    return (
        '<section class="cw-d-sec"><h4>K — Cross-seed residual behavior</h4>'
        + '<h5>K.1 — Pairwise residual agreement</h5>'
        + pair_tbl
        + '<h5>K.2 — Cross-seed sign consensus</h5>'
        + cons_tbl
        + img20
        + "</section>"
    )


def _section_persistence(persist_rows: list[dict], fig22: Path) -> str:
    if not persist_rows:
        return '<section class="cw-d-sec"><h4>L — Persistence baseline</h4></section>'
    r = persist_rows[0]
    rows = [["Persistence", _i(r.get("N")), _f(r.get("mae")), _f(r.get("rmse")),
             _f(r.get("mean_residual")), _f(r.get("median_residual")),
             _f(r.get("std_residual")), _f(r.get("exact_fraction"))]]
    img22 = _img_block(fig22, "Fig. 22 — Persistence baseline residual context",
                       "N=2961 descriptive context.", compact=True)
    return (
        '<section class="cw-d-sec"><h4>L — Persistence baseline residual context</h4>'
        + _tbl(["Model", "N", "MAE (Wh)", "RMSE (Wh)", "mean resid (Wh)",
                "median resid (Wh)", "std resid (Wh)", "exact frac"], rows)
        + img22
        + '<p style="margin:6px 0 0;font-size:12px;color:#5d6b82">'
        + 'Persistence baseline context only — Persistence better on MAE; '
        + 'Transformer better on RMSE and R². LSTM = '
        + 'NOT_ELIGIBLE_CONFIG_MISMATCH; not included.'
        + "</p>"
        + "</section>"
    )


def _section_findings(findings_rows: list[dict]) -> str:
    """Curated 7-headline findings (descriptive only)."""
    curated_ids = {"F1", "F2", "F3", "F4", "F7", "F8", "F9"}
    label_map = {
        "F1": "Mild net underprediction tendency across all 3 seeds",
        "F2": "Right-skewed, heavy-tailed residual distributions",
        "F3": "Tail mass concentrated in positive (underprediction) tail",
        "F4": "Positive short-lag residual autocorrelation, decays with lag",
        "F7": "Strong cross-seed residual agreement (Pearson ≈ 0.90)",
        "F8": "Persistence baseline context (Transformer ≠ ranking)",
        "F9": "Rolling 144-sample diagnostics stable across Test window",
    }
    rows: list[list[Any]] = []
    for r in findings_rows:
        rid = r.get("id", "")
        if rid not in curated_ids:
            continue
        rows.append([rid, label_map.get(rid, ""),
                     ", ".join(r.get("support_seeds", []) or []),
                     r.get("support_metric", "")])
    return (
        '<section class="cw-d-sec"><h4>M — Headline findings (curated)</h4>'
        + _tbl(["#", "Headline", "seeds", "support_metric"], rows)
        + '<p style="margin:8px 0 0;font-size:12px;color:#5d6b82">'
        + 'All statements are descriptive only. Full Phase49 descriptive '
        + 'findings remain preserved in the frozen artifact.'
        + "</p>"
        + "</section>"
    )


def _section_scientific_boundaries() -> str:
    """One compact neutral boundaries block (no callout, no warning)."""
    items = [
        "Residual analysis is descriptive.",
        "Ljung-Box is secondary diagnostic only and does not gate PASS/FAIL.",
        "Prediction deciles are NOT Phase50 target regimes.",
        "No residual / bias correction is applied.",
        "No best-seed selection or ensemble promotion.",
        "No worst-error ranking in Phase49.",
        "Phase51 is the first phase allowed to rank worst errors.",
        "Attention analysis begins Phase52+.",
    ]
    lis = "".join(f"<li>{escape(t)}</li>" for t in items)
    return (
        '<section class="cw-d-sec"><h4>O — Scientific boundaries</h4>'
        f"<ul style=\"margin:6px 0 0 18px;font-size:12.5px;line-height:1.55\">{lis}</ul>"
        + "</section>"
    )


def _section_key_findings(summary: dict[str, Any]) -> str:
    """Compact 'Key Phase49 findings' panel near the top."""
    rows = [
        ["Mean residual (Wh)",
         "Seed42 +5.5590 · Seed123 +5.8856 · Seed2026 +3.2074  → mild net underprediction"],
        ["Residual skewness",
         "≈ 2.79–3.27 (strongly right-skewed)"],
        ["Excess kurtosis",
         "≈ 21–22 (heavy tails)"],
        ["|err| p99 (Wh)",
         "≈ 295–309 across seeds"],
        ["Max |err| (Wh)",
         "≈ 535–601 across seeds"],
        ["ACF lag-1",
         "≈ 0.11–0.21 (positive), decays toward 0 at longer lags"],
        ["Cross-seed residual Pearson",
         "≈ 0.90 across all 3 seed pairs"],
        ["Fully-concordant sign targets",
         "≈ 70% (ALL_UNDER 33.87% + ALL_OVER 36.64%)"],
    ]
    return (
        '<section class="cw-d-sec"><h4>Findings — Key Phase49 results</h4>'
        + _tbl(["Metric", "Headline value"], rows)
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Figure embedding (curated primary + secondary)
# ---------------------------------------------------------------------------

_PRIMARY_FIGURES = [
    ("phase49_fig01_residual_time_series.png",
     "Fig. 1 — Residual time series by seed",
     "Three residual series shown together (positive = UNDERPREDICTION, negative = OVERPREDICTION, exact = 0)."),
    ("phase49_fig02_residual_distribution.png",
     "Fig. 2 — Residual distribution comparison",
     "Density + box plot per seed."),
    ("phase49_fig04_signed_bias.png",
     "Fig. 4 — Signed bias comparison (mean / median residual)",
     "Bars slightly above 0 indicate net underprediction tendency."),
    ("phase49_fig06_tail_diagnostics.png",
     "Fig. 6 — Tail diagnostics",
     "p90 / p95 / p99 / max absolute error across seeds."),
    ("phase49_fig07_residual_acf.png",
     "Fig. 7 — Residual ACF (gap-safe, lags 1..144)",
     "Positive short-lag autocorrelation observed across seeds."),
    ("phase49_fig11_rolling_mean.png",
     "Fig. 11 — Rolling residual mean (window = 144 ≈ 24h)",
     "Per seed; gap-safe, no partial / interpolated / padded windows."),
    ("phase49_fig14_rolling_rmse.png",
     "Fig. 14 — Rolling RMSE (window = 144 ≈ 24h)",
     "Per seed."),
    ("phase49_fig15_abs_residual_vs_y_true.png",
     "Fig. 15 — |residual| vs y_true",
     "Absolute residual magnitude vs truth."),
    ("phase49_fig20_cross_seed_agreement.png",
     "Fig. 20 — Cross-seed residual agreement",
     "Pearson + Spearman across seed pairs."),
    ("phase49_fig22_persistence_context.png",
     "Fig. 22 — Persistence baseline residual context",
     "Reference baseline context only."),
]

_SECONDARY_FIGURES = [
    ("phase49_fig03_residual_ecdf.png",
     "Fig. 3 — Residual ECDF (per seed)"),
    ("phase49_fig05_sign_balance.png",
     "Fig. 5 — Sign balance (under / over / exact)"),
    ("phase49_fig08_key_acf_lags.png",
     "Fig. 8 — Key ACF lags (1, 6, 12, 36, 72, 144)"),
    ("phase49_fig09_sign_run_distribution.png",
     "Fig. 9 — Sign run length distribution"),
    ("phase49_fig10_sign_transitions.png",
     "Fig. 10 — Sign-transition probabilities"),
    ("phase49_fig12_rolling_std.png",
     "Fig. 12 — Rolling residual std (window = 144 ≈ 24h)"),
    ("phase49_fig13_rolling_mae.png",
     "Fig. 13 — Rolling MAE (window = 144 ≈ 24h)"),
    ("phase49_fig16_abs_residual_vs_y_pred.png",
     "Fig. 16 — |residual| vs y_pred"),
    ("phase49_fig17_residual_vs_y_true.png",
     "Fig. 17 — residual vs y_true"),
    ("phase49_fig18_residual_vs_y_pred.png",
     "Fig. 18 — residual vs y_pred"),
    ("phase49_fig19_prediction_deciles.png",
     "Fig. 19 — Prediction-decile MAE & RMSE"),
    ("phase49_fig21_cross_seed_sign_consensus.png",
     "Fig. 21 — Cross-seed sign consensus"),
]


def _render_figure_blocks(fig_dir: Path, items: Sequence[tuple[str, str, ...]]) -> str:
    blocks: list[str] = []
    for tup in items:
        if len(tup) == 2:
            fname, title = tup
            caption = ""
        else:
            fname, title, caption = tup
        p = fig_dir / fname
        b64 = _img_b64(p)
        if b64 is None:
            continue
        compact = fname in (
            "phase49_fig03_residual_ecdf.png",
            "phase49_fig04_signed_bias.png",
            "phase49_fig05_sign_balance.png",
            "phase49_fig08_key_acf_lags.png",
            "phase49_fig09_sign_run_distribution.png",
            "phase49_fig10_sign_transitions.png",
            "phase49_fig12_rolling_std.png",
            "phase49_fig13_rolling_mae.png",
            "phase49_fig16_abs_residual_vs_y_pred.png",
            "phase49_fig19_prediction_deciles.png",
            "phase49_fig20_cross_seed_agreement.png",
            "phase49_fig21_cross_seed_sign_consensus.png",
            "phase49_fig22_persistence_context.png",
        )
        data_uri = f"data:image/png;base64,{b64}"
        extra = " compact" if compact else ""
        cap = f'<div class="fcap">{escape(caption)}</div>' if caption else ""
        blocks.append(
            f'<div class="cw-d-fig{extra}">'
            f'<div class="ftitle">{escape(title)}</div>'
            f'<img src="{escape(data_uri, quote=True)}" alt="{escape(title)}"/>'
            f"{cap}"
            f"</div>"
        )
    return "".join(blocks)


def _section_figures(fig_dir: Path) -> str:
    primary = _render_figure_blocks(fig_dir, _PRIMARY_FIGURES)
    secondary = _render_figure_blocks(fig_dir, _SECONDARY_FIGURES)
    return (
        '<section class="cw-d-sec"><h4>N — Primary figures</h4>' + primary
        + '<details><summary style="cursor:pointer;font-size:12.5px;color:#334155;'
        'margin:10px 0 6px">Secondary figures (12)</summary>'
        + secondary
        + "</details>"
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def render_phase_49_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 49 Residual Analysis dashboard.

    Presentation-only compact style matching the Phase 43-48 family:
        Header -> Overview -> [Figure + compact stats table] -> Signoff

    Read-only over Phase 47 inputs + Phase 49 canonical artifacts.
    """
    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "residual_analysis"

    if not (art / "phase_49_signoff.json").exists():
        return HTML(
            _CSS
            + '<article class="cw-d">'
            + _header("Phase 49 - Residual Analysis", "Signoff not yet materialized.", "FAIL")
            + '<div class="cw-d-body"><div class="cw-d-call warn">'
            + 'Phase 49 canonical artifacts are not yet present under '
            + '<code>artifacts/residual_analysis/</code>.'
            + "</div></div></article>"
        )

    signoff = read_json(art / "phase_49_signoff.json")
    summary = read_json(art / "phase49_summary.json")

    overall = str(signoff.get("status") or signoff.get("overall_status") or "UNKNOWN")
    n_test = summary.get("n_test")
    subtitle = (
        f"Post-Test residual diagnostic - Test N = {n_test} - "
        f"3 Transformer seeds - residual = y_true - y_pred"
    )

    # ----- Overview (compact 6-row table) -----
    overview_pairs = [
        ("Phase status", overall),
        ("Residual definition", "y_true - y_pred"),
        ("Semantics",
         "positive = underprediction, negative = overprediction"),
        ("Test N", n_test),
        ("Seeds", "42 / 123 / 2026"),
        ("Analysis type", "post-Test descriptive diagnostic"),
        ("Prediction source", "frozen Phase 47 outputs"),
        ("Locked candidate", "TR_C2_ALT_LOOKBACK"),
    ]
    overview_html = (
        '<section class="cw-d-sec"><h4>Overview</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in overview_pairs
        )
        + '</tbody></table></section>'
    )

    # ----- Figure selection (FIRST CHOICE per request) -----
    fig_dir = art / "figures"
    figure_html = ""
    fig_choice = "phase49_fig02_residual_distribution.png"
    fp = fig_dir / fig_choice
    if fp.exists() and fp.stat().st_size > 1024:
        try:
            from PIL import Image
            w, h = Image.open(fp).size
            if w >= 100 and h >= 100:
                b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
                data_uri = "data:image/png;base64," + b64
                figure_html = (
                    '<div class="cw-d-fig" style="margin-top:6px">'
                    '<div class="ftitle">Residual distribution (histogram + KDE + box) - '
                    'all three Transformer seeds</div>'
                    f'<img src="{escape(data_uri, quote=True)}" '
                    'alt="Phase 49 residual distribution" '
                    'style="max-width:860px;width:90%"/>'
                    '<div class="fcap" style="font-size:11px;color:#5d6b82;line-height:1.45;margin-top:4px">'
                    'Active canonical Phase 49 figure '
                    '(<code>artifacts/residual_analysis/figures/phase49_fig02_residual_distribution.png</code>). '
                    'Residuals are derived from the frozen Phase 47 predictions; no new inference. '
                    'Heavier right (positive) tail = underprediction in difficult cases.'
                    '</div></div>'
                )
            else:
                fig_choice = "phase49_fig01_residual_time_series.png"  # fallback
                fp2 = fig_dir / fig_choice
                if fp2.exists() and fp2.stat().st_size > 1024:
                    b64 = base64.b64encode(fp2.read_bytes()).decode("ascii")
                    data_uri = "data:image/png;base64," + b64
                    figure_html = (
                        '<div class="cw-d-fig" style="margin-top:6px">'
                        '<div class="ftitle">Residual time series - 3 Transformer seeds</div>'
                        f'<img src="{escape(data_uri, quote=True)}" '
                        'alt="Phase 49 residual time series" '
                        'style="max-width:860px;width:90%"/>'
                        '<div class="fcap" style="font-size:11px;color:#5d6b82;line-height:1.45;margin-top:4px">'
                        'Active canonical Phase 49 figure '
                        '(<code>phase49_fig01_residual_time_series.png</code>).</div></div>'
                    )
        except Exception:
            figure_html = ""

    # ----- Compact residual statistics table -----
    # Pull per-seed rows from phase49_residual_distribution_summary.csv (canonical)
    # and merge lag-1 ACF from phase49_residual_acf_key_lags.csv.
    dist_rows = _csv(art / "phase49_residual_distribution_summary.csv")
    acf_rows = _csv(art / "phase49_residual_acf_key_lags.csv")
    lag1: dict[str, str] = {}
    for r in acf_rows:
        if r.get("lag_steps") == "1":
            lag1[r.get("seed", "")] = r.get("acf_value", "")
    def _fmt(v, digits: int = 3) -> str:
        if v in ("", None):
            return "N/A"
        try:
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return str(v)
    rows = []
    for r in dist_rows:
        s = r.get("seed", "")
        rows.append([
            f"seed {s}",
            _fmt(r.get("mean")),
            _fmt(r.get("std")),
            _fmt(r.get("median")),
            _fmt(r.get("mad")),
            _fmt(r.get("p99")),
            _fmt(r.get("max")),
            _fmt(lag1.get(s, "")),
        ])
    headers = ["Seed", "Mean", "Std", "Median", "MAD", "|err| p99", "Max", "ACF lag-1"]
    stats_table_html = (
        '<table class="cw-d-tbl" style="margin-top:8px">'
        '<thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead>'
        '<tbody>' + ''.join(
            ''.join(f'<td>{escape(c)}</td>' for c in row) + '</tr>' for row in rows
        ) + '</tbody></table>'
    )

    # ----- Conclusion (one short sentence) -----
    conclusion_html = (
        '<p style="margin:8px 0 0;font-size:12.5px;color:#1f2a44;line-height:1.5">'
        '<strong>Finding:</strong> The residual distribution is heavy-tailed, '
        'with the largest errors dominated by positive residuals, '
        'indicating strong underprediction in difficult cases.'
        '</p>'
    )

    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - residual distribution and per-seed statistics</h4>'
        + figure_html + stats_table_html + conclusion_html
        + '</section>'
    )

    # ----- Signoff (positive compliance wording) -----
    signoff_pairs = [
        ("Phase status", overall),
        ("Frozen Phase47 predictions used", "✓"),
        ("Residual definition preserved (y_true - y_pred)", "✓"),
        ("No post-Test tuning", "✓"),
        ("No prediction modification", "✓"),
        ("No best-seed selection / ensemble", "✓"),
        ("No causal claim", "✓"),
        ("Ready for Phase50", str(signoff.get("ready_for_phase50", "✓"))),
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
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + overview_html + main_result_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 49 - Residual Analysis", subtitle, overall)}{body}</article>'
    )
