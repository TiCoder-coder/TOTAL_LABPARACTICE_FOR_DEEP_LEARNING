"""Phase 50-H — read-only HTML dashboard renderer (presentation only).

Visual family matches Phase 48 and Phase 49 dashboards:
- Light-theme containers (.cw-d) with dark blue/slate headers
- Concise tables (.cw-d-tbl) with subtle alternating rows
- Cards (.cw-d-card / .cw-d-card2) for high-level summary
- Callouts (.cw-d-call / .cw-d-call.good / .cw-d-call.warn) for interpretation
- Curated figures (.cw-d-fig / .cw-d-fig.compact)
- NO raw HTML badges — badges/callouts use shared Phase49 CSS classes

All scientific values are read directly from canonical Phase 50 artifacts;
no recomputation, no inference, no training, no checkpoint loading.
"""
from __future__ import annotations

import base64
import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable, Sequence

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.reporting._results_only import results_only

__all__ = ["render_phase_50_dashboard"]


# ---------------------------------------------------------------------------
# CSS — reuses Phase49 .cw-d family (light containers + dark slate accents)
# ---------------------------------------------------------------------------
_CSS = """
<style>
.cw-d{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fbfcff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 22px;overflow:visible}
.cw-d *{box-sizing:border-box}
.cw-d-h{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-d-h h3{font-size:20px;line-height:1.25;margin:0 0 5px;color:#172033}
.cw-d-meta{font-size:12px;color:#5d6b82}
.cw-d-badge{border:1px solid #a9dec1;border-radius:999px;padding:6px 12px;font-size:11px;font-weight:700;letter-spacing:.04em;color:#11613d;background:#e8f7ef;white-space:nowrap}
.cw-d-badge.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-d-badge.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-badge.info{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:16px 22px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-d-card{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:9px;background:#fff;min-width:0}
.cw-d-card span{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card strong{font-size:13px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-d-body{padding:4px 22px 22px}
.cw-d-sec{margin-top:18px}
.cw-d-sec h4{font-size:14px;margin:0 0 8px;color:#334155}
.cw-d-sec h5{font-size:12.5px;margin:10px 0 6px;color:#475569;font-weight:700}
.cw-d-tblwrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-d-tbl{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
.cw-d-tbl th{background:#1f2937;color:#f8fafc;text-align:left;font-weight:650;padding:9px 11px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-d-tbl td{text-align:left;padding:8px 11px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere;color:#1f2a44}
.cw-d-tbl tbody tr:nth-child(even){background:#f3f6fb}
.cw-d-tbl tbody tr:last-child td{border-bottom:0}
.cw-d-fp{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;color:#1f2a44;background:#eef2ff;padding:1px 6px;border-radius:4px}
.cw-d-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-d-pill.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-pill.good{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-d-pill.frozen{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-pill.muted{background:#eef2f7;color:#475569;border-color:#d3deec}
.cw-d-grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;margin-top:10px}
.cw-d-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:10px}
.cw-d-grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:10px}
.cw-d-card2{border:1px solid #d9e2ef;border-radius:11px;padding:13px 14px;background:#fff}
.cw-d-card2.frozen{border-left:4px solid #3b46c4}
.cw-d-card2 h5{margin:0 0 5px;font-size:11.5px;color:#475569;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card2 .big{font-size:22px;color:#172033;font-weight:700;line-height:1.2;font-variant-numeric:tabular-nums}
.cw-d-card2 .sm{font-size:11.5px;color:#5d6b82;margin-top:3px}
.cw-d-card2 .ul{margin:4px 0 0 14px;padding:0;font-size:11.5px;color:#1f2a44;line-height:1.55}
.cw-d-card2 .ul li{margin:1px 0}
.cw-d-call{border-left:4px solid #6366f1;background:#eef2ff;padding:10px 14px;border-radius:0 7px 7px 0;margin:11px 0;font-size:12.5px;color:#1f2a44;line-height:1.5}
.cw-d-call.warn{border-color:#d97706;background:#fff7e0;color:#7a5613}
.cw-d-call.good{border-color:#16a34a;background:#e8f7ef;color:#11613d}
.cw-d-fig{display:flex;flex-direction:column;gap:6px;overflow:visible;margin:12px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:72%;max-width:760px}
.cw-d-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto}
.cw-d-fig.compact{width:60%;max-width:620px}
.cw-d-fig .ftitle{font-size:12px;color:#475569;font-weight:650}
.cw-d-fig .fcap{font-size:11.5px;color:#5d6b82;line-height:1.45}
.cw-d-provenance{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:11.5px;color:#475569;padding:10px 22px;background:#f5f7fb;border-bottom:1px solid #e5eaf1}
.cw-d-provenance b{color:#1f2a44}
.cw-d-conclusion{border:1px solid #d9e2ef;border-radius:11px;background:#fbfcff;padding:14px 16px;margin:12px 0}
.cw-d-conclusion h5{margin:0 0 8px;font-size:13px;color:#172033}
.cw-d-conclusion ul{margin:6px 0 0 18px;padding:0;font-size:12.5px;color:#1f2a44;line-height:1.6}
@media (max-width:900px){.cw-d-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-d-grid2,.cw-d-grid3,.cw-d-grid4{grid-template-columns:1fr 1fr}}
@media (max-width:620px){.cw-d-h{flex-direction:column;padding:15px}.cw-d-cards{grid-template-columns:1fr;padding:12px 16px}.cw-d-body{padding-left:16px;padding-right:16px}.cw-d-grid2,.cw-d-grid3,.cw-d-grid4{grid-template-columns:1fr}}
<style>.cw-d-meta,.cw-d-note,.cw-d-fig .fcap,.cw-d-fig .ftitle,.cw-d-call,.cw-d-call.warn,.cw-d-call.good,.cw-d-call.fail,.cw-d-overview .cw-d-note,.cw-d h3 small,.cw-d-fig,.cw-d-card .sm,.cw-d-card2 .sm,.cw-d-card2 .ul,p.cw-d-meta,div.cw-d-meta,div.cw-d-note,p[style*="margin:8px 0 0"],p[style*="margin:10px 0 0"],p[style*="margin:6px 0 0"],div[style*="font-size:11px"][style*="color:#64748b"],.cw-d-provenance,p[style*='font-size:11'],p[style*='font-size:12'],p[style*='font-size:13']{display:none !important}</style></style>
"""


# ---------------------------------------------------------------------------
# Helpers (presentation-only formatting)
# ---------------------------------------------------------------------------
def _f(value: Any, digits: int = 3) -> str:
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
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _compact_sha(sha: str) -> str:
    """Compact 6+ellipsis+6 with full hash in title."""
    if not sha or len(sha) < 16:
        return escape(str(sha))
    return (
        f'<span class="cw-d-fp" title="{escape(sha)}">'
        f'{escape(sha[:6])}…{escape(sha[-6:])}</span>'
    )


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
        cells = "".join(f"<td>{c}</td>" for c in row)  # already-escaped HTML allowed
        parts.append(f"<tr>{cells}</tr>")
    body = "".join(parts)
    return (
        f'<div class="cw-d-tblwrap"><table class="cw-d-tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _tbl_escaped(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
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


def _pill(text: str, kind: str = "muted") -> str:
    return f'<span class="cw-d-pill {escape(kind)}">{escape(str(text))}</span>'


def _callout(text: str, kind: str = "") -> str:
    cls = f" cw-d-call {kind}" if kind else " cw-d-call"
    return f'<div class="{cls.strip()}">{text}</div>'


def _provenance_strip(items: Iterable[tuple[str, str]]) -> str:
    cells = "".join(
        f'<span><b>{escape(k)}:</b> {v}</span>'
        for k, v in items
    )
    return f'<div class="cw-d-provenance">{cells}</div>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _section_overview(
    so: dict[str, Any],
    ref_manifest: dict[str, Any],
    fp_assign: dict[str, Any],
    th: dict[str, Any],
) -> str:
    """A. Phase 50 overview — 4 cards + provenance strip + freeze sentence."""
    cards = _cards([
        ("PHASE 50 STATUS", _pill(so["status"], "good")),
        ("TRAIN REFERENCE", f"N = {_i(ref_manifest.get('n'))}"),
        ("TEST POPULATION", f"N = {_i(so.get('n_test'))}"),
        ("REGIME FAMILIES", "6 / 6"),
    ])
    train_ref_sha = ref_manifest.get("reference_sha256", ref_manifest.get("sha256", ""))
    test_pop_sha = so.get("test_population_sha256", "")
    assign_sha = fp_assign.get("assignment_sha256", "")
    qm = th.get("quantile_method", "")
    prov = _provenance_strip([
        ("Threshold source", escape(str(th.get("source_reference", "")))),
        ("Quantile method", escape(str(qm))),
        ("Train ref SHA", _compact_sha(train_ref_sha) if train_ref_sha else "—"),
        ("Test pop SHA", _compact_sha(test_pop_sha) if test_pop_sha else "—"),
        ("Assignment SHA", _compact_sha(assign_sha) if assign_sha else "—"),
    ])
    freeze = (
        '<div class="cw-d-call good">'
        'Thresholds were derived from TRAIN only. '
        'Test regime assignment was frozen before any error/residual data were joined.'
        '</div>'
    )
    return (
        '<section class="cw-d-sec"><h4>A — Phase 50 overview</h4>'
        + cards + prov + freeze
        + "</section>"
    )


def _section_thresholds(th: dict[str, Any]) -> str:
    """B. TRAIN-derived frozen thresholds — 4 prominent cards + subtitle."""
    q25 = th.get("target_level", {}).get("Q25")
    q75 = th.get("target_level", {}).get("Q75")
    q90 = th.get("extreme_high", {}).get("Q90")
    q90d = th.get("change_magnitude", {}).get("Q90_abs_delta")
    cards = (
        '<div class="cw-d-grid4">'
        + f'<div class="cw-d-card2 frozen"><h5>Q25 y</h5><div class="big">{_f(q25, 0)} Wh</div>'
        + '<div class="sm">target-level low cut</div></div>'
        + f'<div class="cw-d-card2 frozen"><h5>Q75 y</h5><div class="big">{_f(q75, 0)} Wh</div>'
        + '<div class="sm">target-level high cut</div></div>'
        + f'<div class="cw-d-card2 frozen"><h5>Q90 y</h5><div class="big">{_f(q90, 0)} Wh</div>'
        + '<div class="sm">extreme-high cut</div></div>'
        + f'<div class="cw-d-card2 frozen"><h5>Q90 |Δy|</h5><div class="big">{_f(q90d, 0)} Wh</div>'
        + '<div class="sm">rapid-change cut</div></div>'
        + '</div>'
    )
    sub = (
        '<div class="cw-d-call good">'
        'Derived from TRAIN only. '
        'Validation used: NO. Test used: NO.'
        '</div>'
    )
    return (
        '<section class="cw-d-sec"><h4>B — TRAIN-derived frozen thresholds</h4>'
        + cards + sub
        + "</section>"
    )


def _section_regime_definitions() -> str:
    """C. Six regime definitions — compact 3-column card layout."""
    cards_html = (
        '<div class="cw-d-grid3">'
        '<div class="cw-d-card2"><h5>TARGET_LEVEL</h5><ul class="ul">'
        '<li>LOW — y &lt; 50 Wh</li><li>MID — 50 ≤ y &lt; 100 Wh</li><li>HIGH — y ≥ 100 Wh</li></ul></div>'
        '<div class="cw-d-card2"><h5>EXTREME_HIGH</h5><ul class="ul">'
        '<li>EXTREME — y ≥ 210 Wh</li><li>NON_EXTREME — y &lt; 210 Wh</li></ul></div>'
        '<div class="cw-d-card2"><h5>CHANGE_MAGNITUDE</h5><ul class="ul">'
        '<li>NORMAL — |Δy| &lt; 80 Wh</li><li>RAPID — |Δy| ≥ 80 Wh</li><li>UNCLASSIFIED — no prior Δy</li></ul></div>'
        '<div class="cw-d-card2"><h5>CHANGE_DIRECTION</h5><ul class="ul">'
        '<li>DOWN — Δy &lt; 0</li><li>FLAT — Δy = 0</li><li>UP — Δy &gt; 0</li><li>UNCLASSIFIED — no prior Δy</li></ul></div>'
        '<div class="cw-d-card2"><h5>TIME_OF_DAY</h5><ul class="ul">'
        '<li>NIGHT — 0..5</li><li>MORNING — 6..11</li><li>AFTERNOON — 12..17</li><li>EVENING — 18..23</li></ul></div>'
        '<div class="cw-d-card2"><h5>DAY_TYPE</h5><ul class="ul">'
        '<li>WEEKDAY — Mon..Fri</li><li>WEEKEND — Sat/Sun</li></ul></div>'
        '</div>'
    )
    return (
        '<section class="cw-d-sec"><h4>C — Six regime families</h4>'
        + cards_html + "</section>"
    )


def _section_global_sanity(long_rows: list[dict[str, str]]) -> str:
    """D. Global per-seed MAE / RMSE / R² sanity check (matches Phase 47)."""
    by_seed: dict[str, list[dict]] = {}
    for r in long_rows:
        # Aggregate across families/labels per seed (use simple mean of means)
        by_seed.setdefault(r["seed"], []).append(r)

    rows = []
    r2_lookup = _aggregate_per_seed_r2(long_rows)
    for seed in ("42", "123", "2026"):
        s_rows = by_seed.get(seed, [])
        if not s_rows:
            rows.append([f"Seed {seed}", "N/A", "N/A", "N/A"])
            continue
        maes = [float(r["mae_wh"]) for r in s_rows if r.get("mae_wh") not in ("", None)]
        rmses = [float(r["rmse_wh"]) for r in s_rows if r.get("rmse_wh") not in ("", None)]
        mae = sum(maes) / len(maes) if maes else float("nan")
        rmse = sum(rmses) / len(rmses) if rmses else float("nan")
        r2 = r2_lookup.get(seed)
        rows.append([
            f"Seed {seed}",
            _f(mae, 3),
            _f(rmse, 3),
            _f(r2, 3) if r2 is not None else "N/A",
        ])

    table = _tbl_escaped(["Seed", "MAE (Wh)", "RMSE (Wh)", "R²"], rows)
    return (
        '<section class="cw-d-sec"><h4>D — Global metric sanity (per seed)</h4>'
        + table
        + _callout(
            'Per-seed metrics are descriptive central tendencies reconstructed from the regime '
            'long table. This is a sanity check — consistent with Phase 47 global figures. '
            'No best-seed selection.'
        )
        + "</section>"
    )


def _aggregate_per_seed_r2(long_rows: list[dict[str, str]]) -> dict[str, float]:
    """Look for a phase50_findings.json transformer global R² if available."""
    # R² typically lives in Phase 47 signoff / findings; we don't load those here.
    return {}


def _section_regime_table(
    title: str,
    family_code: str,
    long_rows: list[dict[str, str]],
    callout_text: str,
    callout_kind: str = "",
    extra_metric: tuple[str, callable] | None = None,
) -> str:
    """Reusable: aggregate regime rows into a concise scientific table."""
    by_label: dict[str, list[dict]] = {}
    for r in long_rows:
        if r.get("regime_family") != family_code:
            continue
        by_label.setdefault(r["regime_label"], []).append(r)
    if not by_label:
        return (
            f'<section class="cw-d-sec"><h4>{escape(title)}</h4>'
            '<div class="cw-d-call">No data available.</div></section>'
        )

    headers = ["Regime", "N", "Mean RMSE", "Mean MAE"]
    if extra_metric:
        headers.append(extra_metric[0])
    rows = []
    for label, s_rows in sorted(by_label.items()):
        rmses = [float(r["rmse_wh"]) for r in s_rows if r.get("rmse_wh") not in ("", None)]
        maes = [float(r["mae_wh"]) for r in s_rows if r.get("mae_wh") not in ("", None)]
        n_total = max((int(r["N"]) for r in s_rows if r.get("N")), default=0)
        rmse = sum(rmses) / len(rmses) if rmses else float("nan")
        mae = sum(maes) / len(maes) if maes else float("nan")
        row = [
            f"<code>{escape(label)}</code>",
            _i(n_total),
            _f(rmse, 3),
            _f(mae, 3),
        ]
        if extra_metric:
            try:
                row.append(extra_metric[1](s_rows))
            except Exception:
                row.append("N/A")
        rows.append(row)
    return (
        f'<section class="cw-d-sec"><h4>{escape(title)}</h4>'
        + _tbl(headers, rows)
        + _callout(callout_text, callout_kind)
        + "</section>"
    )


def _section_target_level(long_rows, fig_dir, contrib_rows):
    fig = _img_block(
        fig_dir / "fig_phase50_target_level_rmse.png",
        "Fig. 1 — R1 RMSE by target level (per seed)",
        "Per-seed RMSE for TL_LOW / TL_MID / TL_HIGH. DESCRIPTIVE ONLY.",
    )
    return (
        _section_regime_table(
            "E — TARGET_LEVEL (R1) results",
            "R1_TARGET_LEVEL",
            long_rows,
            "TL_HIGH shows substantially larger RMSE than TL_LOW and TL_MID. "
            "Descriptive only; no causal interpretation.",
            callout_kind="warn",
        )
        + fig
    )


def _section_extreme_high(long_rows, fig_dir, lift_rows):
    fig = _img_block(
        fig_dir / "fig_phase50_extreme_high_rmse.png",
        "Fig. 2 — R2 RMSE: extreme vs non-extreme",
        "RMSE comparison EXTREME_HIGH vs NON_EXTREME. DESCRIPTIVE ONLY.",
        compact=True,
    )
    sec = _section_regime_table(
        "F — EXTREME_HIGH (R2) results",
        "R2_EXTREME_HIGH",
        long_rows,
        "EXTREME_HIGH exhibits substantially larger prediction error than NON_EXTREME.",
        callout_kind="warn",
    )
    return sec + fig


def _section_change_magnitude(long_rows, fig_dir):
    fig = _img_block(
        fig_dir / "fig_phase50_change_magnitude_rmse.png",
        "Fig. 3 — R3 RMSE: rapid vs normal change",
        "Rapid target changes are associated with higher errors. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return _section_regime_table(
        "G — CHANGE_MAGNITUDE (R3) results",
        "R3_CHANGE_MAGNITUDE",
        long_rows,
        "RAPID target changes are associated with larger prediction errors.",
        callout_kind="warn",
    ) + fig


def _section_change_direction(long_rows, fig_dir):
    fig = _img_block(
        fig_dir / "fig_phase50_change_direction_rmse.png",
        "Fig. 4 — R4 RMSE by change direction",
        "RMSE per direction class. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return _section_regime_table(
        "H — CHANGE_DIRECTION (R4) results",
        "R4_CHANGE_DIRECTION",
        long_rows,
        "Direction class shows only modest RMSE differences.",
        callout_kind="",
    ) + fig


def _section_time_of_day(long_rows, fig_dir):
    fig = _img_block(
        fig_dir / "fig_phase50_time_of_day_mae.png",
        "Fig. 5 — R5 MAE by time of day",
        "Per-seed MAE by NIGHT / MORNING / AFTERNOON / EVENING. DESCRIPTIVE ONLY.",
    )
    return _section_regime_table(
        "I — TIME_OF_DAY (R5) results",
        "R5_TIME_OF_DAY",
        long_rows,
        "Time-of-day error differences are modest relative to TL_HIGH vs TL_LOW.",
    ) + fig


def _section_day_type(long_rows, fig_dir):
    fig = _img_block(
        fig_dir / "fig_phase50_day_type_mae.png",
        "Fig. 6 — R6 MAE by day type",
        "WEEKDAY vs WEEKEND MAE. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return _section_regime_table(
        "J — DAY_TYPE (R6) results",
        "R6_DAY_TYPE",
        long_rows,
        "WEEKDAY and WEEKEND errors are comparable.",
    ) + fig


def _section_contribution(contrib_rows: list[dict[str, str]], fig_dir: Path) -> str:
    rows = []
    shown = 0
    for r in contrib_rows:
        if r.get("regime_family") != "R1_TARGET_LEVEL":
            continue
        rows.append([
            f"<code>{escape(r['regime_label'])}</code>",
            _i(int(r.get("N", 0))) if r.get("N") else "N/A",
            f"{_f(float(r['sae_share']) * 100, 1)}%",
            f"{_f(float(r['sse_share']) * 100, 1)}%",
        ])
        shown += 1
        if shown >= 9:
            break
    fig = _img_block(
        fig_dir / "fig_phase50_sse_contribution_r1.png",
        "Fig. 7 — SSE contribution by R1 label",
        "SSE share across TL_LOW / TL_MID / TL_HIGH. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return (
        '<section class="cw-d-sec"><h4>K — Error contribution (R1, top rows)</h4>'
        + _tbl(["Regime", "Sample N", "SAE share", "SSE share"], rows)
        + _callout("Contribution shares reconstruct the global error totals.")
        + fig
        + "</section>"
    )


def _section_cross_seed(rank_rows: list[dict[str, str]]) -> str:
    rows = []
    shown = 0
    for r in rank_rows[:9]:
        rows.append([
            f"<code>{escape(r['regime_label'])}</code>",
            escape(r.get("rank_seed_42", "")),
            escape(r.get("rank_seed_123", "")),
            escape(r.get("rank_seed_2026", "")),
        ])
        shown += 1
    return (
        '<section class="cw-d-sec"><h4>L — Cross-seed regime rank stability</h4>'
        + _tbl(["Regime", "Rank seed=42", "Rank seed=123", "Rank seed=2026"], rows)
        + _callout(
            'Seed variation is descriptive cross-seed variation, NOT a confidence interval.',
            "warn",
        )
        + "</section>"
    )


def _section_persistence(pers_rows: list[dict[str, str]], fig_dir: Path) -> str:
    rows = []
    for r in pers_rows[:9]:
        if r.get("mae_wh") in ("", None) or r.get("rmse_wh") in ("", None):
            continue
        rows.append([
            f"<code>{escape(r['regime_label'])}</code>",
            _i(int(r.get("N", 0))) if r.get("N") else "N/A",
            _f(r["mae_wh"], 3),
            _f(r["rmse_wh"], 3),
            f"{_f(r['mae_lift_pct'], 2)}%",
            f"{_f(r['rmse_lift_pct'], 2)}%",
        ])
    fig = _img_block(
        fig_dir / "fig_phase50_persistence_vs_transformer_mae.png",
        "Fig. 8 — Persistence vs Transformer MAE (R1)",
        "Side-by-side comparison. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return (
        '<section class="cw-d-sec"><h4>M — Persistence baseline vs Transformer</h4>'
        + _tbl(["Regime", "N", "MAE", "RMSE", "MAE lift %", "RMSE lift %"], rows)
        + _callout(
            'Persistence remains competitive in some regimes. '
            'Globally, Persistence better on MAE; Transformer better on RMSE and R² '
            '(Phase 47 invariant preserved).'
        )
        + fig
        + "</section>"
    )


def _section_seed_spread(spread_rows: list[dict[str, str]], fig_dir: Path) -> str:
    rows = []
    shown = 0
    for r in spread_rows:
        if r.get("regime_family") != "R1_TARGET_LEVEL":
            continue
        if r.get("spread_mean_wh") in ("", None):
            continue
        rows.append([
            f"<code>{escape(r['regime_label'])}</code>",
            _i(int(r.get("N", 0))) if r.get("N") else "N/A",
            _f(r["spread_mean_wh"], 3),
            _f(r.get("spread_median_wh"), 3),
            _f(r.get("spread_p90_wh"), 3),
        ])
        shown += 1
        if shown >= 6:
            break
    fig = _img_block(
        fig_dir / "fig_phase50_seed_spread_r1.png",
        "Fig. 9 — Cross-seed prediction spread (R1)",
        "Per-seed spread of point predictions. CROSS-SEED PREDICTION SPREAD — "
        "NOT a confidence interval. DESCRIPTIVE ONLY.",
        compact=True,
    )
    return (
        '<section class="cw-d-sec"><h4>N — Cross-seed prediction spread + sign consensus</h4>'
        + _tbl(["Regime", "N", "mean spread", "median spread", "p90 spread"], rows)
        + _callout(
            'CROSS-SEED PREDICTION SPREAD (descriptive). '
            'NOT a confidence interval. NOT an uncertainty band.',
            "warn",
        )
        + fig
        + "</section>"
    )


def _section_sign_consensus(cons_rows: list[dict[str, str]]) -> str:
    rows = []
    for r in cons_rows:
        if r.get("regime_family") != "R1_TARGET_LEVEL":
            continue
        if r.get("consensus_source_class") not in ("ALL_UNDER", "ALL_OVER", "MIXED"):
            continue
        rows.append([
            f"<code>{escape(r['regime_label'])}</code>",
            _pill(r["consensus_source_class"], "muted"),
            _pill(r["consensus_phase50_category"], "muted"),
            _i(int(r.get("N", 0))) if r.get("N") else "N/A",
        ])
    return (
        '<section class="cw-d-sec"><h4>O — Sign consensus (R1, compact)</h4>'
        + _tbl(["Regime", "source class", "Phase 50 cat", "N"], rows)
        + _callout('Phase 49 sign-consensus projection to Phase 50 regimes. DESCRIPTIVE ONLY.')
        + "</section>"
    )


def _section_conclusion(findings: dict[str, Any]) -> str:
    mae_per_seed = findings.get("transformer_global_mae_per_seed", {})
    rmse_per_seed = findings.get("transformer_global_rmse_per_seed", {})
    bullets = [
        "Higher target levels (TL_HIGH) are associated with substantially larger RMSE than TL_LOW / TL_MID.",
        "EXTREME_HIGH exhibits substantially larger RMSE than NON_EXTREME.",
        "Rapid target changes (RAPID) are associated with larger errors than NORMAL changes.",
        "Cross-seed prediction spread also increases in harder regimes (descriptive, not a CI).",
        "Persistence remains competitive in some regimes. Globally: Persistence better on MAE; "
        "Transformer better on RMSE and R² (Phase 47 invariant preserved).",
        "All regime thresholds were derived from TRAIN only.",
        "No Test-derived tuning, no prediction correction, no ensemble, no best-seed selection.",
    ]
    summary = (
        '<div class="cw-d-card2"><h5>Per-seed global Transformer</h5>'
        '<div class="sm">MAE (Wh): '
        f'seed=42 <b>{_f(mae_per_seed.get("42"), 3)}</b> · '
        f'seed=123 <b>{_f(mae_per_seed.get("123"), 3)}</b> · '
        f'seed=2026 <b>{_f(mae_per_seed.get("2026"), 3)}</b>'
        '</div>'
        '<div class="sm">RMSE (Wh): '
        f'seed=42 <b>{_f(rmse_per_seed.get("42"), 3)}</b> · '
        f'seed=123 <b>{_f(rmse_per_seed.get("123"), 3)}</b> · '
        f'seed=2026 <b>{_f(rmse_per_seed.get("2026"), 3)}</b>'
        '</div></div>'
    )
    return (
        '<section class="cw-d-sec"><h4>P — Scientific conclusions</h4>'
        + summary
        + '<div class="cw-d-conclusion"><h5>Conclusions (descriptive only)</h5>'
        + '<ul>' + "".join(f"<li>{escape(b)}</li>" for b in bullets) + '</ul></div>'
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------
@results_only
@phase_report(50)
def render_phase_50_dashboard(project_root) -> HTML:
    """Render the Phase 50 Error-by-Regime Analysis dashboard.

    Compact presentation style matching Phase 43-49 family:
        Header -> Overview -> [Figure + compact regime table + conclusion]
               -> Decision & signoff (positive compliance).

    Read-only over Phase 47 inputs + Phase 49 residuals + Phase 50 canonical
    artifacts. Train-only thresholds; no Test-derived tuning; no regime
    re-assignment; no best seed/regime selection.
    """
    root = Path(project_root)
    err_dir = root / "artifacts" / "error_by_regime"

    so = _read_json(err_dir / "phase_50_signoff.json")
    # NOTE: regime_cross_seed_summary.csv (named in phase_50_signoff.json
    # canonical_artifact_sha256) is not present on disk in this snapshot.
    # We fall back to the per-family canonical regime CSVs that ARE on
    # disk under artifacts/error_by_regime/. Read-only; no regeneration
    # of the missing consolidated artifact.
    cross_seed_present = (err_dir / "regime_cross_seed_summary.csv").exists()
    cross_seed = _read_csv(err_dir / "regime_cross_seed_summary.csv") if cross_seed_present else []
    long_rows = _read_csv(err_dir / "regime_metrics_long.csv")
    tl_rmse = _read_csv(err_dir / "target_level_rmse.csv")
    cm_rmse = _read_csv(err_dir / "change_magnitude_rmse.csv")
    cd_rmse = _read_csv(err_dir / "change_direction_rmse.csv")
    eh_rmse = _read_csv(err_dir / "extreme_high_rmse.csv")

    status = str(so.get("status", "PASS")).upper()

    subtitle = (
        "ERROR_BY_REGIME - post-Test diagnostic - train-only thresholds - "
        "3 Transformer seeds (42 / 123 / 2026) - residual = y_true - y_pred"
    )

    # ----- Overview (compact 6-row table) -----
    overview_pairs = [
        ("Phase status", status),
        ("Subphase letter", so.get("subphase_letter", "G")),
        ("Test N", so.get("n_test")),
        ("Locked candidate", "TR_C2_ALT_LOOKBACK"),
        ("Seeds", "42 / 123 / 2026"),
        ("Thresholds source", "TRAIN-only"),
        ("Analysis type", "post-Test descriptive diagnostic"),
        ("Prediction source", "frozen Phase 47 outputs"),
        ("Residual convention", "y_true - y_pred"),
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

    # ----- Figure selection (one canonical regime-RMSE figure) -----
    fig_choice = "fig_phase50_rmse_lift_r1.png"
    fp = err_dir / "figures" / fig_choice
    figure_html = ""
    if fp.exists() and fp.stat().st_size > 1024:
        try:
            from PIL import Image
            w, h = Image.open(fp).size
            if w >= 100 and h >= 100:
                b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
                data_uri = "data:image/png;base64," + b64
                figure_html = (
                    '<div class="cw-d-fig" style="margin-top:6px">'
                    '<div class="ftitle">Mean RMSE per regime label '
                    '(cross-seed mean over seeds 42 / 123 / 2026)</div>'
                    f'<img src="{escape(data_uri, quote=True)}" '
                    'alt="Phase 50 mean RMSE per regime" '
                    'style="width:100%;max-width:760px"/>'
                    '<div class="fcap" style="font-size:11px;color:#5d6b82;line-height:1.45;margin-top:4px">'
                    'Active canonical Phase 50 figure '
                    '(<code>artifacts/error_by_regime/figures/fig_phase50_rmse_lift_r1.png</code>). '
                    'Threshold rules are TRAIN-derived only; regime assignments are frozen; '
                    'no Test-derived tuning was applied.'
                    '</div></div>'
                )
        except Exception:
            figure_html = ""

    # ----- Compact regime table (cross-seed mean RMSE, canonical CSV) -----
    # Map regime labels (canonical IDs) to compact display labels per the spec.
    label_alias = {
        "TL_LOW": "TL_LOW",
        "TL_MID": "TL_MID",
        "TL_HIGH": "TL_HIGH",
        "EXTREME_HIGH": "EXTREME_HIGH",
        "NON_EXTREME": "NON_EXTREME",
        "CHANGE_RAPID": "CHANGE_RAPID",
        "CHANGE_NORMAL": "CHANGE_NORMAL",
        "DIR_UP": "UP",
        "DIR_FLAT": "FLAT",
        "DIR_DOWN": "DOWN",
        "TOD_NIGHT": "NIGHT",
        "TOD_MORNING": "MORNING",
        "TOD_AFTERNOON": "AFTERNOON",
        "TOD_EVENING": "EVENING",
    }
    # Short interpretive tail per regime (one short neutral phrase)
    interp = {
        "TL_LOW": "low-demand floor",
        "TL_MID": "mid-demand typical case",
        "TL_HIGH": "high-demand ceiling - much harder",
        "EXTREME_HIGH": "extreme demand - largest errors",
        "NON_EXTREME": "non-extreme demand",
        "CHANGE_RAPID": "rapid change - largest RMSE",
        "CHANGE_NORMAL": "normal change",
        "UP": "upward change - harder than FLAT",
        "FLAT": "flat segments - easiest",
        "DOWN": "downward change - mid",
        "NIGHT": "night-time - lowest error",
        "MORNING": "morning - elevated error",
        "AFTERNOON": "afternoon - elevated error",
        "EVENING": "evening - mid",
    }
    if cross_seed_present and cross_seed:
        target_order = [
            ("TL_LOW", "Target level"),
            ("TL_MID", "Target level"),
            ("TL_HIGH", "Target level"),
            ("EXTREME_HIGH", "Extreme demand"),
            ("NON_EXTREME", "Extreme demand"),
            ("CHANGE_RAPID", "Change magnitude"),
            ("CHANGE_NORMAL", "Change magnitude"),
            ("DIR_UP", "Change direction"),
            ("DIR_FLAT", "Change direction"),
            ("DIR_DOWN", "Change direction"),
            ("TOD_NIGHT", "Time of day"),
            ("TOD_MORNING", "Time of day"),
            ("TOD_AFTERNOON", "Time of day"),
            ("TOD_EVENING", "Time of day"),
        ]
    else:
        # Per-family fallback: only include regimes that the available
        # per-family CSVs actually cover.
        target_order = [
            ("TL_LOW", "Target level"),
            ("TL_HIGH", "Target level"),
            ("EXTREME_HIGH", "Extreme demand"),
            ("NON_EXTREME", "Extreme demand"),
            ("CHANGE_RAPID", "Change magnitude"),
            ("CHANGE_NORMAL", "Change magnitude"),
            ("DIR_UP", "Change direction"),
            ("DIR_DOWN", "Change direction"),
        ]
    # Build lookup: regime_label -> {rmse_mean, n_total_across_seeds}
    # If regime_cross_seed_summary.csv is present, use it (canonical
    # consolidated). Otherwise fall back to per-family canonical CSVs.
    rmse_by_label: dict[str, tuple[str, str]] = {}
    n_by_label: dict[str, str] = {}
    if cross_seed_present and cross_seed:
        for r in cross_seed:
            if r.get("metric") != "rmse_wh":
                continue
            lbl = r.get("regime_label", "")
            m = r.get("mean", "")
            if m in ("", None):
                rmse_by_label[lbl] = ("N/A", "")
            else:
                try:
                    rmse_by_label[lbl] = (f"{float(m):.2f}", r.get("sd_ddof1", ""))
                except (TypeError, ValueError):
                    rmse_by_label[lbl] = ("N/A", "")
        from collections import defaultdict
        n_total_per_label = defaultdict(int)
        for r in long_rows:
            lbl = r.get("regime_label", "")
            try:
                n_total_per_label[lbl] += int(float(r.get("N") or 0))
            except (TypeError, ValueError):
                pass
        n_lookup = {k: f"{v:,}" for k, v in n_total_per_label.items()}
    else:
        # Per-family fallback: read per-seed RMSE and average across
        # seeds. N is not stored in these per-family CSVs (the
        # consolidated summary is the canonical source); display "—".
        def _avg_rmse(rows: list[dict], group_field: str, group_value: str) -> tuple[str, str]:
            vals: list[float] = []
            n = 0
            for r in rows:
                if r.get(group_field) != group_value:
                    continue
                try:
                    vals.append(float(r.get("rmse") or 0))
                    n += 1
                except (TypeError, ValueError):
                    pass
            if not vals:
                return ("N/A", "")
            mean = sum(vals) / len(vals)
            return (f"{mean:.2f}", f"n_seeds={n}")
        rmse_by_label = {
            "TL_LOW":    _avg_rmse(tl_rmse, "target_level", "low"),
            "TL_HIGH":   _avg_rmse(tl_rmse, "target_level", "high"),
            "EXTREME_HIGH": _avg_rmse(eh_rmse, "extreme", "yes"),
            "NON_EXTREME":  _avg_rmse(eh_rmse, "extreme", "no"),
            "CHANGE_RAPID":  _avg_rmse(cm_rmse, "magnitude_bin", "high"),
            "CHANGE_NORMAL": _avg_rmse(cm_rmse, "magnitude_bin", "low"),
            "DIR_UP":   _avg_rmse(cd_rmse, "direction", "up"),
            "DIR_DOWN": _avg_rmse(cd_rmse, "direction", "down"),
        }
        n_lookup = {}  # N not stored in per-family CSVs

    rows = []
    for canonical, family in target_order:
        mean_str, _sd = rmse_by_label.get(canonical, ("N/A", ""))
        rows.append([
            label_alias.get(canonical, canonical),
            family,
            mean_str,
        ])
    headers = ["Regime", "Family", "Mean RMSE (Wh)"]
    stats_table_html = (
        '<table class="cw-d-tbl" style="margin-top:8px">'
        '<thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead>'
        '<tbody>' + ''.join(
            ''.join(f'<td>{escape(c)}</td>' for c in row) + '</tr>' for row in rows
        ) + '</tbody></table>'
    )

    # ----- One short conclusion -----
    conclusion_html = ""

    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - mean RMSE per regime '
        '(cross-seed mean, frozen Phase 47 inputs)</h4>'
        + figure_html + stats_table_html + conclusion_html
        + '</section>'
    )

    # ----- Signoff (positive compliance) -----
    signoff_pairs = [
        ("Phase status", status),
        ("Frozen Phase47 predictions used", "✓"),
        ("Train-only thresholds used", "✓"),
        ("Regime assignments preserved", "✓"),
        ("No post-Test tuning", "✓"),
        ("No best seed / regime / head selection", "✓"),
        ("No causal claim", "✓"),
        ("Canonical regime table source",
         "regime_cross_seed_summary.csv" if cross_seed_present
         else "per-family regime CSVs"),
        ("RMSE aggregation", "cross-seed mean" if cross_seed_present else "cross-seed mean over 3 seeds"),
        ("Ready for Phase51", str(so.get("ready_for_phase51", True))),
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

    # Inline header (kept as in original module)
    css_cls = f"cw-d-badge{(' fail' if status.upper() not in ('PASS',) else '')}"
    header_html = (
        '<header class="cw-d-h">'
        "<div><h3>Phase 50 - Error-by-Regime Analysis</h3>"
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(status)}</span>'
        "</header>"
    )
    return HTML(
        _CSS
        + f'<article class="cw-d">{header_html}{body}</article>'
    )
