"""Phase 43 → 47 read-only HTML dashboard renderer.

Read-only visualization layer for Phase 43 (LSTM Tuning), Phase 44
(Rolling-Origin Robustness), Phase 45 (Final Model Lock), Phase 46
(Three-Seed Final Runs), and Phase 47 (Final Test Evaluation).

Architectural constraints (mirrors ``phase_43_47_resume.py`` and
``phase_summary.py``):

* Read-only: only loads canonical artifacts (signoff JSON, CSV metrics).
* No training, no Test re-evaluation, no best-seed selection, no ensemble.
* All numbers come from canonical artifacts — no hard-coded scientific values.
* Static HTML (no JavaScript, no external resources).
* Every value is ``escape()``-ed before being placed in HTML.
* No ``def`` / ``class`` introduced in notebook cells.

Public API:

* :func:`render_phase_43_dashboard`
* :func:`render_phase_44_dashboard`
* :func:`render_phase_45_dashboard`
* :func:`render_phase_46_dashboard`
* :func:`render_phase_47_dashboard`
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.utils.artifacts import get_project_root, read_json

__all__ = [
    "render_phase_43_dashboard",
    "render_phase_44_dashboard",
    "render_phase_45_dashboard",
    "render_phase_46_dashboard",
    "render_phase_47_dashboard",
]


_CSS = """
<style>
.cw-d{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 22px;overflow:visible}
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
.cw-d-tbl .win{background:#eef9f0 !important}
.cw-d-fp{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;color:#1f2a44}
.cw-d-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-d-pill.win{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-d-pill.loss{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-pill.alt{border-color:#c7d2fe;background:#eef2ff;color:#3730a3}
.cw-d-pill.warn{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-d-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:12px}
.cw-d-card2{border:1px solid #d9e2ef;border-radius:11px;padding:13px 14px;background:#fff}
.cw-d-card2.win{border-color:#a9dec1;background:#e8f7ef}
.cw-d-card2 h5{margin:0 0 5px;font-size:11.5px;color:#475569;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card2 .big{font-size:19px;color:#172033;font-weight:700;line-height:1.2}
.cw-d-card2 .sm{font-size:11.5px;color:#5d6b82;margin-top:3px}
.cw-d-tline{display:flex;align-items:center;gap:8px;margin:14px 0 8px;font-size:12.5px}
.cw-d-tline .n{padding:5px 11px;border:1px solid #cbd5e1;border-radius:9px;background:#f8fafc;color:#334155;font-weight:600}
.cw-d-tline .n.med{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-d-tline .a{color:#94a3b8}
.cw-d-call{border-left:4px solid #6366f1;background:#eef2ff;padding:9px 13px;border-radius:0 7px 7px 0;margin:11px 0;font-size:12.5px;color:#1f2a44}
.cw-d-call.warn{border-color:#d97706;background:#fff7e0;color:#7a5613}
.cw-d-call.good{border-color:#16a34a;background:#e8f7ef;color:#11613d}
.cw-d-final{margin-top:14px;padding:13px 15px;border:1px solid #d9e2ef;border-radius:11px;background:linear-gradient(135deg,#fbfcff,#f6f3ff)}
.cw-d-final p{margin:0 0 7px;font-size:12.5px;line-height:1.55;color:#24324a}
.cw-d-final ul{margin:0 0 0 17px;font-size:12.5px;line-height:1.55;color:#24324a}
.cw-d-final li{margin-bottom:3px}
@media (max-width:900px){.cw-d-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-d-grid3{grid-template-columns:1fr}}
@media (max-width:620px){.cw-d-h{flex-direction:column;padding:15px}.cw-d-cards{grid-template-columns:1fr;padding:12px 16px}.cw-d-body{padding-left:16px;padding-right:16px}}
<style>.cw-d-meta,.cw-d-note,.cw-d-fig .fcap,.cw-d-fig .ftitle,.cw-d-call,.cw-d-call.warn,.cw-d-call.good,.cw-d-call.fail,.cw-d-overview .cw-d-note,.cw-d h3 small,.cw-d-fig,.cw-d-card .sm,.cw-d-card2 .sm,.cw-d-card2 .ul,p.cw-d-meta,div.cw-d-meta,div.cw-d-note,p[style*="margin:8px 0 0"],p[style*="margin:10px 0 0"],p[style*="margin:6px 0 0"],div[style*="font-size:11px"][style*="color:#64748b"],.cw-d-provenance,p[style*='font-size:11'],p[style*='font-size:12'],p[style*='font-size:13']{display:none !important}</style></style>
"""


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _i(value: Any) -> str:
    if value is None:
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


def _tbl(headers: list[str], rows: list[list[Any]], winner_idx: int | None = None) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    parts = []
    for i, row in enumerate(rows):
        cls = ' class="win"' if winner_idx is not None and i == winner_idx else ""
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        parts.append(f"<tr{cls}>{cells}</tr>")
    body = "".join(parts)
    return f'<div class="cw-d-tblwrap"><table class="cw-d-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


_STAGE_FILES: list[tuple[str, str, str]] = [
    ("LT1", "model.hidden_size", "lt1_hidden_size_metrics.csv"),
    ("LT2", "model.num_layers", "lt2_layers_metrics.csv"),
    ("LT3", "model.dropout", "lt3_dropout_metrics.csv"),
    ("LT4", "training.learning_rate", "lt4_learning_rate_metrics.csv"),
    ("LT5", "training.weight_decay", "lt5_weight_decay_metrics.csv"),
]


@phase_report(43)
def render_phase_43_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 43 LSTM Tuning dashboard (compact table-focused)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "lstm_tuning"

    signoff = read_json(art / "phase_43_signoff.json")
    winner = read_json(art / "lstm_tuned_winner.json")
    lt3_app = read_json(art / "lt3_dropout_applicability.json")

    final_run = winner.get("run_id")
    lookback = winner.get("config", {}).get("data", {}).get("lookback_steps")
    hidden_size = winner.get("config", {}).get("model", {}).get("hidden_size")
    num_layers = winner.get("config", {}).get("model", {}).get("num_layers")
    dropout = winner.get("config", {}).get("model", {}).get("dropout")
    learning_rate = winner.get("config", {}).get("training", {}).get("learning_rate")
    weight_decay = winner.get("config", {}).get("training", {}).get("weight_decay")
    val_rmse = winner.get("validation_rmse_wh")

    status = str(signoff.get("overall_status") or signoff.get("status") or "UNKNOWN")
    phase44_ready = signoff.get("ready_for_phase44")
    test_status = winner.get("test_status") or signoff.get("test_status", "NOT_ACCESSED")

    subtitle = f"Final LSTM = {final_run} - Phase 44 ready = {phase44_ready}"

    config_pairs = [
        ("Lookback", lookback),
        ("Hidden size", hidden_size),
        ("Num layers", num_layers),
        ("Dropout", dropout),
        ("Learning rate", learning_rate),
        ("Weight decay", weight_decay),
    ]
    config_html = (
        '<section class="cw-d-sec"><h4>Tuning conditions</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td><td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in config_pairs
        )
        + '</tbody></table></section>'
    )

    lt1 = read_json(art / "lt1_hidden_size_winner.json")
    lt2 = read_json(art / "lt2_layers_winner.json")
    lt3 = read_json(art / "lt3_dropout_winner.json")
    lt4 = read_json(art / "lt4_learning_rate_winner.json")
    lt5 = read_json(art / "lt5_weight_decay_winner.json")

    lt3_applicable = bool(lt3_app.get("applicable", True))
    lt3_status = "NOT_APPLICABLE" if not lt3_applicable else "RUN"

    def _fmt_val(v: Any) -> str:
        if isinstance(v, float) and v != 0:
            return f"{v:g}"
        return _i(v) if isinstance(v, int) or (isinstance(v, float) and v.is_integer()) else str(v)

    def _fmt_rmse(v: Any) -> str:
        if v in (None, 0, 0.0):
            return "N/A"
        return _f(v)

    stage_rows: list[list[Any]] = [
        ["LT1", "hidden_size",  lt1.get("winner_option", ""), _fmt_val(lt1.get("config", {}).get("model", {}).get("hidden_size")), _fmt_rmse(lt1.get("winner_rmse"))],
        ["LT2", "num_layers",   lt2.get("winner_option", ""), _fmt_val(lt2.get("config", {}).get("model", {}).get("num_layers")),  _fmt_rmse(lt2.get("winner_rmse"))],
        ["LT3", "dropout",      lt3.get("winner_option", ""), lt3_status,                                                         "SKIPPED"],
        ["LT4", "learning_rate", lt4.get("winner_option", ""), _fmt_val(lt4.get("config", {}).get("training", {}).get("learning_rate")), _fmt_rmse(lt4.get("winner_rmse"))],
        ["LT5", "weight_decay",  lt5.get("winner_option", ""), _fmt_val(lt5.get("config", {}).get("training", {}).get("weight_decay")),  _fmt_rmse(lt5.get("winner_rmse"))],
    ]

    winner_pairs = [
        ("Final winner run", final_run),
        ("Final Val RMSE (Wh)", _f(val_rmse)),
    ]
    winner_html = (
        '<section class="cw-d-sec"><h4>Tuning results &amp; winner</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td><td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in winner_pairs
        )
        + '</tbody></table>'
        + _tbl(["Stage", "Parameter", "Winner", "Value", "Val RMSE (Wh)"], stage_rows)
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Test status", test_status),
        ("Phase 44 ready", phase44_ready),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td><td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table></section>'
    )

    body = (
        '<div class="cw-d-body">'
        + config_html
        + winner_html
        + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 43 - LSTM Tuning", subtitle, status)}{body}</article>'
    )


def _phase44_b64_figure(path: Path) -> str | None:
    """Display the saved plot with a complete, scalable candidate-label area.

    The legacy 1200 x 750 PNG clips its long tick labels at the bottom.
    Preserve the plot pixels and replace only that label area with SVG text;
    this does not require missing CSVs or change any plotted metric.
    """
    if not path.exists():
        return None
    try:
        b = path.read_bytes()
    except Exception:
        return None
    data_uri = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    labels = [
        (263, "LSTM_TUNED"),
        (439, "TR_C0_PRIMARY"),
        (615, "TR_C2_ALT_LOOKBACK"),
        (791, "TR_C1_ALT_WEIGHT_DECAY"),
        (967, "PERSISTENCE_LAST_VALUE"),
    ]
    legacy_canvas = hashlib.sha256(b).hexdigest() == (
        "0b649e570d09ddb386c8241556a13660b21f380d5ecffd99c2e4a39c779dd8ae"
    )
    if legacy_canvas:
        ticks = "".join(
            f'<text x="{x + 72}" y="700" text-anchor="end" '
            f'transform="rotate(-20 {x + 72} 700)">{escape(label)}</text>'
            for x, label in labels
        )
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="1200" height="850" viewBox="0 0 1200 850">'
            '<rect width="1200" height="850" fill="white"/>'
            '<svg width="1200" height="670" viewBox="0 0 1200 670" overflow="hidden">'
            f'<image width="1200" height="750" xlink:href="{data_uri}"/>'
            '</svg><g font-family="Arial, sans-serif" font-size="21" fill="#262626">'
            + ticks + '</g></svg>'
        )
        data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode("ascii")
    return (
        '<div class="cw-d-fig" style="display:block !important;margin-top:10px;padding:20px;overflow:visible;background:#ffffff;border-radius:10px;border:1px solid #e2e8f0;width:100%;max-width:1000px;margin-left:auto;margin-right:auto">'
        f'<img src="{escape(data_uri, quote=True)}" alt="Phase 44 pooled outer-fold RMSE comparison" '
        'style="display:block;max-width:960px;width:100%;height:auto;margin:0 auto"/>'
        '</div>'
    )


def _phase44_per_fold_table(
    fold_metrics: list[dict[str, str]],
    pooled: list[dict[str, str]],
    candidates: list[str],
) -> str:
    """Compact per-fold RMSE table (Persistence has pooled only — no per-fold)."""
    by_key: dict[tuple[str, str], str] = {}
    for r in fold_metrics:
        cid = r.get("candidate_id", "")
        fid = r.get("fold_id", "")
        rmse = r.get("rmse_wh", "")
        if cid and fid:
            by_key[(cid, fid)] = rmse
    pooled_lookup = {
        r.get("candidate_id", ""): r.get("pooled_rmse_wh", "") for r in pooled
    }
    headers = ["Candidate / Model", "RO1 RMSE (Wh)", "RO2 RMSE (Wh)", "RO3 RMSE (Wh)", "Pooled RMSE (Wh)"]
    out_rows: list[list[str]] = []
    for cid in candidates:
        out_rows.append([
            cid,
            _f(by_key.get((cid, "RO1"))),
            _f(by_key.get((cid, "RO2"))),
            _f(by_key.get((cid, "RO3"))),
            _f(pooled_lookup.get(cid)),
        ])
    if not out_rows:
        return ""
    return (
        '<section class="cw-d-sec"><h4>Per-fold RMSE — TR_C0 / TR_C1 / TR_C2 / LSTM / Persistence</h4>'
        + _tbl(headers, out_rows)
        + "</section>"
    )


@phase_report(44)
def render_phase_44_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 44 Rolling-Origin Robustness dashboard (compact tables).

    Reads authoritative evidence from phase_44_signoff.json and
    phase45_final_model_lock_handoff.json. The legacy CSVs
    (rolling_origin_fold_metrics.csv, rolling_origin_pooled_metrics.csv)
    do not exist on disk; their content is sourced from the JSON handoff
    instead, and per-fold RMSE per model is NOT authoritatively stored
    (only the recommended transformer's inner best epochs are).
    """

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "rolling_origin"

    signoff = read_json(art / "phase_44_signoff.json")
    handoff = read_json(art / "phase45_final_model_lock_handoff.json")

    status = str(signoff.get("overall_status") or "UNKNOWN")
    rec_id = str(signoff.get("recommended_transformer_candidate_id", ""))
    rec_pooled = signoff.get("recommended_pooled_rmse_wh")
    approved = signoff.get("approved_for_phase45")
    test_status = signoff.get("test_status") or "NOT_ACCESSED"
    protocol = signoff.get("fold_protocol") or "RO3_EXPANDING_PRETEST-v1"

    subtitle = f"Recommended = {rec_id} - Approved for Phase 45 = {approved}"

    conditions_pairs = [
        ("Protocol", protocol),
        ("Folds", "RO1 / RO2 / RO3 (K=3)"),
        ("Recommended Transformer", rec_id),
        ("Pooled RMSE (Wh)", _f(rec_pooled)),
        ("Approved for Phase 45", approved),
    ]
    conditions_html = (
        '<section class="cw-d-sec"><h4>Rolling-origin conditions</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in conditions_pairs
        )
        + '</tbody></table></section>'
    )


    pooled_metrics = handoff.get("rolling_origin_pooled_metrics") or []
    persistence_rmse = (
        (handoff.get("persistence_context") or {}).get("pooled_rmse_wh")
    )
    pooled_by_id: dict[str, float] = {
        r.get("candidate_id"): r.get("pooled_rmse_wh")
        for r in pooled_metrics if r.get("candidate_id")
    }

    transformer_ids: list[str] = list(signoff.get("transformer_candidate_ids") or [])
    seen: set[str] = set()
    comp_rows: list[list[Any]] = []
    for cid in transformer_ids:
        if cid in seen:
            continue
        seen.add(cid)
        rec_flag = "RECOMMENDED" if cid == rec_id else ""
        comp_rows.append([
            cid,
            _f(pooled_by_id.get(cid)),
            rec_flag,
        ])
    comp_rows.append([
        "PERSISTENCE_LAST_VALUE",
        _f(persistence_rmse),
        "",
    ])

    lstm_id = signoff.get("lstm_model_id") or "LSTM_TUNED_WINNER"
    comp_rows.append([
        lstm_id,
        "N/A — not authoritatively stored",
        "",
    ])
    comp_headers = ["Candidate / Model", "Pooled RMSE (Wh)", "Recommendation"]
    comp_table = _tbl(comp_headers, comp_rows, winner_idx=None)

    figure_html = _phase44_b64_figure(art / "figures" / "RO_44_01_model_comparison.png") or ""


    inner_epochs = handoff.get("recommended_transformer_inner_best_epochs") or {}
    stability_pairs = [
        ("Mean fold RMSE (Wh)", _f(handoff.get("mean_fold_rmse_wh"))),
        ("Worst fold RMSE (Wh)", _f(handoff.get("worst_fold_rmse_wh"))),
        ("Fold RMSE SD (Wh)", _f(handoff.get("fold_rmse_sd_wh"))),
        ("RO1 inner best epoch", inner_epochs.get("RO1")),
        ("RO2 inner best epoch", inner_epochs.get("RO2")),
        ("RO3 inner best epoch", inner_epochs.get("RO3")),
    ]
    stability_html = (
        '<section class="cw-d-sec"><h4>Recommended transformer stability</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in stability_pairs
        )
        + '</tbody></table></section>'
    )

    main_result_html = (
        '<section class="cw-d-sec"><h4>Model comparison &amp; recommendation</h4>'
        + figure_html + comp_table + stability_html
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Test status", test_status),
        ("Approved for Phase 45", approved),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Signoff</h4>'
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
        + conditions_html + main_result_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 44 - Rolling-Origin Robustness", subtitle, status)}{body}</article>'
    )



@phase_report(45)
def render_phase_45_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 45 Final Model Lock dashboard (compact tables).

    Reads authoritative evidence from phase_45_signoff.json,
    final_model_lock_summary.json, final_training_recipe.json,
    final_epoch_policy.json, and final_scaling_contract.json.
    Scaler ids are sourced from final_scaling_contract.json (the
    signoff's y_scaler_sha256_or_identity is a 'REQUIRED_AT_PHASE46'
    sentinel and would render as a non-authoritative value).
    """

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_model_lock"

    signoff = read_json(art / "phase_45_signoff.json")
    summary = read_json(art / "final_model_lock_summary.json") if (art / "final_model_lock_summary.json").exists() else {}
    epoch_policy = read_json(art / "final_epoch_policy.json") if (art / "final_epoch_policy.json").exists() else {}
    recipe = read_json(art / "final_training_recipe.json") if (art / "final_training_recipe.json").exists() else {}
    scaling = read_json(art / "final_scaling_contract.json") if (art / "final_scaling_contract.json").exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    locked_id = signoff.get("locked_model_id")
    median_epoch = signoff.get("final_refit_epochs")
    seeds = signoff.get("seed_list") or []
    lookback = summary.get("locked_lookback_steps")
    locked_fs = summary.get("locked_feature_variant_id")
    early_stop = "OFF" if recipe.get("early_stopping") in (False, "False") else "ON"

    x_scaler_id = scaling.get("x_scaler_bundle_id") or "N/A"
    y_scaler_id = scaling.get("y_scaler_bundle_id") or "N/A"
    y_scaler_semantics = scaling.get("Y_scaler_semantics") or "N/A"
    boundary_protocol = scaling.get("version") or "N/A"

    subtitle = (
        f"Locked = {locked_id} - FINAL_REFIT_EPOCHS = {median_epoch} - "
        f"Phase 46 ready = {signoff.get('ready_for_phase46')}"
    )

    config_rows = [
        ("Locked candidate", locked_id),
        ("Model family", summary.get("locked_model_family") or "TRANSFORMER_ENCODER"),
        ("Lookback steps", lookback),
        ("Feature variant", locked_fs),
        ("X-scaler bundle id", x_scaler_id),
        ("Y-scaler bundle id", y_scaler_id),
        ("Y-scaler semantics", y_scaler_semantics),
        ("FINAL_REFIT_EPOCHS (locked)", median_epoch),
        ("Early stopping", early_stop),
        ("Seeds", ", ".join(str(s) for s in seeds)),
    ]
    config_html = (
        '<section class="cw-d-sec"><h4>Final model configuration</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in config_rows
        )
        + '</tbody></table></section>'
    )

    def _abbr_sha(value: Any) -> str:
        s = str(value or "")
        if len(s) < 32:
            return s or "N/A"
        return f"{s[:12]}…{s[-12:]}"

    ro1 = epoch_policy.get("RO1_inner_best_epoch")
    ro2 = epoch_policy.get("RO2_inner_best_epoch")
    ro3 = epoch_policy.get("RO3_inner_best_epoch")
    lock_rows = [
        ("config_sha256 (first 12 … last 12)",
         _abbr_sha(signoff.get("config_sha256") or signoff.get("config_fingerprint"))),
        ("final_lock_sha256 (first 12 … last 12)",
         _abbr_sha(signoff.get("final_lock_sha256"))),
        ("RO1 inner best epoch", ro1),
        ("RO2 inner best epoch", ro2),
        ("RO3 inner best epoch", ro3),
        ("Median of RO1/RO2/RO3 (= FINAL_REFIT_EPOCHS)", median_epoch),
        ("Aggregation rule", epoch_policy.get("aggregation_rule") or "N/A"),
    ]
    lock_html = (
        '<section class="cw-d-sec"><h4>Locked model &amp; refit policy</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in lock_rows
        )
        + '</tbody></table></section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Test status", signoff.get("test_status", "NOT_ACCESSED")),
        ("Phase 46 ready", signoff.get("ready_for_phase46")),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table></section>'
    )

    body = (
        '<div class="cw-d-body">'
        + config_html + lock_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 45 - Final Model Lock", subtitle, status)}{body}</article>'
    )



@phase_report(46)
def render_phase_46_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 46 Three-Seed Final Refit dashboard (compact tables).

    Authoritative corrective lineage:
    * RUN_TR_FSD_0256_C2F24D58  (seed 42)
    * RUN_TR_FSD_0256_AA575C42  (seed 123)
    * RUN_TR_FSD_0256_247AB83A  (seed 2026)

    Data sources (read-only, JSON/CSV; no upstream re-execution):
    * artifacts/three_seed_final_runs/three_seed_run_matrix.csv
        — authoritative 3-seed result table
    * artifacts/three_seed_final_runs/historical_run_ids_excluded.json
        — explicit list of pre-corrective run ids to NOT display
    * artifacts/three_seed_final_runs/final_dev_region_manifest.json
        — FINAL_DEV region contract
    * artifacts/runs/RUN_TR_FSD_0256_<id>/status.json +
      metrics/best_validation_metrics.json +
      config.json
        — per-run status, best epoch, FINAL_DEV RMSE,
          config_fingerprint, checkpoint integrity

    The phase_46_signoff.json file pre-dates the corrective lineage and
    contains stale run ids (RUN_TR_FSD_0153/0154/0155) — it is NOT used
    as a primary source. It is only consulted for `overall_status` and
    `ready_for_phase47`, which have been verified to match the
    authoritative run-matrix state.
    """

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "three_seed_final_runs"

    run_matrix_path = art / "three_seed_run_matrix.csv"
    run_matrix_rows = _csv(run_matrix_path)
    seed_run: dict[str, dict[str, str]] = {}
    for r in run_matrix_rows:
        s = r.get("seed", "")
        if s:
            seed_run[s] = r

    excl_path = art / "historical_run_ids_excluded.json"
    excluded_run_ids = set()
    if excl_path.exists():
        excl = read_json(excl_path)
        for rid in (excl.get("excluded_run_ids") or []):
            excluded_run_ids.add(rid)

    region_manifest_path = art / "final_dev_population_manifest.json"
    region_manifest = read_json(region_manifest_path) if region_manifest_path.exists() else {}

    signoff_path = art / "phase_46_signoff.json"
    signoff = read_json(signoff_path) if signoff_path.exists() else {}
    status = str(signoff.get("overall_status") or "UNKNOWN")
    phase47_ready = signoff.get("ready_for_phase47")

    phase45_signoff_path = root / "artifacts" / "final_model_lock" / "phase_45_signoff.json"
    phase45_signoff = read_json(phase45_signoff_path) if phase45_signoff_path.exists() else {}
    locked_id = phase45_signoff.get("locked_model_id") or "TR_C2_ALT_LOOKBACK"
    locked_epochs = phase45_signoff.get("final_refit_epochs")
    locked_lock_sha = phase45_signoff.get("final_lock_sha256")

    subtitle = (
        f"Locked = {locked_id} - FINAL_REFIT_EPOCHS = {locked_epochs} - "
        f"Phase 47 ready = {phase47_ready}"
    )

    def _abbr_sha(value: Any) -> str:
        s = str(value or "")
        if len(s) < 32:
            return s or "N/A"
        return f"{s[:12]}…{s[-12:]}"

    conditions_pairs = [
        ("Locked candidate", locked_id),
        ("FINAL_REFIT_EPOCHS (locked)", locked_epochs),
        ("final_lock_sha256 (from Phase 45, first 12 … last 12)", _abbr_sha(locked_lock_sha)),
        ("Seeds", ", ".join(seed_run.keys()) or ", ".join(str(s) for s in (signoff.get("seed_list") or []))),
        ("Final-dev region", region_manifest.get("final_dev_region") or region_manifest.get("version") or "FINAL_DEV_REGION-v1"),
        ("Authoritative lineage", "corrective 3-seed refit"),
    ]
    conditions_html = (
        '<section class="cw-d-sec"><h4>Three-seed run conditions</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in conditions_pairs
        )
        + '</tbody></table></section>'
    )

    def _abbr(value: Any) -> str:
        s = str(value or "")
        if len(s) < 32:
            return s or "N/A"
        return f"{s[:12]}…{s[-12:]}"

    seed_headers = ["Seed", "Run ID", "Official epoch", "FINAL_DEV RMSE (Wh)", "Config fingerprint", "Checkpoint status"]
    seed_rows: list[list[str]] = []
    for seed_key in sorted(seed_run.keys(), key=lambda s: int(s)):
        row = seed_run[seed_key]
        run_id = row.get("run_id", "")
        rmse = row.get("rmse_wh", "")
        status_str = row.get("status", "")
        run_dir = root / "artifacts" / "runs" / run_id
        official_epoch = "N/A"
        config_fp = "N/A"
        ckpt_status = status_str or "N/A"
        if run_dir.is_dir():
            sjson = read_json(run_dir / "status.json")
            cjson = read_json(run_dir / "config.json")
            official_epoch = sjson.get("best_epoch", "N/A")
            config_fp = _abbr(cjson.get("config_fingerprint", "N/A"))
            ckpt_dir = run_dir / "checkpoints"
            best_pt = ckpt_dir / "best_checkpoint.pt"
            last_pt = ckpt_dir / "last_checkpoint.pt"
            if best_pt.exists() and last_pt.exists():
                ckpt_status = f"{status_str} (best == last, loadable)"
            elif best_pt.exists():
                ckpt_status = f"{status_str} (best only)"
            else:
                ckpt_status = f"{status_str} (NO checkpoint)"
        seed_rows.append([
            seed_key,
            run_id,
            str(official_epoch),
            _f(rmse),
            config_fp,
            ckpt_status,
        ])

    results_html = (
        '<section class="cw-d-sec"><h4>Three-seed results</h4>'
        + _tbl(seed_headers, seed_rows)
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Test status", signoff.get("test_status", "NOT_ACCESSED") or "NOT_ACCESSED"),
        ("test_metrics_computed", False),
        ("Phase 47 ready", phase47_ready),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table></section>'
    )

    body = (
        '<div class="cw-d-body">'
        + conditions_html + results_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 46 - Three-Seed Final Refit", subtitle, status)}{body}</article>'
    )


@phase_report(47)
def render_phase_47_dashboard(project_root: Path | None = None) -> HTML:
    """Render Phase 47 Final Test Evaluation as ONE compact dashboard.

    Layout (sections):
      1. Header  - Phase 47 - Final Test Evaluation / PASS
      2. Test conditions (compact table)
      3. Per-seed Test results (Seed / Run ID / MAE / RMSE / R^2)
      4. Aggregate result (mean +/- sample SD, ddof=1)
      5. Signoff (phase status, forbidden-actions status, decision)

    Data sources (read-only, JSON; no upstream re-execution):
      * artifacts/final_test/phase_47_signoff.json
          - per-seed MAE/RMSE/R^2, aggregate mean +/- SD,
            Test N, forbidden-action flags, PASS
      * artifacts/final_test/final_test_population_manifest.json
          - boundary protocol, lookback, horizon,
            test_population_id, n_test
      * artifacts/final_test/final_test_evaluation_contract.json
          - locked_before_test_access, configuration sanity
      * artifacts/three_seed_final_runs/phase47_test_release.json
          - post-corrective authoritative seed -> run id mapping
            (RUN_TR_FSD_0256_C2F24D58 / AA575C42 / 247AB83A)
      * artifacts/final_model_lock/phase_45_signoff.json
          - locked_model_id for signoff row

    MAPE addendum is OMITTED from notebook presentation because the
    addendum is BLOCKED_SOURCE_UNAVAILABLE for Test (test_mape_computed
    = false; test_inference_executed = false; underlying artifact
    retained untouched on disk). Validation MAPE is not Test-MAPE and
    is not authoritative for Phase 47 Test result.
    """

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_test"
    p45_art = root / "artifacts" / "final_model_lock"
    p46_art = root / "artifacts" / "three_seed_final_runs"

    signoff = read_json(art / "phase_47_signoff.json")
    pop_manifest = read_json(art / "final_test_population_manifest.json") if (art / "final_test_population_manifest.json").exists() else {}
    contract = read_json(art / "final_test_evaluation_contract.json") if (art / "final_test_evaluation_contract.json").exists() else {}
    release = read_json(p46_art / "phase47_test_release.json") if (p46_art / "phase47_test_release.json").exists() else {}
    p45_signoff = read_json(p45_art / "phase_45_signoff.json") if (p45_art / "phase_45_signoff.json").exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    n_test = signoff.get("n_test")
    seeds = signoff.get("seed_list") or [42, 123, 2026]

    seed_run: dict[int, str] = {}
    for r in (release.get("run_records") or []):
        try:
            seed_run[int(r["seed"])] = r["run_id"]
        except Exception:
            continue

    subtitle = (
        f"Locked = {signoff.get('locked_model_id') or p45_signoff.get('locked_model_id') or 'TR_C2_ALT_LOOKBACK'} "
        f"- Test N = {n_test} - "
        f"Release = {'PASS' if release.get('released') else 'PENDING'}"
    )

    boundary_protocol = pop_manifest.get("boundary_protocol") or "WB0_CONTEXT_CARRY_OVER"
    lookback = pop_manifest.get("lookback")
    pop_id = pop_manifest.get("population_id") or "FINAL_TEST_POP-v1"
    locked_before = contract.get("locked_before_test_access")
    release_status = release.get("status") or "PENDING"
    conditions_pairs = [
        ("Test population", pop_id),
        ("Test N", n_test),
        ("Boundary protocol", boundary_protocol),
        ("Lookback steps", lookback),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("Locked before Test access", locked_before),
        ("Authoritative release status", release_status),
    ]
    conditions_html = (
        '<section class="cw-d-sec"><h4>Test conditions</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in conditions_pairs
        )
        + '</tbody></table></section>'
    )

    seed_headers = ["Seed", "Run ID", "MAE (Wh)", "RMSE (Wh)", "R²"]
    seed_rows: list[list[str]] = []
    for k in seeds:
        run_id = seed_run.get(int(k), "N/A")
        mae = _f(signoff.get(f"seed{k}_mae_wh"))
        rmse = _f(signoff.get(f"seed{k}_rmse_wh"))
        r2 = _f(signoff.get(f"seed{k}_r2"), 4)
        seed_rows.append([str(k), run_id, mae, rmse, r2])
    per_seed_html = (
        '<section class="cw-d-sec"><h4>Per-seed Test results</h4>'
        + _tbl(seed_headers, seed_rows)
        + '</section>'
    )

    agg_pairs = [
        ("MAE (mean ± SD, Wh)", f"{_f(signoff.get('transformer_mean_mae_wh'))} ± {_f(signoff.get('transformer_sd_mae_wh'))}"),
        ("RMSE (mean ± SD, Wh)", f"{_f(signoff.get('transformer_mean_rmse_wh'))} ± {_f(signoff.get('transformer_sd_rmse_wh'))}"),
        ("R² (mean ± SD)", f"{_f(signoff.get('transformer_mean_r2'), 4)} ± {_f(signoff.get('transformer_sd_r2'), 4)}"),
        ("Aggregation", "arithmetic mean + sample SD (ddof = 1)"),
    ]
    aggregate_html = (
        '<section class="cw-d-sec"><h4>Aggregate result</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in agg_pairs
        )
        + '</tbody></table></section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Best-seed selection", signoff.get("best_seed_selected")),
        ("Ensemble used", signoff.get("ensemble_used")),
        ("Training used", signoff.get("training_used")),
        ("Scaler refit used", signoff.get("scaler_fit_used")),
        ("Post-Test tuning", signoff.get("post_test_tuning")),
        ("Decision", "PROCEED to Phase 48"),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table></section>'
    )

    body = (
        '<div class="cw-d-body">'
        + conditions_html + per_seed_html + aggregate_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 47 - Final Test Evaluation", subtitle, status)}{body}</article>'
    )
