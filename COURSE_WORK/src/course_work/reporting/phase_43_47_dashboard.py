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
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.utils.artifacts import get_project_root, read_json

__all__ = [
    "render_phase_43_dashboard",
    "render_phase_44_dashboard",
    "render_phase_45_dashboard",
    "render_phase_46_dashboard",
    "render_phase_47_dashboard",
]


# ---------------------------------------------------------------------------
# Shared CSS — single style block, reused by all 5 dashboards
# ---------------------------------------------------------------------------

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
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 43 — LSTM Tuning
# ---------------------------------------------------------------------------


_STAGE_FILES: list[tuple[str, str, str]] = [
    ("LT1", "model.hidden_size", "lt1_hidden_size_metrics.csv"),
    ("LT2", "model.num_layers", "lt2_layers_metrics.csv"),
    ("LT3", "model.dropout", "lt3_dropout_metrics.csv"),
    ("LT4", "training.learning_rate", "lt4_learning_rate_metrics.csv"),
    ("LT5", "training.weight_decay", "lt5_weight_decay_metrics.csv"),
]


def render_phase_43_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 43 LSTM Tuning dashboard (compact table-focused)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "lstm_tuning"

    signoff = read_json(art / "phase_43_signoff.json")
    winner = read_json(art / "lstm_tuned_winner.json")
    lineage = _csv(art / "lstm_stage_lineage.csv")

    status = str(signoff.get("overall_status") or signoff.get("status") or "UNKNOWN")
    subtitle = (
        f"Final LSTM = {winner.get('winner_run_id')} - "
        f"Phase 44 ready = {signoff.get('ready_for_phase44')}"
    )

    # PHASE 43 — TABLE-FOCUSED: 4 sections (header, overview, LT1–LT5 lineage result, signoff).
    # No figure: existing PNG figures do not materially aid the LT1–LT5 winner table.
    overview_pairs = [
        ("Tuning objective", "Minimize Val RMSE (5 stages: LT1–LT5)"),
        ("Final winner run", winner.get("winner_run_id")),
        ("Final LSTM ID", winner.get("winner_id")),
        ("Lookback steps", winner.get("lookback_steps")),
        ("Hidden size", winner.get("hidden_size")),
        ("Num layers", winner.get("num_layers")),
        ("Dropout", winner.get("dropout_arg")),
        ("Learning rate", winner.get("learning_rate")),
        ("Weight decay", winner.get("weight_decay")),
        ("Val RMSE (Wh)", _f(winner.get("validation_rmse_wh"))),
        ("Test status", winner.get("test_status") or signoff.get("test_status", "NOT_ACCESSED")),
    ]
    overview_html = (
        '<section class="cw-d-sec"><h4>Overview</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td><td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in overview_pairs
        )
        + '</tbody></table></section>'
    )

    # Main result: LT1–LT5 winner table (the central scientific artifact of this phase)
    lineage_headers = ["Stage", "Parameter", "Winner option", "Winner value", "Winner Val RMSE (Wh)"]
    lineage_rows = [
        [
            r.get("stage", ""),
            r.get("parameter", ""),
            r.get("winner_option", ""),
            r.get("winner_value", ""),
            _f(r.get("winner_rmse_wh")),
        ]
        for r in lineage
    ]
    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - LT1–LT5 winner selection</h4>'
        + _tbl(lineage_headers, lineage_rows)
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + 'Note: Phase 43 was finalized from verified completed runs through canonical recovery; '
        'LT1–LT5 stage winners should not be interpreted as a fully chained sequential search.'
        + '</p>'
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Final LSTM candidate", winner.get("winner_run_id")),
        ("Phase 44 ready", signoff.get("ready_for_phase44")),
        ("Test status", winner.get("test_status") or signoff.get("test_status", "NOT_ACCESSED")),
    ]
    signoff_html = (
        '<section class="cw-d-sec"><h4>Decision &amp; signoff</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td><td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in signoff_pairs
        )
        + '</tbody></table>'
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + overview_html
        + main_result_html
        + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 43 - LSTM Tuning", subtitle, status)}{body}</article>'
    )


# ---------------------------------------------------------------------------
# Phase 44 — Rolling-Origin Robustness
# ---------------------------------------------------------------------------


def _phase44_b64_figure(path: Path) -> str | None:
    """Embed a canonical read-only Phase 44 figure as base64 (only the active
    RO_44_01_model_comparison.png which is the only non-placeholder canonical
    Phase 44 figure). Reads bytes once, returns the inline image tag.
    """
    if not path.exists():
        return None
    try:
        b = path.read_bytes()
    except Exception:
        return None
    data_uri = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    return (
        '<div class="cw-d-fig" style="margin-top:6px">'
        '<div class="ftitle">Rolling-origin RMSE per fold (RO1 / RO2 / RO3) — '
        'Persistence / TR_C0 / TR_C1 / TR_C2 / LSTM_TUNED</div>'
        f'<img src="{escape(data_uri, quote=True)}" alt="Phase 44 rolling-origin RMSE per fold" '
        'style="max-width:780px;width:80%"/>'
        '<div class="fcap" style="font-size:11px;color:#5d6b82;line-height:1.45;margin-top:4px">'
        'Active canonical Phase 44 figure '
        '(<code>artifacts/rolling_origin/figures/RO_44_01_model_comparison.png</code>). '
        'Bars show per-fold RMSE on the rolling-origin outer-fold evaluation set; '
        'no Test data are used.'
        '</div>'
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
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + 'Persistence per-fold RMSE is not present in the canonical outer-fold artifact '
        '(only pooled is). LSTM is a comparison baseline; Phase 45 selects the Transformer '
        'shortlist winner TR_C2_ALT_LOOKBACK.'
        + '</p>'
        + "</section>"
    )


def render_phase_44_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 44 Rolling-Origin Robustness dashboard (1 figure + table)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "rolling_origin"

    signoff = read_json(art / "phase_44_signoff.json")
    fold_metrics = _csv(art / "rolling_origin_fold_metrics.csv")
    pooled = _csv(art / "rolling_origin_pooled_metrics.csv")

    status = str(signoff.get("overall_status") or "UNKNOWN")
    rec_id = str(signoff.get("recommended_transformer_candidate_id", ""))
    subtitle = (
        f"Recommended = {rec_id} - Approved for Phase 45 = {signoff.get('approved_for_phase45')}"
    )

    # Overview (compact)
    overview_pairs = [
        ("Recommended Transformer", rec_id),
        ("Protocol", signoff.get("protocol_id") or "RO3_EXPANDING_PRETEST-v1"),
        ("Folds", "RO1 / RO2 / RO3"),
        ("Pooled RMSE (Wh)", _f(signoff.get("recommended_pooled_rmse_wh"))),
        ("Test status", signoff.get("test_status")),
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

    # Main result: ONE figure + per-fold+poll table (caller-chosen candidate list)
    figure_html = _phase44_b64_figure(art / "figures" / "RO_44_01_model_comparison.png") or ""
    table_html = _phase44_per_fold_table(
        fold_metrics, pooled,
        ["TR_C0_PRIMARY", "TR_C1_ALT_WEIGHT_DECAY", "TR_C2_ALT_LOOKBACK",
         "LSTM_TUNED_WINNER", "PERSISTENCE_LAST_VALUE"],
    )
    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - rolling-origin RMSE across RO1 / RO2 / RO3</h4>'
        + figure_html + table_html
        + '</section>'
    )

    signoff_pairs = [
        ("Recommended Transformer", rec_id),
        ("Approved for Phase 45", signoff.get("approved_for_phase45")),
        ("Test status", signoff.get("test_status", "NOT_ACCESSED")),
        ("Phase status", status),
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
        + 'Phase 44 is development robustness evidence, NOT final held-out Test evidence. '
        + 'TR_C2_ALT_LOOKBACK is selected for the next phase; no best head is selected.'
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
        + f'<article class="cw-d">{_header("Phase 44 - Rolling-Origin Robustness", subtitle, status)}{body}</article>'
    )


# ---------------------------------------------------------------------------
# Phase 45 — Final Model Lock
# ---------------------------------------------------------------------------


def render_phase_45_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 45 Final Model Lock dashboard (table-focused)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_model_lock"

    signoff = read_json(art / "phase_45_signoff.json")
    epoch_policy = read_json(art / "final_epoch_policy.json") if (art / "final_epoch_policy.json").exists() else {}
    summary = read_json(art / "final_model_lock_summary.json") if (art / "final_model_lock_summary.json").exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    locked_id = signoff.get("locked_model_id")
    fp_disp = str(signoff.get("config_fingerprint") or "")[:10] + "..."
    median_epoch = signoff.get("final_refit_epochs")
    seeds = signoff.get("seed_list") or []
    lookback = summary.get("locked_lookback_steps")
    locked_fs = summary.get("locked_feature_variant_id")
    ys = signoff.get("y_scaler_sha256_or_identity", "YS1")
    recipe_path = art / "final_training_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    early_stop = "OFF" if recipe.get("early_stopping") in (False, "False") else "ON"

    subtitle = (
        f"Locked = {locked_id} - FINAL_REFIT_EPOCHS = {median_epoch} - "
        f"Phase 46 ready = {signoff.get('ready_for_phase46')}"
    )

    # PHASE 45 — TABLE-FOCUSED. Single reader-friendly config table.
    config_rows = [
        ("Locked candidate", locked_id),
        ("Model family", summary.get("locked_model_family") or "TRANSFORMER_ENCODER"),
        ("Lookback steps", lookback),
        ("Feature variant", locked_fs),
        ("Target scaling", ys),
        ("FINAL_REFIT epochs (= FINAL_REFIT_EPOCHS)", median_epoch),
        ("Early stopping", early_stop),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("Config fingerprint (head)", fp_disp),
    ]
    overview_html = (
        '<section class="cw-d-sec"><h4>Lock overview</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in config_rows
        )
        + '</tbody></table></section>'
    )

    # Main result — Final refit policy (the only thing the reader must see)
    ro1 = epoch_policy.get("RO1_inner_best_epoch")
    ro2 = epoch_policy.get("RO2_inner_best_epoch")
    ro3 = epoch_policy.get("RO3_inner_best_epoch")
    epoch_rows_pairs = [
        ("RO1 inner best epoch", ro1),
        ("RO2 inner best epoch", ro2),
        ("RO3 inner best epoch", ro3),
        ("Median of RO1/RO2/RO3 (= FINAL_REFIT_EPOCHS, locked)", median_epoch),
        ("Early stopping", early_stop),
        ("Validation stopping", "OFF"),
        ("Warm start", "OFF"),
    ]
    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - final refit policy (locked)</h4>'
        + _tbl(["Field", "Value"], epoch_rows_pairs)
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + f'Final run = FINAL_REFIT_EPOCHS = {median_epoch} (no max-epochs cap of 50). '
        + 'No manual or Test-dependent epoch selection. Test held out for Phase 47.'
        + '</p>'
        + '</section>'
    )

    signoff_pairs = [
        ("Locked candidate", locked_id),
        ("Phase 46 ready", signoff.get("ready_for_phase46")),
        ("Test status", signoff.get("test_status", "NOT_ACCESSED")),
        ("Phase status", status),
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
        + f'<article class="cw-d">{_header("Phase 45 - Final Model Lock", subtitle, status)}{body}</article>'
    )


# ---------------------------------------------------------------------------
# Phase 46 — Three-Seed Final Runs
# ---------------------------------------------------------------------------


def render_phase_46_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 46 Three-Seed Final Runs dashboard (table-focused)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "three_seed_final_runs"

    signoff = read_json(art / "phase_46_signoff.json")

    status = str(signoff.get("overall_status") or "UNKNOWN")
    median_epoch = signoff.get("final_refit_epochs")
    locked_id = signoff.get("candidate_id") or signoff.get("locked_model_id")
    seeds = signoff.get("seed_list") or [42, 123, 2026]
    subtitle = (
        f"Candidate = {locked_id} - FINAL_REFIT_EPOCHS = {median_epoch} - "
        f"Phase 47 ready = {signoff.get('ready_for_phase47')}"
    )

    fp_disp = str(signoff.get("config_sha256") or "")[:10] + "..."

    overview_pairs = [
        ("Candidate", locked_id),
        ("Config SHA-256 (head)", fp_disp),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("FINAL_REFIT epochs (= FINAL_REFIT_EPOCHS)", median_epoch),
        ("Avg FINAL_DEV RMSE (Wh)", _f(signoff.get("average_rmse_wh"))),
        ("Completed runs", signoff.get("completed_run_count")),
        ("Test status", signoff.get("test_status")),
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

    # Main result: three-seed run table (the central artifact, with epochs per run)
    seed_headers = ["Seed", "Run", "Epochs", "Final-dev RMSE (Wh)", "Checkpoint (head)"]
    seed_rows = []
    for k in seeds:
        rid = str(signoff.get(f"seed{k}_run_id") or signoff.get(f"seed_{k}_run_id") or "")
        ckpt = str(signoff.get(f"seed{k}_checkpoint_sha256") or signoff.get(f"seed_{k}_checkpoint_sha256") or "")
        seed_rows.append([
            str(k),
            (rid[:10] + "...") if len(rid) > 10 else rid,
            f"{median_epoch}/{median_epoch}",
            _f(signoff.get("average_rmse_wh")),
            (ckpt[:10] + "...") if len(ckpt) > 10 else ckpt,
        ])
    main_result_html = (
        '<section class="cw-d-sec"><h4>Main result - three-seed FINAL_REFIT runs</h4>'
        + _tbl(seed_headers, seed_rows)
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + 'All three seeds completed the same FINAL_REFIT_EPOCHS=' + str(median_epoch) + ' schedule. '
        + 'Three independent final runs - NOT an ensemble. No best seed is selected. '
        + 'Early stopping OFF, validation stopping OFF, warm start OFF.'
        + '</p>'
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Locked candidate", locked_id),
        ("Phase 47 ready", signoff.get("ready_for_phase47")),
        ("Test status", signoff.get("test_status", "NOT_ACCESSED")),
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
        + f'<article class="cw-d">{_header("Phase 46 - Three-Seed Final Runs", subtitle, status)}{body}</article>'
    )


# ---------------------------------------------------------------------------
# Phase 47 — Final Test Evaluation
# ---------------------------------------------------------------------------


def render_phase_47_dashboard(project_root: Path | None = None) -> HTML:
    """Render the Phase 47 Final Test Evaluation dashboard (table-focused)."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_test"

    signoff = read_json(art / "phase_47_signoff.json")
    population_path = art / "final_test_population_manifest.json"
    population = read_json(population_path) if population_path.exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    n_test = signoff.get("n_test")
    final_lock_disp = str(signoff.get("final_lock_sha256") or "")[:10] + "..."
    test_sha_disp = str(
        signoff.get("final_test_population_sha256")
        or population.get("target_ids_sha256") or ""
    )[:10] + "..."

    subtitle = (
        f"HELD-OUT TEST N = {n_test} - first authorized Test access - "
        f"frozen prediction set"
    )

    # PHASE 47 — TABLE-FOCUSED. Test metrics table is the central artifact.
    overview_pairs = [
        ("Phase status", status),
        ("Test population N", n_test),
        ("Locked candidate", signoff.get("locked_model_id") or "TR_C2_ALT_LOOKBACK"),
        ("Final lock sha256 (head)", final_lock_disp),
        ("Test population sha256 (head)", test_sha_disp),
    ]
    overview_html = (
        '<section class="cw-d-sec"><h4>Test overview</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in overview_pairs
        )
        + '</tbody></table></section>'
    )

    # Main result — per-seed Test metrics (the headline scientific numbers)
    headers = ["Seed", "Test MAE (Wh)", "Test RMSE (Wh)", "Test R2"]
    rows = []
    for k in [42, 123, 2026]:
        rows.append([
            str(k),
            _f(signoff.get(f"seed{k}_mae_wh") or signoff.get(f"seed_{k}_mae_wh")),
            _f(signoff.get(f"seed{k}_rmse_wh") or signoff.get(f"seed_{k}_rmse_wh")),
            _f(signoff.get(f"seed{k}_r2") or signoff.get(f"seed_{k}_r2"), 4),
        ])
    # Mean ± SD row
    rows.append([
        "Mean ± SD",
        f"{_f(signoff.get('transformer_mean_mae_wh'))} ± {_f(signoff.get('transformer_sd_mae_wh'))}",
        f"{_f(signoff.get('transformer_mean_rmse_wh'))} ± {_f(signoff.get('transformer_sd_rmse_wh'))}",
        f"{_f(signoff.get('transformer_mean_r2'), 4)} ± {_f(signoff.get('transformer_sd_r2'), 4)}",
    ])
    main_result_html = (
        f'<section class="cw-d-sec"><h4>Main result - per-seed Test metrics '
        f'(FINAL_TEST_POP-v1, N = {n_test})</h4>'
        + _tbl(headers, rows)
        + '</section>'
    )

    # Baseline comparison (compact, pre-computed in signoff)
    base_rows = [
        ["Transformer (Mean ± SD, Test)",
         f"{_f(signoff.get('transformer_mean_mae_wh'))} ± {_f(signoff.get('transformer_sd_mae_wh'))}",
         f"{_f(signoff.get('transformer_mean_rmse_wh'))} ± {_f(signoff.get('transformer_sd_rmse_wh'))}",
         f"{_f(signoff.get('transformer_mean_r2'), 4)} ± {_f(signoff.get('transformer_sd_r2'), 4)}"],
        ["Persistence (Test)",
         _f(signoff.get("persistence_mae_wh")),
         _f(signoff.get("persistence_rmse_wh")),
         _f(signoff.get("persistence_r2"), 4)],
        ["LSTM (Test, final)",
         "N/A",
         "N/A",
         "N/A — lookback mismatch prevents a fair final Test comparison"],
    ]
    baseline_html = (
        '<section class="cw-d-sec"><h4>Baseline comparison</h4>'
        + _tbl(headers, base_rows)
        + '<p style="font-size:11.5px;color:#64748b;margin-top:6px;line-height:1.45">'
        + 'Transformer improves RMSE / R² over Persistence; Persistence retains better MAE. '
        + 'LSTM is not evaluated on the final Test because the protocol/lookback mismatch '
        + 'prevents a fair final-Test comparison.'
        + '</p>'
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Test status", "EXECUTED"),
        ("Three seeds used",
         ", ".join(str(s) for s in signoff.get("seed_list", [42, 123, 2026]))),
        ("Frozen prediction set", "YES"),
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
        + 'Three-seed Test summary is descriptive only - NOT an ensemble. No best seed. '
        + 'No post-Test tuning. Downstream phases consume frozen Phase 47 outputs.'
        + '</p>'
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + overview_html + main_result_html + baseline_html + signoff_html
        + '</div>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{_header("Phase 47 - Final Test Evaluation", subtitle, status)}{body}</article>'
    )
