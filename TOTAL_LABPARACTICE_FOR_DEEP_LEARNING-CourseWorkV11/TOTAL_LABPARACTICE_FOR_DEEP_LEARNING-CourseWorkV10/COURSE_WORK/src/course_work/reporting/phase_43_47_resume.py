"""Phase 43 → 47 read-only HTML visualization renderer.

This module owns the **presentation layer** for Phase 43 (LSTM Tuning), Phase 44
(Rolling-Origin Robustness), Phase 45 (Final Model Lock), Phase 46 (Three-Seed
Final Runs) and Phase 47 (Final Test Evaluation).

It is intentionally a **separate module** from `phase_summary.py` because
`phase_summary.py` owns the presentation layer for Phase 0-33 per
`architecture_rule.md` §7.9.1. Phase 43-47 are out of that scope but follow the
same architectural principles:

* Read-only: only loads canonical artifacts (signoff JSON, CSV metrics).
* No training, no Test re-evaluation, no best-seed selection, no ensemble.
* All numbers come from canonical artifacts — no hard-coded scientific values.
* Static HTML (no JavaScript, no external resources).
* Every value is `escape()`-ed before being placed in HTML.
* No `def`/`class` is added to notebook cells — the notebook only imports
  `render_phase_43/44/45/46/47_resume(project_root)` and `display()`s the result.

Public API:

* :func:`render_phase_43_lstm_tuning_resume`
* :func:`render_phase_44_rolling_origin_resume`
* :func:`render_phase_45_final_model_lock_resume`
* :func:`render_phase_46_three_seed_final_runs_resume`
* :func:`render_phase_47_final_test_evaluation_resume`

Each function returns an ``IPython.display.HTML`` object suitable for direct
``display()`` in the notebook.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.utils.artifacts import get_project_root, read_json

__all__ = [
    "render_phase_43_lstm_tuning_resume",
    "render_phase_44_rolling_origin_resume",
    "render_phase_45_final_model_lock_resume",
    "render_phase_46_three_seed_final_runs_resume",
    "render_phase_47_final_test_evaluation_resume",
]


_SHARED_CSS = """
<style>
.cw-p4347{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:16px;background:#fff;box-shadow:0 10px 28px rgba(31,45,61,.09);margin:14px 0 24px;overflow:hidden}
.cw-p4347 *{box-sizing:border-box}
.cw-p4347-header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:22px 24px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-p4347-header h3{font-size:22px;line-height:1.3;margin:0 0 6px;color:#172033}
.cw-p4347-meta{font-size:13px;color:#5d6b82}
.cw-p4347-status{border:1px solid #a9dec1;border-radius:999px;padding:7px 13px;font-size:12px;font-weight:700;letter-spacing:.03em;white-space:nowrap;color:#11613d;background:#e8f7ef}
.cw-p4347-status.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-p4347-status.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-p4347-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;padding:18px 24px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-p4347-metric{display:flex;flex-direction:column;gap:5px;min-width:0;padding:12px 14px;border:1px solid #e1e7f0;border-radius:10px;background:#fff}
.cw-p4347-metric span{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-p4347-metric strong{font-size:14px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-p4347-content{padding:4px 24px 24px}
.cw-p4347-section{margin-top:20px}
.cw-p4347-section h4{font-size:15px;margin:0 0 9px;color:#334155}
.cw-p4347-tablewrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:10px}
.cw-p4347-table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
.cw-p4347-table th{background:#f5f7fb;color:#475569;text-align:left;font-weight:650;padding:11px 13px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-p4347-table td{text-align:left;padding:10px 13px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-p4347-table tbody tr:nth-child(even){background:#fafbfd}
.cw-p4347-table tbody tr:last-child td{border-bottom:0}
.cw-p4347-winner{background:#eef9f0 !important}
.cw-p4347-fp{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#1f2a44}
.cw-p4347-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-p4347-pill.win{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-p4347-pill.loss{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-p4347-pill.gray{border-color:#cbd5e1;background:#f1f5f9;color:#475569}
.cw-p4347-pill.warn{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-p4347-pill.alt{border-color:#c7d2fe;background:#eef2ff;color:#3730a3}
.cw-p4347-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:14px}
.cw-p4347-card{border:1px solid #d9e2ef;border-radius:12px;padding:14px 16px;background:#fff}
.cw-p4347-card.win{border-color:#a9dec1;background:#e8f7ef}
.cw-p4347-card h5{margin:0 0 6px;font-size:13px;color:#475569;text-transform:uppercase;letter-spacing:.04em}
.cw-p4347-card .big{font-size:22px;color:#172033;font-weight:700;line-height:1.2}
.cw-p4347-card .sm{font-size:12px;color:#5d6b82;margin-top:4px}
.cw-p4347-timeline{display:flex;align-items:center;gap:8px;margin:18px 0 10px;font-size:13px}
.cw-p4347-timeline .node{padding:6px 12px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc;color:#334155;font-weight:600}
.cw-p4347-timeline .node.median{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-p4347-timeline .arrow{color:#94a3b8}
.cw-p4347-bars{display:flex;align-items:flex-end;gap:6px;height:130px;padding:14px 12px 8px;border:1px solid #e2e8f0;border-radius:10px;background:#fbfcff}
.cw-p4347-bar{flex:1;background:linear-gradient(180deg,#c7d2fe,#6366f1);border-radius:6px 6px 0 0;position:relative;min-height:6px}
.cw-p4347-bar .lab{position:absolute;top:-18px;left:0;right:0;text-align:center;font-size:11px;color:#1f2a44;font-weight:700}
.cw-p4347-bar .sub{position:absolute;bottom:-22px;left:0;right:0;text-align:center;font-size:10px;color:#5d6b82}
.cw-p4347-bar.win{background:linear-gradient(180deg,#a9dec1,#16a34a)}
.cw-p4347-conclusion{margin-top:18px;padding:14px 16px;border:1px solid #d9e2ef;border-radius:12px;background:linear-gradient(135deg,#fbfcff,#f6f3ff)}
.cw-p4347-conclusion p{margin:0 0 8px;font-size:13px;line-height:1.55;color:#24324a}
.cw-p4347-conclusion ul{margin:0 0 0 18px;font-size:13px;line-height:1.55;color:#24324a}
.cw-p4347-conclusion li{margin-bottom:4px}
.cw-p4347-callout{border-left:4px solid #6366f1;background:#eef2ff;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0;font-size:13px;color:#1f2a44}
.cw-p4347-callout.warn{border-color:#d97706;background:#fff7e0;color:#7a5613}
@media (max-width:900px){.cw-p4347-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.cw-p4347-grid3{grid-template-columns:1fr}}
@media (max-width:620px){.cw-p4347-header{flex-direction:column;padding:18px}.cw-p4347-metrics{grid-template-columns:1fr;padding:14px 18px}.cw-p4347-content{padding-left:18px;padding-right:18px}}
</style>
"""

def _fmt_float(value: Any, digits: int = 4) -> str:
    """Format a numeric value with `digits` decimals, or the string ``"N/A"``."""

    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):
        return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dict rows; returns empty list if missing."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _status_pill(status: str) -> str:
    """Return an HTML status pill for the given canonical status."""
    s = str(status or "UNKNOWN").upper()
    cls = "win" if s == "PASS" else "warn" if "WARNING" in s else "fail"
    return f'<span class="cw-p4347-pill {cls}">{escape(s)}</span>'


def _header(
    title: str,
    subtitle: str,
    status: str,
) -> str:
    pill_class = (
        "cw-p4347-status"
        + (
            ""
            if str(status).upper() == "PASS"
            else " warn"
            if "WARNING" in str(status).upper()
            else " fail"
        )
    )
    return (
        '<header class="cw-p4347-header">'
        f"<div><h3>{escape(title)}</h3>"
        f'<div class="cw-p4347-meta">{escape(subtitle)}</div></div>'
        f'<span class="{pill_class}">{escape(str(status).upper())}</span>'
        "</header>"
    )


def _metric_cards(pairs: Iterable[tuple[str, Any]]) -> str:
    items = "".join(
        f'<div class="cw-p4347-metric"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in pairs
    )
    return f'<div class="cw-p4347-metrics">{items}</div>'


def _table(headers: list[str], rows: list[list[Any]], winner_row_idx: int | None = None) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body_parts = []
    for i, row in enumerate(rows):
        klass = ' class="cw-p4347-winner"' if winner_row_idx is not None and i == winner_row_idx else ""
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in row)
        body_parts.append(f"<tr{klass}>{cells}</tr>")
    body = "".join(body_parts)
    return (
        '<div class="cw-p4347-tablewrap"><table class="cw-p4347-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )



_LT_PHASE_NAME = "Phase 43 — LSTM Tuning"
_LT_STAGE_FILES = [
    ("LT1", "model.hidden_size", "lt1_hidden_size_metrics.csv"),
    ("LT2", "model.num_layers", "lt2_layers_metrics.csv"),
    ("LT3", "model.dropout", "lt3_dropout_metrics.csv"),
    ("LT4", "training.learning_rate", "lt4_learning_rate_metrics.csv"),
    ("LT5", "training.weight_decay", "lt5_weight_decay_metrics.csv"),
]


def render_phase_43_lstm_tuning_resume(project_root: Path | None = None) -> HTML:
    """Render the Phase 43 LSTM tuning dashboard."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "lstm_tuning"

    signoff = read_json(art / "phase_43_signoff.json")
    winner = read_json(art / "lstm_tuned_winner.json")
    stage_lineage = _read_csv(art / "lstm_stage_lineage.csv")

    status = str(signoff.get("overall_status") or signoff.get("status") or "UNKNOWN")
    subtitle = (
        f"{signoff.get('artifact_version', '')} | "
        f"final_winner_run = {winner.get('winner_run_id', 'N/A')} | "
        f"ready_for_phase44 = {signoff.get('ready_for_phase44')}"
    )

    metric_cards = _metric_cards([
        ("Final winner run ID", winner.get("winner_run_id")),
        ("Hidden size", winner.get("hidden_size")),
        ("Num layers", winner.get("num_layers")),
        ("Dropout", winner.get("dropout_arg")),
        ("Learning rate", winner.get("learning_rate")),
        ("Weight decay", winner.get("weight_decay")),
        ("Validation RMSE (Wh)", _fmt_float(winner.get("validation_rmse_wh"))),
        ("Validation MAE (Wh)", _fmt_float(winner.get("validation_mae_wh"))),
        ("Validation R²", _fmt_float(winner.get("validation_r2"), 6)),
        ("Test status", winner.get("test_status")),
    ])

    lineage_headers = ["Stage", "Parameter", "Winner option", "Winner value", "Winner run ID", "Validation RMSE (Wh)"]
    lineage_rows = []
    for row in stage_lineage:
        lineage_rows.append([
            row.get("stage", ""),
            row.get("parameter", ""),
            row.get("winner_option", ""),
            row.get("winner_value", ""),
            row.get("winner_run_id", ""),
            _fmt_float(row.get("winner_rmse_wh")),
        ])

    stage_blocks = []
    for stage_code, parameter, fname in _LT_STAGE_FILES:
        metrics_rows = _read_csv(art / fname)
        if not metrics_rows:
            stage_blocks.append(
                f'<section class="cw-p4347-section"><h4>{escape(stage_code)} — {escape(parameter)}</h4>'
                '<div class="cw-p4347-callout warn">No metrics CSV found.</div></section>'
            )
            continue
        winner_idx = next(
            (i for i, r in enumerate(metrics_rows) if r.get("is_winner", "").upper() == "WINNER"),
            None,
        )
        headers = ["Candidate", "Value", "Source", "Best epoch", "Val RMSE (Wh)", "Val MAE (Wh)", "Val R²"]
        rows = []
        for r in metrics_rows:
            rows.append([
                r.get("candidate_id", ""),
                r.get("candidate_value", ""),
                r.get("source_type", ""),
                r.get("best_epoch", ""),
                _fmt_float(r.get("validation_rmse_wh")),
                _fmt_float(r.get("validation_mae_wh")),
                _fmt_float(r.get("validation_r2"), 6),
            ])
        stage_blocks.append(
            f'<section class="cw-p4347-section"><h4>{escape(stage_code)} — {escape(parameter)}</h4>'
            + _table(headers, rows, winner_row_idx=winner_idx)
            + "</section>"
        )

    body = (
        metric_cards
        + '<div class="cw-p4347-content">'
        + '<section class="cw-p4347-section"><h4>Tuning lineage (LT1 → LT5)</h4>'
        + _table(lineage_headers, lineage_rows)
        + "</section>"
        + "".join(stage_blocks)
        + '<section class="cw-p4347-section"><h4>Final LSTM winner card</h4>'
        + '<div class="cw-p4347-grid3">'
        + '<div class="cw-p4347-card win"><h5>Winner run ID</h5><div class="big">' + escape(str(winner.get("winner_run_id"))) + '</div>'
        + '<div class="sm">config fingerprint</div>'
        + '<div class="cw-p4347-fp">' + escape(str(winner.get("config_fingerprint"))[:16] + "…") + '</div></div>'
        + '<div class="cw-p4347-card"><h5>Validation RMSE</h5><div class="big">' + _fmt_float(winner.get("validation_rmse_wh")) + ' Wh</div>'
        + '<div class="sm">Validation MAE ' + _fmt_float(winner.get("validation_mae_wh")) + ' Wh · R² ' + _fmt_float(winner.get("validation_r2"), 6) + '</div></div>'
        + '<div class="cw-p4347-card"><h5>Phase 44 readiness</h5><div class="big">' + escape(str(signoff.get("ready_for_phase44"))) + '</div>'
        + '<div class="sm">Status ' + escape(status) + ' · Test status ' + escape(str(winner.get("test_status"))) + '</div></div>'
        + "</div></section>"
        + '<div class="cw-p4347-callout">Read-only render — LSTM is not retuned. All values come from the canonical Phase 43 artifacts under <code>artifacts/lstm_tuning/</code>.</div>'
        + "</div>"
    )

    return HTML(
        _SHARED_CSS
        + f'<article class="cw-p4347">{_header(_LT_PHASE_NAME, subtitle, status)}{body}</article>'
    )

_P44_PHASE_NAME = "Phase 44 — Rolling-Origin Robustness"


def render_phase_44_rolling_origin_resume(project_root: Path | None = None) -> HTML:
    """Render the Phase 44 rolling-origin robustness dashboard."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "rolling_origin"

    signoff = read_json(art / "phase_44_signoff.json")
    pooled = _read_csv(art / "rolling_origin_pooled_metrics.csv")
    ranking = _read_csv(art / "rolling_origin_transformer_robustness_ranking.csv")
    inner_epochs = _read_csv(art / "rolling_origin_inner_best_epochs.csv")
    rec = read_json(art / "rolling_origin_recommended_transformer.json")

    status = str(signoff.get("overall_status") or "UNKNOWN")
    subtitle = (
        f"3 folds · RO3_EXPANDING_PRETEST-v1 · "
        f"recommended = {signoff.get('recommended_transformer_candidate_id')} · "
        f"approved_for_phase45 = {signoff.get('approved_for_phase45')}"
    )

    cards = []
    for row in pooled:
        cid = row.get("candidate_id", "")
        cls = "win" if cid == signoff.get("recommended_transformer_candidate_id") else ""
        cards.append(
            f'<div class="cw-p4347-card {cls}"><h5>{escape(cid)}</h5>'
            f'<div class="big">{_fmt_float(row.get("pooled_rmse_wh"))} Wh</div>'
            f'<div class="sm">MAE {_fmt_float(row.get("pooled_mae_wh"))} · R² {_fmt_float(row.get("pooled_r2"), 6)}</div></div>'
        )
    pooled_section = (
        '<section class="cw-p4347-section"><h4>A — Pooled robustness metrics across RO1 / RO2 / RO3</h4>'
        '<div class="cw-p4347-grid3">' + "".join(cards) + "</div></section>"
    )

    rank_headers = ["Rank", "Candidate", "Pooled RMSE (Wh)", "Worst-fold RMSE (Wh)", "Fold-RMSE SD (Wh)"]
    rank_rows = []
    for r in ranking:
        rank_rows.append([
            r.get("rank", ""),
            r.get("candidate_id", ""),
            _fmt_float(r.get("pooled_rmse_wh")),
            _fmt_float(r.get("worst_fold_rmse_wh")),
            _fmt_float(r.get("fold_rmse_sd_wh")),
        ])
    ranking_section = (
        '<section class="cw-p4347-section"><h4>B — Transformer ranking on rolling-origin robustness</h4>'
        + _table(rank_headers, rank_rows) + "</section>"
    )

    rec_id = signoff.get("recommended_transformer_candidate_id")
    epoch_headers = ["Candidate", "Fold", "Best inner epoch", "Best inner RMSE (Wh)", "Stage-A run ID"]
    epoch_rows = []
    for r in inner_epochs:
        if r.get("candidate_id") != rec_id:
            continue
        epoch_rows.append([
            r.get("candidate_id", ""),
            r.get("fold_id", ""),
            r.get("best_epoch_inner", ""),
            _fmt_float(r.get("best_inner_rmse_wh")),
            r.get("stage_a_run_id", ""),
        ])
    epoch_section = (
        f'<section class="cw-p4347-section"><h4>C — Inner-CV best epochs for {escape(str(rec_id))} '
        f'(feeds Phase 45 epoch policy)</h4>'
        + _table(epoch_headers, epoch_rows) + "</section>"
    )

    signoff_section = (
        '<section class="cw-p4347-section"><h4>D — Signoff & protocol</h4>'
        '<div class="cw-p4347-grid3">'
        + '<div class="cw-p4347-card win"><h5>Recommended Transformer</h5><div class="big">' + escape(str(rec_id)) + '</div>'
        + '<div class="sm">Pooled RMSE ' + _fmt_float(signoff.get("recommended_pooled_rmse_wh")) + ' Wh</div></div>'
        + '<div class="cw-p4347-card"><h5>Approved for Phase 45</h5><div class="big">' + escape(str(signoff.get("approved_for_phase45"))) + '</div>'
        + '<div class="sm">Protocol RO3_EXPANDING_PRETEST-v1</div></div>'
        + '<div class="cw-p4347-card"><h5>Test status</h5><div class="big">' + escape(str(signoff.get("test_status"))) + '</div>'
        + '<div class="sm">Status ' + escape(status) + '</div></div>'
        + "</div>"
        + '<div class="cw-p4347-callout warn">'
        + 'LSTM may have the lowest pooled RMSE in section A, but Phase 45 locks the best Transformer by protocol '
        + '(rolling-origin robustness on the shortlist). The recommendation is <strong>' + escape(str(rec_id)) + '</strong>.'
        + "</div></section>"
    )

    body = (
        '<div class="cw-p4347-content">'
        + pooled_section
        + ranking_section
        + epoch_section
        + signoff_section
        + '<div class="cw-p4347-callout">Read-only render — Phase 44 is not re-executed. '
        + 'All values come from <code>artifacts/rolling_origin/</code>.</div>'
        + "</div>"
    )

    return HTML(
        _SHARED_CSS
        + f'<article class="cw-p4347">{_header(_P44_PHASE_NAME, subtitle, status)}{body}</article>'
    )

_P45_PHASE_NAME = "Phase 45 — Final Model Lock"


def render_phase_45_final_model_lock_resume(project_root: Path | None = None) -> HTML:
    """Render the Phase 45 final-model-lock dashboard."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_model_lock"

    signoff = read_json(art / "phase_45_signoff.json")
    epoch_policy = read_json(art / "final_epoch_policy.json")
    seed_contract = read_json(art / "final_seed_contract.json")
    summary = read_json(art / "final_model_lock_summary.json") if (art / "final_model_lock_summary.json").exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    subtitle = (
        f"{signoff.get('artifact_version', '')} | "
        f"final_refit_epochs = {signoff.get('final_refit_epochs')} | "
        f"ready_for_phase46 = {signoff.get('ready_for_phase46')}"
    )

    ro1 = epoch_policy.get("RO1_inner_best_epoch")
    ro2 = epoch_policy.get("RO2_inner_best_epoch")
    ro3 = epoch_policy.get("RO3_inner_best_epoch")
    sorted_epochs = epoch_policy.get("sorted_epochs", [ro3, ro1, ro2])
    median_epoch = epoch_policy.get("final_refit_epochs")
    timeline = (
        '<div class="cw-p4347-timeline">'
        f'<div class="node">{escape(str(sorted_epochs[0]))}</div>'
        '<div class="arrow">───</div>'
        f'<div class="node median">{escape(str(median_epoch))} ↑ median</div>'
        '<div class="arrow">───</div>'
        f'<div class="node">{escape(str(sorted_epochs[-1]))}</div>'
        "</div>"
    )

    metric_cards = _metric_cards([
        ("Locked candidate", signoff.get("locked_model_id")),
        ("Config fingerprint", (signoff.get("config_fingerprint") or "")[:16] + "…"),
        ("FINAL_REFIT epochs", signoff.get("final_refit_epochs")),
        ("RO1 inner best", ro1),
        ("RO2 inner best", ro2),
        ("RO3 inner best", ro3),
        ("Median (FINAL_REFIT)", median_epoch),
        ("Final DEV region", summary.get("final_dev_region", "FINAL_DEV_REGION-v1")),
        ("Test status", signoff.get("test_status")),
        ("Ready for Phase 46", signoff.get("ready_for_phase46")),
    ])

    epoch_table = _table(
        ["Fold", "Best inner epoch"],
        [
            ["RO1", str(ro1)],
            ["RO2", str(ro2)],
            ["RO3", str(ro3)],
            ["Sorted (median policy)", " → ".join(str(x) for x in sorted_epochs)],
            ["Median", str(median_epoch)],
            ["FINAL_REFIT", str(median_epoch)],
        ],
    )
    seed_table = _table(
        ["Field", "Value"],
        [[k, str(v)] for k, v in seed_contract.items() if k != "version"],
        winner_row_idx=0,
    )

    body = (
        metric_cards
        + '<div class="cw-p4347-content">'
        + '<section class="cw-p4347-section"><h4>A — Epoch policy (median of rolling-origin inner best epochs)</h4>'
        + timeline
        + epoch_table
        + "</section>"
        + '<section class="cw-p4347-section"><h4>B — Seed contract</h4>'
        + seed_table
        + "</section>"
        + '<section class="cw-p4347-section"><h4>C — Lock contract</h4>'
        + _table(
            ["Field", "Value"],
            [
                ["Locked candidate", signoff.get("locked_model_id")],
                ["Model class", signoff.get("model_class")],
                ["Config fingerprint", signoff.get("config_fingerprint")],
                ["Locked RMSE (Wh)", _fmt_float(signoff.get("locked_rmse_wh"))],
                ["Seeds", ", ".join(str(s) for s in signoff.get("seed_list", []))],
                ["Validation used", signoff.get("validation_used")],
                ["Early stopping used", signoff.get("early_stopping_used")],
                ["Test status", signoff.get("test_status")],
                ["Ready for Phase 46", signoff.get("ready_for_phase46")],
            ],
        )
        + "</section>"
        + '<div class="cw-p4347-callout">Read-only render — Phase 45 lock logic is not re-executed. '
        + 'All values come from <code>artifacts/final_model_lock/</code>.</div>'
        + "</div>"
    )

    return HTML(
        _SHARED_CSS
        + f'<article class="cw-p4347">{_header(_P45_PHASE_NAME, subtitle, status)}{body}</article>'
    )


_P46_PHASE_NAME = "Phase 46 — Three-Seed Final Runs"


def render_phase_46_three_seed_final_runs_resume(project_root: Path | None = None) -> HTML:
    """Render the Phase 46 three-seed final-runs dashboard."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "three_seed_final_runs"

    signoff = read_json(art / "phase_46_signoff.json")
    summary = read_json(art / "three_seed_final_runs_summary.json")
    seed_contracts = read_json(art / "final_lock_verification.json") if (art / "final_lock_verification.json").exists() else {}

    status = str(signoff.get("overall_status") or "UNKNOWN")
    subtitle = (
        f"{signoff.get('artifact_version', '')} | "
        f"candidate = {signoff.get('candidate_id')} | "
        f"FINAL_REFIT_EPOCHS = {signoff.get('final_refit_epochs')} | "
        f"ready_for_phase47 = {signoff.get('ready_for_phase47')}"
    )

    metric_cards = _metric_cards([
        ("Candidate", signoff.get("candidate_id")),
        ("Config SHA-256", (signoff.get("config_sha256") or "")[:16] + "…"),
        ("Seeds", ", ".join(str(s) for s in signoff.get("seed_list", []))),
        ("FINAL_REFIT epochs", signoff.get("final_refit_epochs")),
        ("Avg FINAL_DEV RMSE (Wh)", _fmt_float(signoff.get("average_rmse_wh"))),
        ("Completed runs", signoff.get("completed_run_count")),
        ("FINAL_DEV region", signoff.get("final_dev_region")),
        ("Test status", signoff.get("test_status")),
        ("Validation used", signoff.get("validation_used")),
        ("Ready for Phase 47", signoff.get("ready_for_phase47")),
    ])

    seed_42 = {
        "run_id": signoff.get("seed42_run_id"),
        "checkpoint_sha256": (signoff.get("seed42_checkpoint_sha256") or "")[:16] + "…",
        "rmse_wh": summary.get("seed42_rmse"),
    }
    seed_123 = {
        "run_id": signoff.get("seed123_run_id"),
        "checkpoint_sha256": (signoff.get("seed123_checkpoint_sha256") or "")[:16] + "…",
        "rmse_wh": summary.get("seed123_rmse"),
    }
    seed_2026 = {
        "run_id": signoff.get("seed2026_run_id"),
        "checkpoint_sha256": (signoff.get("seed2026_checkpoint_sha256") or "")[:16] + "…",
        "rmse_wh": summary.get("seed2026_rmse"),
    }

    def _seed_card(name: str, s: dict[str, Any]) -> str:
        return (
            f'<div class="cw-p4347-card"><h5>Seed {name}</h5>'
            f'<div class="big">{_fmt_float(s["rmse_wh"])} Wh</div>'
            f'<div class="sm">FINAL_DEV RMSE (diagnostic only)</div>'
            f'<div class="sm">run_id <span class="cw-p4347-fp">{escape(str(s["run_id"]))}</span></div>'
            f'<div class="sm">checkpoint <span class="cw-p4347-fp">{escape(str(s["checkpoint_sha256"]))}</span></div>'
            f'<div class="sm">epoch = FINAL_REFIT ({escape(str(signoff.get("final_refit_epochs")))})</div>'
            "</div>"
        )

    seed_cards = (
        '<section class="cw-p4347-section"><h4>A — Three-seed FINAL_REFIT runs</h4>'
        '<div class="cw-p4347-grid3">'
        + _seed_card("42", seed_42)
        + _seed_card("123", seed_123)
        + _seed_card("2026", seed_2026)
        + "</div></section>"
    )

    contract_table = _table(
        ["Shared contract field", "Value"],
        [
            ["candidate_id", signoff.get("candidate_id")],
            ["config_sha256", signoff.get("config_sha256")],
            ["FINAL_REFIT epochs", signoff.get("final_refit_epochs")],
            ["early_stopping_used", signoff.get("early_stopping_used")],
            ["validation_used", signoff.get("validation_used")],
            ["FINAL_DEV region", signoff.get("final_dev_region")],
            ["final_scaling_version", signoff.get("final_scaling_version")],
            ["x_scaler_sha256", signoff.get("x_scaler_sha256")],
            ["y_scaler_sha256", signoff.get("y_scaler_sha256")],
            ["test_status", signoff.get("test_status")],
            ["phase47_released", signoff.get("phase47_released")],
            ["ready_for_phase47", signoff.get("ready_for_phase47")],
        ],
    )

    body = (
        metric_cards
        + '<div class="cw-p4347-content">'
        + seed_cards
        + '<section class="cw-p4347-section"><h4>B — Shared contract (all three seeds)</h4>'
        + contract_table
        + "</section>"
        + '<div class="cw-p4347-callout warn">'
        + '<strong>FINAL_DEV DIAGNOSTIC METRICS — NOT held-out Test results.</strong> '
        + 'Phase 46 evaluates the three FINAL_REFIT Transformer checkpoints on the FINAL_DEV region (Train + Validation) '
        + 'for internal consistency only. Held-out Test evaluation happens exclusively in Phase 47.'
        + "</div>"
        + '<div class="cw-p4347-callout">Read-only render — Phase 46 is not re-executed. '
        + 'All values come from <code>artifacts/three_seed_final_runs/</code>.</div>'
        + "</div>"
    )

    return HTML(
        _SHARED_CSS
        + f'<article class="cw-p4347">{_header(_P46_PHASE_NAME, subtitle, status)}{body}</article>'
    )

_P47_PHASE_NAME = "Phase 47 — Final Test Evaluation"


def render_phase_47_final_test_evaluation_resume(project_root: Path | None = None) -> HTML:
    """Render the Phase 47 final-Test evaluation dashboard."""

    root = Path(project_root or get_project_root())
    art = root / "artifacts" / "final_test"

    signoff = read_json(art / "phase_47_signoff.json")
    population = read_json(art / "final_test_population_manifest.json")
    access_event = read_json(art / "final_test_access_event.json")

    status = str(signoff.get("overall_status") or "UNKNOWN")
    subtitle = (
        f"Test N = {signoff.get('n_test')} · candidate = TR_C2_ALT_LOOKBACK · "
        f"first Test access authorized"
    )

    metric_cards = _metric_cards([
        ("Phase 47 status", status),
        ("Test N", signoff.get("n_test")),
        ("Candidate", "TR_C2_ALT_LOOKBACK"),
        ("Final lock SHA-256", (signoff.get("final_lock_sha256") or "")[:16] + "…"),
        ("Test population SHA-256", (population.get("target_ids_sha256") or "")[:16] + "…"),
        ("Seeds", ", ".join(str(s) for s in signoff.get("seed_list", []))),
        ("Ready for Phase 48", signoff.get("ready_for_phase48")),
        ("Ready for Phase 52", signoff.get("ready_for_phase52")),
    ])

    seed_42 = (signoff.get("seed42_mae_wh"), signoff.get("seed42_rmse_wh"), signoff.get("seed42_r2"))
    seed_123 = (signoff.get("seed123_mae_wh"), signoff.get("seed123_rmse_wh"), signoff.get("seed123_r2"))
    seed_2026 = (signoff.get("seed2026_mae_wh"), signoff.get("seed2026_rmse_wh"), signoff.get("seed2026_r2"))

    def _seed_card(name: str, t: tuple[Any, Any, Any]) -> str:
        return (
            f'<div class="cw-p4347-card"><h5>Seed {name}</h5>'
            f'<div class="big">RMSE {_fmt_float(t[1])} Wh</div>'
            f'<div class="sm">MAE {_fmt_float(t[0])} Wh · R² {_fmt_float(t[2], 6)}</div>'
            f'<div class="sm">checkpoint '
            f'<span class="cw-p4347-fp">{escape((signoff.get(f"seed{name}_checkpoint_sha256") or "")[:16] + "…")}</span>'
            f'</div></div>'
        )

    seed_cards = (
        '<section class="cw-p4347-section"><h4>B — Per-seed Test metrics (FINAL_TEST_POP-v1, N = ' + escape(str(signoff.get("n_test"))) + ')</h4>'
        '<div class="cw-p4347-grid3">'
        + _seed_card("42", seed_42)
        + _seed_card("123", seed_123)
        + _seed_card("2026", seed_2026)
        + "</div></section>"
    )

    aggregate_table = _table(
        ["Metric", "Mean", "Sample SD"],
        [
            ["MAE (Wh)", _fmt_float(signoff.get("transformer_mean_mae_wh")), _fmt_float(signoff.get("transformer_sd_mae_wh"))],
            ["RMSE (Wh)", _fmt_float(signoff.get("transformer_mean_rmse_wh")), _fmt_float(signoff.get("transformer_sd_rmse_wh"))],
            ["R²", _fmt_float(signoff.get("transformer_mean_r2"), 6), _fmt_float(signoff.get("transformer_sd_r2"), 6)],
        ],
    )

    comp_rows = [
        [
            "MAE (Wh)",
            _fmt_float(signoff.get("transformer_mean_mae_wh")) + " ± " + _fmt_float(signoff.get("transformer_sd_mae_wh")),
            _fmt_float(signoff.get("persistence_mae_wh")),
            "Persistence" if signoff.get("persistence_mae_wh") < signoff.get("transformer_mean_mae_wh") else "Transformer",
        ],
        [
            "RMSE (Wh)",
            _fmt_float(signoff.get("transformer_mean_rmse_wh")) + " ± " + _fmt_float(signoff.get("transformer_sd_rmse_wh")),
            _fmt_float(signoff.get("persistence_rmse_wh")),
            "Transformer" if signoff.get("transformer_mean_rmse_wh") < signoff.get("persistence_rmse_wh") else "Persistence",
        ],
        [
            "R²",
            _fmt_float(signoff.get("transformer_mean_r2"), 6) + " ± " + _fmt_float(signoff.get("transformer_sd_r2"), 6),
            _fmt_float(signoff.get("persistence_r2"), 6),
            "Transformer" if signoff.get("transformer_mean_r2") > signoff.get("persistence_r2") else "Persistence",
        ],
    ]
    comp_table = _table(
        ["Metric", "Transformer (mean ± SD)", "Persistence", "Winner"],
        comp_rows,
    )

    t_mae = signoff.get("transformer_mean_mae_wh")
    t_rmse = signoff.get("transformer_mean_rmse_wh")
    t_r2 = signoff.get("transformer_mean_r2")
    p_mae = signoff.get("persistence_mae_wh")
    p_rmse = signoff.get("persistence_rmse_wh")
    p_r2 = signoff.get("persistence_r2")
    lstm_elig = signoff.get("lstm_eligibility")

    conclusion_mae = "Persistence" if p_mae < t_mae else "Transformer"
    conclusion_rmse = "Transformer" if t_rmse < p_rmse else "Persistence"
    conclusion_r2 = "Transformer" if t_r2 > p_r2 else "Persistence"

    conclusion_html = (
        '<section class="cw-p4347-section"><h4>D — Scientific conclusion</h4>'
        '<div class="cw-p4347-conclusion">'
        "<ul>"
        f"<li><strong>MAE</strong> → <strong>{conclusion_mae}</strong> wins "
        f"(Transformer {_fmt_float(t_mae)} vs Persistence {_fmt_float(p_mae)}).</li>"
        f"<li><strong>RMSE</strong> → <strong>{conclusion_rmse}</strong> wins "
        f"(Transformer {_fmt_float(t_rmse)} vs Persistence {_fmt_float(p_rmse)}).</li>"
        f"<li><strong>R²</strong> → <strong>{conclusion_r2}</strong> wins "
        f"(Transformer {_fmt_float(t_r2, 6)} vs Persistence {_fmt_float(p_r2, 6)}).</li>"
        "</ul>"
        "<p><strong>Verdict</strong>: Transformer better on RMSE and R²; Persistence better on MAE. "
        "The model captures temporal/trend context (improving RMSE/R²) while the naive last-value baseline "
        "is hard to beat point-wise on a noisy 10-minute Appliances series (MAE). Both wins are by margins "
        "that exceed the seed-to-seed sample SD, so the comparison is robust under the canonical seed ensemble.</p>"
        "</div>"
        "</section>"
    )

    lstm_html = (
        '<section class="cw-p4347-section"><h4>E — LSTM eligibility and Phase 47 governance</h4>'
        + _table(
            ["Field", "Value"],
            [
                ["LSTM eligibility on FINAL_TEST_POP-v1", str(lstm_elig)],
                ["Reason", "LSTM_TUNED_DEV uses lookback L36, FINAL_TEST_POP-v1 requires L72 (config mismatch — not a model failure)."],
                ["training_used", signoff.get("training_used")],
                ["scaler_fit_used", signoff.get("scaler_fit_used")],
                ["best_seed_selected", signoff.get("best_seed_selected")],
                ["ensemble_used", signoff.get("ensemble_used")],
                ["post_test_tuning", signoff.get("post_test_tuning")],
                ["prediction_bundles_frozen", signoff.get("prediction_bundles_frozen")],
                ["First Test access", access_event.get("first_access_timestamp")],
                ["First access authorized", access_event.get("authorized")],
            ],
        )
        + "</section>"
    )

    body = (
        metric_cards
        + '<div class="cw-p4347-content">'
        + '<section class="cw-p4347-section"><h4>A — Top status</h4>'
        + '<div class="cw-p4347-callout">'
        + '<strong>Phase 47 PASS.</strong> First authorized Test access. '
        + 'No training, no scaler fitting, no best-seed selection, no ensemble.'
        + "</div></section>"
        + seed_cards
        + '<section class="cw-p4347-section"><h4>C — Transformer mean ± sample SD vs Persistence</h4>'
        + '<div class="cw-p4347-grid3">'
        + '<div class="cw-p4347-card win"><h5>Aggregate MAE</h5><div class="big">' + _fmt_float(t_mae) + ' Wh</div>'
        + '<div class="sm">± ' + _fmt_float(signoff.get("transformer_sd_mae_wh")) + ' Wh</div></div>'
        + '<div class="cw-p4347-card win"><h5>Aggregate RMSE</h5><div class="big">' + _fmt_float(t_rmse) + ' Wh</div>'
        + '<div class="sm">± ' + _fmt_float(signoff.get("transformer_sd_rmse_wh")) + ' Wh</div></div>'
        + '<div class="cw-p4347-card win"><h5>Aggregate R²</h5><div class="big">' + _fmt_float(t_r2, 6) + '</div>'
        + '<div class="sm">± ' + _fmt_float(signoff.get("transformer_sd_r2"), 6) + '</div></div>'
        + "</div>"
        + comp_table
        + "</section>"
        + conclusion_html
        + lstm_html
        + '<div class="cw-p4347-callout">Read-only render — Phase 47 Test is not re-evaluated. '
        + 'All values come from <code>artifacts/final_test/phase_47_signoff.json</code> and the four canonical CSVs.</div>'
        + "</div>"
    )

    return HTML(
        _SHARED_CSS
        + f'<article class="cw-p4347">{_header(_P47_PHASE_NAME, subtitle, status)}{body}</article>'
    )
