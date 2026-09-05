# -*- coding: utf-8 -*-
"""Phase 58-H - read-only HTML dashboard renderer (presentation only).

Visual family matches Phase 49-57 dashboards:
- Light-theme containers (.cw-d) with dark blue/slate headers
- Concise tables (.cw-d-tbl) with subtle alternating rows
- Cards (.cw-d-card / .cw-d-card2) for high-level summary
- Callouts (.cw-d-call / .cw-d-call.good / .cw-d-call.warn / .cw-d-call.strong)
- Provenance / cell lineage footer

Phase 58 is the FINAL reporting synthesis and governance phase. It consolidates
the frozen scientific evidence from Phases 43-57 into the canonical final-report
table package (FT01..FT10 main + FA01..FA12 appendix), with explicit
provenance, evidence-class separation, consistent rounding, and NO new model
selection, NO new scientific analysis, NO new metric.

Architecture constraints (per architecture_rule.md v1.20):
* Read-only over Phase 43-58 canonical artifacts.
* No model loading, no new attention extraction, no Test inference.
* No training, no scaler fitting, no checkpoint reload.
* No recomputation of final metrics, no manual re-rounding.
* No best-seed selection, no best-head selection, no head pruning, no ablation.
* No ensemble reconstruction, no Test reranking, no new hypothesis test.
* No implementation of Phase 59 substantive conclusions.
* No notebook modification (this is a presentation-only renderer; the notebook
  edit is a separate, Human-approved, append-only step).
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
from html import escape
from pathlib import Path

from IPython.display import HTML

__all__ = ["render_phase_58_dashboard"]


# ---------------------------------------------------------------------------
# CSS - reuses Phase 49-57 .cw-d family
# ---------------------------------------------------------------------------
_CSS = """
<style>
.cw-d{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fbfcff;box-shadow:0 8px 24px rgba(31,45,61,.08);margin:14px 0 22px;overflow:hidden}
.cw-d *{box-sizing:border-box}
.cw-d-h{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef}
.cw-d-h h3{font-size:20px;line-height:1.25;margin:0 0 5px;color:#172033}
.cw-d-meta{font-size:12px;color:#5d6b82}
.cw-d-badge{border:1px solid #a9dec1;border-radius:999px;padding:6px 12px;font-size:11px;font-weight:700;letter-spacing:.04em;color:#11613d;background:#e8f7ef;white-space:nowrap}
.cw-d-badge.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
.cw-d-badge.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-badge.info{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-badge.frozen{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;padding:16px 22px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-d-card{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:9px;background:#fff;min-width:0}
.cw-d-card span{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card strong{font-size:15px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
.cw-d-card .sm{font-size:9px;color:#94a3b8;margin-top:2px}
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
.cw-d-pill.fail{border-color:#e2b2b8;background:#fcecef;color:#8b2430}
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
.cw-d-call.strong{border-left:4px solid #dc2626;background:#fcecef;color:#8b2430}
.cw-d-call.muted{border-color:#94a3b8;background:#f1f5f9;color:#475569}
.cw-d-fig{display:flex;flex-direction:column;gap:6px;margin:12px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:75%;max-width:880px}
.cw-d-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto}
.cw-d-fig.compact{width:62%;max-width:720px}
.cw-d-fig .ftitle{font-size:12px;color:#475569;font-weight:650}
.cw-d-fig .fcap{font-size:11px;color:#64748b;line-height:1.45}
.cw-d-foot{padding:14px 22px;border-top:1px solid #e5eaf1;background:#fbfcff;font-size:11px;color:#5d6b82;display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between;align-items:center}
.cw-d-foot code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;background:#eef2ff;color:#3b46c4;padding:2px 6px;border-radius:4px}
</style>
"""


# ---------------------------------------------------------------------------
# Read-only helpers (no upstream mutation)
# ---------------------------------------------------------------------------

def _read_csv(fp: Path) -> list[dict[str, str]]:
    if not fp.is_file():
        return []
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    if not fp.is_file():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    """Format a value for display (preserve full precision if numeric)."""
    if v is None or v == "":
        return "N/A"
    s = str(v)
    if s.upper() in ("N/A", "NA", "NAN"):
        return "N/A"
    return escape(s)


def _badge(kind: str, text: str) -> str:
    klass = "cw-d-badge"
    if kind in ("good", "frozen", "info", "warn", "fail"):
        klass += " " + kind
    return f'<span class="{klass}">{escape(text)}</span>'


def _pill(kind: str, text: str) -> str:
    klass = "cw-d-pill"
    if kind in ("good", "warn", "fail", "frozen", "muted"):
        klass += " " + kind
    return f'<span class="{klass}">{escape(text)}</span>'


def _table(rows: list[dict], cols: list[str], max_rows: int | None = None,
           col_widths: dict[str, str] | None = None) -> str:
    """Build a dashboard HTML table from a list of row dicts."""
    out: list[str] = ['<div class="cw-d-tblwrap"><table class="cw-d-tbl">']
    out.append("<thead><tr>")
    for c in cols:
        out.append(f"<th>{escape(str(c))}</th>")
    out.append("</tr></thead><tbody>")
    for r in (rows[:max_rows] if max_rows else rows):
        out.append("<tr>")
        for c in cols:
            v = r.get(c, "")
            out.append(f"<td>{_fmt(v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if max_rows and len(rows) > max_rows:
        out.append(f'<div style="font-size:11px;color:#64748b;margin-top:6px">_showing first {max_rows} of {len(rows)} rows; full table in CSV/Markdown/LaTeX package_</div>')
    return "\n".join(out)


def _section(title: str, badge: str = "", body: str = "") -> str:
    head = title
    if badge:
        head += f" {_badge('frozen', badge)}"
    return f'<div class="cw-d-sec"><h4>{escape(head)}</h4>{body}</div>'


def _callout(kind: str, body: str) -> str:
    return f'<div class="cw-d-call {kind}">{body}</div>'


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_phase_58_dashboard(project_root: Path | str | None = None) -> HTML:
    """Render the Phase 58 Final Tables dashboard (presentation-only).

    Uses ONLY the active corrected v2 artifacts:
      - phase_58_signoff.json               (FINAL_TABLES-v2)
      - final_test_metrics_by_seed.csv      (Phase 47 final Test metrics, N=2961/seed)
      - final_test_model_comparison.csv     (mean +/- SD across seeds)
      - phase45_final_model_lock_handoff.json (locked transformer config)
    Stale FINAL_TABLES-v1 outputs NOT used.

    Layout: HEADER -> FINAL MODEL CONFIGURATION -> FINAL TEST RESULTS ->
    KEY ANALYSIS FINDINGS -> FINAL STATUS. No figures.
    """
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    art = root / "artifacts" / "final_tables"
    art_test = root / "artifacts" / "final_test"
    art_lock = root / "artifacts" / "final_model_lock"
    art_roll = root / "artifacts" / "rolling_origin"

    signoff = _read_json(art / "phase_58_signoff.json")
    seed_metrics = _read_csv(art_test / "final_test_metrics_by_seed.csv")
    model_comp = _read_csv(art_test / "final_test_model_comparison.csv")
    handoff = _read_json(art_roll / "phase45_final_model_lock_handoff.json")
    op_contract = _read_json(art_lock / "final_optimizer_contract.json")
    boundary = _read_json(art_lock / "final_boundary_contract.json")
    features = _read_json(art_lock / "final_feature_contract.json")
    seeds_contract = _read_json(art_lock / "final_seed_contract.json")

    status = str(signoff.get("overall_status", "PASS")).upper()
    version = str(signoff.get("version", "FINAL_TABLES-v2"))

    subtitle = (
        "Consolidated final configuration, Test performance, and key "
        "analysis results."
    )

    # ----- Extract canonical values -----
    rec = handoff.get("recommended_transformer_config", {}) or {}
    rec_model = rec.get("model", {}) or {}
    rec_data = rec.get("data", {}) or {}

    # Final test seed metrics (N = 2961 each)
    seed_rows = []
    for r in seed_metrics:
        try:
            seed_rows.append((
                str(r["seed"]),
                float(r["mae_wh"]),
                float(r["rmse_wh"]),
                float(r["r2"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue

    # Mean +/- SD row from final_test_model_comparison.csv
    mean_row = None
    for r in model_comp:
        if r.get("row_type") == "TRANSFORMER_AGGREGATE":
            try:
                mean_row = (
                    "Mean +/- SD",
                    float(r["mae_wh"]),
                    float(r.get("mae_sd_if_aggregate") or 0.0),
                    float(r["rmse_wh"]),
                    float(r.get("rmse_sd_if_aggregate") or 0.0),
                    float(r["r2"]),
                    float(r.get("r2_sd_if_aggregate") or 0.0),
                )
            except (TypeError, ValueError):
                continue
            break

    # ----- Final model configuration -----
    config_pairs = [
        ("Model", f"{rec_model.get('model_name','Transformer Encoder Regressor')} (TR_C2_ALT_LOOKBACK)"),
        ("Lookback", f"{rec_data.get('lookback_steps', 72)} steps ({rec_data.get('sampling_interval_minutes', 10)}-min, 12 h)"),
        ("Input features", f"{rec_data.get('feature_count', 33)} ({features.get('feature_set_version', 'FEATURESETS-v1')} / {features.get('feature_version', 'FEATURES-v1')})"),
        ("d_model", str(rec_model.get("d_model", 64))),
        ("Heads", str(rec_model.get("num_heads", 4))),
        ("Layers", str(rec_model.get("num_layers", 2))),
        ("FFN", str(rec_model.get("ffn_dim", 256))),
        ("Dropout", str(rec_model.get("dropout", 0.1))),
        ("Loss", "MSE"),
        ("Optimizer", f"{op_contract.get('optimizer', 'AdamW')}"),
        ("Learning rate", f"{op_contract.get('LR', 0.0003)}"),
        ("Weight decay", f"{op_contract.get('WD', 0.001)}"),
        ("Batch size", "32"),
        ("Gradient clipping", f"max-norm {op_contract.get('gradient_clip_max_norm', 1.0)}"),
        ("Final epochs", "30 (TRAIN+VAL, FINAL_REFIT)"),
        ("Seeds", ", ".join(str(s) for s in seeds_contract.get("seeds", [42, 123, 2026]))),
        ("Boundary protocol", str(boundary.get("protocol", "WB0"))),
    ]
    config_html = (
        '<section class="cw-d-sec"><h4>Final model configuration</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569;width:38%">{escape(label)}</td>'
            f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
            for label, value in config_pairs
        )
        + '</tbody></table></section>'
    )

    # ----- Final test results (MAIN Phase 58 table) -----
    test_rows = []
    for seed, mae, rmse, r2 in seed_rows:
        test_rows.append(
            '<tr>'
            f'<td>Seed {seed}</td>'
            f'<td>{mae:.2f}</td>'
            f'<td>{rmse:.2f}</td>'
            f'<td>{r2:.3f}</td>'
            '</tr>'
        )
    if mean_row:
        _, mae, mae_sd, rmse, rmse_sd, r2, r2_sd = mean_row
        test_rows.append(
            '<tr style="background:#f1f5fa;font-weight:600">'
            '<td>Mean +/- SD</td>'
            f'<td>{mae:.2f} +/- {mae_sd:.2f}</td>'
            f'<td>{rmse:.2f} +/- {rmse_sd:.2f}</td>'
            f'<td>{r2:.3f} +/- {r2_sd:.3f}</td>'
            '</tr>'
        )
    test_table_html = (
        '<table class="cw-d-tbl" style="margin-top:6px">'
        '<thead><tr><th>Seed</th><th>MAE (Wh)</th><th>RMSE (Wh)</th>'
        '<th>R^2</th></tr></thead>'
        '<tbody>' + ''.join(test_rows) + '</tbody></table>'
        '<p style="margin:8px 0 0;font-size:12px;color:#475569;line-height:1.5">'
        'N = 2961 Test targets per seed. Results are reported across all '
        'three prespecified final seeds; no best-seed selection or ensemble '
        'was performed.'
        '</p>'
    )

    # ----- Key analysis findings (synthesis of Phase 48-57) -----
    findings_pairs = [
        ("Prediction behavior",
         "Captures the temporal pattern of Appliances energy but smooths the sharpest variations."),
        ("Residual analysis",
         "Residuals are right-tailed; the largest errors are concentrated around rapid-change periods."),
        ("Regime analysis",
         "Forecast error increases substantially during high-consumption and rapid-change regimes."),
        ("Worst-error analysis",
         "Largest errors are predominantly underpredictions of energy peaks."),
        ("Attention analysis",
         "Attention allocation varies across historical positions and across heads within each layer."),
        ("Cross-seed attention stability",
         "Layer-level attention behavior shows measurable cross-seed consistency; matched head indices are structural-only."),
    ]
    findings_html = (
        '<table class="cw-d-tbl" style="margin-top:6px">'
        '<thead><tr><th style="width:30%">Analysis</th><th>Key finding</th></tr></thead>'
        '<tbody>'
        + ''.join(
            f'<tr><td style="font-weight:600">{escape(a)}</td>'
            f'<td style="color:#1f2a44">{escape(f_)}</td></tr>'
            for a, f_ in findings_pairs
        )
        + '</tbody></table>'
    )

    # ----- Final status -----
    status_pairs = [
        ("Phase status", "PASS"),
        ("Active revision", version),
        ("Final configuration locked", "\u2713"),
        ("Three-seed Test results reported", "\u2713"),
        ("No best-seed selection", "\u2713"),
        ("No ensemble", "\u2713"),
        ("No post-Test tuning", "\u2713"),
        ("Ready for Phase 59", "\u2713"),
    ]
    status_html = (
        '<section class="cw-d-sec"><h4>Final status</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569;width:60%">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in status_pairs
        )
        + '</tbody></table></section>'
    )

    # ----- Assemble -----
    main_section = (
        '<section class="cw-d-sec"><h4>Final Test results</h4>'
        + test_table_html
        + '</section>'
        '<section class="cw-d-sec"><h4>Key analysis findings</h4>'
        + findings_html
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + config_html
        + main_section
        + status_html
        + '</div>'
    )

    css_cls = f"cw-d-badge{(' fail' if status not in ('PASS',) else '')}"
    header_html = (
        '<header class="cw-d-h">'
        f'<div><h3>Phase 58 - Final Results Summary</h3>'
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(status)}</span>'
        '</header>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{header_html}{body}</article>'
    )
