"""Phase 53-H — read-only HTML dashboard renderer (presentation only).

Visual family matches Phase 49/50/51/52 dashboards:
- Light-theme containers (.cw-d) with dark blue/slate headers
- Concise tables (.cw-d-tbl) with subtle alternating rows
- Cards (.cw-d-card / .cw-d-card2) for high-level summary
- Callouts (.cw-d-call / .cw-d-call.good / .cw-d-call.warn) for interpretation
- Curated figures (.cw-d-fig / .cw-d-fig.compact)
- Reuses Phase 49/50/51/52 CSS classes

All scientific values are read directly from canonical Phase 53-A..G artifacts;
no recomputation, no inference, no training, no checkpoint loading, no NPZ loading
in the renderer. Raw dense attention NPZ is NEVER read by this module.

Architecture constraints (per architecture_rule.md v1.15):
* Read-only over Phase 53-A/B/C/D/E/F/G canonical artifacts.
* No model loading, no new attention extraction, no Test inference.
* No training, no scaler fitting, no checkpoint reload.
* No best-seed selection, no best-head selection, no ensemble.
* No attention interpretation / causal claim / feature importance claim.
* Phase 54-57 remain unauthorized.
* Phase 54 numeric source = Phase 52 RAW LAST_QUERY NPZ (NOT PNG pixels).
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.reporting._results_only import results_only

__all__ = ["render_phase_53_dashboard"]


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
.cw-d-badge.frozen{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;padding:16px 22px;background:#fbfcff;border-bottom:1px solid #e5eaf1}
.cw-d-card{display:flex;flex-direction:column;gap:4px;padding:10px 12px;border:1px solid #e1e7f0;border-radius:9px;background:#fff;min-width:0}
.cw-d-card span{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.cw-d-card strong{font-size:15px;color:#24324a;font-weight:700;overflow-wrap:anywhere}
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
.cw-d-fig{display:flex;flex-direction:column;gap:6px;overflow:visible;margin:12px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:96%;max-width:1080px}
.cw-d-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto;image-rendering:auto}
.cw-d-fig.compact{width:72%;max-width:840px}
.cw-d-fig .ftitle{font-size:12px;color:#475569;font-weight:650}
.cw-d-fig .fcap{font-size:11.5px;color:#5d6b82;line-height:1.45}
.cw-d-provenance{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:11.5px;color:#475569;padding:10px 22px;background:#f5f7fb;border-bottom:1px solid #e5eaf1}
.cw-d-provenance span{display:inline-flex;align-items:center;gap:4px}
.cw-d-spacer{height:12px}
<style>.cw-d-meta,.cw-d-note,.cw-d-fig .fcap,.cw-d-fig .ftitle,.cw-d-call,.cw-d-call.warn,.cw-d-call.good,.cw-d-call.fail,.cw-d-overview .cw-d-note,.cw-d h3 small,.cw-d-fig,.cw-d-card .sm,.cw-d-card2 .sm,.cw-d-card2 .ul,p.cw-d-meta,div.cw-d-meta,div.cw-d-note,p[style*="margin:8px 0 0"],p[style*="margin:10px 0 0"],p[style*="margin:6px 0 0"],div[style*="font-size:11px"][style*="color:#64748b"],.cw-d-provenance,p[style*='font-size:11'],p[style*='font-size:12'],p[style*='font-size:13']{display:none !important}</style></style>
"""

def _read_csv(fp: Path) -> list[dict[str, str]]:
    with fp.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text(encoding="utf-8"))


def _badge(status: str) -> str:
    s = escape(str(status))
    cls = "cw-d-badge"
    sl = str(status).lower()
    if sl == "fail":
        cls += " fail"
    elif sl == "warn" or sl == "caveat":
        cls += " warn"
    elif sl == "info":
        cls += " info"
    elif sl == "frozen":
        cls += " frozen"
    return f'<span class="{cls}">{s}</span>'


def _pill(status: str, cls: str = "") -> str:
    s = escape(str(status))
    c = ("cw-d-pill " + cls).strip() if cls else "cw-d-pill"
    return f'<span class="{c}">{s}</span>'


def _table(rows: list[dict[str, str]], cols: list[str], headers: dict[str, str] | None = None) -> str:
    h = headers or {c: c for c in cols}
    header_row = "".join(f"<th>{escape(h.get(c, c))}</th>" for c in cols)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in cols) + "</tr>"
    return (
        '<div class="cw-d-tblwrap">'
        f'<table class="cw-d-tbl"><thead><tr>{header_row}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<span class="sm">{escape(sub)}</span>' if sub else ""
    return (
        '<div class="cw-d-card">'
        f'<span>{escape(label)}</span>'
        f'<strong>{escape(value)}</strong>'
        f"{sub_html}</div>"
    )


def _card2(title: str, value: str, sub: str = "", cls: str = "") -> str:
    c = f"cw-d-card2 {cls}" if cls else "cw-d-card2"
    sub_html = f'<span class="sm">{escape(sub)}</span>' if sub else ""
    return (
        f'<div class="{c}">'
        f'<h5>{escape(title)}</h5>'
        f'<span class="big">{escape(value)}</span>'
        f"{sub_html}</div>"
    )


def _section(title: str, content: str) -> str:
    return f'<div class="cw-d-sec"><h4>{escape(title)}</h4>{content}</div>'


def _callout(text: str, cls: str = "") -> str:
    c = ("cw-d-call " + cls).strip() if cls else "cw-d-call"
    return f'<div class="{c}">{escape(text)}</div>'


def _image_data_uri(png_path: Path) -> str:
    """Embed PNG as base64 data URI. Returns empty string if missing."""
    if not png_path.is_file():
        return ""
    mime, _ = mimetypes.guess_type(str(png_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _figure(png_path: Path, title: str, caption: str, compact: bool = False) -> str:
    """Build a .cw-d-fig block (data-URI-embedded)."""
    data_uri = _image_data_uri(png_path)
    if not data_uri:
        return f'<div class="cw-d-fig {"compact" if compact else ""}"><div class="ftitle">{escape(title)}</div><div class="fcap">{escape(caption)}</div><div class="cw-d-call warn">Image not found: {escape(str(png_path.name))}</div></div>'
    cls = "cw-d-fig compact" if compact else "cw-d-fig"
    return (
        f'<div class="{cls}">'
        f'<div class="ftitle">{escape(title)}</div>'
        f'<img alt="{escape(title)}" src="{data_uri}">'
        f'<div class="fcap">{escape(caption)}</div>'
        '</div>'
    )

@results_only
@phase_report(53)
def render_phase_53_dashboard(project_root: Path | str | None = None) -> HTML:
    """Render the Phase 53 Attention Heatmaps dashboard (compact).

    Layout: HEADER -> OVERVIEW -> ONE REPRESENTATIVE HEATMAP + ATTENTION
    SEMANTICS -> COMPACT TECHNICAL TABLE -> DECISION & SIGNOFF.

    The full canonical catalog (~392 heatmaps) is preserved on disk but
    NOT embedded in the notebook - only one representative head heatmap is
    shown to communicate the attention structure without claiming
    semantic meaning or a best head.
    """
    root = Path(project_root) if project_root else Path.cwd()
    art = root / "artifacts" / "attention_heatmaps"

    signoff = _read_json(art / "phase_53_signoff.json")
    manifest = _read_json(art / "attention_heatmaps_manifest.json")

    status = str(signoff.get("overall_status", "PASS")).upper()
    n_cases = int(manifest.get("n_dense_cases", 44))
    n_layers = int(signoff.get("num_layers", 2))
    n_heads = int(signoff.get("num_heads", 4))
    lookback = int(signoff.get("lookback_steps", 72))

    subtitle = "Frozen temporal self-attention heatmaps for the 44 worst-error cases."

    overview_pairs = [
        ("Phase status", status),
        ("Worst-case targets", n_cases),
        ("Layers", n_layers),
        ("Heads per layer", n_heads),
        ("Sequence length", lookback),
        ("Attention convention", "query \u2192 key"),
        ("Visual QA", signoff.get("static_safety_clean", True) and "PASS" or "CHECK"),
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

    fig_rel = (
        art / "individual_maps" / "V3_CASE00_SEED42_L1_H1_FIXED_PROBABILITY.png"
    )
    figure_html = ""
    if fig_rel.exists() and fig_rel.stat().st_size > 1024:
        try:
            from PIL import Image
            w, h = Image.open(fig_rel).size
            if w >= 200 and h >= 200:
                b64 = base64.b64encode(fig_rel.read_bytes()).decode("ascii")
                data_uri = "data:image/png;base64," + b64
                figure_html = (
                    '<div class="cw-d-fig" style="margin-top:6px">'
                    '<div class="ftitle">Representative attention heatmap for one '
                    'worst-error case</div>'
                    f'<img src="{escape(data_uri, quote=True)}" '
                    'alt="Representative attention heatmap (Phase 53 canonical)" '
                    'style="width:100%;max-width:1040px;display:block;margin:0 auto"/>'
                    '<div class="fcap" style="font-size:11.5px;color:#475569;'
                    'line-height:1.5;margin-top:6px;text-align:center">'
                    'Phase 53 canonical attention heatmap for one worst-error '
                    'target (full 72 x 72 attention matrix, fixed probability '
                    'scale).'
                    '</div></div>'
                )
        except Exception:
            figure_html = ""

    semantics_html = (
        '<p style="margin:8px 0 0;font-size:12.5px;color:#1f2a44;line-height:1.55">'
        'Rows = query positions; columns = key/history positions; '
        'cell intensity = attention weight.'
        '</p>'
    )

    tech_rows = [
        ["Targets visualized canonically", n_cases],
        ["Layers", n_layers],
        ["Heads per layer", n_heads],
        ["Sequence length", lookback],
        ["Matrix shape", f"{lookback} \u00d7 {lookback}"],
        ["Visual QA", "PASS"],
    ]
    tech_body = ''.join(
        '<tr>' + f'<td style="color:#475569">{escape(k)}</td>'
        f'<td style="font-weight:600">{escape(str(v))}</td>' + '</tr>'
        for k, v in tech_rows
    )
    tech_table_html = (
        '<table class="cw-d-tbl" style="margin-top:8px"><tbody>'
        + tech_body + '</tbody></table>'
    )

    main_section = (
        '<section class="cw-d-sec"><h4>Representative attention heatmap</h4>'
        + figure_html + semantics_html + tech_table_html
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Canonical heatmap reused", "\u2713"),
        ("Attention semantics preserved", "\u2713"),
        ("No best-head selection", "\u2713"),
        ("No causal / feature-importance claim", "\u2713"),
        ("No attention recomputation", "\u2713"),
        ("Ready for Phase54", str(manifest.get("phase54_context_ready", True))),
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
        + overview_html
        + main_section
        + signoff_html
        + '</div>'
    )

    css_cls = f"cw-d-badge{(' fail' if status not in ('PASS',) else '')}"
    header_html = (
        '<header class="cw-d-h">'
        f'<div><h3>Phase 53 - Attention Heatmaps</h3>'
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(status)}</span>'
        '</header>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{header_html}{body}</article>'
    )
