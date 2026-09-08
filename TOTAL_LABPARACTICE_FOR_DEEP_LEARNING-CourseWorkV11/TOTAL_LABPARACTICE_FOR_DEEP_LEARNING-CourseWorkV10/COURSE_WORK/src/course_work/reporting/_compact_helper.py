"""Compact dashboard helper used by Phase 53-59 simplified renderers.

A presentation-only layer that produces Phase-42-style compact dashboards:
header + overview + main result table + signoff.

All values come from canonical Phase artifacts (signoff JSON + CSV metrics).
"""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML

from course_work.utils.artifacts import read_json

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
.cw-d-body{padding:18px 22px 22px}
.cw-d-sec{margin-top:14px}
.cw-d-sec h4{font-size:13.5px;margin:0 0 7px;color:#334155;font-weight:600}
.cw-d-tblwrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-d-tbl{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
.cw-d-tbl th{background:#f5f7fb;color:#475569;text-align:left;font-weight:650;padding:9px 11px;border-bottom:1px solid #dfe6ef;white-space:nowrap}
.cw-d-tbl td{text-align:left;padding:8px 11px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-d-tbl tbody tr:nth-child(even){background:#fafbfd}
.cw-d-tbl tbody tr:last-child td{border-bottom:0}
.cw-d-tbl .win{background:#eef9f0 !important}
.cw-d-overview{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 14px;margin:0 0 12px}
.cw-d-overview .row{display:flex;justify-content:space-between;font-size:12.5px;padding:4px 0;border-bottom:1px solid #edf1f5;gap:10px}
.cw-d-overview .row .l{color:#475569;flex-shrink:0}
.cw-d-overview .row .v{color:#172033;font-weight:600;text-align:right;overflow-wrap:anywhere}
.cw-d-note{font-size:12px;color:#475569;margin-top:10px;line-height:1.5;padding:9px 12px;background:#fbfcff;border-left:3px solid #cbd5e1;border-radius:0 7px 7px 0}
@media (max-width:720px){.cw-d-h{flex-direction:column;padding:15px}.cw-d-body{padding:14px 16px 18px}.cw-d-overview{grid-template-columns:1fr}}
<style>.cw-d-meta,.cw-d-note,.cw-d-fig .fcap,.cw-d-fig .ftitle,.cw-d-call,.cw-d-call.warn,.cw-d-call.good,.cw-d-call.fail,.cw-d-overview .cw-d-note,.cw-d h3 small,.cw-d-card .sm,.cw-d-card2 .sm,.cw-d-card2 .ul,.mape-note,.mape-cards,p.cw-d-meta,div.cw-d-meta,div.cw-d-note,p[style*="font-size:11"],p[style*="font-size:12"],p[style*="font-size:13"],p[style*='font-size:11'],p[style*='font-size:12'],p[style*='font-size:13']{display:none !important}</style></style>
"""


def _status_class(status: str) -> str:
    s = (status or "UNKNOWN").upper()
    if s == "PASS":
        return ""
    if "WARNING" in s:
        return "warn"
    return "fail"


def _header(title: str, subtitle: str, status: str) -> str:
    cls = _status_class(status)
    css_cls = f"cw-d-badge {cls}" if cls else "cw-d-badge"
    return (
        '<header class="cw-d-h">'
        f"<div><h3>{escape(title)}</h3>"
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(str(status).upper())}</span>'
        "</header>"
    )


def _overview_table(pairs: list[tuple[str, Any]]) -> str:
    """Render an overview as a single 2-column table (Phase 42-style)."""
    rows = "".join(
        f'<tr><td style="color:#475569">{escape(str(label))}</td>'
        f'<td style="font-weight:600;color:#172033">{escape(str(value))}</td></tr>'
        for label, value in pairs
    )
    return f'<table class="cw-d-tbl"><tbody>{rows}</tbody></table>'


def _tbl(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a generic HTML table."""
    ths = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<div class="cw-d-tblwrap"><table class="cw-d-tbl"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV file safely."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def render_compact_phase_dashboard(
    title: str,
    subtitle: str,
    status: str,
    overview_pairs: list[tuple[str, Any]],
    main_section_title: str,
    main_section_html: str,
    signoff_pairs: list[tuple[str, Any]],
    note: str,
) -> HTML:
    """Render a Phase-42-style compact dashboard.

    Parameters
    ----------
    title : str
        Dashboard title (e.g., "Phase 53 - Attention Heatmaps").
    subtitle : str
        Subtitle line below title (one short factual statement).
    status : str
        PASS / FAIL / WARNING / etc.
    overview_pairs : list of (label, value)
        Compact 2-column overview rows.
    main_section_title : str
        Section heading for main result.
    main_section_html : str
        Pre-rendered HTML for the main result section body (a single table).
    signoff_pairs : list of (label, value)
        Compact 2-column signoff rows.
    note : str
        Essential caveat (single short neutral note).
    """
    body = (
        '<div class="cw-d-body">'
        + '<section class="cw-d-sec"><h4>Overview</h4>'
        + _overview_table(overview_pairs)
        + '</section>'
        + '<section class="cw-d-sec"><h4>' + escape(main_section_title) + '</h4>'
        + main_section_html
        + '</section>'
        + '<section class="cw-d-sec"><h4>Decision & signoff</h4>'
        + _overview_table(signoff_pairs)
        + f'<div class="cw-d-note">{escape(note)}</div>'
        + '</section>'
        + '</div>'
    )
    return HTML(
        _CSS
        + f'<article class="cw-d">{_header(title, subtitle, status)}{body}</article>'
    )
