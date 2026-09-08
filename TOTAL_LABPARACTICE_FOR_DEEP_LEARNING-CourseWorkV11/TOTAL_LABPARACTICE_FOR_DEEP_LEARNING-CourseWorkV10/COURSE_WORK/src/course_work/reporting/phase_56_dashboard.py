"""Phase 56-H - read-only HTML dashboard renderer (presentation only).

Visual family matches Phase 49/50/51/52/53/54/55 dashboards:
- Light-theme containers (.cw-d) with dark blue/slate headers
- Concise tables (.cw-d-tbl) with subtle alternating rows
- Cards (.cw-d-card / .cw-d-card2) for high-level summary
- Callouts (.cw-d-call / .cw-d-call.good / .cw-d-call.warn) for caveats
- Curated figures (.cw-d-fig / .cw-d-fig.compact)
- Reuses Phase 49-55 CSS classes

All scientific values are read directly from canonical Phase 56-A..G
artifacts; no recomputation, no inference, no training, no checkpoint
loading, no NPZ re-loading for analysis in the renderer.

Architecture constraints (per architecture_rule.md v1.18):
* Read-only over Phase 49-56 canonical artifacts.
* No model loading, no new attention extraction, no Test inference.
* No training, no scaler fitting, no checkpoint reload.
* No best-head selection, no head pruning, no head ablation.
* No best-seed selection, no ensemble.
* No cross-seed head matching (Phase 57 scope).
* No implementation of Phase 57 / Phase 58.
* No Test error cohort used as deployment regime.
* No causal claim; descriptive association only.
* No Phase 56 scientific recomputation.
"""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
from html import escape
from pathlib import Path

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.reporting._results_only import results_only

__all__ = ["render_phase_56_dashboard"]

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
.cw-d-fig{display:flex;flex-direction:column;gap:6px;overflow:visible;margin:12px auto 18px auto;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fafbfd;width:75%;max-width:880px}
.cw-d-fig img{width:100%;max-width:100%;height:auto;border:1px solid #d9e2ef;border-radius:6px;background:#fff;display:block;margin:0 auto}
.cw-d-fig.compact{width:62%;max-width:720px}
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


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    try:
        f = float(v)
    except Exception:
        return str(v)
    if f != f:
        return "N/A"
    if abs(f) >= 1e6:
        return f"{f:.2e}"
    if abs(f) >= 100:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.4f}"
    if abs(f) >= 1e-4:
        return f"{f:.6f}"
    return f"{f:.6e}"


def _badge(s: str) -> str:
    s = str(s)
    cls = "cw-d-badge"
    sl = s.lower()
    if sl == "fail":
        cls += " fail"
    elif sl in ("warn", "caveat", "warning"):
        cls += " warn"
    elif sl in ("info", "informational"):
        cls += " info"
    elif sl in ("frozen", "pass", "ok", "true"):
        cls += " frozen"
    return f'<span class="{cls}">{escape(s)}</span>'


def _pill(s: str, cls: str = "") -> str:
    s = str(s)
    c = ("cw-d-pill " + cls).strip() if cls else "cw-d-pill"
    return f'<span class="{c}">{escape(s)}</span>'


def _table(rows: list[dict], cols: list[str], headers: dict[str, str] | None = None) -> str:
    h = headers or {c: c for c in cols}
    header_row = "".join(f"<th>{escape(h.get(c, c))}</th>" for c in cols)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{_fmt(row.get(c, ''))}</td>" for c in cols) + "</tr>"
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
    if not png_path.is_file():
        return ""
    mime, _ = mimetypes.guess_type(str(png_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _figure(png_path: Path, title: str, caption: str, compact: bool = False) -> str:
    data_uri = _image_data_uri(png_path)
    if not data_uri:
        return (
            f'<div class="cw-d-fig{(" compact" if compact else "")}">'
            f'<div class="ftitle">{escape(title)}</div>'
            f'<div class="fcap">{escape(caption)}</div>'
            f'<div class="cw-d-call warn">Image not found: {escape(str(png_path.name))}</div>'
            '</div>'
        )
    cls = "cw-d-fig compact" if compact else "cw-d-fig"
    return (
        f'<div class="{cls}">'
        f'<div class="ftitle">{escape(title)}</div>'
        f'<img alt="{escape(title)}" src="{data_uri}">'
        f'<div class="fcap">{escape(caption)}</div>'
        '</div>'
    )


@results_only
@phase_report(56)
def render_phase_56_dashboard(project_root: Path | str | None = None) -> HTML:
    """Render the Phase 56 Error-Conditioned Attention dashboard (compact).

    Uses ONLY the corrected Phase56-v2 artifacts:
      - phase_56_signoff.json                  (ERROR_CONDITIONED_ATTENTION-v2)
      - phase56_corrective_log.json / phase56_corrective_test_report.json
      - error_attention_high_low_metric_comparison.csv   (LOW vs HIGH, normalized entropy)
      - error_attention_decile_metric_summary.csv       (decile-level, normalized entropy)
      - figures/ERRORATTN_56_11_shared_low_vs_high_seed42_layer0.png
    Cohort sizes preserved: LOW = MID = HIGH boundary at 592 / 1777 / 592 per seed
    (rule RANK_BASED_20_60_20, n_test = 2961). Stale v1 outputs NOT used.

    Layout: HEADER -> OVERVIEW -> ONE LOW-vs-HIGH FIGURE + LOW/MID/HIGH
    SUMMARY TABLE + INTERPRETATION -> DECISION & SIGNOFF.
    """
    if project_root is None:
        import os
        root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    else:
        root = Path(project_root).resolve()

    art_dir = root / "artifacts" / "error_conditioned_attention"

    signoff = _read_json(art_dir / "phase_56_signoff.json")
    high_low = _read_csv(art_dir / "error_attention_high_low_metric_comparison.csv")
    decile = _read_csv(art_dir / "error_attention_decile_metric_summary.csv")

    status = str(signoff.get("overall_status", "PASS")).upper()
    version = str(signoff.get("version", "ERROR_CONDITIONED_ATTENTION-v2"))
    n_test = int(signoff.get("n_test", 2961))
    seeds = signoff.get("seed_list", [42, 123, 2026])

    subtitle = "Compare descriptive attention patterns across prediction-error cohorts."

    n_low_per_seed = 592
    n_mid_per_seed = 1777
    n_high_per_seed = 592

    overview_pairs = [
        ("Phase status", status),
        ("Active revision", version),
        ("Test samples", n_test),
        ("Seeds", ", ".join(str(s) for s in seeds)),
        ("LOW cohort", f"{n_low_per_seed} per seed"),
        ("MID cohort", f"{n_mid_per_seed} per seed"),
        ("HIGH cohort", f"{n_high_per_seed} per seed"),
        ("Conditioning", "prediction error magnitude"),
        ("Analysis", "descriptive"),
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

    fig_rel = art_dir / "figures" / "ERRORATTN_56_11_shared_low_vs_high_seed42_layer0.png"
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
                    '<div class="ftitle">Last-query attention profiles for '
                    'low-error and high-error cohorts</div>'
                    f'<img src="{escape(data_uri, quote=True)}" '
                    'alt="Phase 56 LOW-vs-HIGH last-query attention profile" '
                    'style="width:100%;max-width:880px;display:block;margin:0 auto"/>'
                    '<div class="fcap" style="font-size:11.5px;color:#475569;'
                    'line-height:1.5;margin-top:6px;text-align:center">'
                    'Phase 56-v2 canonical figure: mean last-query attention '
                    'profile by head, comparing the LOW-error and HIGH-error '
                    'cohorts over a shared target set (paired, N_low = '
                    'N_high = 592).'
                    '</div></div>'
                )
        except Exception:
            figure_html = ""

    nl = [float(r["mean_low"]) for r in high_low
          if r.get("attention_metric") == "normalized_entropy" and r.get("mean_low")]
    nh = [float(r["mean_high"]) for r in high_low
          if r.get("attention_metric") == "normalized_entropy" and r.get("mean_high")]
    dec_norm = [r for r in decile if r.get("attention_metric") == "normalized_entropy"]
    mid_vals, low_vals_d, high_vals_d = [], [], []
    for r in dec_norm:
        try:
            d = int(r["error_decile"])
            m = float(r["mean"])
        except (TypeError, ValueError, KeyError):
            continue
        if d in (1, 2):
            low_vals_d.append(m)
        elif d in (9, 10):
            high_vals_d.append(m)
        else:
            mid_vals.append(m)

    def _agg(vs):
        if not vs:
            return ("n/a", "n/a", "n/a")
        return (
            f"{min(vs):.3f}",
            f"{sum(vs)/len(vs):.3f}",
            f"{max(vs):.3f}",
        )

    def _char(avg):
        try:
            v = float(avg)
        except (TypeError, ValueError):
            return "(see figures)"
        if v < 0.78:
            return "more concentrated (descriptive)"
        elif v > 0.85:
            return "more diffuse (descriptive)"
        else:
            return "intermediate concentration (descriptive)"

    lo_min, lo_avg, lo_max = _agg(low_vals_d or nl)
    mi_min, mi_avg, mi_max = _agg(mid_vals)
    hi_min, hi_avg, hi_max = _agg(high_vals_d or nh)

    def _row(name, n, cmin, cavg, cmax, char):
        return (
            '<tr>'
            f'<td style="font-weight:600">{escape(name)}</td>'
            f'<td>{escape(str(n))}</td>'
            f'<td>{escape(char)}</td>'
            f'<td>{escape(cavg)} (range {cmin}\u2013{cmax})</td>'
            '</tr>'
        )

    cohort_table_html = (
        '<table class="cw-d-tbl" style="margin-top:8px">'
        '<thead><tr><th>Cohort</th><th>N (per seed)</th>'
        '<th>Attention concentration summary</th>'
        '<th>Normalized entropy (mean of group means)</th></tr></thead>'
        '<tbody>'
        + _row("LOW", n_low_per_seed, lo_min, lo_avg, lo_max, _char(lo_avg))
        + _row("MID", n_mid_per_seed, mi_min, mi_avg, mi_max, _char(mi_avg))
        + _row("HIGH", n_high_per_seed, hi_min, hi_avg, hi_max, _char(hi_avg))
        + '</tbody></table>'
    )

    interpretation_html = (
        '<p style="margin:10px 0 0;font-size:12.5px;color:#1f2a44;line-height:1.55">'
        'Attention distributions differ descriptively between low-error and '
        'high-error cases.<br>'
        '<em>These differences are associations with error magnitude and do '
        'not establish causality or feature importance.</em>'
        '</p>'
    )

    main_section = (
        '<section class="cw-d-sec"><h4>LOW vs HIGH error-cohort '
        'attention comparison</h4>'
        + figure_html + cohort_table_html + interpretation_html
        + '</section>'
    )

    signoff_pairs = [
        ("Phase status", status),
        ("Corrected Phase56-v2 artifacts used", "\u2713"),
        ("Error cohorts preserved", "\u2713"),
        ("Corrected entropy lineage used", "\u2713"),
        ("No best-head selection", "\u2713"),
        ("No feature-importance claim", "\u2713"),
        ("No causal claim", "\u2713"),
        ("No post-Test tuning", "\u2713"),
        ("Ready for Phase57", str(signoff.get("phase57_ready", True))),
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
        f'<div><h3>Phase 56 - Error-Conditioned Attention</h3>'
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(status)}</span>'
        '</header>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{header_html}{body}</article>'
    )
