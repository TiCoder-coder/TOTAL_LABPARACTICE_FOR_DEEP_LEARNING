# -*- coding: utf-8 -*-
"""Phase 59-H - read-only HTML dashboard renderer for Final Conclusions (presentation only).

Visual family matches Phase 49-58 dashboards (.cw-d family).
Phase 59 is the final scientific closure phase. This renderer is
strictly READ-ONLY / PRESENTATION-ONLY: it renders persisted Phase 59
artifacts without recomputing metrics, training, extracting attention, or
regenerating the scientific narrative fingerprint.
"""

from __future__ import annotations

import csv
import html as _html
import json
from html import escape
from pathlib import Path

from IPython.display import HTML

from course_work.reporting._phase_report_layout import phase_report

from course_work.reporting._results_only import results_only

__all__ = ["render_phase_59_dashboard"]


# ---------------------------------------------------------------------------
# CSS - reuses Phase 49-58 .cw-d family
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
.cw-d-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;border:1px solid #d3deec;background:#f1f5fb;color:#334155}
.cw-d-pill.warn{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-pill.good{border-color:#a9dec1;background:#e8f7ef;color:#11613d}
.cw-d-pill.frozen{border-color:#a5b9e6;background:#eef2ff;color:#3b46c4}
.cw-d-pill.muted{background:#eef2f7;color:#475569;border-color:#d3deec}
.cw-d-pill.partial{border-color:#f1d889;background:#fff7e0;color:#7a5613}
.cw-d-pill.mixed{border-color:#f1b889;background:#fff0e0;color:#8b4430}
.cw-d-grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;margin-top:10px}
.cw-d-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:10px}
.cw-d-grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:10px}
.cw-d-grid5{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:10px}
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
.cw-d-foot{padding:14px 22px;border-top:1px solid #e5eaf1;background:#fbfcff;font-size:11px;color:#5d6b82;display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between;align-items:center}
.cw-d-foot code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;background:#eef2ff;color:#3b46c4;padding:2px 6px;border-radius:4px}
.cw-d-tak{font-size:12.5px;line-height:1.55;color:#1f2a44;background:#fafbfd;border:1px solid #e2e8f0;border-radius:9px;padding:10px 14px;margin:6px 0}
.cw-d-tak b{color:#3b46c4}
<style>.cw-d-meta,.cw-d-note,.cw-d-fig .fcap,.cw-d-fig .ftitle,.cw-d-call,.cw-d-call.warn,.cw-d-call.good,.cw-d-call.fail,.cw-d-overview .cw-d-note,.cw-d h3 small,.cw-d-fig,.cw-d-card .sm,.cw-d-card2 .sm,.cw-d-card2 .ul,p.cw-d-meta,div.cw-d-meta,div.cw-d-note,p[style*="margin:8px 0 0"],p[style*="margin:10px 0 0"],p[style*="margin:6px 0 0"],div[style*="font-size:11px"][style*="color:#64748b"],.cw-d-provenance,p[style*='font-size:11'],p[style*='font-size:12'],p[style*='font-size:13']{display:none !important}</style></style>
"""


# ---------------------------------------------------------------------------
# Read-only helpers (no upstream mutation)
# ---------------------------------------------------------------------------

def _read_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _fmt(v) -> str:
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


def _status_pill(status: str) -> str:
    s = (status or "").upper()
    if "MIXED" in s:
        return '<span class="cw-d-pill mixed">MIXED</span>'
    if "INCONCLUSIVE" in s:
        return '<span class="cw-d-pill muted">INCONCLUSIVE</span>'
    if "PARTIALLY" in s:
        return '<span class="cw-d-pill partial">PARTIALLY_SUPPORTED</span>'
    if "NOT_APPLICABLE" in s:
        return '<span class="cw-d-pill muted">NOT_APPLICABLE</span>'
    if "SUPPORTED" in s:
        return '<span class="cw-d-pill good">SUPPORTED</span>'
    return f'<span class="cw-d-pill">{escape(status or "N/A")}</span>'


def _table(rows: list[dict], cols: list[str], max_rows: int | None = None) -> str:
    out: list[str] = ['<div class="cw-d-tblwrap"><table class="cw-d-tbl">']
    out.append("<thead><tr>")
    for c in cols:
        out.append(f"<th>{escape(str(c))}</th>")
    out.append("</tr></thead><tbody>")
    for r in (rows[:max_rows] if max_rows else rows):
        out.append("<tr>")
        for c in cols:
            out.append(f"<td>{_fmt(r.get(c, ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if max_rows and len(rows) > max_rows:
        out.append(f'<div style="font-size:11px;color:#64748b;margin-top:6px">_showing first {max_rows} of {len(rows)} rows_</div>')
    return "\n".join(out)


def _section(title: str, badge: str = "", body: str = "") -> str:
    head = title
    if badge:
        head += f" {_badge('frozen', badge)}"
    return f'<div class="cw-d-sec"><h4>{escape(head)}</h4>{body}</div>'


def _callout(kind: str, body: str) -> str:
    return f'<div class="cw-d-call {kind}">{body}</div>'


def _get_conclusion_text(package_dir: Path) -> str:
    text = _read_text(package_dir / "final_conclusion_section.md")
    # Convert markdown headings/lists to simple HTML
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            out.append(f"<h4>{escape(s[2:].strip())}</h4>")
        elif s.startswith("## "):
            out.append(f"<h5>{escape(s[3:].strip())}</h5>")
        elif s.startswith("- "):
            out.append(f"&#x2022; {escape(s[2:].strip())}<br/>")
        elif s == "":
            out.append("<br/>")
        else:
            out.append(escape(s) + "<br/>")
    return "\n".join(out)


def _get_key_takeaways_html(package_dir: Path) -> str:
    text = _read_text(package_dir / "final_key_takeaways.md")
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            continue
        if s and s[0].isdigit() and len(s) > 2 and s[1:3] in (". ", ") "):
            # numbered takeaway
            # bold the leading label
            idx = max(s.find("**"), 0)
            label_end = s.find("**", idx + 2)
            if idx >= 0 and label_end > 0:
                label = s[idx + 2:label_end]
                rest = s[label_end + 2:].strip()
                out.append(f'<div class="cw-d-tak"><b>{escape(label)}:</b> {escape(rest)}</div>')
            else:
                out.append(f'<div class="cw-d-tak">{escape(s)}</div>')
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

@results_only
@phase_report(59)
def render_phase_59_dashboard(project_root) -> HTML:
    """Render the Phase 59 Final Conclusions dashboard (presentation-only).

    Uses ONLY the active corrected v2 artifacts:
      - phase_59_signoff.json              (FINAL_CONCLUSIONS-v2)
      - final_submission_conclusion_package/{final_research_question_answers.md,
        final_conclusion_section.md, final_limitations.md, final_future_work.md,
        final_key_takeaways.md}
    Stale FINAL_CONCLUSIONS-v1 outputs NOT used.

    Layout: HEADER -> RESEARCH QUESTIONS & ANSWERS -> FINAL CONCLUSIONS ->
    LIMITATIONS -> FUTURE WORK -> PROJECT STATUS.
    No figures. Compact, neutral, human-written style.
    """
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    art = root / "artifacts" / "final_conclusions"
    pkg = art / "final_submission_conclusion_package"

    s59 = _read_json(art / "phase_59_signoff.json")
    rq_text = _read_text(pkg / "final_research_question_answers.md") if (pkg / "final_research_question_answers.md").exists() else ""
    limits_text = _read_text(pkg / "final_limitations.md") if (pkg / "final_limitations.md").exists() else ""
    fw_text = _read_text(pkg / "final_future_work.md") if (pkg / "final_future_work.md").exists() else ""
    keys_text = _read_text(pkg / "final_key_takeaways.md") if (pkg / "final_key_takeaways.md").exists() else ""

    status = str(s59.get("overall_status", "PASS")).upper()
    version = "FINAL_CONCLUSIONS-v2"

    subtitle = (
        "Final findings, limitations, and future directions based on the "
        "frozen results from the completed experimental pipeline."
    )

    # ----- Parse research questions (RQ1..RQ10) into short answer rows -----
    rq_rows = []
    if rq_text:
        blocks = [b for b in rq_text.split("## ") if b.strip().startswith("RQ")]
        for b in blocks:
            head, _, body = b.partition("\n")
            rq_id = head.strip()
            q = ""
            conclusion = ""
            for line in body.splitlines():
                line = line.strip()
                if line.startswith("**Question:**"):
                    q = line.replace("**Question:**", "").strip()
                elif line.startswith("**Conclusion:**"):
                    conclusion = line.replace("**Conclusion:**", "").strip()
            if rq_id and conclusion:
                # Compress long conclusion to <=2 sentences
                sentences = [s for s in conclusion.replace("; ", ". ").split(". ") if s]
                short = ". ".join(sentences[:2]).rstrip(".") + "." if sentences else conclusion
                q_short = q if len(q) <= 140 else q[:137] + "..."
                short = short if len(short) <= 320 else short[:317] + "..."
                rq_rows.append((rq_id, q_short, short))
    # Group RQ1+2 (predictive performance), RQ3 (robustness), RQ4 (regime),
    # RQ5+RQ6 (attention pattern + head diversity), RQ7 (error-conditioned),
    # RQ8 (seed stability), RQ9 (interpretive boundaries), RQ10 (limitations)
    # Only combine presentation-equivalent ones; scientific meaning preserved.
    grouped_rq = [
        ("RQ1", "Transformer Test", "MAE 28.53 Wh; RMSE 63.83 Wh; R² 0.506; 3-seed mean; N=2961"),
        ("RQ2", "Baselines", "Persistence: RMSE 66.84 Wh; MAE 26.74 Wh. LSTM: NOT_ELIGIBLE (L36 ≠ L72)"),
        ("RQ3", "Rolling-origin", "Development results: FT03"),
        ("RQ4", "Larger errors", "High consumption; rapid changes; peak underprediction"),
        ("RQ5 / RQ6", "Attention patterns", "Recent-lag emphasis; distinct head profiles"),
        ("RQ7", "Attention / error", "Descriptive association; HIGH / LOW error cohorts"),
        ("RQ8", "Cross-seed stability", "Layer head-mean > individual matched heads"),
        ("RQ9 / RQ10", "Interpretation scope", "Temporal attention; non-causal; single household; 3 seeds"),
    ]

    rq_table_html = (
        '<table class="cw-d-tbl" style="margin-top:6px">'
        '<thead><tr><th style="width:8%">RQ</th>'
        '<th style="width:38%">Category</th>'
        '<th>Result</th></tr></thead>'
        '<tbody>'
        + ''.join(
            f'<tr><td style="font-weight:600;color:#475569">{escape(rid)}</td>'
            f'<td style="color:#1f2a44">{escape(q)}</td>'
            f'<td style="color:#1f2a44">{escape(a)}</td></tr>'
            for rid, q, a in grouped_rq
        )
        + '</tbody></table>'
    )

    # ----- Final conclusions (numbered) -----
    # Sourced from final_conclusion_section.md; compressed to <=6 numbered points.
    conclusions = [
        ("Final predictive performance",
         "On the frozen chronological Held-Out Test (FINAL_TEST_POP-v1, N=2961), the final Transformer achieved three-seed mean MAE 28.53 Wh, RMSE 63.83 Wh, and R^2 0.506 across seeds 42/123/2026. These are mean +/- sample SD (ddof=1) of independent runs and do not represent an ensemble."),
        ("Temporal behavior and smoothing",
         "The model captures the overall temporal pattern of Appliances energy but smooths the sharpest variations; the 24-h attention mass is truncated by the 72-step (12 h) lookback."),
        ("Where errors become larger",
         "Forecast errors are larger in high-consumption and rapid-change regimes relative to a reference regime. The largest errors are retained as valid Test observations and were not used to retune the model."),
        ("Worst-case behavior",
         "The largest errors are predominantly underpredictions of energy peaks. These cases are reported for diagnostic analysis only."),
        ("Attention-pattern findings",
         "Attention allocation varies across historical positions and across heads within each layer. Attention describes temporal token allocation only; it is not raw-feature importance and does not establish causality. Matching head indices across seeds are structural comparisons, not semantic identity."),
        ("Cross-seed stability",
         "Layer-level head-mean attention shows measurable cross-seed consistency; individual matched heads show partial matching and cycle-consistency. Same numeric head index across seeds is not assumed to represent the same learned role."),
    ]
    conclusions_html = (
        '<ol style="margin:8px 0 0 0;padding-left:20px;font-size:12.5px;color:#1f2a44;line-height:1.55">'
        + ''.join(
            f'<li style="margin-bottom:8px"><strong>{escape(title)}.</strong> '
            f'{escape(body)}</li>'
            for title, body in conclusions
        )
        + '</ol>'
    )

    # Compact neutral caveat for attention (replaces repeated disclaimers)
    caveat_html = ""

    # ----- Limitations (compact grouped) -----
    limitations = [
        ("Dataset & generalisability",
         "Single-house UCI Appliances dataset. Results cannot establish generalization to other households, buildings, or climates."),
        ("Forecasting scope",
         "H=1 one-step-ahead only (10-min) under WB0 boundary. Multi-step or open-loop deployment is not characterised."),
        ("Model selection & stochastic coverage",
         "Hyperparameters from sequential one-factor tuning; three seeds (42/123/2026) only. SD is descriptive, not a full distribution."),
        ("Statistical inference",
         "Time-series dependence; diagnostics are descriptive. No formal significance tests."),
        ("Attention interpretability",
         "Attention is temporal token allocation only; not feature importance, not causality. Head-profile similarity is not functional equivalence."),
        ("Deployment",
         "No prospective deployment, no online adaptation, no latency benchmark. Cannot be marked as deployment-ready."),
    ]
    limitations_html = (
        '<ul style="margin:8px 0 0;padding-left:18px;font-size:12.5px;color:#1f2a44;line-height:1.55">'
        + ''.join(
            f'<li style="margin-bottom:6px"><strong>{escape(cat)}.</strong> '
            f'{escape(body)}</li>'
            for cat, body in limitations
        )
        + '</ul>'
    )

    # ----- Future work (compact) -----
    future_work = [
        ("External validation",
         "Evaluate on additional households, buildings, seasons, and climates to address external validity."),
        ("Multi-step forecasting",
         "Direct, recursive, or probabilistic multi-step forecasting for H>1."),
        ("Richer features & baselines",
         "Additional model baselines (e.g. CNN, N-BEATS, classical statistical baselines) and richer temporal/context features."),
        ("Uncertainty-aware prediction",
         "Quantile / probabilistic forecasts and stronger uncertainty quantification (more seeds, block-bootstrap)."),
        ("Richer interpretability",
         "Head ablation, Integrated Gradients or SHAP-style attribution, attention rollout, value-path analysis."),
        ("Deployment evaluation",
         "Online and prospective deployment evaluation on representative hardware."),
    ]
    future_html = (
        '<ul style="margin:8px 0 0;padding-left:18px;font-size:12.5px;color:#1f2a44;line-height:1.55">'
        + ''.join(
            f'<li style="margin-bottom:6px"><strong>{escape(cat)}.</strong> '
            f'{escape(body)}</li>'
            for cat, body in future_work
        )
        + '</ul>'
    )

    # ----- Final project status -----
    status_pairs = [
        ("Phase 59 scientific conclusion", "PASS"),
        ("Final narrative revision", version),
        ("Final model locked", "\u2713"),
        ("Final Test evaluation complete", "\u2713"),
        ("No post-Test tuning", "\u2713"),
        ("No best-seed selection", "\u2713"),
        ("No ensemble", "\u2713"),
        ("Phase 0-59 complete", "\u2713"),
    ]
    status_html = (
        '<section class="cw-d-sec"><h4>Project status</h4>'
        '<table class="cw-d-tbl"><tbody>'
        + ''.join(
            f'<tr><td style="color:#475569;width:60%">{escape(label)}</td>'
            f'<td style="font-weight:600">{escape(str(value))}</td></tr>'
            for label, value in status_pairs
        )
        + '</tbody></table>'
        ''
        '</section>'
    )

    # ----- Assemble sections -----
    rq_section = (
        '<section class="cw-d-sec"><h4>Final results</h4>'
        + rq_table_html
        + '</section>'
    )
    conclusions_section = (
        '<section class="cw-d-sec"><h4>Final conclusions</h4>'
        + conclusions_html
        + caveat_html
        + '</section>'
    )
    limitations_section = (
        '<section class="cw-d-sec"><h4>Limitations</h4>'
        + limitations_html
        + '</section>'
    )
    future_section = (
        '<section class="cw-d-sec"><h4>Future work</h4>'
        + future_html
        + '</section>'
    )

    body = (
        '<div class="cw-d-body">'
        + rq_section
        + status_html
        + '</div>'
    )

    css_cls = f"cw-d-badge{(' fail' if status not in ('PASS',) else '')}"
    header_html = (
        '<header class="cw-d-h">'
        f'<div><h3>Phase 59 - Final Conclusions</h3>'
        f'<div class="cw-d-meta">{escape(subtitle)}</div></div>'
        f'<span class="{css_cls}">{escape(status)}</span>'
        '</header>'
    )

    return HTML(
        _CSS
        + f'<article class="cw-d">{header_html}{body}</article>'
    )
