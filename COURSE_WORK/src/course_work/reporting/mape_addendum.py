import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import HTML

from course_work.metric_addendum.contract import ARTIFACT_ROOT
from course_work.utils.artifacts import get_project_root, read_json


def _display_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        if math.isnan(value):
            return "Not available"
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)


def _status_class(value: Any) -> str:
    status = str(value).upper()
    if status == "PASS" or status == "DEFINED":
        return "mape-pass"
    if status in {"PARTIAL", "PASS_WITH_LIMITATIONS", "PASS_WITH_BLOCKED_TEST_EXTENSION", "NOT_APPLICABLE"}:
        return "mape-partial"
    return "mape-blocked"


def _table(headers: list[str], rows: list[list[Any]], status_column: int | None = None) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            rendered = escape(_display_value(value))
            if status_column == index:
                rendered = f'<span class="mape-state {_status_class(value)}">{rendered}</span>'
            cells.append(f"<td>{rendered}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="mape-table-wrap"><table class="mape-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def render_mape_addendum(project_root: Path | None = None) -> HTML:
    root = Path(project_root or get_project_root()).resolve()
    artifact_root = root / ARTIFACT_ROOT
    contract = read_json(artifact_root / "mape_metric_contract.json")
    summary = read_json(artifact_root / "validation_mape_summary.json")
    test_status = read_json(artifact_root / "final_test_mape_status.json")
    signoff = read_json(artifact_root / "mape_addendum_signoff.json")
    validation = pd.read_csv(artifact_root / "validation_mape_by_run.csv")
    test_sources = pd.read_csv(artifact_root / "final_test_source_audit.csv")
    audit_path = artifact_root / "ml_pipeline_compliance_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else {"overall_status": "NOT_AVAILABLE", "status_counts": {}, "records": []}
    metric = contract["metric_contract"]
    validation_rows = [
        [
            row.run_id,
            row.model_family,
            row.model_id,
            float(row.mape_pct),
            int(row.n_samples),
            int(row.zero_target_count),
            row.mape_status,
        ]
        for row in validation.itertuples(index=False)
    ]
    test_rows = [
        [row.source_id, row.path, row.expected_rows, row.actual_rows, row.status]
        for row in test_sources.itertuples(index=False)
    ]
    audit_rows = [
        [row["order"], row["area"], row["status"], row["finding"], row["evidence"]]
        for row in audit.get("records", [])
    ]
    contract_rows = [
        ["Metric", metric["display_name"]],
        ["Formula", metric["formula"]],
        ["Unit", metric["result_unit"]],
        ["Direction", metric["direction"]],
        ["Role", metric["role"]],
        ["Zero-target policy", metric["zero_target_policy"]],
        ["Selection metric remains", summary["selection_metric"]],
    ]
    counts = audit.get("status_counts", {})
    cards = [
        ["Validation sources", summary["validation_sources_discovered"]],
        ["MAPE defined", summary["validation_mape_defined"]],
        ["Transformer runs", summary["transformer_run_count"]],
        ["Cross-config aggregation", summary["cross_configuration_mape_aggregation"]],
        ["Seed mean and SD", summary["seed_mean_sd_status"]],
        ["Pipeline PASS checks", counts.get("PASS", 0)],
    ]
    cards_html = "".join(
        f'<div class="mape-card"><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>'
        for label, value in cards
    )
    html = f"""
<style>
.mape-dashboard{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;border:1px solid #dbe3ee;border-radius:16px;background:#fff;box-shadow:0 10px 28px rgba(31,45,61,.09);margin:14px 0 24px;overflow:hidden}}
.mape-dashboard *{{box-sizing:border-box}}
.mape-header{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;padding:22px 24px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #dbe3ee}}
.mape-header h3{{font-size:22px;line-height:1.3;margin:0 0 5px;color:#172033}}
.mape-header p{{margin:0;color:#5d6b82;font-size:13px}}
.mape-badge{{white-space:nowrap;border-radius:999px;padding:8px 13px;font-size:12px;font-weight:700;letter-spacing:.03em;border:1px solid #efd18a;color:#7a4b00;background:#fff5dc}}
.mape-body{{padding:4px 24px 24px}}
.mape-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:20px}}
.mape-card{{display:flex;flex-direction:column;gap:5px;padding:12px 14px;border:1px solid #e1e7f0;border-radius:10px;background:#fbfcff}}
.mape-card span{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}}
.mape-card strong{{font-size:15px;color:#24324a;overflow-wrap:anywhere}}
.mape-section{{margin-top:22px}}
.mape-section h4{{font-size:15px;margin:0 0 10px;color:#334155}}
.mape-section p{{font-size:13px;line-height:1.55;color:#475569}}
.mape-table-wrap{{overflow:auto;max-height:520px;border:1px solid #e2e8f0;border-radius:10px}}
.mape-table{{border-collapse:collapse;width:100%;table-layout:auto;font-size:12.5px;background:#fff}}
.mape-table th{{position:sticky;top:0;z-index:2;background:#263f68;color:#fff;text-align:left;font-weight:650;padding:10px 12px;white-space:nowrap}}
.mape-table td{{text-align:left;padding:10px 12px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}}
.mape-table tbody tr:nth-child(even){{background:#f8fafe}}
.mape-state{{display:inline-block;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:700;white-space:nowrap}}
.mape-pass{{color:#11613d;background:#e8f7ef;border:1px solid #a9dec1}}
.mape-partial{{color:#7a4b00;background:#fff5dc;border:1px solid #efd18a}}
.mape-blocked{{color:#8f2430;background:#fdecef;border:1px solid #efb3ba}}
.mape-note{{border-left:3px solid #d09a2d;background:#fffaf0;padding:10px 12px;border-radius:0 8px 8px 0}}
@media (max-width:720px){{.mape-header{{flex-direction:column}}.mape-body{{padding-left:12px;padding-right:12px}}}}
</style>
<article class="mape-dashboard">
<header class="mape-header"><div><h3>Supplementary MAPE Metric Addendum</h3><p>{escape(str(signoff["artifact_version"]))} | Validation evidence only where signed sources exist</p></div><span class="mape-badge">{escape(str(signoff["status"]))}</span></header>
<div class="mape-body">
<div class="mape-cards">{cards_html}</div>
<section class="mape-section"><h4>Metric contract</h4>{_table(["Field", "Value"], contract_rows)}</section>
<section class="mape-section"><h4>Validation MAPE by available prediction artifact</h4>{_table(["Run ID", "Model family", "Model", "MAPE (%)", "Samples", "Zero targets", "Status"], validation_rows, 6)}</section>
<section class="mape-section"><h4>Final Test source gate</h4><p class="mape-note">Test MAPE status: <strong>{escape(str(test_status["status"]))}</strong>. No Test inference or checkpoint execution was performed.</p>{_table(["Source", "Expected path", "Expected rows", "Actual rows", "Status"], test_rows, 4)}</section>
<section class="mape-section"><h4>Machine-learning pipeline compliance audit</h4><p>Overall status: <strong>{escape(str(audit["overall_status"]))}</strong>. Partial and blocked items are retained as explicit limitations.</p>{_table(["Order", "Area", "Status", "Finding", "Evidence"], audit_rows, 2)}</section>
</div>
</article>
"""
    return HTML(html)
