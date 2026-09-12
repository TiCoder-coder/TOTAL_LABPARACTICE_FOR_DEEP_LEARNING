"""Read-only notebook views for retained historical evidence.

Nothing in this module materializes phases or repairs scientific artifacts.
It renders an existing processing log when usable, then falls back to an
explicit inventory and summary of the retained canonical files.
"""
from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from IPython.display import HTML

from course_work.reporting.phase_summary import (
    LOG_FILENAMES,
    PHASE_NAMES,
    SOURCE_SPECS,
    build_phase_resume_log,
    render_phase_log,
    render_phase_resume_log,
)
from course_work.utils.artifacts import read_json, sha256_file


_USABLE_LOG_STATUSES = {"PASS", "PASS_WITH_WARNING", "VALID_REUSABLE"}
_V10_PRESENTATION_RECOVERY_ROOT = Path(
    "artifacts/notebook_presentation_recovery/courseworkv10_phase54_57"
)


def _render_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return escape(str(value))


def _table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_render_value(row.get(column, ''))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return (
        f'<section class="cw-frozen-section"><h4>{escape(title)}</h4>'
        '<div class="cw-frozen-scroll"><table class="cw-frozen-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>"
    )


def _scalar_rows(payload: dict[str, Any], limit: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not isinstance(nested_value, (dict, list)):
                    rows.append({"Field": f"{key}.{nested_key}", "Value": nested_value})
        elif not isinstance(value, list):
            rows.append({"Field": key, "Value": value})
        if len(rows) >= limit:
            break
    return rows[:limit]


def _artifact_rows(root: Path, paths: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for role, relative in paths:
        path = root / relative
        rows.append(
            {
                "Role": role,
                "Artifact": relative,
                "Available": path.is_file(),
                "SHA-256": sha256_file(path) if path.is_file() else "NOT_AVAILABLE",
            }
        )
    return rows


def _read_preview(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {"value": payload}
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {"row_count": len(rows), "columns": list(rows[0]) if rows else []}
    return {}


def _fallback_dashboard(
    *,
    root: Path,
    title: str,
    subtitle: str,
    artifacts: Iterable[tuple[str, str]],
    status: str = "FROZEN_EVIDENCE",
) -> HTML:
    artifacts = tuple(artifacts)
    inventory = _artifact_rows(root, artifacts)
    details = []
    for role, relative in artifacts:
        path = root / relative
        if not path.is_file() or path.suffix not in {".json", ".csv"}:
            continue
        rows = _scalar_rows(_read_preview(path))
        if rows:
            details.append(_table(f"{role}: {path.name}", rows))
    css = """
<style>
.cw-frozen{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;border:1px solid #dbe3ee;border-radius:14px;background:#fff;margin:14px 0 24px;overflow:hidden}
.cw-frozen *{box-sizing:border-box}.cw-frozen header{display:flex;justify-content:space-between;gap:16px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f8f5ff);border-bottom:1px solid #dbe3ee}
.cw-frozen h3{margin:0 0 5px;font-size:20px}.cw-frozen p{margin:0;color:#5d6b82;font-size:13px}.cw-frozen .badge{height:max-content;padding:6px 11px;border-radius:999px;background:#fff5dc;border:1px solid #efd18a;color:#7a4b00;font-size:11px;font-weight:700}
.cw-frozen-content{padding:2px 22px 22px}.cw-frozen-section{margin-top:18px}.cw-frozen-section h4{margin:0 0 8px;font-size:14px}.cw-frozen-scroll{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px}
.cw-frozen-table{border-collapse:collapse;width:100%;font-size:12.5px}.cw-frozen-table th{background:#f7f9fc;text-align:left;padding:9px 11px;white-space:nowrap}.cw-frozen-table td{padding:9px 11px;border-top:1px solid #edf1f5;vertical-align:top;overflow-wrap:anywhere}.cw-frozen-table tbody tr:nth-child(even){background:#fbfcfe}
</style>
"""
    content = _table("Frozen artifact inventory", inventory) + "".join(details)
    return HTML(
        f'{css}<article class="cw-frozen"><header><div><h3>{escape(title)}</h3>'
        f'<p>{escape(subtitle)}</p></div><span class="badge">{escape(status)}</span></header>'
        f'<div class="cw-frozen-content">{content}</div></article>'
    )


def render_frozen_phase_evidence(phase_id: int, project_root: Path | str) -> HTML:
    """Render a retained phase log without rebuilding or writing artifacts."""
    root = Path(project_root).resolve()
    if phase_id not in LOG_FILENAMES:
        raise ValueError(f"Unsupported phase: {phase_id}")
    # The canonical Phase 6 presentation spec intentionally contains figures
    # only.  The notebook additionally needs an auditable tabular inventory.
    if phase_id == 6:
        return _fallback_dashboard(
            root=root,
            title=f"Phase {phase_id} - {PHASE_NAMES[phase_id]}",
            subtitle="Frozen EDA inventory; figures below remain the canonical presentation outputs.",
            artifacts=SOURCE_SPECS[phase_id],
        )
    log_path = root / "docs" / "save_log_in_processing" / LOG_FILENAMES[phase_id]
    if log_path.is_file():
        log = read_json(log_path)
        if str(log.get("status", "")).upper() in _USABLE_LOG_STATUSES:
            if 23 <= phase_id <= 41:
                return render_phase_resume_log(log)
            return render_phase_log(log)
    # A stale/blocked saved processing log must not force the notebook into a
    # generic frozen inventory.  For selective sweep phases, build the current
    # read-only audit in memory and render its real state (including BLOCKED and
    # the exact discrepancies).  This does not save a processing log or execute
    # a scientific phase.
    if 23 <= phase_id <= 41:
        return render_phase_resume_log(build_phase_resume_log(phase_id, root))
    return _fallback_dashboard(
        root=root,
        title=f"Phase {phase_id} - {PHASE_NAMES[phase_id]}",
        subtitle="Read-only fallback over retained evidence; no phase execution or artifact regeneration.",
        artifacts=SOURCE_SPECS[phase_id],
    )


def render_historical_analysis_summary(
    phase_id: int,
    project_root: Path | str,
    summary_path: str,
    signoff_path: str,
) -> HTML:
    """Render verified historical summary/signoff evidence without recomputation."""
    root = Path(project_root).resolve()
    return _fallback_dashboard(
        root=root,
        title=f"Phase {phase_id} - historical summary",
        subtitle="Verified summary/signoff view; detailed dashboard CSVs are unavailable in this snapshot.",
        artifacts=(("summary", summary_path), ("signoff", signoff_path)),
        status="HISTORICAL_SUMMARY_ONLY",
    )


def render_verified_courseworkv10_dashboard(
    phase_id: int,
    project_root: Path | str,
) -> HTML:
    """Render a byte-verified Phase 54-57 dashboard preserved in CourseWorkV10.

    The snapshots are presentation evidence extracted from the committed V10
    notebook.  Before returning one, this function validates both its checksum
    and every active source artifact recorded in the recovery manifest.  It
    never invokes analysis, model loading, inference, or Test-data readers.
    """
    if phase_id not in {54, 55, 56, 57}:
        raise ValueError(f"Unsupported recovered dashboard phase: {phase_id}")
    root = Path(project_root).resolve()
    recovery_root = root / _V10_PRESENTATION_RECOVERY_ROOT
    manifest = read_json(recovery_root / "manifest.json")
    if manifest.get("artifact_version") != "NOTEBOOK_PRESENTATION_RECOVERY-v1":
        raise RuntimeError("Invalid CourseWorkV10 presentation recovery manifest")
    if manifest.get("scientific_recomputation") is not False:
        raise RuntimeError("Recovered dashboard is not presentation-only evidence")
    entry = manifest.get("phases", {}).get(str(phase_id))
    if not isinstance(entry, dict):
        raise RuntimeError(f"Missing recovered dashboard manifest entry: Phase {phase_id}")
    for relative, expected in entry.get("active_evidence_sha256", {}).items():
        source = root / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(
                f"Recovered Phase {phase_id} dashboard source mismatch: {relative}"
            )
    snapshot = root / str(entry["snapshot_path"])
    if not snapshot.is_file() or sha256_file(snapshot) != entry["snapshot_sha256"]:
        raise RuntimeError(f"Recovered Phase {phase_id} dashboard checksum mismatch")
    return HTML(snapshot.read_text(encoding="utf-8"))


def render_frozen_analysis_summary(
    phase_id: int,
    project_root: Path | str,
    summary_path: str,
    signoff_path: str,
) -> HTML:
    """Backward-compatible alias for the historical summary renderer."""
    return render_historical_analysis_summary(
        phase_id,
        project_root,
        summary_path,
        signoff_path,
    )
