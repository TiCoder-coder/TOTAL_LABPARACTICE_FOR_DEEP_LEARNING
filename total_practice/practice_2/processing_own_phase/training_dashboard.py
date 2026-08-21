"""Artifact-only HTML training dashboard for Practice 2.

This module never imports the model, data loaders, training loop, or final
evaluation code.  It renders existing JSON/JSONL/CSV/log artifacts only.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TrainingDashboard:
    """Notebook-displayable self-contained HTML dashboard."""

    html: str
    saved_path: Optional[Path] = None

    def _repr_html_(self) -> str:
        return self.html

    def __str__(self) -> str:
        return self.html


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"Missing artifact: {path.name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return {}


def read_metrics_jsonl(path: Path, warnings: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Read complete metric rows; tolerate only a malformed final line."""
    messages = warnings if warnings is not None else []
    if not path.is_file():
        messages.append(f"Missing artifact: {path.name}")
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                messages.append("Ignored an incomplete final metrics.jsonl row")
                break
            raise ValueError(f"Malformed metrics.jsonl row {index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"metrics.jsonl row {index + 1} is not an object")
        rows.append(row)
    return rows


def _read_csv(path: Path, warnings: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        warnings.append(f"Missing artifact: {path.name}")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return []


def _resolve_selected_run(
    runs_dir: Path,
    selection: Mapping[str, Any],
    selected_run: Optional[str | Path],
    warnings: list[str],
) -> Optional[Path]:
    if selected_run is not None:
        candidate = Path(selected_run).expanduser()
        if not candidate.is_absolute():
            candidate = runs_dir / candidate
        if candidate.is_dir():
            return candidate.resolve()
        warnings.append(f"Selected run does not exist: {candidate}")
        return None

    checkpoint = selection.get("selected_checkpoint")
    if checkpoint:
        candidate = Path(str(checkpoint)).expanduser().parent
        if candidate.is_dir():
            return candidate.resolve()

    run_id = selection.get("selected_experiment")
    if run_id and (runs_dir / str(run_id)).is_dir():
        return (runs_dir / str(run_id)).resolve()

    metric_files = list(runs_dir.glob("*/metrics.jsonl")) if runs_dir.is_dir() else []
    if metric_files:
        warnings.append("Selection could not be resolved; showing the newest metrics run")
        return max(metric_files, key=lambda item: item.stat().st_mtime).parent.resolve()
    warnings.append("No saved training run could be resolved")
    return None


def _summary_for_run(run_dir: Optional[Path], warnings: list[str]) -> dict[str, Any]:
    if run_dir is None:
        return {}
    modern = run_dir / "summary.json"
    legacy = run_dir / f"{run_dir.name}_summary.json"
    if modern.is_file():
        return _read_json(modern, warnings)
    return _read_json(legacy, warnings)


def _device_from_log(run_dir: Optional[Path]) -> Optional[str]:
    if run_dir is None:
        return None
    for log_path in sorted(run_dir.glob("*.log")):
        try:
            match = re.search(
                r"Using device:\s*([^\s|]+)",
                log_path.read_text(encoding="utf-8", errors="replace"),
                flags=re.IGNORECASE,
            )
        except OSError:
            continue
        if match:
            return match.group(1).lower()
    return None


def load_training_dashboard_data(visualization_data_path: str | Path) -> dict[str, Any]:
    """Read the single aggregate JSON used by the dashboard renderer."""
    path = Path(visualization_data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visualization data does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid visualization data JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Visualization data must be a JSON object")
    required = {"selected_run", "training_history", "hyperparameter_search", "checkpoints", "final_evaluation", "error_analysis", "sources"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Visualization data is missing sections: {', '.join(missing)}")
    return data


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if abs(number) <= 1:
        number *= 100
    return f"{number:.2f}%"


def _duration(value: Any) -> str:
    seconds = _number(value)
    if seconds is None:
        return "—"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes:d}m {secs:02d}s"


def _escape(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "—"))


def _series_svg(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[tuple[str, str, str]],
    y_label: str,
    markers: Sequence[tuple[int, str, str]] = (),
    zero_line: bool = False,
    log_scale: bool = False,
) -> str:
    available = [item for item in series if any(_number(row.get(item[0])) is not None for row in rows)]
    if not rows or not available:
        return f'<section class="p2d-chart"><h3>{_escape(title)}</h3><p class="p2d-empty">No saved data available.</p></section>'
    width, height = 680, 280
    left, right, top, bottom = 62, 22, 22, 46
    epochs = [_number(row.get("epoch")) for row in rows]
    valid_epochs = [value for value in epochs if value is not None]
    values = [
        value
        for key, _label, _css in available
        for row in rows
        if (value := _number(row.get(key))) is not None and (not log_scale or value > 0)
    ]
    if not valid_epochs or not values:
        return f'<section class="p2d-chart"><h3>{_escape(title)}</h3><p class="p2d-empty">No saved data available.</p></section>'
    x_min, x_max = min(valid_epochs), max(valid_epochs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    transformed = [math.log10(value) if log_scale else value for value in values]
    y_min, y_max = min(transformed), max(transformed)
    if zero_line and not log_scale:
        y_min, y_max = min(y_min, 0.0), max(y_max, 0.0)
    padding = (y_max - y_min) * 0.08 or max(abs(y_max) * 0.08, 1.0)
    y_min -= padding
    y_max += padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        mapped = math.log10(value) if log_scale else value
        return top + (y_max - mapped) / (y_max - y_min) * (height - top - bottom)

    grid = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * (height - top - bottom)
        raw = y_max - ratio * (y_max - y_min)
        label = 10 ** raw if log_scale else raw
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="p2d-grid"/>'
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{label:.3g}</text>'
        )
    paths = []
    legends = []
    for key, label, css_class in available:
        points = []
        for row in rows:
            epoch = _number(row.get("epoch"))
            value = _number(row.get(key))
            if epoch is None or value is None or (log_scale and value <= 0):
                continue
            points.append((sx(epoch), sy(value)))
        if not points:
            continue
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(points)
        )
        paths.append(f'<path d="{path}" class="p2d-line {css_class}"/>')
        legends.append(f'<span><i class="{css_class}"></i>{_escape(label)}</span>')
    marker_lines = []
    for epoch, label, css_class in markers:
        if x_min <= epoch <= x_max:
            x = sx(float(epoch))
            marker_lines.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="p2d-marker {css_class}"/>'
                f'<text x="{x+4:.1f}" y="{top+13}" class="p2d-marker-label">{_escape(label)}</text>'
            )
    zero = ""
    if zero_line and y_min <= 0 <= y_max:
        y = sy(0)
        zero = f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="p2d-zero"/>'
    return (
        f'<section class="p2d-chart"><h3>{_escape(title)}</h3>'
        f'<div class="p2d-legend">{"".join(legends)}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f'<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" class="p2d-frame"/>'
        f'{"".join(grid)}{zero}{"".join(paths)}{"".join(marker_lines)}'
        f'<text x="{(left+width-right)/2:.1f}" y="{height-8}" text-anchor="middle">Epoch</text>'
        f'<text transform="translate(15 {(top+height-bottom)/2:.1f}) rotate(-90)" text-anchor="middle">{_escape(y_label)}</text>'
        '</svg></section>'
    )


def _overview(data: Mapping[str, Any]) -> str:
    run = data.get("selected_run", {})
    items = [
        ("Run ID", run.get("run_id")),
        ("Experiment", run.get("experiment_id")),
        ("Model", run.get("model")),
        ("Training mode", run.get("training_mode")),
        ("Device", run.get("device")),
        ("Best epoch", run.get("best_epoch")),
        ("Total epochs", run.get("total_epochs")),
        ("Best Validation loss", _fmt(run.get("best_val_loss"), 6)),
        ("Best Validation accuracy", _pct(run.get("best_val_accuracy"))),
        ("Training duration", _duration(run.get("training_duration"))),
    ]
    return '<div class="p2d-overview">' + "".join(
        f'<div class="p2d-stat"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in items
    ) + "</div>"


def _ranking_table(data: Mapping[str, Any]) -> str:
    search = data.get("hyperparameter_search", {})
    rows = search.get("ranking", [])
    if not rows:
        return '<p class="p2d-empty">Hyperparameter ranking is unavailable.</p>'
    selected = str(search.get("selected_experiment", ""))
    body = []
    for row in rows:
        run_id = row.get("run_id", "")
        is_selected = run_id == selected
        head = row.get("hidden_layers", "[]")
        learning_rate = f'{row.get("head_learning_rate", "—")} / {row.get("backbone_learning_rate", "—")}'
        body.append(
            f'<tr class="{"p2d-selected" if is_selected else ""}">'
            f'<td>{_escape(row.get("rank"))}</td><td>{_escape(run_id)}</td>'
            f'<td>{_escape(learning_rate)}</td><td>{_escape(head)}</td>'
            f'<td>{_escape(row.get("best_epoch"))}</td><td>{_fmt(row.get("best_val_loss"), 6)}</td>'
            f'<td>{_pct(row.get("best_val_accuracy"))}</td><td>{"Selected" if is_selected else ""}</td></tr>'
        )
    return (
        '<div class="p2d-table-wrap"><table><thead><tr><th>Rank</th><th>Experiment</th>'
        '<th>Head / backbone LR</th><th>Classifier head</th><th>Best epoch</th>'
        '<th>Validation loss</th><th>Validation accuracy</th><th>Selection</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _checkpoint_table(data: Mapping[str, Any]) -> str:
    checkpoints = data.get("checkpoints", {})
    body = []
    for alias, name in (("latest", "latest.pt"), ("best", "best.pt"), ("best_val_loss", "best_val_loss.pt"), ("best_val_accuracy", "best_val_accuracy.pt")):
        detail = checkpoints.get(alias, {})
        available = bool(detail.get("exists"))
        size = detail.get("size_bytes")
        size_text = f"{int(size) / (1024 * 1024):.1f} MB" if _number(size) is not None else "—"
        sha = str(detail.get("sha256", ""))
        body.append(
            f'<tr><td>{_escape(name)}</td><td class="{"p2d-ok" if available else "p2d-missing"}">'
            f'{"✓ Available" if available else "— Missing"}</td><td>{_escape(detail.get("epoch"))}</td>'
            f'<td>{_fmt(detail.get("val_loss"), 6)}</td><td>{_pct(detail.get("val_accuracy"))}</td>'
            f'<td>{_escape(size_text)}</td><td><code>{_escape(sha[:12] if sha else "—")}</code></td></tr>'
        )
    return (
        '<div class="p2d-table-wrap"><table><thead><tr><th>Checkpoint</th><th>Status</th>'
        '<th>Epoch</th><th>Val loss</th><th>Val accuracy</th><th>Size</th><th>SHA256</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _final_evaluation(data: Mapping[str, Any]) -> str:
    summary = data.get("final_evaluation", {})
    items = [
        ("Test accuracy", _pct(summary.get("test_accuracy"))),
        ("Test loss", _fmt(summary.get("test_loss"), 6)),
        ("Macro precision", _fmt(summary.get("macro_precision"), 6)),
        ("Macro recall", _fmt(summary.get("macro_recall"), 6)),
        ("Macro F1", _fmt(summary.get("macro_f1"), 6)),
    ]
    return '<p class="p2d-readonly">Saved Final Test artifacts only — no inference is executed.</p><div class="p2d-final">' + "".join(
        f'<div><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
        for label, value in items
    ) + "</div>"


def _confusion_table(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return '<p class="p2d-empty">Saved confusion matrix is unavailable.</p>'
    columns = [column for column in rows[0] if column != "true_label"]
    maximum = max((_number(value) or 0 for row in rows for value in (row.get(column) for column in columns)), default=1) or 1
    body = []
    for row in rows:
        label = row.get("true_label") or next(iter(row.values()), "")
        cells = []
        for column in columns:
            value = _number(row.get(column)) or 0
            opacity = 0.08 + 0.72 * value / maximum
            cells.append(f'<td style="--heat:{opacity:.3f}">{int(value)}</td>')
        body.append(f'<tr><th>{_escape(label)}</th>{"".join(cells)}</tr>')
    return (
        '<div class="p2d-table-wrap p2d-confusion"><table><thead><tr><th>True ↓ / Predicted →</th>'
        + "".join(f'<th>{_escape(column)}</th>' for column in columns)
        + f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _error_analysis(data: Mapping[str, Any]) -> str:
    analysis = data.get("error_analysis", {})
    report = [row for row in (analysis.get("classification_report") or []) if row.get("class") not in {"accuracy", "macro avg", "weighted avg"}]
    report_body = "".join(
        f'<tr><td>{_escape(row.get("class"))}</td><td>{_fmt(row.get("precision"))}</td>'
        f'<td>{_fmt(row.get("recall"))}</td><td>{_fmt(row.get("f1-score"))}</td>'
        f'<td>{_escape(row.get("support"))}</td></tr>'
        for row in report
    )
    report_html = (
        '<div class="p2d-table-wrap"><table><thead><tr><th>Class</th><th>Precision</th>'
        '<th>Recall</th><th>F1</th><th>Support</th></tr></thead>'
        f'<tbody>{report_body}</tbody></table></div>'
        if report_body else '<p class="p2d-empty">Saved classification report is unavailable.</p>'
    )
    errors = analysis.get("highest_confidence_errors") or []
    error_body = "".join(
        f'<tr><td>{_escape(row.get("true_label"))}</td><td>{_escape(row.get("predicted_label"))}</td>'
        f'<td>{_pct(row.get("confidence"))}</td></tr>' for row in errors[:10]
    )
    errors_html = (
        '<div class="p2d-table-wrap"><table><thead><tr><th>Ground truth</th><th>Prediction</th>'
        f'<th>Confidence</th></tr></thead><tbody>{error_body}</tbody></table></div>'
        if error_body else '<p class="p2d-empty">Saved prediction errors are unavailable.</p>'
    )
    return (
        '<div class="p2d-error-grid"><section><h3>Per-class report</h3>' + report_html
        + '</section><section><h3>Highest-confidence saved errors</h3>' + errors_html + "</section></div>"
    )


def build_training_dashboard_html(data: Mapping[str, Any]) -> str:
    """Build a self-contained HTML fragment from loaded dashboard data."""
    rows = data.get("training_history", [])
    best_loss_epoch = int(data.get("selected_run", {}).get("best_epoch") or 0)
    best_acc_epoch = 0
    if rows:
        best_acc_epoch = int(max(rows, key=lambda row: (_number(row.get("val_accuracy")) or -math.inf, -(_number(row.get("val_loss")) or math.inf))).get("epoch", 0))
    normalized_rows = []
    for row in rows:
        copy = dict(row)
        normalized_rows.append(copy)
    lr_names = [name for name in ("head_lr", "backbone_lr") if any(_number(row.get(name)) is not None for row in normalized_rows)]
    markers = [(best_loss_epoch, "Best loss", "p2d-c3"), (best_acc_epoch, "Best accuracy", "p2d-c4")]
    warning_html = ""
    if data.get("warnings"):
        warning_html = '<details class="p2d-warnings"><summary>Artifact notes</summary><ul>' + "".join(
            f'<li>{_escape(message)}</li>' for message in data["warnings"]
        ) + "</ul></details>"
    css = """
<style>
#practice2-training-dashboard{--bg:#f7f8fa;--surface:#fff;--text:#172033;--muted:#617087;--border:#d9dee8;--grid:#e7eaf0;--accent:#2457a6;--soft:#eaf1fb;--ok:#167447;--bad:#a13b3b;--c1:#2457a6;--c2:#d97925;--c3:#aa3d61;--c4:#148168;--c5:#7453a6;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--bg);padding:22px;border:1px solid var(--border);border-radius:12px;line-height:1.45}
@media(prefers-color-scheme:dark){#practice2-training-dashboard{--bg:#111722;--surface:#182131;--text:#edf2fa;--muted:#aeb9ca;--border:#344055;--grid:#2a3547;--accent:#8bb8ff;--soft:#1d3150;--ok:#65d59a;--bad:#ff9b9b;--c1:#8bb8ff;--c2:#ffb36b;--c3:#f48bab;--c4:#65d59a;--c5:#c0a1ff}}
#practice2-training-dashboard *{box-sizing:border-box}#practice2-training-dashboard h1{margin:0;font-size:1.65rem;font-weight:500}#practice2-training-dashboard h2{margin:30px 0 12px;font-size:1.2rem;font-weight:500;border-bottom:1px solid var(--border);padding-bottom:7px}#practice2-training-dashboard h3{margin:8px 0;font-size:1rem;font-weight:500}#practice2-training-dashboard .p2d-subtitle{color:var(--muted);margin:4px 0 18px}.p2d-overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}.p2d-stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:11px 12px;min-width:0}.p2d-stat span,.p2d-final span{display:block;color:var(--muted);font-size:.82rem}.p2d-stat strong,.p2d-final strong{display:block;margin-top:4px;font-weight:500;overflow-wrap:anywhere}.p2d-chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.p2d-chart{min-width:0}.p2d-chart svg{display:block;width:100%;height:auto}.p2d-chart svg text{fill:var(--muted);font-size:11px}.p2d-frame{fill:var(--surface);stroke:var(--border)}.p2d-grid{stroke:var(--grid);stroke-width:1}.p2d-zero{stroke:var(--muted);stroke-width:1}.p2d-line{fill:none;stroke-width:2}.p2d-c1{stroke:var(--c1);color:var(--c1)}.p2d-c2{stroke:var(--c2);color:var(--c2)}.p2d-c3{stroke:var(--c3);color:var(--c3)}.p2d-c4{stroke:var(--c4);color:var(--c4)}.p2d-c5{stroke:var(--c5);color:var(--c5)}.p2d-marker{stroke-width:1.25;stroke-dasharray:5 4}.p2d-marker-label{fill:var(--text)!important}.p2d-legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.82rem;margin-bottom:3px}.p2d-legend i{display:inline-block;width:15px;border-top:3px solid currentColor;margin-right:5px;vertical-align:middle}.p2d-table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:var(--surface)}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{background:var(--soft);font-weight:500}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}.p2d-selected{background:var(--soft);font-weight:500}.p2d-ok{color:var(--ok)}.p2d-missing{color:var(--bad)}code{color:inherit}.p2d-final{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.p2d-final>div{border-left:3px solid var(--accent);padding:7px 11px;background:var(--surface)}.p2d-readonly,.p2d-empty{color:var(--muted)}.p2d-error-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.p2d-confusion td{background:color-mix(in srgb,var(--accent) calc(var(--heat)*100%),var(--surface));text-align:center}.p2d-warnings{margin-top:18px;color:var(--muted)}.p2d-warnings summary{cursor:pointer}.p2d-warnings li{margin:4px 0}@media(max-width:760px){#practice2-training-dashboard{padding:14px}.p2d-chart-grid,.p2d-error-grid{grid-template-columns:1fr}}
</style>"""
    lr_series = [(name, name.replace("_", " ").title(), f"p2d-c{index % 5 + 1}") for index, name in enumerate(lr_names)]
    return (
        '<div id="practice2-training-dashboard">' + css
        + '<h1>Practice 2 — Training Dashboard</h1>'
        + '<p class="p2d-subtitle">Saved-artifact view of the selected CIFAR-10 transfer-learning run.</p>'
        + _overview(data)
        + '<h2>Training curves</h2><div class="p2d-chart-grid">'
        + _series_svg("Loss", normalized_rows, (("train_loss", "Train loss", "p2d-c1"), ("val_loss", "Validation loss", "p2d-c2")), "Loss", markers)
        + _series_svg("Accuracy", normalized_rows, (("train_accuracy", "Train accuracy", "p2d-c1"), ("val_accuracy", "Validation accuracy", "p2d-c2")), "Accuracy (%)", markers)
        + _series_svg("Learning rate", normalized_rows, lr_series, "Learning rate", log_scale=True)
        + _series_svg("Generalization", normalized_rows, (("train_accuracy", "Train accuracy", "p2d-c1"), ("val_accuracy", "Validation accuracy", "p2d-c2"), ("generalization_gap", "Generalization gap", "p2d-c5")), "Percentage points", zero_line=True)
        + '</div><h2>Hyperparameter search</h2>' + _ranking_table(data)
        + '<h2>Checkpoint status</h2>' + _checkpoint_table(data)
        + '<h2>Saved Final Evaluation</h2>' + _final_evaluation(data)
        + '<h2>Error analysis</h2><h3>Saved confusion matrix</h3>' + _confusion_table(data.get("error_analysis", {}).get("confusion_matrix") or [])
        + _error_analysis(data) + warning_html + '</div>'
    )


def render_training_dashboard(
    visualization_data_path: str | Path,
    save_path: Optional[str | Path] = None,
) -> TrainingDashboard:
    """Render only the supplied aggregate JSON; never inspect run folders."""
    data = load_training_dashboard_data(visualization_data_path)
    dashboard_html = build_training_dashboard_html(data)
    destination = Path(save_path).expanduser().resolve() if save_path else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(dashboard_html, encoding="utf-8")
        temporary.replace(destination)
    return TrainingDashboard(dashboard_html, destination)
