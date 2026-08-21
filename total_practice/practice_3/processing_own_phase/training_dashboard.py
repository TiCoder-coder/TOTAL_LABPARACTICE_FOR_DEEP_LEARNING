"""Self-contained, artifact-only Practice 3 HTML dashboard."""
from __future__ import annotations

import html, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class TrainingDashboard:
    html: str
    saved_path: Path | None = None
    def _repr_html_(self) -> str: return self.html
    def __str__(self) -> str: return self.html

def load_training_dashboard_data(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file(): raise FileNotFoundError(f"Visualization data does not exist: {source}")
    try: data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise ValueError(f"Invalid visualization data JSON: {source}") from exc
    required = {"experiments", "training_history", "experiment_comparison", "winner", "final_holdout", "error_analysis", "custom_inference", "save_reload_verification", "sources"}
    if not isinstance(data, dict) or required - data.keys(): raise ValueError(f"Visualization data missing sections: {sorted(required - data.keys())}")
    return data

def _e(value: Any) -> str: return html.escape(str("—" if value in (None, "") else value))
def _n(value: Any, digits: int = 4) -> str:
    try: number = float(value)
    except (TypeError, ValueError): return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"
def _pct(value: Any) -> str:
    try: return f"{float(value)*100:.2f}%"
    except (TypeError, ValueError): return "—"
def _duration(value: Any) -> str:
    try: seconds = int(round(float(value)))
    except (TypeError, ValueError): return "—"
    return f"{seconds//60}m {seconds%60:02d}s"
def _pending(section: dict[str, Any]) -> str: return '<p class="p3-empty">Artifact not generated yet</p>' if not section.get("available") else ""

def _curve(data: dict[str, Any], key: str, title: str) -> str:
    colors = ["#2563eb", "#0891b2", "#7c3aed", "#dc2626", "#ea580c", "#16a34a"]
    series = []
    for index, run in enumerate(data["experiments"]):
        points = [(r.get("epoch"), r.get(key)) for r in run.get("history", []) if isinstance(r.get("epoch"), (int, float)) and isinstance(r.get(key), (int, float))]
        if points: series.append((run["short_id"], points, colors[index]))
    if not series: return f'<section><h3>{_e(title)}</h3><p class="p3-empty">Artifact not generated yet</p></section>'
    xs = [x for _, pts, _ in series for x, _ in pts]; ys = [y for _, pts, _ in series for _, y in pts]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys); xmax = xmax if xmax != xmin else xmin + 1; pad = (ymax-ymin)*.08 or .01; ymin -= pad; ymax += pad
    sx=lambda x: 54+(x-xmin)/(xmax-xmin)*596; sy=lambda y: 18+(ymax-y)/(ymax-ymin)*212
    grid=''.join(f'<line x1="54" y1="{18+i*53}" x2="650" y2="{18+i*53}" class="grid"/><text x="48" y="{22+i*53}" text-anchor="end">{ymax-i*(ymax-ymin)/4:.3f}</text>' for i in range(5))
    paths=''.join(f'<polyline points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2.3"/>' for _,pts,color in series)
    legend=''.join(f'<span><i style="background:{color}"></i>{name}</span>' for name,_,color in series)
    return f'<section class="p3-chart"><h3>{_e(title)}</h3><div class="legend">{legend}</div><svg viewBox="0 0 680 270" role="img" aria-label="{_e(title)}"><rect x="54" y="18" width="596" height="212" class="frame"/>{grid}{paths}<text x="352" y="260" text-anchor="middle">Epoch</text></svg></section>'

def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return '<div class="p3-table"><table><thead><tr>'+''.join(f'<th>{_e(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def _render(data: dict[str, Any]) -> str:
    experiments, comparison, winner = data["experiments"], data["experiment_comparison"], data["winner"]
    overview = _table(["Run", "Stage", "Status", "Changed variable", "Value", "Best epoch", "Val loss", "Val accuracy", "Val F1"], [[_e(r["run_id"]),_e(r["stage"]),_e(r["status"]),_e(r["experiment_type"]),_e(r["controlled_value"]),_e(r["best_epoch"]),_n(r["validation_loss"],6),_pct(r["validation_accuracy"]),_pct(r["validation_f1"])] for r in experiments])
    hyper = _table(["Run", "Learning rate", "Weight decay", "Dropout", "Fine-tuning", "Runtime", "Device"], [[_e(r["short_id"]),_e(r["learning_rate"]),_e(r["weight_decay"]),_e(r["dropout"]),_e(r["fine_tuning_strategy"]),_duration(r["runtime_seconds"]),_e(r["device"])] for r in experiments])
    ranking = _table(["Rank", "Run", "Validation loss", "Accuracy", "Precision", "Recall", "F1"], [[_e(r["rank"]),_e(r["run_id"]),_n(r["validation_loss"],6),_pct(r["validation_accuracy"]),_pct(r["validation_precision"]),_pct(r["validation_recall"]),_pct(r["validation_f1"])] for r in comparison.get("ranking", [])])
    holdout = data["final_holdout"]; hv = holdout.get("values") or {}
    holdout_html = _pending(holdout) or '<div class="stats">'+''.join(f'<div><span>{label}</span><strong>{value}</strong></div>' for label,value in [("Loss",_n(hv.get("holdout_loss"),6)),("Accuracy",_pct(hv.get("holdout_accuracy"))),("Precision",_pct(hv.get("holdout_precision"))),("Recall",_pct(hv.get("holdout_recall"))),("F1",_pct(hv.get("holdout_f1")))])+'</div>'
    error = data["error_analysis"]; cm = error.get("confusion_matrix") or {}
    error_html = _pending(error) or f'<div class="cm"><span>True ↓ / Predicted →</span><b>Negative</b><b>Positive</b><b>Negative</b><strong>{_e(cm.get("TN"))}</strong><strong>{_e(cm.get("FP"))}</strong><b>Positive</b><strong>{_e(cm.get("FN"))}</strong><strong>{_e(cm.get("TP"))}</strong></div><p>{_e(error.get("error_count"))} misclassifications; {_e(error.get("high_confidence_error_count"))} high-confidence errors.</p>'
    custom = data["custom_inference"]; records = custom.get("records") or []
    custom_html = _pending(custom) or _table(["Sample", "Category", "Prediction", "Confidence", "Expected match"], [[_e(r.get("sample_id")),_e(r.get("intended_category")),_e(r.get("predicted_label_name",r.get("predicted_label"))),_pct(r.get("confidence")),_e(r.get("match_expected"))] for r in records])
    save = data["save_reload_verification"]; sv=save.get("values") or {}
    save_html = _pending(save) or _table(["Status", "Parameter mismatches", "Prediction matches", "Max probability difference"], [[_e(sv.get("verification_status")),_e(sv.get("parameter_mismatch_count")),f'{_e(sv.get("prediction_match_count"))} / {_e(sv.get("sentences_compared"))}',_e(sv.get("max_probability_difference"))]])
    return f'''<div id="p3-dashboard"><style>
#p3-dashboard{{font-family:Inter,system-ui,sans-serif;color:#172033;background:#f8fafc;padding:22px;border:1px solid #dbe3ef;border-radius:14px}}#p3-dashboard h1{{margin:0 0 4px;font-size:28px}}#p3-dashboard h2{{margin:28px 0 10px;font-size:19px;border-bottom:2px solid #2563eb;padding-bottom:6px}}#p3-dashboard h3{{font-size:15px;margin:0 0 7px}}#p3-dashboard .subtitle{{color:#64748b;margin:0 0 18px}}.p3-table{{overflow:auto}}#p3-dashboard table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}#p3-dashboard th,#p3-dashboard td{{padding:8px;border:1px solid #dbe3ef;text-align:left;white-space:nowrap}}#p3-dashboard th{{background:#e8eef8}}#p3-dashboard .stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}#p3-dashboard .stats div{{background:white;border:1px solid #dbe3ef;padding:12px}}#p3-dashboard .stats span{{display:block;color:#64748b;font-size:11px}}#p3-dashboard .stats strong{{font-size:18px}}#p3-dashboard .charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.p3-chart{{background:white;border:1px solid #dbe3ef;padding:10px}}.p3-chart svg{{width:100%;height:auto}}.p3-chart text{{font-size:10px;fill:#475569}}.frame{{fill:none;stroke:#94a3b8}}.grid{{stroke:#e2e8f0}}.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:11px}}.legend i{{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}}.cm{{display:grid;grid-template-columns:180px repeat(2,110px);max-width:420px;text-align:center}}.cm>*{{padding:10px;border:1px solid #cbd5e1}}.cm strong{{background:#dbeafe;font-size:18px}}.p3-empty{{padding:16px;background:#fff7ed;border-left:4px solid #f97316}}@media(max-width:760px){{#p3-dashboard .charts,#p3-dashboard .stats{{grid-template-columns:1fr}}}}
</style><h1>Practice 3 — Training & Evaluation Dashboard</h1><p class="subtitle">Saved artifacts only · protocol {_e(data.get("protocol_version"))} · generated {_e(data.get("generated_at"))}</p>
<h2>1. Experiment Overview E1–E6</h2>{overview}<h2>2. Hyperparameter & Runtime Comparison</h2>{hyper}<h2>3. Validation Ranking</h2>{ranking}
<h2>4. Selected Winner</h2><div class="stats"><div><span>Run</span><strong>{_e(winner.get("run_id"))}</strong></div><div><span>Best epoch</span><strong>{_e(winner.get("best_epoch"))}</strong></div><div><span>Validation loss</span><strong>{_n(winner.get("validation_loss"),6)}</strong></div><div><span>Validation accuracy</span><strong>{_pct(winner.get("validation_accuracy"))}</strong></div><div><span>Validation F1</span><strong>{_pct(winner.get("validation_f1"))}</strong></div></div>
<h2>5–9. Saved Training Curves</h2><div class="charts">{_curve(data,"train_loss","Training Loss")}{_curve(data,"validation_loss","Validation Loss")}{_curve(data,"validation_accuracy","Validation Accuracy")}{_curve(data,"validation_f1","Validation F1")}</div>
<h2>10. Final Holdout Metrics</h2>{holdout_html}<h2>11. Confusion Matrix & Error Analysis</h2>{error_html}<h2>12. Custom Inference</h2>{custom_html}<h2>13. Save / Reload Verification</h2>{save_html}</div>'''

def render_training_dashboard(visualization_data_path: str | Path, save_path: str | Path | None = None) -> TrainingDashboard:
    """Render only the aggregate JSON; never search runs or load a model."""
    rendered = _render(load_training_dashboard_data(visualization_data_path))
    destination = Path(save_path).expanduser().resolve() if save_path else None
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Practice 3 Dashboard</title></head><body>'+rendered+'</body></html>', encoding="utf-8")
    return TrainingDashboard(rendered, destination)
