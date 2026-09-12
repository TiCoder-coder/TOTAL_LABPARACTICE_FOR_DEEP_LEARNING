"""Read-only notebook renderer for the closed Model Improvement V2 track."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from IPython.display import HTML

__all__ = ["render_model_improvement_v2_closure_dashboard"]


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def render_model_improvement_v2_closure_dashboard(project_root: Path | str) -> HTML:
    root = Path(project_root).resolve()
    path = root / "artifacts/model_improvement_v2/model_improvement_v2_final_closure.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 closure artifact: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("model_improvement_v2_status") != "COMPLETE":
        raise RuntimeError("Model Improvement V2 closure is not COMPLETE")
    dev = document["final_development_metrics"]
    bench = document["post_hoc_benchmark"]
    v2 = bench["ensemble_metrics"]
    v1 = bench["v1_historical_transformer_mean"]
    persistence = bench["persistence_metrics"]
    policy = document["final_policy"]
    debts = "".join(f"<li>{escape(item)}</li>" for item in document["reporting_debt"])
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{_fmt(values['rmse_wh'])}</td>"
        f"<td>{_fmt(values['mae_wh'])}</td><td>{_fmt(values['r2'])}</td></tr>"
        for name, values in (
            ("V2 equal-weight ensemble", v2),
            ("V1 historical Transformer mean", v1),
            ("Persistence last value", persistence),
        )
    )
    html = f"""
    <div style="font-family:system-ui;border:1px solid #dbe4f0;border-radius:12px;padding:20px;background:#fbfdff">
      <div style="display:flex;justify-content:space-between;gap:16px">
        <div><h3 style="margin:0">Model Improvement V2 — Final Closure</h3>
        <p><code>{escape(policy['policy_id'])}</code>; seeds {policy['seeds']}; equal weights.</p></div>
        <strong style="color:#11613d">COMPLETE</strong>
      </div>
      <p><b>Development:</b> RMSE {_fmt(dev['pooled_rmse_wh'])} Wh · MAE {_fmt(dev['pooled_mae_wh'])} Wh · R² {_fmt(dev['pooled_r2'])}</p>
      <p style="color:#8a5a00"><b>{escape(bench['label'])}</b> — old Test, N={bench['sample_count']}; not an unbiased unseen-Test result.</p>
      <table style="border-collapse:collapse;width:100%"><thead><tr><th style="text-align:left">Policy</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{rows}</tbody></table>
      <p><b>Provenance:</b> {escape(document['provenance']['status'])} · post-Test retuning: FALSE · best-seed selection: FALSE.</p>
      <details><summary>Remaining non-blocking reporting debt</summary><ul>{debts}</ul></details>
    </div>
    """
    return HTML(html)
