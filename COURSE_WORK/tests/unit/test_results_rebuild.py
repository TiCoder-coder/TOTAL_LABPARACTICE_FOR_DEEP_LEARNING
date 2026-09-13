import json
import hashlib
from pathlib import Path

from course_work.reporting.results_rebuild import (
    render_verified_phase_result,
    render_verified_v2_results,
)


ROOT = Path(__file__).resolve().parents[2]


def test_every_phase_43_59_renders_nonempty_table_without_evidence_metadata():
    for phase_id in range(43, 60):
        html = render_verified_phase_result(phase_id, ROOT).data
        assert html.count("<table") == 3
        assert "Phase status" in html
        assert "Configuration / analysis" in html
        assert "Results / decision" in html
        assert "Evidence:" not in html
        assert "artifacts/" not in html
        assert "<script" not in html.lower()
        assert "raw json" not in html.lower()


def test_phase47_uses_current_verified_per_seed_and_baseline_metrics():
    html = render_verified_phase_result(47, ROOT).data
    for value in ("61.262482", "63.259959", "62.423633", "61.608937", "66.836915"):
        assert value in html
    for value in ("24.187279", "23.147923", "20.222825", "21.519219", "21.513353"):
        assert value in html
    assert "V2 Seed 42" in html
    assert "V2 Equal-weight Ensemble" in html
    assert "V1 historical Transformer" not in html
    assert "MAPE %" in html
    assert "Presentation lineage" not in html
    assert "Best seed" not in html


def test_phase47_adds_verified_v2_mape_artifact_when_present(tmp_path):
    destination = tmp_path / "artifacts/model_improvement_v2/post_hoc_v2_benchmark/step17_metrics_with_mape.json"
    destination.parent.mkdir(parents=True)
    model_ids = [
        "V2_FINAL_SEED_42",
        "V2_FINAL_SEED_123",
        "V2_FINAL_SEED_2026",
        "V2_FINAL_EQUAL_WEIGHT_ENSEMBLE",
        "PERSISTENCE_LAST_VALUE",
    ]
    destination.write_text(json.dumps({
        "schema": "MODEL_IMPROVEMENT_V2_STEP17_METRICS_WITH_MAPE-v1",
        "benchmark_label": "POST_HOC_V2_BENCHMARK_MAPE_ADDENDUM",
        "post_test_retuning": False,
        "inference_executed": False,
        "metrics": [
            {"model_id": model_id, "rmse_wh": 1.0, "mae_wh": 0.5, "mape_pct": 12.5, "mape_status": "DEFINED", "r2": 0.9}
            for model_id in model_ids
        ],
    }), encoding="utf-8")
    manifest = destination.with_name("step17_mape_addendum_manifest.json")
    source_manifest = destination.with_name("step17_benchmark_manifest.json")
    source_manifest.write_text("{}", encoding="utf-8")
    source_metrics = destination.with_name("step17_metrics.json")
    source_metrics.write_text("{}", encoding="utf-8")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["source_metrics_sha256"] = hashlib.sha256(source_metrics.read_bytes()).hexdigest()
    destination.write_text(json.dumps(payload), encoding="utf-8")
    manifest.write_text(json.dumps({
        "status": "PASS",
        "metrics_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    signoff = destination.with_name("step17_mape_addendum_signoff.json")
    signoff.write_text(json.dumps({
        "status": "PASS",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    html = render_verified_phase_result(47, tmp_path).data
    assert "V2 Equal-weight Ensemble" in html
    assert "12.500000" in html
    assert "Presentation lineage" not in html


def test_v2_result_chain_is_complete_without_evidence_metadata():
    html = render_verified_v2_results(ROOT).data
    for label in ("E01", "E20", "Step 14A", "Step 14B", "Step 16", "Step 17", "Closure"):
        assert label in html
    assert "61.608937" in html
    assert "COMPLETE" in html
    assert html.count("<table") == 3
    assert "Phase status" in html
    assert "Configuration / analysis" in html
    assert "Results / decision" in html
    assert "Evidence:" not in html
    assert "artifacts/" not in html


def test_notebook_phase_1_42_source_matches_locked_rebuild_hash():
    notebook = json.loads((ROOT / "notebook_course_work/CourseWork.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    phase43_index = next(i for i, source in enumerate(sources) if source.startswith("## Phase 43"))
    assert phase43_index == 117
    # Output and execution metadata may be refreshed read-only.  Lock the
    # scientific narrative and executable source, not transient UI metadata.
    prefix = json.dumps(
        [
            {
                "cell_type": cell.get("cell_type"),
                "id": cell.get("id"),
                "source": cell.get("source", []),
            }
            for cell in notebook["cells"][:phase43_index]
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(prefix).hexdigest() == (
        "f3d57cb71a38d40d73362a87930314ea75dbdb2ed9411024c72386a0adc0fad9"
    )
