import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from course_work.reporting.phase_summary import (
    LOG_FILENAMES,
    PRESENTATION_SPECS,
    PRESENTATION_VERSION,
    build_phase_33_transformer_configuration,
    build_phase_processing_log,
    render_dataframe_table,
    render_phase_33_transformer_configuration,
    render_phase_log,
)


class PhaseSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]

    def test_every_phase_builds_a_versioned_processing_log(self) -> None:
        for phase_id in range(15):
            log = build_phase_processing_log(phase_id, self.root)
            self.assertEqual(log["presentation_version"], PRESENTATION_VERSION)
            self.assertEqual(log["phase_id"], phase_id)
            self.assertIn(log["status"], {"PASS", "PASS_WITH_WARNING"})
            self.assertTrue(log["summary"])
            self.assertTrue(log["sections"])
            self.assertTrue(log["source_artifacts"])
            self.assertIn("input_checksums", log["technical_details"])
            self.assertIn("output_checksums", log["technical_details"])

    def test_every_phase_renders_html_without_rewriting_processing_logs(self) -> None:
        for phase_id in range(15):
            log = build_phase_processing_log(phase_id, self.root)
            rendered = render_phase_log(log)
            self.assertIn(f"Phase {phase_id} -", rendered.data)
            if phase_id == 6:
                self.assertNotIn("<table", rendered.data)
            else:
                self.assertIn("<table", rendered.data)
            path = self.root / "docs/save_log_in_processing" / LOG_FILENAMES[phase_id]
            self.assertTrue(path.is_file())

    def test_rendering_escapes_untrusted_values(self) -> None:
        log = build_phase_processing_log(0, self.root)
        log["phase_name"] = '<script>alert("unsafe")</script>'
        rendered = render_phase_log(log).data
        self.assertNotIn('<script>alert("unsafe")</script>', rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_phase_5_renders_chronological_split_outputs(self) -> None:
        log = build_phase_processing_log(5, self.root)
        rendered = render_phase_log(log).data
        fingerprint = log["technical_details"]["global_split_fingerprint"]
        self.assertNotIn(fingerprint, rendered)
        self.assertNotIn("Phase summary", rendered)
        self.assertIn("Chronological membership", rendered)
        self.assertIn("Split allocation", rendered)

    def test_phase_11_renders_policy_booleans_as_yes_or_no(self) -> None:
        log = build_phase_processing_log(11, self.root)
        rendered = render_phase_log(log).data
        self.assertIn("<td>Yes</td>", rendered)
        self.assertIn("<td>No</td>", rendered)
        self.assertNotIn("<td>FAIL</td>", rendered)

    def test_minimal_presentation_policy_is_complete(self) -> None:
        self.assertEqual(set(PRESENTATION_SPECS), set(range(47)))
        forbidden = (
            "Warnings and discrepancies",
            "Technical details",
            "Source artifacts",
            "Technical lineage",
            "Training smoke test",
        )
        expected_titles = {
            1: ("Environment overview", "Core package versions"),
            2: ("Dataset overview",),
            3: ("Schema overview", "Critical schema checks"),
            4: ("Temporal coverage", "Critical temporal checks"),
            5: ("Chronological membership", "Split allocation"),
            7: ("Feature overview", "Engineered feature registry"),
            8: ("Feature-set registry",),
            9: ("X scaler bundles", "Target scaling options"),
            10: ("Window contract", "Common target population"),
            11: ("Dataset population", "Loader policy"),
            12: ("Metric registry", "Evaluation policy"),
            13: ("Registry state", "Core safeguards"),
            14: ("Validation performance", "Baseline contract"),
        }
        for phase_id in range(1, 15):
            log = build_phase_processing_log(phase_id, self.root)
            rendered = render_phase_log(log).data
            self.assertTrue(all(value not in rendered for value in forbidden))
            for title in expected_titles.get(phase_id, ()):
                self.assertIn(title, rendered)
            self.assertIn("input_checksums", log["technical_details"])
            self.assertIn("output_checksums", log["technical_details"])

    def test_phase_8_hides_fingerprints_from_visible_registry(self) -> None:
        log = build_phase_processing_log(8, self.root)
        rendered = render_phase_log(log).data
        self.assertNotIn("Fingerprint", rendered)
        self.assertTrue(log["technical_details"])

    def test_header_hides_creation_timestamp(self) -> None:
        log = build_phase_processing_log(1, self.root)
        rendered = render_phase_log(log).data
        self.assertNotIn(log["created_at"], rendered)

    def test_phase_14_renders_only_decision_facing_baseline_outputs(self) -> None:
        log = build_phase_processing_log(14, self.root)
        rendered = render_phase_log(log).data
        self.assertIn("Validation performance", rendered)
        self.assertIn("Baseline contract", rendered)
        self.assertIn("26.1622", rendered)
        self.assertIn("66.4297", rendered)
        self.assertIn("0.481326", rendered)
        self.assertNotIn(log["technical_details"]["prediction_fingerprint"], rendered)
        self.assertNotIn(log["technical_details"]["experiment_config_fingerprint"], rendered)
        self.assertNotIn("Technical details", rendered)
        self.assertNotIn("Warnings and discrepancies", rendered)
        self.assertEqual(log["technical_details"]["run_status"], "COMPLETED")
        self.assertFalse(log["technical_details"]["test_access_authorized"])

    def test_every_visible_table_has_one_layout_contract(self) -> None:
        for phase_id in range(15):
            rendered = render_phase_log(build_phase_processing_log(phase_id, self.root)).data
            table_count = rendered.count("<table")
            layout_count = rendered.count('data-layout="compact"') + rendered.count('data-layout="wide"')
            self.assertEqual(layout_count, table_count)
            if phase_id == 6:
                self.assertEqual(table_count, 0)

    def test_phase_14_uses_compact_two_and_three_column_layouts(self) -> None:
        rendered = render_phase_log(build_phase_processing_log(14, self.root)).data
        self.assertEqual(rendered.count('class="cw-phase-table cw-phase-table-compact"'), 2)
        self.assertIn('data-column-count="3"', rendered)
        self.assertIn('data-column-count="2"', rendered)
        self.assertNotIn('data-layout="wide"', rendered)
        self.assertIn("table-layout:auto", rendered)
        self.assertIn("text-align:left", rendered)
        self.assertIn("width:1%", rendered)
        self.assertIn("white-space:nowrap", rendered)
        self.assertIn("overflow-wrap:anywhere", rendered)

    def test_phase_5_split_uses_wide_scrollable_layout(self) -> None:
        rendered = render_phase_log(build_phase_processing_log(5, self.root)).data
        self.assertEqual(rendered.count('class="cw-phase-table cw-phase-table-wide"'), 1)
        self.assertIn('data-layout="wide"', rendered)
        self.assertIn('data-column-count="6"', rendered)
        self.assertIn("overflow-x:auto", rendered)
        self.assertIn("width:max-content", rendered)
        self.assertIn("min-width:100%", rendered)

    def test_phase_rendering_does_not_mutate_processing_log(self) -> None:
        log = build_phase_processing_log(14, self.root)
        before = json.dumps(log, ensure_ascii=False, sort_keys=True)
        render_phase_log(log)
        after = json.dumps(log, ensure_ascii=False, sort_keys=True)
        self.assertEqual(after, before)

    def test_invalid_phase_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_phase_processing_log(47, self.root)

    def test_dataframe_renderer_is_safe_scrollable_and_non_mutating(self) -> None:
        frame = pd.DataFrame({"value": [1.23456, 2.34567], "label": ["safe", "<script>unsafe</script>"]})
        before = frame.copy(deep=True)
        rendered = render_dataframe_table(
            frame,
            title="Sample table",
            subtitle="Validated sample rows",
            metrics={"Dataset rows": 19735},
            max_height=320,
        ).data
        pd.testing.assert_frame_equal(frame, before)
        self.assertIn("cw-data-view", rendered)
        self.assertNotIn("cw-phase-table", rendered)
        self.assertIn("overflow:auto", rendered)
        self.assertIn("Dataset rows", rendered)
        self.assertIn("1.2346", rendered)
        self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", rendered)
        self.assertNotIn("<script>unsafe</script>", rendered)

    def test_dataframe_renderer_rejects_invalid_layout_options(self) -> None:
        frame = pd.DataFrame({"value": [1]})
        with self.assertRaises(ValueError):
            render_dataframe_table(frame, "Title", "Subtitle", max_height=0)
        with self.assertRaises(ValueError):
            render_dataframe_table(frame, "Title", "Subtitle", precision=-1)

    def test_phase_33_transformer_configuration_uses_canonical_artifacts(self) -> None:
        view = build_phase_33_transformer_configuration(self.root)
        self.assertEqual(view["status"], "PASS")
        self.assertEqual(view["current_reference_run_id"], "RUN_TR_S09_0016_AE0FB819")
        self.assertEqual(view["validation_rmse_wh"], 58.08190056355405)
        self.assertEqual(view["validation_mae_wh"], 27.595002038670813)
        self.assertEqual(view["validation_r2"], 0.6034923842647237)
        self.assertEqual(len(view["configuration_rows"]), 19)
        rows = {row["Component"]: row for row in view["configuration_rows"]}
        self.assertEqual(rows["Feature set"]["Current value"], "FS2_TF1, 33 features")
        self.assertEqual(rows["Target scaling"]["Current value"], "YS1, Train-only StandardScaler")
        self.assertEqual(rows["Lookback"]["Current value"], "L36, 6 hours")
        self.assertEqual(rows["d_model"]["Current value"], 64)
        self.assertEqual(rows["Boundary protocol"]["Current value"], "WB0_CONTEXT_CARRY_OVER")

    def test_phase_33_transformer_configuration_rejects_artifact_mismatch(self) -> None:
        relative_paths = (
            Path("artifacts/sweeps/S11_d_model/phase_33_signoff.json"),
            Path("artifacts/sweeps/S11_d_model/s11_d_model_winner.json"),
            Path("artifacts/sweeps/S11_d_model/s11_reference_update.json"),
            Path("artifacts/runs/RUN_TR_S09_0016_AE0FB819/config.json"),
            Path("artifacts/feature_sets/feature_set_registry.json"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in relative_paths:
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / relative_path, destination)
            winner_path = root / "artifacts/sweeps/S11_d_model/s11_d_model_winner.json"
            winner = json.loads(winner_path.read_text(encoding="utf-8"))
            winner["winner_rmse_wh"] = 60.0
            winner_path.write_text(json.dumps(winner), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Validation RMSE values disagree"):
                build_phase_33_transformer_configuration(root)

    def test_phase_33_transformer_configuration_renders_static_html(self) -> None:
        rendered = render_phase_33_transformer_configuration(self.root).data
        required_values = (
            "Transformer Configuration after Phase 33",
            "RUN_TR_S09_0016_AE0FB819",
            "58.08190056355405 Wh",
            "27.595002038670813 Wh",
            "0.603492",
            "FS2_TF1, 33 features",
            "YS1, Train-only StandardScaler",
            "L36, 6 hours",
            "Current configuration",
            "Canonical lineage",
        )
        self.assertTrue(all(value in rendered for value in required_values))
        self.assertNotIn("<script", rendered.lower())
        self.assertNotIn("jupyter.widget", rendered.lower())
        self.assertNotIn("checksum", rendered.lower())
        self.assertNotIn("fingerprint", rendered.lower())
        self.assertEqual(rendered.count("<tbody>"), 2)


if __name__ == "__main__":
    unittest.main()
