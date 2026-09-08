"""Presentation regression checks; no training or inference."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from course_work.reporting._phase_report_layout import apply_layout


class PhaseReportLayoutTests(unittest.TestCase):
    def test_image_and_result_values_survive_reordering(self):
        html = '''<h3>Phase 49 - Residual Analysis</h3>
        <h4>Overview</h4><table><tbody>
        <tr><td>Test N</td><td>2961</td></tr>
        <tr><td>Seeds</td><td>42 / 123 / 2026</td></tr>
        </tbody></table><h4>Main result</h4>
        <img src="data:image/png;base64,QUJD" alt="Residual distribution" style="width:80%"/>
        <table><thead><tr><th>Mean</th></tr></thead><tbody><tr><td>-5.559</td></tr></tbody></table>
        <h4>Decision &amp; signoff</h4><table><tbody>
        <tr><td>Ready for Phase50</td><td>False</td></tr></tbody></table>'''
        with tempfile.TemporaryDirectory() as root:
            output = apply_layout(html, 49, root)
        headings = re.findall(r'<h4>(.*?)</h4>', output)
        self.assertEqual(headings, ['Overview', 'Residual analysis conditions',
                                   'Residual distribution &amp; statistics', 'Signoff'])
        self.assertIn('data:image/png;base64,QUJD', output)
        self.assertIn('<td>-5.559</td>', output)
        self.assertIn('<td>False</td>', output)
        self.assertIn('>UNKNOWN</span>', output)
        self.assertNotIn('>PASS</span>', output)
        self.assertLess(output.index('Residual analysis conditions'), output.index('<img'))

    def test_source_status_overrides_old_display_and_missing_rows_are_explicit(self):
        html = '''<h3>Phase 51 - Worst-Error Analysis</h3>
        <h4>Main result</h4><table><thead><tr><th>Seed</th><th>RMSE</th></tr></thead><tbody></tbody></table>
        <h4>Signoff</h4><table><tbody><tr><td>Phase status</td><td>PASS</td></tr></tbody></table>'''
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'artifacts/worst_error_analysis/phase51_signoff.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'phase51_status': 'FAIL'}))
            output = apply_layout(html, 51, root)
        self.assertIn('>FAIL</span>', output)
        self.assertNotIn('>PASS<', output)
        self.assertIn('colspan="2">N/A — source rows unavailable', output)

    def test_svg_chart_and_wide_phase48_figures_are_preserved(self):
        html = '''<h3>Phase 48 - Prediction Analysis</h3>
        <img src="data:image/svg+xml;base64,QUJD" alt="Actual &amp; predicted"/>
        <img src="data:image/png;base64,REVG" alt="Last 24 hours"/>'''
        with tempfile.TemporaryDirectory() as root:
            output = apply_layout(html, 48, root)
        self.assertEqual(output.count('<img '), 2)
        self.assertIn('data:image/svg+xml;base64,QUJD', output)
        self.assertEqual(output.count('class="cw-report-figure wide"'), 2)
        self.assertNotIn('cw-d', output)


if __name__ == '__main__':
    unittest.main()
