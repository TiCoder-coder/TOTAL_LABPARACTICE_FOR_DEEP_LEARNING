"""Four-section presentation for Phase 43–59; no model or metric computation."""
from functools import wraps
from html import escape, unescape
from pathlib import Path
import json
import re

from IPython.display import HTML
from course_work.utils.artifacts import get_project_root

SECTIONS = {
    43: ('Tuning conditions', 'Tuning results & winner', 'lstm_tuning'),
    44: ('Rolling-origin conditions', 'Model comparison & recommendation', 'rolling_origin'),
    45: ('Final model configuration', 'Locked model & refit policy', 'final_model_lock'),
    46: ('Three-seed run conditions', 'Training results', 'three_seed_final_runs'),
    47: ('Test evaluation conditions', 'Final Test results', 'final_test'),
    48: ('Prediction analysis conditions', 'Actual vs predicted & behavior metrics', 'prediction_analysis'),
    49: ('Residual analysis conditions', 'Residual distribution & statistics', 'residual_analysis'),
    50: ('Regime definitions', 'Error-by-regime results', 'error_by_regime'),
    51: ('Worst-case selection conditions', 'Worst-case predictions & errors', 'worst_error_analysis'),
    52: ('Attention extraction overview', 'Extracted attention summary', 'attention_extraction'),
    53: ('Heatmap configuration', 'Attention heatmaps & summary', 'attention_heatmaps'),
    54: ('Last-query analysis conditions', 'Last-query attention results', 'last_query_attention'),
    55: ('Head comparison conditions', 'Attention head comparison', 'head_comparison'),
    56: ('Error-cohort definitions', 'Error-conditioned attention results', 'error_conditioned_attention'),
    57: ('Cross-seed comparison conditions', 'Attention stability results', 'seed_stability_attention'),
    58: ('Final reporting scope', 'Consolidated results', 'final_tables'),
    59: ('Conclusion scope', 'Final findings', 'final_conclusions'),
}

CSS = '''<style>
.cw-report{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;border:1px solid #d9e2ef;border-radius:14px;background:#fff;margin:14px 0 22px;overflow:visible}
.cw-report *{box-sizing:border-box}
.cw-report-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 22px;background:linear-gradient(135deg,#eef4ff,#f7f4ff);border-bottom:1px solid #d9e2ef;border-radius:14px 14px 0 0}
.cw-report-header h3{font-size:20px;margin:0;line-height:1.3}
.cw-report-status{font-size:11px;font-weight:700;padding:6px 12px;border:1px solid #f1d889;border-radius:999px;background:#fff7e0;color:#7a5613;white-space:nowrap}
.cw-report-status.pass{background:#e8f7ef;border-color:#a9dec1;color:#11613d}
.cw-report-status.fail{background:#fcecef;border-color:#e2b2b8;color:#8b2430}
.cw-report-body{padding:4px 22px 22px}
.cw-report-section{margin-top:18px}
.cw-report-section h4{font-size:14px;color:#334155;margin:0 0 8px}
.cw-report-section h5{font-size:12px;color:#475569;margin:12px 0 8px}
.cw-report-table-wrap{overflow-x:auto;border:1px solid #e2e8f0;border-radius:9px;margin:8px 0}
.cw-report table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
.cw-report th,.cw-report td{text-align:left;padding:8px 11px;border-bottom:1px solid #edf1f5;vertical-align:top;line-height:1.45;overflow-wrap:anywhere}
.cw-report th{background:#f5f7fb;color:#475569;font-weight:650}
.cw-report tbody tr:nth-child(even){background:#fafbfd}
.cw-report tr:last-child td{border-bottom:0}
.cw-report-figure{margin:12px auto 18px;padding:12px 14px;border:1px solid #e2e8f0;border-radius:11px;background:#fff;width:100%;max-width:1000px}
.cw-report-figure.wide{max-width:1200px}
.cw-report-figure img{display:block;width:100%;max-width:960px;height:auto;margin:auto}
.cw-report-figure.wide img{max-width:1200px}
@media(max-width:620px){.cw-report-header{padding:14px;align-items:flex-start}.cw-report-body{padding:4px 12px 16px}.cw-report-header h3{font-size:17px}}
</style>'''


def _text(value):
    return unescape(re.sub(r'<[^>]*>', '', value)).strip()


def _pairs(table):
    rows = []
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table, re.S):
        cells = re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S)
        if len(cells) == 2:
            rows.append(tuple(_text(c) for c in cells))
    return rows


def _table(rows):
    rows = rows or [('Data', 'N/A')]
    return ('<div class="cw-report-table-wrap"><table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _section(title, content):
    return f'<section class="cw-report-section"><h4>{escape(title)}</h4>{content}</section>'


def _condition(label):
    return bool(re.search(
        r'seed|protocol|folds|lookback|hidden|num layers|dropout|learning rate|weight decay|'
        r'feature|scaling|epochs|early stopping|warm start|validation stopping|fingerprint|sha|'
        r'residual|semantics|analysis type|prediction source|ranking|set status|layers|heads|'
        r'sequence|extraction|model state|attention convention|history|last-query|comparison|'
        r'cohort|conditioning|head matching|^analysis$|threshold|d_model|ffn|optimizer|loss|'
        r'active revision|narrative revision|tuning objective', label, re.I))


def apply_layout(html, phase, root):
    """Reassemble the existing renderer's tables and image tags without recomputing results."""
    condition_title, result_title, folder = SECTIONS[phase]
    path = Path(root) / 'artifacts' / folder / f'phase_{phase}_signoff.json'
    if phase == 51:
        path = path.with_name('phase51_signoff.json')
    signoff = json.loads(path.read_text()) if path.exists() else {}
    status = str(signoff.get('overall_status') or signoff.get(f'phase{phase}_status')
                 or signoff.get('status') or 'UNKNOWN').upper()
    title_match = re.search(r'<h3\b[^>]*>(.*?)</h3>', html, re.S)
    title = _text(title_match[1]) if title_match else f'Phase {phase}'
    overview, conditions, decisions, result_pairs, result_tables = [], [], [], [], []
    headings = [(m.start(), _text(m[1])) for m in re.finditer(r'<h[45]\b[^>]*>(.*?)</h[45]>', html, re.S)]
    for match in re.finditer(r'<table\b[^>]*>.*?</table>', html, re.S):
        heading = next((h for pos, h in reversed(headings) if pos < match.start()), '')
        table = match[0]
        is_overview = 'overview' in heading.lower() or (phase == 58 and heading == 'Final model configuration')
        is_signoff = bool(re.search(r'signoff|final status|project status', heading, re.I))
        is_condition_section = heading == condition_title
        if is_overview or is_signoff:
            for label, value in _pairs(table):
                pair = (label, value)
                if re.search(r'phase.*status|phase 59 scientific conclusion', label, re.I):
                    continue 
                if is_signoff:
                    if label in ('Locked candidate', 'Test population N') and phase == 47:
                        overview.append(pair)
                    elif _condition(label) and label in ('Active revision', 'Final narrative revision'):
                        conditions.append(pair)
                    else:
                        decisions.append(pair)
                elif _condition(label):
                    conditions.append(pair)
                elif re.search(r'rmse|winner|recommended|completed runs|visual QA', label, re.I):
                    result_pairs.append(pair)
                else:
                    overview.append(pair)
        elif is_condition_section:
            for label, value in _pairs(table):
                if _condition(label):
                    conditions.append((label, value))
                else:
                    result_pairs.append((label, value))
        else:
            table = table.replace('cw-d', 'cw-report').replace('cw-p48', 'cw-report')
            if not re.search(r'<td\b', table):
                count = max(1, len(re.findall(r'<th\b', table)))
                table = table.replace('</tbody>', f'<tr><td colspan="{count}">N/A — source rows unavailable</td></tr></tbody>')
            result_tables.append(table)

    overview.insert(0, ('Phase', title.split(' - ', 1)[-1]))
    if phase == 47:
        seeds = sorted({int(m[1]) for key in signoff
                        if (m := re.match(r'seed(\d+)_', key))})
        if seeds:
            conditions.append(('Seeds', ', '.join(map(str, seeds))))
        overview.append(('Dataset', 'Held-out Test'))
    if phase in (58, 59):
        if signoff.get('seed_list'):
            conditions.append(('Seeds', ', '.join(map(str, signoff['seed_list']))))
    if not conditions:
        conditions = [('Artifact version', signoff.get('version') or signoff.get('artifact_version') or 'N/A')]
    if phase == 48:
        for key, label in [('source_bundles_verified', 'Source bundles verified'),
                           ('aligned_target_population', 'Aligned target population'),
                           ('ready_for_phase49', 'Ready for Phase 49')]:
            decisions.append((label, signoff.get(key, 'N/A')))
    if phase in (47, 58, 59) and signoff.get('source_phase46_version' if phase == 47 else 'source_phase58_version'):
        key = 'source_phase46_version' if phase == 47 else 'source_phase58_version'
        conditions.append(('Source version', signoff[key]))

    seen = set()
    def unique(rows):
        output = []
        for label, value in rows:
            key = (str(label), str(value))
            if key not in seen:
                seen.add(key)
                output.append((label, value))
        return output
    result_pairs = unique(result_pairs)
    overview, conditions, decisions = unique(overview), unique(conditions), unique(decisions)
    figures = []
    for image in re.findall(r'<img\b[^>]*>', html):
        image = re.sub(r'\sstyle="[^"]*"', '', image)
        wide = ' wide' if phase == 48 else ''
        alt = re.search(r'\balt="([^"]*)"', image)
        label = f'<h5>{escape(unescape(alt[1]))}</h5>' if alt else ''
        figures.append(f'<div class="cw-report-figure{wide}">{label}{image}</div>')
    results = ''.join(figures)
    if result_pairs:
        results += _table(result_pairs)
    results += ''.join(f'<div class="cw-report-table-wrap">{table}</div>' for table in result_tables)
    if not results:
        results = _table([('Results', 'N/A')])
    for note in re.findall(r'<div class="cw-d-note"[^>]*>(.*?)</div>', html, re.S):
        if 'not available' in _text(note):
            results += _table([('Source table', _text(note).replace(' not available.', ': N/A'))])
    decisions.insert(0, ('Phase status', status))
    color = 'pass' if status == 'PASS' else 'fail' if status == 'FAIL' else 'unknown'
    header = (f'<header class="cw-report-header"><h3>{escape(title)}</h3>'
              f'<span class="cw-report-status {color}">{escape(status)}</span></header>')
    body = (_section('Overview', _table(overview))
            + _section(condition_title, _table(conditions))
            + _section(result_title, results)
            + _section('Signoff', _table(decisions)))
    return CSS + f'<article class="cw-report" data-phase="{phase}">{header}<div class="cw-report-body">{body}</div></article>'


def phase_report(phase):
    def decorate(renderer):
        @wraps(renderer)
        def render(*args, **kwargs):
            root = args[0] if args else kwargs.get('project_root')
            root = Path(root or get_project_root())
            return HTML(apply_layout(renderer(*args, **kwargs).data, phase, root))
        return render
    return decorate
