import nbformat
nb = nbformat.read('total_practice/practice_2_2/notebooks/04_canonical_report.ipynb', as_version=4)

def replace_cell(idx, old, new):
    if old in nb.cells[idx].source:
        nb.cells[idx].source = nb.cells[idx].source.replace(old, new)

# Cell 3: import visualization_utils
if 'vu.inject_theme_css()' not in nb.cells[3].source:
    nb.cells[3].source += '\nvu.inject_theme_css()'

# Update Cell 14 (transform table) to use style_transform_table
replace_cell(14, 'vu.style_generic_table(transform_audit)', 'vu.style_transform_table(transform_audit)')

# Update Cell 31 (test metrics table) to use style_test_metrics_table
replace_cell(31, 'vu.style_generic_table(final_table)', 'vu.style_test_metrics_table(final_table)')

nbformat.write(nb, 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb')
