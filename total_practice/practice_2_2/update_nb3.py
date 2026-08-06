import nbformat
nb = nbformat.read('total_practice/practice_2_2/notebooks/04_canonical_report.ipynb', as_version=4)

def replace_cell(idx, old, new):
    if old in nb.cells[idx].source:
        nb.cells[idx].source = nb.cells[idx].source.replace(old, new)

# Cell 8:
cell_8_code = '''
model_use = manifest.loc[manifest["use_for_model"]]
split_counts = model_use["split"].value_counts().reindex(["Train", "Validation", "Test"])
display(vu.render_summary_cards([
    ("Train", f"{split_counts['Train']:,}", "ACTIVE", "primary"),
    ("Validation", f"{split_counts['Validation']:,}", "ACTIVE", "primary"),
    ("Test", f"{split_counts['Test']:,}", "LOCKED", "danger"),
    ("Total", f"{split_counts.sum():,}", "ORIGINALS", "muted")
]))
display(split_counts.rename("model_use_originals").to_frame())
display(pd.DataFrame({
    "dataset_fingerprint": [split_summary["dataset_fingerprint_sha256"]],
    "split_fingerprint": [split_summary["split_fingerprint_sha256"]],
    "generated_excluded": [int(manifest["is_generated"].sum())],
    "quarantine": [int(manifest["quarantined"].sum())],
}))
assert split_counts.to_dict() == {"Train": 2016, "Validation": 438, "Test": 440}
'''
nb.cells[8].source = cell_8_code.strip()

# Cell 28:
cell_28_code = '''
display(vu.render_summary_cards([
    ("Selected", selection["selected_experiment_id"], "WINNER", "success"),
    ("Val Accuracy", f'{selection["validation_accuracy"]*100:.2f}%' if selection["validation_accuracy"] < 1 else f'{selection["validation_accuracy"]:.2f}%', "BEST", "success"),
    ("Gap", f'{selection.get("generalization_gap", 20):.2f}%', "OVERFIT", "danger")
]))
display(pd.Series(selection, name="frozen_selection").to_frame())
'''
nb.cells[28].source = cell_28_code.strip()

nbformat.write(nb, 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb')
