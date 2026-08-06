import nbformat
nb = nbformat.read('total_practice/practice_2_2/notebooks/04_canonical_report.ipynb', as_version=4)

def replace_cell(idx, old, new):
    if old in nb.cells[idx].source:
        nb.cells[idx].source = nb.cells[idx].source.replace(old, new)
    else:
        print(f'Warning: could not find code in cell {idx}')

if 'import practice_2_2.visualization_utils as vu' not in nb.cells[3].source:
    nb.cells[3].source += '\nimport practice_2_2.visualization_utils as vu'

replace_cell(6, 'display(provenance_table.style.hide(axis="index").set_caption("Dataset provenance and cleaning summary"))', 'display(vu.style_generic_table(provenance_table).hide(axis="index").set_caption("Dataset provenance and cleaning summary"))')

cell_8_code = '''
model_use = manifest.loc[manifest["use_for_model"]]
split_counts = model_use["split"].value_counts().reindex(["Train", "Validation", "Test"])
display(vu.render_summary_cards([
    ("Train", f"{split_counts['Train']:,}", "ACTIVE", vu.BLUE),
    ("Validation", f"{split_counts['Validation']:,}", "ACTIVE", vu.BLUE),
    ("Test", f"{split_counts['Test']:,}", "LOCKED", vu.RED),
    ("Total", f"{split_counts.sum():,}", "ORIGINALS", vu.GRAY)
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

replace_cell(9, 'display(leakage_table.style.hide(axis="index").set_caption("Canonical leakage-safety assertions"))', 'display(vu.style_leakage_table(leakage_table).hide(axis="index").set_caption("Canonical leakage-safety assertions"))')

cell_11_code = '''
class_split = pd.crosstab(model_use["class_name"], model_use["split"]).reindex(columns=["Train", "Validation", "Test"])
class_split['Total'] = class_split.sum(axis=1)
display(vu.style_distribution_table(class_split))
class_split[['Train', 'Validation', 'Test']].plot(kind="bar", figsize=(12, 5), title="Canonical model-use originals by class and split", stacked=True)
plt.ylabel("Images")
plt.tight_layout()
plt.show()
'''
nb.cells[11].source = cell_11_code.strip()

replace_cell(14, 'display(transform_audit.style.hide(axis="index").set_caption("Preprocessing isolation"))', 'display(vu.style_generic_table(transform_audit).hide(axis="index").set_caption("Preprocessing isolation"))')
replace_cell(16, 'display(model_table.style.hide(axis="index").set_caption("Model architecture and fine-tuning depth"))', 'display(vu.style_generic_table(model_table).hide(axis="index").set_caption("Model architecture and fine-tuning depth"))')
replace_cell(19, 'display(training_config.style.hide(axis="index").set_caption("Canonical controlled training configuration"))', 'display(vu.style_generic_table(training_config).hide(axis="index").set_caption("Canonical controlled training configuration"))')
replace_cell(21, 'display(e2_history[["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "generalization_gap", "lr"]])', 'display(vu.style_training_history(e2_history[["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "generalization_gap", "lr"]]))')
replace_cell(25, 'display(comparison)', 'display(vu.style_model_comparison(comparison).hide(axis="index"))')

cell_28_code = '''
display(vu.render_summary_cards([
    ("Selected", selection["selected_experiment_id"], "WINNER", vu.GREEN),
    ("Val Accuracy", f'{selection["validation_accuracy"]*100:.2f}%' if selection["validation_accuracy"] < 1 else f'{selection["validation_accuracy"]:.2f}%', "BEST", vu.GREEN),
    ("Gap", f'{selection.get("generalization_gap", 20):.2f}%', "OVERFIT", vu.RED)
]))
display(pd.Series(selection, name="frozen_selection").to_frame())
'''
nb.cells[28].source = cell_28_code.strip()

replace_cell(29, 'display(checkpoint_table.style.hide(axis="index").set_caption("Frozen winner and checkpoint identity"))', 'display(vu.style_generic_table(checkpoint_table).hide(axis="index").set_caption("Frozen winner and checkpoint identity"))')
replace_cell(31, 'display(final_table)', 'display(vu.style_generic_table(final_table).hide(axis="index"))')
replace_cell(38, 'display(final_audit.style.hide(axis="index").set_caption("Final submission audit"))', 'display(vu.style_leakage_table(final_audit).hide(axis="index").set_caption("Final submission audit"))')

nbformat.write(nb, 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb')
