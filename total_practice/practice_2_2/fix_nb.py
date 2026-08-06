import nbformat

nb_path = 'total_practice/practice_2_2/notebooks/04_canonical_report.ipynb'
nb = nbformat.read(nb_path, as_version=4)

# Replace the variables in the source code of the cells
for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        source = source.replace('vu.BLUE', "'#1f77b4'")
        source = source.replace('vu.RED', "'#d62728'")
        source = source.replace('vu.GRAY', "'gray'")
        source = source.replace('vu.GREEN', "'#2ca02c'")
        cell.source = source

nbformat.write(nb, nb_path)
print("Notebook variables fixed.")
