# Practice 1 Notebook Links

This VS Code helper opens an exact notebook cell when a link in
`description/description_own_phase` is clicked.

## Install

From this directory:

```bash
code --install-extension practice1-notebook-links-0.1.0.vsix --force
```

The extension registers the URI authority:

```text
vscode://ticoder.practice1-notebook-links/open-cell
```

Each documentation link supplies a workspace-relative notebook path and a
zero-based cell index. The extension opens the notebook, selects that cell and
reveals it at the top of the notebook editor.
