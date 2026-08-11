const vscode = require('vscode');

const NOTEBOOK_QUERY_KEY = 'notebook';
const CELL_QUERY_KEY = 'cell';

function parseRelativeNotebookPath(value) {
  if (!value || value.startsWith('/') || value.startsWith('\\')) {
    throw new Error('The notebook query must be a workspace-relative path.');
  }

  const segments = value.split('/').filter(Boolean);
  if (segments.length === 0 || segments.some((segment) => segment === '..')) {
    throw new Error('The notebook query contains an invalid path.');
  }
  return segments;
}

async function resolveNotebookUri(relativePath) {
  const normalizedSuffix = `/${relativePath}`;
  const openNotebook = vscode.workspace.notebookDocuments.find(
    (document) => document.uri.path.endsWith(normalizedSuffix),
  );
  if (openNotebook) {
    return openNotebook.uri;
  }

  const segments = parseRelativeNotebookPath(relativePath);
  for (const folder of vscode.workspace.workspaceFolders || []) {
    const candidate = vscode.Uri.joinPath(folder.uri, ...segments);
    try {
      await vscode.workspace.fs.stat(candidate);
      return candidate;
    } catch {
      continue;
    }
  }

  throw new Error(
    `Cannot find ${relativePath}. Open the LAB&PRACTICE folder in VS Code.`,
  );
}

async function openNotebookCell(uri) {
  if (uri.path !== '/open-cell') {
    throw new Error(`Unsupported notebook-link path: ${uri.path}`);
  }

  const query = new URLSearchParams(uri.query);
  const relativePath = query.get(NOTEBOOK_QUERY_KEY);
  const cellValue = query.get(CELL_QUERY_KEY);
  const cellIndex = Number(cellValue);

  if (!Number.isInteger(cellIndex) || cellIndex < 0) {
    throw new Error(`Invalid cell index: ${cellValue}`);
  }

  const notebookUri = await resolveNotebookUri(relativePath);
  const notebook = await vscode.workspace.openNotebookDocument(notebookUri);
  if (cellIndex >= notebook.cellCount) {
    throw new Error(
      `Cell ${cellIndex} is outside notebook range 0-${notebook.cellCount - 1}.`,
    );
  }

  const selection = new vscode.NotebookRange(cellIndex, cellIndex + 1);
  const editor = await vscode.window.showNotebookDocument(notebook, {
    preserveFocus: false,
    preview: false,
    selections: [selection],
  });
  editor.selection = selection;
  editor.selections = [selection];
  editor.revealRange(selection, vscode.NotebookEditorRevealType.AtTop);

  console.log(
    `[practice1-notebook-links] opened ${notebookUri.fsPath} at cell ${cellIndex}`,
  );
}

function activate(context) {
  const handler = {
    async handleUri(uri) {
      try {
        await openNotebookCell(uri);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`[practice1-notebook-links] ${message}`);
        await vscode.window.showErrorMessage(`Notebook link failed: ${message}`);
      }
    },
  };

  context.subscriptions.push(vscode.window.registerUriHandler(handler));
  console.log('[practice1-notebook-links] URI handler activated');
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
