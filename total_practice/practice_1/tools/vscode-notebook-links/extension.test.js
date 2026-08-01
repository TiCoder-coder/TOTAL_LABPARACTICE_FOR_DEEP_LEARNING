const assert = require('assert');
const Module = require('module');

const originalLoad = Module._load;
let registeredHandler;
let showOptions;
let revealedRange;
let revealType;

class NotebookRange {
  constructor(start, end) {
    this.start = start;
    this.end = end;
  }
}

const editor = {
  selection: undefined,
  selections: [],
  revealRange(range, type) {
    revealedRange = range;
    revealType = type;
  },
};

const vscodeMock = {
  workspace: {
    notebookDocuments: [],
    workspaceFolders: [
      {
        uri: {
          path: '/workspace',
          fsPath: '/workspace',
        },
      },
    ],
    fs: {
      async stat() {
        return { type: 1 };
      },
    },
    async openNotebookDocument(uri) {
      return {
        uri,
        cellCount: 89,
      };
    },
  },
  Uri: {
    joinPath(base, ...segments) {
      const path = `${base.path}/${segments.join('/')}`;
      return {
        path,
        fsPath: path,
      };
    },
  },
  window: {
    registerUriHandler(handler) {
      registeredHandler = handler;
      return { dispose() {} };
    },
    async showNotebookDocument(notebook, options) {
      showOptions = options;
      return editor;
    },
    async showErrorMessage(message) {
      throw new Error(message);
    },
  },
  NotebookRange,
  NotebookEditorRevealType: {
    AtTop: 3,
  },
};

Module._load = function load(request, parent, isMain) {
  if (request === 'vscode') {
    return vscodeMock;
  }
  return originalLoad(request, parent, isMain);
};

async function run() {
  const extension = require('./extension');
  const context = { subscriptions: [] };
  extension.activate(context);

  assert.ok(registeredHandler);
  await registeredHandler.handleUri({
    path: '/open-cell',
    query:
      'notebook=total_practice%2Fpractice_1%2Fpractice_1.ipynb&cell=7',
  });

  assert.strictEqual(showOptions.preview, false);
  assert.strictEqual(showOptions.preserveFocus, false);
  assert.strictEqual(showOptions.selections[0].start, 7);
  assert.strictEqual(showOptions.selections[0].end, 8);
  assert.strictEqual(editor.selection.start, 7);
  assert.strictEqual(editor.selection.end, 8);
  assert.strictEqual(editor.selections[0].start, 7);
  assert.strictEqual(editor.selections[0].end, 8);
  assert.strictEqual(revealedRange.start, 7);
  assert.strictEqual(revealedRange.end, 8);
  assert.strictEqual(revealType, vscodeMock.NotebookEditorRevealType.AtTop);

  Module._load = originalLoad;
  console.log('Notebook URI handler test passed for Cell 7.');
}

run().catch((error) => {
  Module._load = originalLoad;
  console.error(error);
  process.exitCode = 1;
});
