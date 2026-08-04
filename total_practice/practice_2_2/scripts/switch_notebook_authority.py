"""One-time Phase 6 mechanical rewrite of the canonical report setup cell."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/04_canonical_report.ipynb"

SETUP = '''import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

cwd = Path.cwd().resolve()
candidates = [cwd, *cwd.parents]
project_root = next(
    (p for p in candidates if (p / "src" / "practice_2_2").is_dir()
     and (p / "configs" / "canonical_registry.json").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Open this notebook from the canonical practice_2_2 project")

src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from practice_2_2.resources import (
    load_canonical_registry,
    resolve_dataset_root,
    resolve_final_test_dir,
    resolve_resource,
    resolve_split_manifest,
    verify_canonical_resources,
)

registry = load_canonical_registry()
assert registry.get("authority_status") == "active"
authority = verify_canonical_resources(verify_dataset_contents=False)
assert authority["test_loader_constructed"] is False
assert authority["test_evaluated"] is False

data_root = resolve_dataset_root()
split_root = resolve_split_manifest().parent
canonical_root = resolve_resource("canonical_output_dir", require_file=False)
e3_root = resolve_resource("e3_output", require_file=False)
e4_root = resolve_resource("e4_output", require_file=False)
final_root = resolve_final_test_dir()

required = [
    split_root / "split_manifest.csv", split_root / "split_summary.json",
    canonical_root / "config_snapshot.json", canonical_root / "training_histories.json",
    canonical_root / "validation_comparison.csv", canonical_root / "lineage_manifest.json",
    e3_root / "phase_2_5_report.json", e4_root / "phase_2_6_report.json",
    e3_root / "training_history_e3.json", e4_root / "training_history_e4.json",
    final_root / "final_selection.json", final_root / "final_test_summary.json",
    final_root / "final_test_metrics_by_class.csv", final_root / "final_test_confusion_matrix.csv",
    final_root / "final_test_lineage.json", final_root / "pre_test_verification.json",
    final_root / "FINAL_TEST_COMPLETED.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Canonical artifacts are missing; do not regenerate Final Test. Missing:\\n" + "\\n".join(missing))

manifest = pd.read_csv(split_root / "split_manifest.csv")
split_summary = json.loads((split_root / "split_summary.json").read_text())
config = json.loads((canonical_root / "config_snapshot.json").read_text())
histories = json.loads((canonical_root / "training_histories.json").read_text())
lineage = json.loads((canonical_root / "lineage_manifest.json").read_text())
e3_report = json.loads((e3_root / "phase_2_5_report.json").read_text())
e4_report = json.loads((e4_root / "phase_2_6_report.json").read_text())
e3_history = json.loads((e3_root / "training_history_e3.json").read_text())
e4_history = json.loads((e4_root / "training_history_e4.json").read_text())
selection = json.loads((final_root / "final_selection.json").read_text())
final_summary = json.loads((final_root / "final_test_summary.json").read_text())
final_lineage = json.loads((final_root / "final_test_lineage.json").read_text())
pre_test_gate = json.loads((final_root / "pre_test_verification.json").read_text())
guard = json.loads((final_root / "FINAL_TEST_COMPLETED.json").read_text())
assert guard["FINAL_TEST_COMPLETED"] and guard["final_test_evaluation_count"] == 1
assert selection["test_used_for_selection"] is False
display(Markdown("**Artifact gate: PASS — active registry resolves the new canonical layout; Final Test guard remains locked.**"))
'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    matches = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "repo_root = next" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one legacy setup cell, found {len(matches)}")
    matches[0]["source"] = SETUP.splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
