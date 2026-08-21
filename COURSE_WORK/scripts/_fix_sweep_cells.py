"""Fix the 8 sweep cells in CourseWork.ipynb.

Bug: the matplotlib chart block sits at the top-level after `else:` but
references df_xx which is only defined inside the `if SWEEP_PATH.exists():`
branch. When the file is missing, the chart block tries to use df_xx and
crashes with NameError.

Fix: move the matplotlib block inside the `if SWEEP_PATH.exists():` branch,
and replace the `variant` column reference with `condition` (which is the
actual column produced by run_single_condition.py).
"""
import json
from pathlib import Path

NB = Path("/Users/mac/Documents/study/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/notebook_course_work/CourseWork.ipynb")

sweep_specs = [
    (23, "s1_feature_set", "S1 Feature-Set Sweep", "bar"),
    (24, "s2_time_feature", "S2 Time-Feature Sweep", "bar"),
    (25, "s3_target_scaling", "S3 Target-Scaling Sweep", "bar"),
    (26, "s4_lookback",       "S4 Lookback Sweep", "plot"),
    (27, "s5_pooling",        "S5 Pooling Sweep", "bar"),
    (28, "s6_activation",     "S6 Activation Sweep", "bar"),
    (29, "s7_batch_size",     "S7 Batch-Size Sweep", "bar"),
    (30, "s8_learning_rate",  "S8 Learning-Rate Sweep", "bar"),
]

BAR_TEMPLATE = """# Phase {pid}: Load and display sweep/diagnostic results
import pandas as pd
from pathlib import Path
from IPython.display import display

SWEEP_PATH = Path(PROJECT_ROOT) / "artifacts/sweeps/{dir}/results.csv"

if SWEEP_PATH.exists():
    df_{pid} = pd.read_csv(SWEEP_PATH)
    if "variant" not in df_{pid}.columns and "condition" in df_{pid}.columns:
        df_{pid} = df_{pid}.rename(columns={{"condition": "variant"}})
    display(render_dataframe_table(
        df_{pid},
        title="Phase {pid} - {title}",
        subtitle="Summary table",
        metrics={{"Rows": len(df_{pid})}},
        max_height=320,
    ))
    if "val_rmse" in df_{pid}.columns:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xcol = "condition" if "condition" in df_{pid}.columns else "variant"
        ax.bar(df_{pid}[xcol].astype(str), df_{pid}["val_rmse"])
        ax.set_xlabel("Variant")
        ax.set_ylabel("Validation RMSE (Wh)")
        ax.set_title("Phase {pid} - {title}")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
else:
    print(f"Results not found: {{SWEEP_PATH}}")
    print("Run the corresponding sweep training script first to populate results.")
"""

PLOT_TEMPLATE = """# Phase {pid}: Load and display sweep/diagnostic results
import pandas as pd
from pathlib import Path
from IPython.display import display

SWEEP_PATH = Path(PROJECT_ROOT) / "artifacts/sweeps/{dir}/results.csv"

if SWEEP_PATH.exists():
    df_{pid} = pd.read_csv(SWEEP_PATH)
    if "variant" not in df_{pid}.columns and "condition" in df_{pid}.columns:
        df_{pid} = df_{pid}.rename(columns={{"condition": "variant"}})
    display(render_dataframe_table(
        df_{pid},
        title="Phase {pid} - {title}",
        subtitle="Summary table",
        metrics={{"Rows": len(df_{pid})}},
        max_height=320,
    ))
    if "val_rmse" in df_{pid}.columns:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        xcol = "condition" if "condition" in df_{pid}.columns else "variant"
        ax.plot(df_{pid}[xcol].astype(str), df_{pid}["val_rmse"], marker="o")
        ax.set_xlabel(xcol)
        ax.set_ylabel("Validation RMSE (Wh)")
        ax.set_title("Phase {pid} - {title}")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
else:
    print(f"Results not found: {{SWEEP_PATH}}")
    print("Run the corresponding sweep training script first to populate results.")
"""


def to_source(text: str) -> list[str]:
    """Convert text to nbformat source lines (each line ending with \\n)."""
    lines = text.split("\n")
    # If split produced empty trailing string because text ended with \n, drop it
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [line + "\n" for line in lines]


with NB.open() as f:
    nb = json.load(f)

fixes = []
for phase_id, subdir, title, kind in sweep_specs:
    marker = f"SWEEP_PATH = Path(PROJECT_ROOT) / \"artifacts/sweeps/{subdir}/results.csv\""
    template = PLOT_TEMPLATE if kind == "plot" else BAR_TEMPLATE
    new_src = template.format(pid=phase_id, dir=subdir, title=title)
    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell.get("source"), list) else cell.get("source", "")
        if marker not in src:
            continue
        cell["source"] = to_source(new_src)
        cell["outputs"] = []  # clear stale outputs
        found = True
        fixes.append(phase_id)
        break
    if not found:
        print(f"WARNING: Could not find cell for Phase {phase_id} ({subdir})")

with NB.open("w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed phases: {fixes}")
print("Notebook saved.")
