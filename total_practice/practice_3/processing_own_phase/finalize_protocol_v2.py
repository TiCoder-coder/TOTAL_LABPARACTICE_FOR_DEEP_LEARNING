"""Read-only final audit and documentation freeze for Practice 3 v2 Part 2."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, RESULT_DIR
from .dataset_protocol_v2 import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    validate_split_manifest,
)
from .experiment_protocol_v2 import (
    LEARNING_RATES,
    PART_2_1_PLAN,
    PROTOCOL_VERSION,
    RUN_IDS,
    V2_RESULT_DIR,
    atomic_write_json,
    sha256_file,
    sha256_payload,
    validate_protocol_manifest,
    validate_run_configs,
)
from .experiment_registry_v2 import validate_registry
from .holdout_guard_v2 import validate_initial_holdout_state


EXPECTED_EXECUTION_HASH = "fdfcbb87b20d0bc618890a51c70a9689d618682b5a7cdf8edf05b638fa347055"
EXPECTED_V1_WEIGHT_HASH = "22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660"
AUDIT_PATH = V2_RESULT_DIR / "part_02_final_audit.json"
README_PATH = V2_RESULT_DIR / "PART_02_PROTOCOL_README.md"
REPORT_PATH = V2_RESULT_DIR / "part_02_completion_report.md"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _git_unchanged(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT.parent.parent)
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=PROJECT_ROOT.parent.parent,
        check=False,
    )
    return completed.returncode == 0


def _execution_components(
    protocol: dict[str, Any],
    split: dict[str, Any],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "part_2_1_plan_hash": sha256_file(PART_2_1_PLAN),
        "protocol_manifest_hash": protocol["protocol_manifest_hash"],
        "dataset_split_manifest_hash": split["split_manifest_hash"],
        "run_config_hashes": {
            config["run_id"]: config["config_hash"] for config in configs
        },
        "ranking": protocol["ranking"],
    }


def _render_readme(audit: dict[str, Any]) -> str:
    return f"""# Practice 3 v2 — Frozen Experiment Protocol

## Protocol

- Version: `{audit['protocol_version']}`
- Execution protocol hash: `{audit['execution_protocol_hash']}`
- Protocol drift: `false`

## Development split

| Split | Count | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 7,676 | 3,838 | 3,838 |
| Validation | 960 | 480 | 480 |
| Holdout | 960 | 480 | 480 |

Source: original official Train + Validation. The official v1 Test is excluded from v2.

## Experiments

- `p3v2_lr_2e-5`: learning rate `2e-5`
- `p3v2_lr_3e-5`: learning rate `3e-5`
- `p3v2_lr_5e-5`: learning rate `5e-5`

Only learning rate differs experimentally.

## Training contract

- Generic DistilBERT binary classifier and tokenizer
- max length 80, truncation and dynamic padding
- maximum 10 epochs
- Early Stopping on Validation loss: patience 2, threshold `1e-6`
- fused AdamW, linear scheduler, weight decay 0.01, gradient clipping 1.0
- Train/eval batch sizes 16/32

## Monitoring

TensorBoard paths are isolated under `runs/practice_3_v2/<run_id>/`. Required future scalars: Train loss, Validation loss, Validation Accuracy, Validation F1, learning rate, epoch and global step. No event has been written before Part 3.

## Ranking

Validation only:

1. lowest Validation loss;
2. within `1e-6`, higher Validation F1;
3. within `1e-6`, higher Validation Accuracy;
4. earlier best epoch;
5. lexical run ID as deterministic final fallback.

## Current status

```text
Experiments = NOT_STARTED
Runs = PLANNED / PLANNED / PLANNED
Winner = NOT_SELECTED
Holdout = SEALED
holdout_evaluation_count = 0
```

## Holdout limitation

The v2 Holdout is untouched relative to the frozen v2 protocol, but its records originate from the v1 official Train/Validation source. It is not globally unseen in the complete project history and must not be described as a new external Test set.

## Historical v1 separation

Practice 3 v1 remains historical: `checkpoint-1068`, Test evaluation count 1, and preserved weight/package SHA-256 `{EXPECTED_V1_WEIGHT_HASH}`. Its Test result is not used by v2.

## Technical debt

- Transformers `logging_dir` future deprecation: non-blocking; identical supported behavior for all runs.
- MPS pinned-memory warning: non-blocking unless real training reveals a functional issue.
- Newly initialized generic classification head: expected before fine-tuning, not a bug.

## Readiness

Part 2 final audit: **PASS**. Ready for Part 3 real controlled training: **YES**. No training, ranking, winner selection or Holdout evaluation has occurred.
"""


def _render_report(audit: dict[str, Any]) -> str:
    return f"""# Practice 3 v2 — Part 2 Completion Report

## Part 2.1

Protocol design: PASS. The LR candidates, deterministic split, fixed variables, Early Stopping, Validation-only ranking, registry, winner lock and one-time Holdout contract were pre-registered.

## Part 2.2

Infrastructure: PASS. Separate v2 metadata, deterministic split manifest, three immutable configs, PLANNED registry, TensorBoard contract, ranking/winner/cache helpers and sealed Holdout state were implemented and synthetically verified.

## Part 2.3

Pre-training integration: PASS. Real v2 Train/Validation, tokenizer, dynamic batches, three fresh models, TrainingArguments, callbacks, Trainers, fused optimizers and schedulers were constructed without training. Execution hash: `{audit['execution_protocol_hash']}`.

## Part 2.4

Final audit/freeze: PASS. Frozen hashes and values match; registry/Holdout/v1 state remain unchanged; no metric, winner, event or checkpoint exists for v2.

## Frozen protocol

`practice_3_v2.0`; learning rates `2e-5/3e-5/5e-5`; max 10 epochs; patience 2; threshold/tolerance `1e-6`.

## Dataset boundary

Train 7,676; Validation 960; Holdout 960 SEALED. Official v1 Test is excluded. Duplicate and cross-split overlap counts are zero.

## Experiment configs

Three deterministic PLANNED configs. Identity/path/hash fields are run-specific; learning rate is the only experimental difference.

## Training contract

Fresh DistilBERT/model/optimizer/scheduler per run, binary metrics, Validation loss selection, epoch evaluation/save/logging and no shared mutable run state.

## TensorBoard contract

Three isolated directories and the seven required future scalar series. No pre-training event file exists.

## Holdout safety

No winner manifest; access denied; evaluation count zero. The Holdout is protocol-relative rather than globally novel.

## Historical v1 separation

v1 Test count remains one and checkpoint/package hashes remain `{EXPECTED_V1_WEIGHT_HASH}`. v1 evidence is not reused as v2 final evaluation.

## Technical debt

`logging_dir` deprecation and MPS pinned-memory warnings are non-blocking. Generic classification-head initialization is expected.

## Final readiness

- protocol drift: false
- training performed: false
- winner selected: false
- Holdout materialized/evaluated: false
- ready for Part 3: **YES**
"""


def finalize_part_02() -> dict[str, Any]:
    """Validate frozen metadata and write documentation only."""
    protocol = _load(V2_RESULT_DIR / "protocol_manifest.json")
    split = _load(V2_RESULT_DIR / "dataset_split_manifest.json")
    registry = _load(V2_RESULT_DIR / "experiment_registry.json")
    state = _load(V2_RESULT_DIR / "holdout_access_state.json")
    readiness = _load(V2_RESULT_DIR / "pretraining_readiness_manifest.json")
    configs = [
        _load(V2_RESULT_DIR / "experiments" / run_id / "run_config.json")
        for run_id in RUN_IDS
    ]
    validate_protocol_manifest(protocol)
    validate_split_manifest(split)
    validate_run_configs(configs)
    validate_registry(registry, configs, require_planned=True)
    validate_initial_holdout_state(state)

    readiness_hash = readiness.pop("readiness_manifest_hash")
    readiness_self_hash_valid = readiness_hash == sha256_payload(readiness)
    components = _execution_components(protocol, split, configs)
    recomputed_execution_hash = sha256_payload(components)
    registry_hash = sha256_payload(registry)

    result_fields = (
        "stopped_epoch", "best_epoch", "best_val_loss", "best_val_accuracy",
        "best_val_precision", "best_val_recall", "best_val_f1",
        "best_checkpoint", "runtime_seconds",
    )
    event_files = list((PROJECT_ROOT / "runs" / "practice_3_v2").glob("**/events.out.tfevents*"))
    experiment_extra_files = [
        path
        for path in (V2_RESULT_DIR / "experiments").glob("**/*")
        if path.is_file() and path.name != "run_config.json"
    ]
    notebook_path = PROJECT_ROOT / "notebook_practice_3" / "practice_3.ipynb"
    v1_manifest_path = RESULT_DIR / "phase_11_evaluation" / "phase_11_evaluation_manifest.json"
    v1_manifest = _load(v1_manifest_path)
    checkpoint = RESULT_DIR / "phase_09_training" / "checkpoints" / "checkpoint-1068" / "model.safetensors"
    package = RESULT_DIR / "phase_14_saved_model" / "model.safetensors"

    checks = {
        "part_2_1_pass": protocol["part_2_1_plan_sha256"] == sha256_file(PART_2_1_PLAN),
        "part_2_2_pass": protocol["optimizer_preflight"]["status"] == "SUPPORTED",
        "part_2_3_pass": readiness["status"] == "PASS" and readiness["ready_for_training"] is True,
        "readiness_self_hash_valid": readiness_self_hash_valid,
        "execution_hash_unchanged": (
            recomputed_execution_hash == EXPECTED_EXECUTION_HASH
            and readiness["execution_protocol_hash"] == EXPECTED_EXECUTION_HASH
            and readiness["freeze_components"] == components
        ),
        "protocol_values_frozen": (
            protocol["protocol_version"] == PROTOCOL_VERSION
            and tuple(protocol["learning_rates"]) == LEARNING_RATES
            and protocol["max_epochs"] == 10
            and protocol["early_stopping"] == {
                "patience": 2,
                "threshold": 1e-6,
                "metric": "validation_loss",
                "greater_is_better": False,
            }
            and protocol["ranking"]["tolerance"] == 1e-6
        ),
        "dataset_protocol_valid": (
            split["development_pool_count"] == 9596
            and split["split_counts"] == EXPECTED_SPLIT_COUNTS
            and split["class_counts"] == EXPECTED_CLASS_COUNTS
            and split["split_manifest_hash"] == "4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13"
            and split["within_development_exact_duplicate_count"] == 0
            and all(value == 0 for value in split["source_id_overlap"].values())
            and all(value == 0 for value in split["exact_text_cross_split_overlap"].values())
        ),
        "official_test_loaded_false": (
            split["official_test_loaded"] is False
            and readiness["official_test_loaded"] is False
        ),
        "holdout_sealed": (
            protocol["holdout_status"] == "SEALED"
            and state["holdout_access_allowed"] is False
            and state["holdout_evaluation_count"] == 0
            and readiness["holdout_materialized"] is False
            and split["holdout_content_in_manifest"] is False
        ),
        "registry_three_planned": (
            [record["run_id"] for record in registry["runs"]] == list(RUN_IDS)
            and all(record["status"] == "PLANNED" for record in registry["runs"])
        ),
        "real_metrics_absent": (
            all(
                all(record[field] is None for field in result_fields)
                for record in registry["runs"]
            )
            and not experiment_extra_files
        ),
        "run_config_hashes_frozen": (
            readiness["run_config_hashes"]
            == {config["run_id"]: config["config_hash"] for config in configs}
        ),
        "optimizer_scheduler_frozen": all(
            config["optimizer"] == "ADAMW_TORCH_FUSED"
            and config["scheduler"] == "linear"
            and config["weight_decay"] == 0.01
            and config["gradient_clipping"] == 1.0
            for config in configs
        ),
        "early_stopping_frozen": all(
            config["max_epochs"] == 10
            and config["early_stopping_patience"] == 2
            and config["early_stopping_threshold"] == 1e-6
            and config["selection_metric"] == "validation_loss"
            and config["greater_is_better"] is False
            and config["evaluation_strategy"] == "epoch"
            and config["save_strategy"] == "epoch"
            for config in configs
        ),
        "tensorboard_ready_no_events": (
            not event_files
            and len({config["tensorboard_log_dir"] for config in configs}) == 3
            and protocol["tensorboard"]["event_files_created"] is False
        ),
        "run_isolation": (
            len({report["output_dir"] for report in readiness["trainer_reports"]}) == 3
            and len({report["tensorboard_log_dir"] for report in readiness["trainer_reports"]}) == 3
        ),
        "ranking_contract": (
            protocol["ranking"]["primary"] == "validation_loss"
            and protocol["ranking"]["secondary"] == "validation_f1"
            and protocol["ranking"]["tertiary"] == "validation_accuracy"
            and protocol["ranking"]["fourth"] == "earlier_best_epoch"
            and readiness["real_ranking_performed"] is False
        ),
        "winner_absent": (
            protocol["winner_status"] == "NOT_SELECTED"
            and not (V2_RESULT_DIR / "winner_manifest.json").exists()
            and readiness["real_winner_selected"] is False
        ),
        "v1_test_count_one": v1_manifest["test_evaluation_count"] == 1,
        "v1_checkpoint_preserved": (
            sha256_file(checkpoint) == EXPECTED_V1_WEIGHT_HASH
            and sha256_file(package) == EXPECTED_V1_WEIGHT_HASH
        ),
        "v1_artifacts_preserved": (
            _git_unchanged(v1_manifest_path)
            and _git_unchanged(notebook_path)
        ),
        "no_training": (
            readiness["training_performed"] is False
            and readiness["trainer_train_called"] is False
            and readiness["backward_called"] is False
            and readiness["optimizer_step_called"] is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Part 2.4 final audit failed: {checks}")

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "part_2_1_status": "PASS",
        "part_2_2_status": "PASS",
        "part_2_3_status": "PASS",
        "part_2_4_status": "PASS",
        "execution_protocol_hash": recomputed_execution_hash,
        "protocol_drift": False,
        "dataset_protocol_valid": True,
        "dataset_counts": split["split_counts"],
        "official_test_loaded": False,
        "holdout_status": "SEALED",
        "holdout_materialized": False,
        "holdout_evaluation_count": 0,
        "registry_status": "NOT_STARTED",
        "run_statuses": {
            record["run_id"]: record["status"] for record in registry["runs"]
        },
        "registry_hash": registry_hash,
        "real_metrics_exist": False,
        "optimizer_status": "SUPPORTED",
        "early_stopping_status": "FROZEN",
        "tensorboard_status": "READY_NO_EVENTS",
        "run_isolation_status": "PASS",
        "ranking_status": "FROZEN_NOT_RUN",
        "winner_status": "NOT_SELECTED",
        "v1_test_evaluation_count": 1,
        "v1_checkpoint_preserved": True,
        "v1_artifacts_preserved": True,
        "training_performed": False,
        "checks": checks,
        "technical_debt": {
            "transformers_logging_dir_deprecation": "NON_BLOCKING",
            "mps_pinned_memory_warning": "NON_BLOCKING",
            "generic_classification_head_initialization": "EXPECTED_NOT_A_BUG",
        },
        "documentation": {
            "protocol_readme": str(README_PATH.relative_to(PROJECT_ROOT)),
            "completion_report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
        "ready_for_part_3": True,
    }
    audit["audit_hash"] = sha256_payload(audit)
    atomic_write_json(AUDIT_PATH, audit)
    README_PATH.write_text(_render_readme(audit), encoding="utf-8")
    REPORT_PATH.write_text(_render_report(audit), encoding="utf-8")
    return audit


if __name__ == "__main__":
    result = finalize_part_02()
    print({
        "part_2_4_status": result["part_2_4_status"],
        "execution_protocol_hash": result["execution_protocol_hash"],
        "ready_for_part_3": result["ready_for_part_3"],
    })

