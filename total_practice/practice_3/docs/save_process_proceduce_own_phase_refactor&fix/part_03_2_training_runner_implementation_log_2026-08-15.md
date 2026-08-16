# PART 3.2 — Training Runner Implementation Log

Date: 2026-08-15  
Scope: implementation and safe verification only; no real training.

## Files audited

- `processing_own_phase/experiment_runner_v2.py`
- `processing_own_phase/experiment_protocol_v2.py`
- `processing_own_phase/dataset_protocol_v2.py`
- `processing_own_phase/experiment_registry_v2.py`
- `processing_own_phase/holdout_guard_v2.py`
- `processing_own_phase/pretraining_integration_v2.py`
- `processing_own_phase/authorize_training_v2.py`
- Part 3.1 authorization and all frozen v2 manifests/configs

The requested Part 3.2 plan file existed as a zero-byte file. It was restored from the approved task contract before implementation.

## Files changed

- `docs/plan-doc/plan_before_process/part_03_2_real_controlled_training_plan_2026-08-15.md`
- `processing_own_phase/experiment_runner_v2.py`
- `processing_own_phase/experiment_registry_v2.py`
- `tests/test_training_runner_v2.py`
- this log

## Implementation

- Added one explicit `run_experiment(run_id: str)` execution interface.
- Added a required `--run-id` CLI restricted to the three frozen run IDs.
- Added authorization self-hash and execution-protocol-hash verification.
- Added config, registry, execution order, Holdout seal, official-Test exclusion, winner absence and stale-output guards.
- Reused the locked Train/Validation materializer; no Holdout/Test provider is referenced by the runner.
- Added deterministic seed reset and fresh tokenizer/model/Trainer/optimizer/scheduler construction per invocation.
- Added isolated epoch checkpoint and TensorBoard configuration.
- Added normalized training history, deterministic within-run best-epoch selection, checkpoint SHA-256 and run summary writers.
- Added file-locked atomic registry transitions: `PLANNED -> RUNNING -> COMPLETED`, or `RUNNING -> FAILED` on an exception.
- No cross-run ranking or winner selection was implemented or executed.

## Safe verification

Command:

```bash
cd total_practice/practice_3
../../.venv/bin/python -m unittest -v tests.test_training_runner_v2 tests.test_protocol_v2
```

Result: 17 tests PASS. Tests covered syntax/import, CLI parsing and rejection, read-only real metadata preflight, synthetic history/epoch selection, synthetic checkpoint hashing, and lifecycle persistence exclusively in a temporary registry.

The actual CLI import path was also verified from repository root with `--help`:

```bash
.venv/bin/python -m total_practice.practice_3.processing_own_phase.experiment_runner_v2 --help
```

## Final safety state

- `Trainer.train()` executed by agent: false
- real experiment executed: false
- registry statuses: all `PLANNED`
- real checkpoints/metrics/events created: false
- Holdout evaluation count: 0
- official Test loaded: false
- frozen protocol modified: false

## User command for the first authorized run

From repository root:

```bash
source .venv/bin/activate
python -m total_practice.practice_3.processing_own_phase.experiment_runner_v2 --run-id p3v2_lr_2e-5
```
