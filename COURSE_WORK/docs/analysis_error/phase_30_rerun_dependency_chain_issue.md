# Phase 30 Rerun Dependency Chain Issue

## 1. Requested outcome

Prepare a safe refactor and terminal workflow that reruns Phase 30 without rerunning the notebook and without repeating valid scientific work.

## 2. Current Phase 30 state

```text
Phase 30 state = UPSTREAM_INVALID
Effective action = BLOCK
Verified LR conditions = none
Missing LR conditions = LR1, LR2, LR3
```

Current Phase 30 canonical files:

```text
artifacts/sweeps/s8_learning_rate/sweep_manifest.json = present
artifacts/sweeps/s8_learning_rate/phase_30_signoff.json = present but stale
artifacts/sweeps/s8_learning_rate/results.csv = missing
artifacts/sweeps/s8_learning_rate/live_sweep_results.jsonl = missing
artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json = missing
artifacts/sweeps/s8_learning_rate/s8_reference_update.json = missing
```

The old sign-off declares a missing `results.csv`. Its `PASS` status is therefore not reusable.

## 3. Phase 29 handoff state

```text
Phase 29 state = UPSTREAM_INVALID
Verified batch conditions = none
Missing batch conditions = B32, B64
Invalid batch condition = B32
```

The existing B32 record is incomplete:

```text
Run ID = RUN_TR_S07_0006_3181A7D4
Registry record = present
config.json = present
status.json = present
best_validation_metrics.json = present
training.log = missing
best_checkpoint.pt = missing
```

The Phase 29 live result contains only B32. It does not provide B64 evidence and cannot support a valid two-condition winner selection.

Missing Phase 29 handoff files:

```text
artifacts/sweeps/s7_batch_size/results.csv
artifacts/sweeps/s7_batch_size/s7_batch_winner.json
artifacts/sweeps/s7_batch_size/s7_reference_update.json
```

## 4. Earlier dependency state

Read-only inspection reports every sweep from Phase 23 through Phase 30 as `UPSTREAM_INVALID`.

The dependency chain is:

```text
Phase 22 diagnostics
Phase 23 S1
Phase 24 S2
Phase 25 S3
Phase 26 S4
Phase 27 S5
Phase 28 S6
Phase 29 S7
Phase 30 S8
```

Phase 22 is also missing:

```text
artifacts/learning_diagnostics/learning_diagnostics_summary.csv
```

Deep verification after implementation also found incomplete signed evidence in both baseline sources required by Phase 22.

Phase 20 is missing its canonical checkpoint and multiple audit tables. Phase 21 is missing its canonical checkpoint, run summary and multiple audit tables. Therefore the earliest broken scientific dependency is Phase 20, not merely the missing Phase 22 summary.

Phase 22 recovery must stop and report the incomplete Phase 20 and Phase 21 evidence. It must not rebuild diagnostics from partial histories.

Each sweep directory from S1 through S7 is missing its canonical `results.csv` and downstream winner/reference handoff files.

## 5. Environment state

The current signed environment belongs to another filesystem and interpreter:

```text
Signed project root = /Users/mac/Documents/study/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK
Signed Python = 3.10.21
Signed executable = /opt/homebrew/Cellar/python@3.10/3.10.21/Frameworks/Python.framework/Versions/3.10/bin/python3.10
Signed device = mps
```

The signed executable no longer exists on the current machine.

The current project runtime is:

```text
Current project root = /Users/voanhnhat-ticoder-coder/Documents/DEEP_LEARNING/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK
Current venv Python = 3.10.11
Current executable = /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10
```

The environment gate correctly reports:

```text
PYTHON_VERSION_MISMATCH
PYTHON_EXECUTABLE_MISMATCH
```

## 6. Why Phase 30 cannot be run directly

Phase 30 must reuse the exact Phase 29 winner as LR2 and run only LR1 and LR3 as new conditions.

The exact Phase 29 winner cannot currently be proven because:

- the S7 comparison is incomplete;
- required run evidence is missing;
- the canonical winner and reference handoff do not exist;
- Phase 29 itself has invalid upstream dependencies;
- the current runtime does not match the signed environment.

Running `--phase-id 30` now would either block or require bypassing scientific contracts. A bypass is not acceptable.

## 7. Required correction direction

The safe solution is a dependency-aware recovery workflow that:

1. reconciles the environment on the current machine;
2. validates Phase 20 and Phase 21 source evidence before Phase 22;
3. audits Phase 22 through Phase 30 in order;
4. reuses only complete registry evidence with valid checksums;
5. reruns only conditions whose required evidence is missing or invalid;
6. finalizes each phase before unlocking the next phase;
7. writes canonical results, winner, reference and sign-off artifacts;
8. regenerates derived JSON and static HTML only after canonical validation;
9. never accesses Test.

## 8. Scientific execution impact

The current evidence cannot guarantee a Phase 30-only rerun.

The upper bound is a recovery of the full missing sweep chain. The actual number of training runs must be calculated after the environment and Phase 22 recovery gates pass.

No training command should be issued until the recovery runner and canonical finalizer are implemented and verified.
