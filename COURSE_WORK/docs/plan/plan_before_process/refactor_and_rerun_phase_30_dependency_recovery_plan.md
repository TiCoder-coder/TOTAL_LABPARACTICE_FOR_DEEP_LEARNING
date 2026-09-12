# Refactor and Rerun Phase 30 Dependency Recovery Plan

## 1. Objective

Implement a clean terminal-owned recovery flow that reaches a valid Phase 30 result without rerunning the notebook and without rerunning any condition that still has complete canonical evidence.

## 2. Non-negotiable rules

```text
No Test access
No fabricated CSV, winner, reference or sign-off
No trust in stale PASS labels when declared files are missing
No use of processing logs as scientific source of truth
No retraining of a complete exact-match reference
No Phase 30 execution before Phase 29 is valid
No downstream phase finalization before all registered conditions are valid
No deletion of existing evidence
```

## 3. Target architecture

### 3.1 Read-only recovery inspector

Add a recovery inspector under:

```text
src/course_work/experiments/sweep_recovery.py
```

Responsibilities:

- inspect Phase 22 through the requested target phase;
- classify every artifact and condition as reusable, missing, invalid or blocked;
- calculate the earliest invalid dependency;
- calculate the minimal required execution set;
- detect stale sign-offs;
- detect environment identity mismatch;
- produce a deterministic recovery plan without writing scientific artifacts.

### 3.2 Canonical sweep finalizer

Add or extend the sweep finalization layer under:

```text
src/course_work/sweeps/sweep_results.py
```

Responsibilities for Phase 23 through Phase 30:

- load only verified completed run evidence;
- require every registered condition exactly once;
- create canonical results CSV;
- apply the declared Validation RMSE selection rule and exact tie rule;
- create the phase-specific winner artifact;
- create the next-phase reference update;
- create a sign-off whose checksums cover all required outputs;
- confirm Test remains untouched;
- reject incomplete or ambiguous sweeps.

Phase 30 finalization must create:

```text
artifacts/sweeps/s8_learning_rate/results.csv
artifacts/sweeps/s8_learning_rate/s8_learning_rate_winner.json
artifacts/sweeps/s8_learning_rate/s8_reference_update.json
artifacts/sweeps/s8_learning_rate/phase_30_signoff.json
```

The Phase 30 handoff must include:

```text
winner_lr_id
winner_learning_rate
winner_run_id
winner_config_fingerprint
winner_optimizer_config_fingerprint
weight_decay_state = WD1_1E-4
approved_for_phase31 = true
test_status = FORBIDDEN or LOCKED
```

### 3.3 Dependency-aware terminal runner

Extend the terminal runner with:

```text
--target-phase 30
--with-dependencies
--audit-only
--dry-run
--foreground
```

The runner must process one phase at a time:

```text
inspect
execute missing conditions only
verify every new run
finalize canonical artifacts
verify phase sign-off and handoff
unlock next phase
```

If a phase fails verification, the runner must stop before the next phase.

### 3.4 Environment reconciliation

Add an explicit environment recovery stage before scientific execution.

The stage must:

- preserve the existing signed environment files;
- record the current runtime identity;
- require MPS or CUDA before training;
- verify the selected notebook kernel and terminal interpreter refer to the same runtime;
- create an approved current-machine environment revision;
- propagate that environment identity into new run configs;
- never silently overwrite the historical environment evidence.

The recovery strategy must be implemented and verified before any training command is provided.

## 4. Sequential implementation plan

### Step 1. Capture recovery baseline

Record checksums and presence states for:

```text
artifacts/environment
artifacts/learning_diagnostics
artifacts/experiments
artifacts/runs
artifacts/sweeps/s1_feature_set
artifacts/sweeps/s2_time_feature
artifacts/sweeps/s3_target_scaling
artifacts/sweeps/s4_lookback
artifacts/sweeps/s5_pooling
artifacts/sweeps/s6_activation
artifacts/sweeps/s7_batch_size
artifacts/sweeps/s8_learning_rate
docs/save_log_in_processing/phase_22_*.json through phase_30_*.json
notebook_course_work/CourseWork.ipynb
```

Verification:

- baseline is read-only;
- existing evidence remains unchanged;
- notebook outputs remain unchanged.

### Step 2. Implement environment recovery audit

Add read-only environment identity inspection and a current-machine recovery proposal.

Verification:

- old environment evidence remains preserved;
- mismatched version, executable, root and device are displayed separately;
- training remains blocked while MPS or CUDA is unavailable.

### Step 3. Implement dependency-chain inspection

Calculate the earliest invalid phase and the exact missing/invalid conditions from Phase 22 to Phase 30.

Verification:

- current workspace resolves Phase 22 as the first broken canonical dependency;
- Phase 30 cannot be selected without dependencies;
- audit mode performs no writes.

### Step 4. Implement canonical finalization

Build phase-specific results, winner, reference and sign-off artifacts exclusively from verified registry evidence.

Verification for every finalized phase:

- expected conditions complete;
- no duplicate valid condition;
- all required artifacts exist and match registry checksums;
- full-precision Validation metrics used;
- exact tie rule applied;
- winner run configuration matches the selected factor;
- Test records absent;
- output checksums reload correctly.

### Step 5. Refactor the terminal runner

Add dependency-aware resume behavior.

Verification:

- audit-only writes nothing;
- dry-run lists only unresolved phases and conditions;
- execution never enters a downstream phase before upstream sign-off passes;
- reference conditions are reused only when their complete evidence exists;
- invalid references are not silently accepted;
- interruption can resume from the last verified condition.

### Step 6. Validate Phase 20 and Phase 21, then repair Phase 22

Attempt to rebuild diagnostics only from complete Phase 20 and Phase 21 histories.

Current deep verification confirms both signed baseline artifact sets are incomplete. The recovery runner must stop and report the exact missing Phase 20 and Phase 21 files before changing Phase 22.

Verification:

```text
learning_diagnostics_summary.csv present
Phase 22 output checksums valid
Phase 22 sign-off reusable
```

The Phase 22 repair gate must additionally verify:

```text
Phase 20 sign-off outputs complete and checksum-valid
Phase 21 sign-off outputs complete and checksum-valid
Canonical best checkpoints present
Canonical learning histories present
No partial baseline evidence used for diagnostics
```

### Step 7. Recover Phase 23 through Phase 29

For each phase in order:

1. reuse complete exact-match run evidence;
2. run only missing or invalid conditions;
3. verify new run artifacts immediately;
4. finalize results, winner, reference and sign-off;
5. regenerate the derived log and static HTML;
6. verify the next-phase handoff before continuing.

### Step 8. Run Phase 30

After Phase 29 passes:

- load the selected B* from `s7_reference_update.json`;
- reuse the exact S7 winner as LR2 only if all S8 fixed fields match;
- run LR1 and LR3 if missing;
- run LR2 only if its reference evidence is invalid and the Phase 30 contract permits recovery;
- compare full-precision Validation RMSE;
- create the Phase 30 winner and Phase 31 handoff;
- preserve the Test firewall.

### Step 9. Regenerate Phase 30 and Phase 31 presentation

Regenerate only:

```text
docs/save_log_in_processing/phase_30_s8_learning_rate_log.json
docs/save_log_in_processing/phase_31_s9_weight_decay_log.json
```

The notebook remains presentation-only. The user only reruns the Phase 30, Phase 31 and all-log display cells if a refreshed stored output is wanted.

### Step 10. Final verification

Required checks:

```text
Phase 22 through Phase 29 = valid reusable
Phase 30 = PASS or PASS_WITH_WARNING
LR1, LR2, LR3 = verified
Phase 30 winner unique
approved_for_phase31 = true
Phase 31 handoff valid
Test access = forbidden
notebook processing code = absent
widget MIME = absent
static HTML = valid
```

## 5. Planned terminal interface

After the refactor passes tests, the initial safe command will be:

```bash
cd "/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK"
PYTHONPATH=src ../venv/bin/python scripts/run_all_pending.py --target-phase 30 --with-dependencies --audit-only
```

The dry-run command will be:

```bash
PYTHONPATH=src ../venv/bin/python scripts/run_phase_background.py sweep --target-phase 30 --with-dependencies --dry-run
```

The actual background execution command will be:

```bash
PYTHONPATH=src ../venv/bin/python scripts/run_phase_background.py sweep --target-phase 30 --with-dependencies
```

These commands are planned interfaces. They must not be used until their implementation and verification are complete.

## 6. Stop and resume protocol

After the user starts the background command:

```bash
PYTHONPATH=src ../venv/bin/python scripts/run_phase_background.py --status
```

To inspect recent logs:

```bash
PYTHONPATH=src ../venv/bin/python scripts/run_phase_background.py --logs --tail 100
```

After the process completes, the user provides:

```text
terminal exit status
background status JSON
last 100 stdout lines
last 100 stderr lines
```

The next verification will inspect only newly created or changed scientific artifacts and will not rerun the completed conditions.

## 7. Current authorization boundary

```text
Plan creation = complete
Read-only audit = complete
Recovery implementation = not started
Scientific execution = not started
Terminal training command = not yet released for execution
```
