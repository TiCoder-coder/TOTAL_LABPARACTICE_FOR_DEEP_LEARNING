# Learning-Curve Diagnostics (Phase 22)

## Overview

Phase 22 performs learning-curve diagnostics on LSTM B0 and Transformer B0 training histories to identify:
- Optimization dynamics (convergence, instability, plateau)
- Overfitting/underfitting patterns
- Gradient health
- Potential hypotheses for future sweep phases

## Source Runs

| Model | Run ID | Best Epoch | Val RMSE (Wh) |
|-------|--------|------------|---------------|
| LSTM | RUN_LS_LS_0013_63C7E5ED | 6 | 60.446 |
| Transformer | RUN_TR_B0_0014_00EF3A31 | 12 | 61.031 |

## Key Findings

### LSTM B0
- **Pattern**: Early convergence (epoch 6) with subsequent overfitting-like development
- **Action**: Monitor regularization factors (S9, S10)

### Transformer B0
- **Pattern**: Late convergence (epoch 12) with healthy learning dynamics
- **Action**: No immediate concerns

## Hypotheses Generated

| ID | Factor | Pre-registered Phase | Confidence |
|----|--------|---------------------|------------|
| HYP_001 | Regularization (WD, Dropout) | S9, S10 | MEDIUM |
| HYP_002 | Epoch Budget | S16 | MEDIUM |
| HYP_003 | Gradient Clipping | S17 | MEDIUM |

## Test Status

- **Status**: LOCKED
- **Test Access**: FORBIDDEN
- **Clearance**: PASS (no critical findings)

## Upstream Contracts

Required contracts:
- `LSTM_BASELINE-v1` (Phase 20)
- `TRANSFORMER_B0-v1` (Phase 21)

## Downstream

Phase 23 (S1 Feature-Set Sweep) requires `LEARNING_DIAGNOSTICS-v1` clearance.

## Files

- `learning_diagnostics_run_contract.json` - Diagnostic contract
- `learning_diagnostics_summary.json` - Diagnostic summary
- `learning_diagnostics_audit.csv` - Integrity checks
- `learning_diagnostics_discrepancies.json` - Discrepancies
- `phase_22_signoff.json` - Phase signoff
