# S17 Gradient-Clipping Sweep

This directory contains canonical Phase 39 evidence for GC0/GC1 under fixed D64, H4, head_dim16, N2, F256.
Only gradient clipping differs between conditions. All other variables are frozen from Phase 38.

GC0: clipping OFF (TRAIN_NEW)
GC1: global L2 max_norm=1.0 (REUSE_REFERENCE)

Strict full-precision Validation RMSE selects the winner, with GC0 winning only on exact tie.
