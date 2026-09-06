"""
Pipeline execution scripts organized by phase.

This package contains all official pipeline runner scripts:
- p39_*: Phase 39 - Compliance and sweep cleanup
- p40_*: Phase 40 - RevIN preparation and strict best
- p42_*: Phase 42 - Candidate synthesis
- p43_*: Phase 43 - LSTM tuning and harness
- p44_*: Phase 44 - Rolling origin robustness
- p45_*: Phase 45 - Final model lock
- p46_*: Phase 46 - Three seed runs and verification
- p47_*: Phase 47 - Final test evaluation

Utility scripts (run_*, setup_*, recover_*, sweep_*) remain in the top-level scripts/ directory.
"""

__all__ = [
    # Phase 39
    "p39_cleanup_unauthorized_runs",
    "p39_compliance_corrective", 
    "p39_finalize",
    "p39_strict_best_gc0",
    # Phase 40
    "p40_finalize",
    "p40_prepare_revin",
    "p40_strict_best_rn1",
    # Phase 42
    "p42_candidate_synthesis",
    # Phase 43
    "p43_disposable_harness",
    "p43_dry_run",
    "p43_lstm_tuning",
    "p43_one_epoch_harness",
    "p43_stage_simulation",
    # Phase 44
    "p44_pretrain_gate",
    "p44_quarantine_invalid_official",
    "p44_rolling_origin",
    # Phase 45
    "p45_final_model_lock",
    "p45_pretrain_gate",
    # Phase 46
    "p46_archive_historical_checkpoints",
    "p46_final_dev_metric_smoke_test",
    "p46_pre_train_schema_simulation",
    "p46_pretrain_gate",
    "p46_registration_preflight",
    "p46_three_seed_runs",
    # Phase 47
    "p47_final_test_evaluation",
    "p47_pretest_gate",
]
