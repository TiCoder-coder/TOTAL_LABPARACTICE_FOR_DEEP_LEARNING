"""Phase 44 rolling-origin robustness — reusable core modules."""

from course_work.rolling_origin.populations import (
    ROBASE_VERSION,
    WINDOWPOP_VERSION,
    extract_robase_population,
    compute_population_fingerprint,
    extract_robase_val_ids,
    extract_robase_train_ids,
)
from course_work.rolling_origin.folds import (
    FoldId,
    FoldDefinition,
    build_rolling_folds,
    validate_fold_temporal_ordering,
    serialize_fold_manifest,
)
from course_work.rolling_origin.scaling import (
    FoldLocalScalerBundle,
    fit_fold_a_x_scaler,
    fit_fold_a_y_scaler,
    fit_fold_b_x_scaler,
    fit_fold_b_y_scaler,
    build_bundle,
    serialize_scaler_bundle,
    load_scaler_bundle,
    ScalerFitAudit,
    record_scaler_fit_audit,
)
from course_work.rolling_origin.persistence import (
    PersistenceBundle,
    build_prior_history_lookup,
    compute_persistence_bundle,
)
from course_work.rolling_origin.pooling import (
    PooledMetrics,
    MacroRobustnessMetrics,
    compute_pooled_metrics,
    compute_macro_metrics,
)
from course_work.rolling_origin.ranking import (
    TransformerRankingEntry,
    rank_transformers,
)
from course_work.rolling_origin.refit_engine import (
    RefitEngine,
    StageBResult,
)
from course_work.rolling_origin.stages import (
    StageAResult,
    StageCResult,
    train_stage_a,
    evaluate_stage_c,
    load_fold_subset_loader,
)
from course_work.rolling_origin.artifacts import (
    Phase44Artifacts,
    write_all_artifacts,
    write_manifest,
    write_contract,
    write_fold_manifest,
    write_fold_table,
    write_population_audit,
    write_candidate_matrix,
    write_compatibility_audit,
    write_scaling_contract,
    write_scaler_fit_audit,
    write_temporal_leakage_tests,
    write_common_target_audit,
    write_inner_selection_run_registry,
    write_inner_best_epochs,
    write_refit_run_registry,
    write_refit_epoch_audit,
    write_initialization_audit,
    write_sample_order_audit,
    write_runtime_diagnostics,
    write_outer_predictions,
    write_pooled_predictions,
    write_fold_metrics,
    write_pooled_metrics,
    write_macro_metrics,
    write_pairwise_effects,
    write_fold_ranks,
    write_transformer_ranking,
    write_model_family_comparison,
    write_findings,
    write_recommended_transformer,
    write_phase45_handoff,
    write_tests,
    write_discrepancies,
    write_summary,
    write_report,
    write_readme,
    write_signoff,
)
from course_work.rolling_origin.consistency import (
    Phase44Consistency,
    run_all_consistency_checks,
)
from course_work.rolling_origin.preflight import (
    PreflightResult,
    run_preflight,
)
from course_work.rolling_origin.candidate_loader import (
    CandidateSpec,
    LSTM_CANDIDATE_ID,
    PERSISTENCE_CANDIDATE_ID,
    PERSISTENCE_VERSION,
    load_candidates,
)
from course_work.rolling_origin.payloads import (
    StageRegistrationPayload,
    build_all_payloads,
    build_stage_a_payload,
    build_stage_b_payload,
)
from course_work.rolling_origin.persistence_probe import (
    FirstTargetProbe,
    run_persistence_probe,
)
from course_work.rolling_origin.probe import (
    PopulationProbeResult,
    ScalerProbeResult,
    run_population_probe,
)
from course_work.rolling_origin.pipeline import (
    PipelineConfig,
    PipelineResult,
    run_pipeline,
)
from course_work.rolling_origin.rehearsal import (
    RehearsalResult,
    run_rehearsal,
)
from course_work.rolling_origin.failure_resume import (
    FailureScenario,
    simulate_failure,
)

__all__ = [
    # populations
    "ROBASE_VERSION",
    "WINDOWPOP_VERSION",
    "extract_robase_population",
    "compute_population_fingerprint",
    "extract_robase_val_ids",
    "extract_robase_train_ids",
    # folds
    "FoldId",
    "FoldDefinition",
    "build_rolling_folds",
    "validate_fold_temporal_ordering",
    "serialize_fold_manifest",
    # scaling
    "FoldLocalScalerBundle",
    "ScalerFitAudit",
    "fit_fold_a_x_scaler",
    "fit_fold_a_y_scaler",
    "fit_fold_b_x_scaler",
    "fit_fold_b_y_scaler",
    "build_bundle",
    "serialize_scaler_bundle",
    "load_scaler_bundle",
    "record_scaler_fit_audit",
    # persistence
    "PersistenceBundle",
    "build_prior_history_lookup",
    "compute_persistence_bundle",
    # pooling
    "PooledMetrics",
    "MacroRobustnessMetrics",
    "compute_pooled_metrics",
    "compute_macro_metrics",
    # ranking
    "TransformerRankingEntry",
    "rank_transformers",
    # refit_engine
    "RefitEngine",
    "StageBResult",
    # stages
    "StageAResult",
    "StageCResult",
    "train_stage_a",
    "evaluate_stage_c",
    "load_fold_subset_loader",
    # artifacts
    "Phase44Artifacts",
    "write_all_artifacts",
    "write_manifest",
    "write_contract",
    "write_fold_manifest",
    "write_fold_table",
    "write_population_audit",
    "write_candidate_matrix",
    "write_compatibility_audit",
    "write_scaling_contract",
    "write_scaler_fit_audit",
    "write_temporal_leakage_tests",
    "write_common_target_audit",
    "write_inner_selection_run_registry",
    "write_inner_best_epochs",
    "write_refit_run_registry",
    "write_refit_epoch_audit",
    "write_initialization_audit",
    "write_sample_order_audit",
    "write_runtime_diagnostics",
    "write_outer_predictions",
    "write_pooled_predictions",
    "write_fold_metrics",
    "write_pooled_metrics",
    "write_macro_metrics",
    "write_pairwise_effects",
    "write_fold_ranks",
    "write_transformer_ranking",
    "write_model_family_comparison",
    "write_findings",
    "write_recommended_transformer",
    "write_phase45_handoff",
    "write_tests",
    "write_discrepancies",
    "write_summary",
    "write_report",
    "write_readme",
    "write_signoff",
    # consistency
    "Phase44Consistency",
    "run_all_consistency_checks",
    # preflight / rehearsal
    "PreflightResult",
    "run_preflight",
    "RehearsalResult",
    "run_rehearsal",
]
