"""Phase 54 — Last-Query Attention Analysis (LAST_QUERY_ATTENTION-v1).

Phase 54 is a strictly quantitative last-query attention analysis over the
frozen Phase 52 raw last-query NPZ for FINAL_TEST_POP-v1 (N=2961) × 3 seeds ×
2 layers × 4 heads.

Hard rules (per architecture_rule.md v1.16 and Phase 54 detail):
* Raw source = Phase 52 float32 last-query NPZ only; NO PNG digitization.
* Last query = A[:, :, L-1, :].
* Raw axis order = [target, layer, head, source].
* Source position 0 = oldest; source position L-1 = newest.
* Forecast target is NOT an attention token.
* Lag mapping = H + (L-1-p), H = 1.
* No new attention extraction, no Test inference, no model load, no training,
  no scaler fitting, no optimizer/backward, no checkpoint reload.
* No best-seed/head selection; no head clustering; no head ablation;
  no cross-seed head matching; no error-conditioned analysis;
  no regime-conditioned analysis; no feature-importance or causal claims.
* Phase 55-57 implementation is UNAUTHORIZED. Only handoff files may be written.
* Notebook modification is FORBIDDEN (Phase 54-H deferred).
"""

from .sources import FrozenSources54, load_frozen_sources_54, load_last_query_per_seed, lag_steps_array
from .contract import AnalysisContract, build_analysis_contract
from .integrity import (
    per_vector_integrity_audit,
    phase52_summary_reconstruction_audit,
    target_order_audit,
    lag_mapping_audit,
)
from .metrics import (
    compute_vector_metrics,
    compute_top1_with_tie_rule,
    compute_top5_mass,
    compute_recent_masses,
)
from .coverage import (
    compute_coverage_radii,
    compute_lag_bin_masses,
    NON_OVERLAP_BINS,
)
from .aggregations import (
    build_per_head_metric_summary,
    build_temporal_profiles_by_lag,
)
from .profiles import (
    build_layer_head_mean_profile,
    build_seed_overall_profile,
)
from .report_cases import resolve_report_cases, write_report_case_artifacts
from .figures import (
    render_all_core_figures,
    render_report_case_figures,
)
from .findings import write_findings
from .discrepancies import write_discrepancies
from .handoffs import write_phase55_handoff, write_phase56_context_handoff, write_phase57_context_handoff
from .signoff import write_phase54_signoff
from .orchestrator import run_phase54
from .finalize_phase54 import finalize_phase54

__all__ = [
    "FrozenSources54",
    "load_frozen_sources_54",
    "AnalysisContract",
    "build_analysis_contract",
    "per_vector_integrity_audit",
    "phase52_summary_reconstruction_audit",
    "target_order_audit",
    "lag_mapping_audit",
    "compute_vector_metrics",
    "compute_top1_with_tie_rule",
    "compute_top5_mass",
    "compute_recent_masses",
    "compute_coverage_radii",
    "compute_lag_bin_masses",
    "NON_OVERLAP_BINS",
    "build_per_head_metric_summary",
    "build_temporal_profiles_by_lag",
    "build_layer_head_mean_profile",
    "build_seed_overall_profile",
    "resolve_report_cases",
    "write_report_case_artifacts",
    "render_all_core_figures",
    "render_report_case_figures",
    "write_findings",
    "write_discrepancies",
    "write_phase55_handoff",
    "write_phase56_context_handoff",
    "write_phase57_context_handoff",
    "write_phase54_signoff",
    "run_phase54",
    "finalize_phase54",
]
