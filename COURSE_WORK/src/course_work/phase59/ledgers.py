# -*- coding: utf-8 -*-
"""Phase 59 — outcome matrix + limitation + future-work ledgers."""

from __future__ import annotations

from . import constants as C


# ---------------------------------------------------------------------------
# Outcome matrix (12 topics × {outcome_status, evidence, claim_level, caveat})
# ---------------------------------------------------------------------------

def build_outcome_matrix() -> list[dict]:
    """Per-topic outcome record (12 topics)."""
    topics = [
        ("Final Test performance", C.SUPPORTED, "FT02",
         "phase_47_test_metrics", C.LEVEL_3,
         "Three-seed mean ± sample SD held-out Test metrics reported from FT02.",
         "Three-seed summary is descriptive, NOT an ensemble."),
        ("Transformer vs Persistence", C.MIXED, "FT02",
         "phase_47_test_metrics", C.LEVEL_3,
         "Persistence and final Transformer RMSE compared; direction depends on the metric.",
         "Both rows reported even when unfavorable."),
        ("Transformer vs LSTM", C.MIXED, "FT02",
         "phase_47_test_metrics", C.LEVEL_3,
         "Tuned LSTM was not evaluated on FINAL_TEST_POP-v1 due to lookback mismatch.",
         "LSTM absence is documented as an upstream protocol decision, not a Phase 59 choice."),
        ("Temporal robustness", C.PARTIALLY_SUPPORTED, "FT03",
         "phase_44_rolling_origin", C.LEVEL_2,
         "Pooled RMSE and per-fold variability reported as development evidence.",
         "Rolling-origin is DEVELOPMENT_EVIDENCE only."),
        ("Error concentration", C.SUPPORTED, "FT05",
         "phase_51_worst_case", C.LEVEL_1,
         "A small subset of Test observations contributes a disproportionate share of squared error.",
         "Worst cases retained as valid observations."),
        ("Regime difficulty", C.PARTIALLY_SUPPORTED, "FT05",
         "phase_50_regime", C.LEVEL_1,
         "Forecast accuracy varied across frozen Train-defined regimes.",
         "Regime does NOT cause error."),
        ("Recent-history attention", C.SUPPORTED, "FT06",
         "phase_54_last_query", C.LEVEL_2,
         "Substantial last-query mass allocated to recent positions (recent-1h, recent-6h).",
         "Lookback = 72 = 12 h truncates 24-h mass."),
        ("Long-history attention", C.PARTIALLY_SUPPORTED, "FT06",
         "phase_54_last_query", C.LEVEL_2,
         "Long-history mass depends on the layer and head.",
         "Lag 24-h mass is truncated by lookback."),
        ("Head diversity", C.SUPPORTED, "FT07",
         "phase_55_head_comparison", C.LEVEL_1,
         "Different heads within each layer learned non-identical temporal profiles.",
         "Similarity ≠ functional redundancy."),
        ("Error-attention association", C.PARTIALLY_SUPPORTED, "FT08",
         "phase_56_error_conditioned", C.LEVEL_1,
         "Attention metrics were associated with forecast-error magnitude.",
         "Associations are descriptive; HIGH/LOW are NOT deployment regimes."),
        ("Attention seed stability", C.PARTIALLY_SUPPORTED, "FT09",
         "phase_57_seed_stability", C.LEVEL_2,
         "Layer head-mean evidence more consistent across seeds than individual matched-head.",
         "Same-index heads NOT semantically equivalent; matching ambiguity surfaced."),
        ("Head semantic stability", C.PARTIALLY_SUPPORTED, "FT09",
         "phase_57_seed_stability", C.LEVEL_2,
         "Cycle consistency is partial (Layer 0 = 1/4; Layer 1 = 4/4).",
         "Three seeds provide limited stochastic evidence; cycle inconsistency limits head-level semantic interpretation."),
    ]
    rows = []
    for topic, status, table, src, lvl, summary, caveat in topics:
        rows.append({
            "topic": topic,
            "outcome_status": status,
            "primary_evidence": table,
            "secondary_evidence": src,
            "claim_level": lvl,
            "summary": summary,
            "caveat": caveat,
            "status": "ACTIVE",
        })
    return rows


# ---------------------------------------------------------------------------
# Limitation ledger (L1..L6)
# ---------------------------------------------------------------------------

def build_limitation_ledger() -> list[dict]:
    """Frozen canonical limitation categories with mandatory-in-report flags."""
    rows = [
        {
            "limitation_id": "L1A_DATASET_SINGLE_HOUSE",
            "category": C.LIMITATION_CATEGORIES[0],
            "limitation": (
                "Single-house UCI Appliances dataset; no multi-house, multi-climate, "
                "or multi-building validation."
            ),
            "source": "FA12 + Phase 0-43 dataset documentation",
            "why_it_matters": "External validity cannot be established from a single dwelling.",
            "claim_scope_restricted": "Performance / attention findings cannot be extrapolated to other households.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW2",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L1B_DATASET_TEMPORAL_SPAN",
            "category": C.LIMITATION_CATEGORIES[0],
            "limitation": "Limited temporal span; no seasonal coverage beyond the recorded months.",
            "source": "FA12 + Phase 0-43 dataset documentation",
            "why_it_matters": "Year-round / multi-seasonal behavior is unobserved.",
            "claim_scope_restricted": "Findings cannot be extended to other seasons / climates.",
            "mandatory_main_text": "NO",
            "future_work_id": "FW2",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L2A_FORECAST_H1",
            "category": C.LIMITATION_CATEGORIES[1],
            "limitation": "H=1 one-step-ahead only (10 min); no recursive multi-step evaluation.",
            "source": "FT01 + FA12",
            "why_it_matters": "Long-horizon forecasting requires a different protocol.",
            "claim_scope_restricted": "Performance does not characterize multi-step forecasts.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW1",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L2B_WB0",
            "category": C.LIMITATION_CATEGORIES[1],
            "limitation": (
                "Final primary boundary is WB0: one-step forecasting assumes previously "
                "observed target values become available for subsequent predictions."
            ),
            "source": "FT01 + FA12",
            "why_it_matters": "Open-loop multi-step deployment is not what the current model targets.",
            "claim_scope_restricted": "Cannot imply fully open-loop multi-step forecasting.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW1",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L3A_SEQUENTIAL_TUNING",
            "category": C.LIMITATION_CATEGORIES[2],
            "limitation": (
                "Sequential one-factor tuning rather than full Cartesian / global "
                "hyperparameter search; selected hyperparameters do not guarantee the "
                "global optimum."
            ),
            "source": "Phase 43 + FT01",
            "why_it_matters": "Alternative tuning strategies could yield different selected hyperparameters.",
            "claim_scope_restricted": "Cannot claim optimal hyperparameters.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW3",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L3B_THREE_SEEDS",
            "category": C.LIMITATION_CATEGORIES[2],
            "limitation": (
                "Only three final seeds (42, 123, 2026); SD is descriptive only "
                "and does not characterize the full distribution over random initializations."
            ),
            "source": "FT01 + FT02 + FA12",
            "why_it_matters": "Three-seed SD is a limited stochastic robustness check.",
            "claim_scope_restricted": "Stability findings are limited to the three predefined seeds.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW4",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L3C_PRE_TEST_EVIDENCE_REUSE",
            "category": C.LIMITATION_CATEGORIES[2],
            "limitation": (
                "Pre-Test development evidence (rolling-origin, validation) was reused "
                "for model selection; final Test was evaluated once after lock."
            ),
            "source": "FT01 + FT03 + FA12",
            "why_it_matters": "Test is single-shot after lock; cannot be re-evaluated under redesign.",
            "claim_scope_restricted": "Cannot be used as a retuning signal.",
            "mandatory_main_text": "NO",
            "future_work_id": "FW4",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L4A_TIME_SERIES_DEPENDENCE",
            "category": C.LIMITATION_CATEGORIES[3],
            "limitation": (
                "Time-series dependence; naive iid significance claims are not warranted. "
                "Diagnostics are descriptive."
            ),
            "source": "FT04 + FA12",
            "why_it_matters": "Cannot use 'statistically significant' without a formal test.",
            "claim_scope_restricted": "Effects reported as descriptive only.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW7",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L4B_NO_FORMAL_UNCERTAINTY",
            "category": C.LIMITATION_CATEGORIES[3],
            "limitation": (
                "No formal confidence intervals; the three-seed SD is descriptive only."
            ),
            "source": "FT02 + FA12",
            "why_it_matters": "Cannot quote confidence intervals without an upstream test.",
            "claim_scope_restricted": "Effects reported as descriptive only.",
            "mandatory_main_text": "NO",
            "future_work_id": "FW9",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L5A_ATTENTION_TEMPORAL_ALLOCATION",
            "category": C.LIMITATION_CATEGORIES[4],
            "limitation": (
                "Attention describes temporal token allocation only — NOT raw-feature "
                "importance and NOT causal explanation."
            ),
            "source": "FT06/FT07/FT08/FT09 + FA12",
            "why_it_matters": "Misreading attention as feature importance or as causal would overclaim.",
            "claim_scope_restricted": "Cannot be used for feature importance / causal claims.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW5",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L5B_HEAD_MATCHING_NOT_FUNCTIONAL",
            "category": C.LIMITATION_CATEGORIES[4],
            "limitation": (
                "Head-profile similarity does NOT prove functional equivalence; "
                "value / output projections can differ; matching ambiguity and partial "
                "cycle consistency are surfaced."
            ),
            "source": "FT07 + FT09 + FA12",
            "why_it_matters": "Head pruning / ablation implication would overclaim.",
            "claim_scope_restricted": "No head-pruning recommendation.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW5",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L5C_LAST_QUERY_ONE_VIEW",
            "category": C.LIMITATION_CATEGORIES[4],
            "limitation": (
                "Last-query is one token-level view; full attention maps are available "
                "only for the dense Phase 51 case set."
            ),
            "source": "FT06 + FA12",
            "why_it_matters": "Last-query may hide multimodality or layer interactions.",
            "claim_scope_restricted": "Full-map interpretations must reference dense-case analysis.",
            "mandatory_main_text": "NO",
            "future_work_id": "FW5",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L6A_DEPLOYMENT_GENERALIZATION",
            "category": C.LIMITATION_CATEGORIES[5],
            "limitation": (
                "No prospective deployment, no online adaptation, no multi-house "
                "validation, no computational latency benchmark."
            ),
            "source": "FA12 + FT01",
            "why_it_matters": "Deployment claims require separate evidence.",
            "claim_scope_restricted": "Cannot be marked as deployment-ready.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW8",
            "status": "ACTIVE",
        },
        {
            "limitation_id": "L6B_EXTERNAL_GENERALIZATION",
            "category": C.LIMITATION_CATEGORIES[5],
            "limitation": (
                "Results cannot establish generalization to other households, buildings, "
                "climates, or energy systems without external validation."
            ),
            "source": "FA12",
            "why_it_matters": "External generalization is out of scope of the current project.",
            "claim_scope_restricted": "Cannot be generalized externally.",
            "mandatory_main_text": "YES",
            "future_work_id": "FW2",
            "status": "ACTIVE",
        },
    ]
    return rows


# ---------------------------------------------------------------------------
# Future-work ledger (linked to limitations)
# ---------------------------------------------------------------------------

def build_future_work_ledger() -> list[dict]:
    """Future-work items explicitly labeled as not performed; each linked to a limitation."""
    rows = [
        {
            "future_work_id": "FW1_MULTI_STEP",
            "future_work": (
                "Direct multi-horizon, recursive, seq2seq / decoder, or probabilistic "
                "multi-step forecasting for H>1."
            ),
            "linked_limitation_id": "L2A_FORECAST_H1",
            "motivation": "Current scope is H=1; multi-step deployment needs a new protocol.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P2",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW2_EXTERNAL_VALIDATION",
            "future_work": (
                "Evaluate on additional households / buildings, different seasons / climates, "
                "and out-of-distribution periods."
            ),
            "linked_limitation_id": "L1A_DATASET_SINGLE_HOUSE",
            "motivation": "Current single-house dataset cannot establish external validity.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P1",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW3_ADDITIONAL_BASELINES",
            "future_work": "Additional model baselines (e.g. CNN, N-BEATS, classical statistical baselines).",
            "linked_limitation_id": "L3A_SEQUENTIAL_TUNING",
            "motivation": "Current comparison set is Persistence + tuned LSTM + Transformer.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P5",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW4_MORE_SEEDS",
            "future_work": (
                "More seeds and stronger uncertainty characterization (e.g. tens of seeds, "
                "bootstrap CI on rolling-origin folds)."
            ),
            "linked_limitation_id": "L3B_THREE_SEEDS",
            "motivation": "Three-seed SD is descriptive only.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P4",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW5_HEAD_ABLATION_ATTRIBUTION",
            "future_work": (
                "Head ablation, Integrated Gradients or SHAP-style feature attribution, "
                "attention rollout, counterfactual temporal masking, and value-path analysis."
            ),
            "linked_limitation_id": "L5A_ATTENTION_TEMPORAL_ALLOCATION",
            "motivation": "Attention is descriptive; functional attribution requires explicit ablation.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P3",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW6_FEATURE_ATTRIBUTION",
            "future_work": "Feature-level attribution methods on the input side.",
            "linked_limitation_id": "L5A_ATTENTION_TEMPORAL_ALLOCATION",
            "motivation": "Attention is not feature importance.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P3",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW7_BLOCK_AWARE_INFERENCE",
            "future_work": (
                "Temporal block-aware statistical inference (e.g. block-bootstrap, Newey-West "
                "covariance) to address time-series dependence."
            ),
            "linked_limitation_id": "L4A_TIME_SERIES_DEPENDENCE",
            "motivation": "Current diagnostics are descriptive; formal inference needs block-aware methods.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P4",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW8_ONLINE_DEPLOYMENT",
            "future_work": "Online / rolling deployment evaluation with prospective data.",
            "linked_limitation_id": "L6A_DEPLOYMENT_GENERALIZATION",
            "motivation": "No prospective deployment is part of this project.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P2",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW9_UNCERTAINTY_FORECASTING",
            "future_work": "Uncertainty-aware forecasting (quantile / probabilistic outputs).",
            "linked_limitation_id": "L4B_NO_FORMAL_UNCERTAINTY",
            "motivation": "Current protocol produces point forecasts only.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P3",
            "status": "FUTURE",
        },
        {
            "future_work_id": "FW10_EFFICIENCY_LATENCY",
            "future_work": "Efficiency / latency analysis on representative hardware.",
            "linked_limitation_id": "L6A_DEPLOYMENT_GENERALIZATION",
            "motivation": "Deployment latency / efficiency is not benchmarked.",
            "not_performed_in_current_project": "TRUE",
            "requires_new_evaluation_cycle": "TRUE",
            "priority": "P5",
            "status": "FUTURE",
        },
    ]
    return rows
