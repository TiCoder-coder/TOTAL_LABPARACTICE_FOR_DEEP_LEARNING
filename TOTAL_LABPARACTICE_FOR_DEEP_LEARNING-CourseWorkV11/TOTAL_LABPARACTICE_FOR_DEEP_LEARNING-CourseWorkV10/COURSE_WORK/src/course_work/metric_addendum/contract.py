from pathlib import Path
from typing import Any

from course_work.evaluation.metrics import build_mape_metric_contract
from course_work.utils.artifacts import canonical_json_bytes, sha256_bytes


ADDENDUM_VERSION = "POSTHOC-MAPE-v1"
ARTIFACT_ROOT = Path("artifacts/metric_addendum/mape")
PROCESS_LOG_PATH = Path("docs/save_log_in_processing/mape_metric_addendum_log.json")
COMPLIANCE_LOG_PATH = Path("docs/save_log_in_processing/ml_pipeline_compliance_audit_log.json")


def build_addendum_contract() -> dict[str, Any]:
    payload = {
        "artifact_version": ADDENDUM_VERSION,
        "metric_contract": build_mape_metric_contract(),
        "scope": "POSTHOC_SUPPLEMENTARY_METRIC",
        "validation_source_policy": "EXISTING_SIGNED_PREDICTION_ARTIFACTS_ONLY",
        "test_source_policy": "EXISTING_FROZEN_CHECKSUM_MATCHED_PREDICTIONS_ONLY",
        "test_missing_policy": "BLOCKED_SOURCE_UNAVAILABLE",
        "selection_metric_unchanged": "rmse_wh",
        "selection_decisions_mutable": False,
        "training_authorized": False,
        "test_inference_authorized": False,
        "checkpoint_loading_authorized": False,
        "scaler_fitting_authorized": False,
        "frozen_phase_mutation_authorized": False,
        "phase_59_narrative_mutation_authorized": False,
    }
    payload["addendum_contract_fingerprint"] = sha256_bytes(canonical_json_bytes(payload))
    return payload
