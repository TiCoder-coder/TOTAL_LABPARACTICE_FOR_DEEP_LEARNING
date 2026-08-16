# Practice 3 Result Provenance

JSON verification/manifest files are the source of truth. Phase result Markdown pages are generated from those JSON files by `processing_own_phase/result_documentation.py`; they provide navigation and interpretation, not a second metric source.

| Phase | Result page | Authoritative evidence |
|---:|---|---|
| 0 | Notebook Phase 0 | `phase_00_pipeline_uml.dot` |
| 1 | Notebook Phase 1 | `2026-08-10_phase01-environment-log.json` |
| 2 | Notebook Phase 2 | No standalone artifact |
| 3 | Notebook Phase 3 | No standalone artifact |
| 4 | `phase_04_result.md` | `phase_04_dataset_summary.json` |
| 5 | `phase_05_result.md` | `phase_05_eda_summary.json`, token statistics, EDA figures |
| 6 | `phase_06_result.md` | `phase_06_preprocessing_verification.json` |
| 7 | `phase_07_result.md` | `phase_07_model_verification.json` |
| 8 | `phase_08_result.md` | `phase_08_metrics_training_configuration_verification.json` |
| 9 | `phase_09_result.md` | `phase_09_training/phase_09_training_manifest.json` and related records/checkpoints |
| 10 | `phase_10_result.md` | `phase_10_learning_curve_analysis.json` and two figures |
| 11 | `phase_11_result.md` | `phase_11_evaluation/phase_11_evaluation_manifest.json` and frozen evaluation/predictions |
| 12 | `phase_12_result.md` | `phase_12_error_analysis.json`, confusion matrix, deterministic samples |
| 13 | `phase_13_result.md` | `phase_13_inference_examples.json` |
| 14 | `phase_14_result.md` | `phase_14_save_reload_verification.json` and package manifest |
| 15 | `phase_15_result.md` | `phase_15_final_summary.json` |

Trace a result in this order:

```text
phase source module → authoritative artifact → notebook phase output → phase result page
```

Finalization of `docs/current_flow/` is deferred until Practice 3 Protocol v2 is implemented.

