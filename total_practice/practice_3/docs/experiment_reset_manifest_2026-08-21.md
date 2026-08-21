# Practice 3 Experiment Reset Manifest — 2026-08-21

This manifest records the destructive reset authorized for the active Practice
3 v2.3 experiment workflow. It is intentionally Git-readable and contains no
model weights.

## Pre-reset inventory

- `runs/practice_3_v2_3/`: 460 files, approximately 36 GB.
- `docs/result/practice_3_v2_3/`: 78 files, approximately 257 MB.
- `save_logs/`: 3 files, approximately 204 KB.

## Experiment identities present before reset

| Experiment | Pre-reset config SHA256 |
|---|---|
| E1_lr_1e-5 | `87596fc2ddd83be9b54f0f67bb3602bd13033d3dbdf023e911df48258d6568e8` |
| E2_lr_2e-5 | `f231c4c87bca25e96128ec4dbd33c1968f56b89ca81e296b0c096c459c273f61` |
| E3_lr_3e-5 | `2904b5f96bfdc555a0189af08fd0bcefde0cae0401778d13a965273be292e40f` |
| E4_weight_decay_0.05 | `05f00d0e26e0a6f70d493e6fb7d85748bcffebc84db7dba3e93c5ab9c2a838c7` |
| E5_classifier_dropout_0.20 | `330ab151d2972fc9d790b0640c3c8b92b13da4a6c234ffd5b426d875ef6c5a2a` |
| E5b_classifier_dropout_0.40 | `31e0f28c463de1be4897fcded8176228944e2b668474cf5faf1fcc60d0999a35` |
| E6_staged_finetune | `ebe44efcc2416774a4b480bfd07e5136fe93e7940ade224253cbba8bc7f7256f` |
| E6b_staged_finetune | `84eb897dacdace934847cbbedf383c9bfbefa64941eeabe53af59d441628d664` |
| E6c_staged_finetune | `c3598a852c4f0d3afd2defbf5c55987d8aa0aa2772ebdc0f26bd639e381ad834` |
| E7_weight_decay_0.10 | `ccea2ee021a42bd05903018c2ba8ab500f68b8cca99b68b3de0407dd901ce7cd` |

## Exact deletion targets

The following complete directory trees are reset. Every file below each path,
including checkpoints, optimizer/scheduler state, Trainer state, TensorBoard
events, histories and summaries, is a deletion target:

- `runs/practice_3_v2_3/`
- `save_logs/`
- `docs/result/practice_3_v2_3/experiments/`
- `docs/result/practice_3_v2_3/figures/`
- `docs/result/practice_3_v2_3/final_saved_model/`

The following old result/report artifacts are also deletion targets:

- `custom_inference_analysis.md`
- `custom_inference_inputs.json`
- `custom_inference_results.csv`
- `custom_inference_results.json`
- `excluded_experiments.json`
- `final_holdout_artifact_manifest.json`
- `final_holdout_error_analysis.md`
- `final_holdout_evaluation.md`
- `final_holdout_metrics.json`
- `final_saved_model_manifest.json`
- `final_summary.md`
- `final_validation_experiment_analysis.md`
- `final_validation_ranking.csv`
- `final_validation_ranking.json`
- `final_validation_winner_lock.json`
- `high_confidence_holdout_errors.csv`
- `holdout_claim.json`
- `holdout_errors.csv`
- `holdout_predictions.csv`
- `parameter_usage_audit.md`
- `save_reload_comparison.csv`
- `save_reload_verification.json`
- `save_reload_verification.md`
- `stage_a_selection_report.json`
- `staged_finetune_validation.json`
- `validation_vs_holdout_comparison.json`
- `experiment_registry.json.lock`

The stale post-training phase reports are also deletion targets because they
describe results/checkpoints that no longer exist:

- `docs/result/phase_08_hyperparameter_search_results.md`
- `docs/result/phase_09_learning_curve_results.md`
- `docs/result/phase_10_holdout_evaluation_results.md`
- `docs/result/phase_11_error_analysis_results.md`
- `docs/result/phase_12_custom_inference_results.md`
- `docs/result/phase_13_save_reload_verification_results.md`

## Preserved foundations

- Source modules, including the three repaired modules.
- Dataset split identity/fingerprint and tokenizer validation evidence.
- Protocol documentation and zero-training validation evidence.
- Exercise 1 pretrained-sentiment artifact.
- Notebook source cells; stale executed outputs are cleared separately.

## Clean active sequence after reset

`E1_lr_1e-5 → E2_lr_2e-5 → E3_lr_3e-5 → E4_weight_decay_0.05 → E5_classifier_dropout_0.40 → E6_staged_finetune`

All six records start at `PLANNED`. E5 adopts the effective E5b dropout
configuration; E6 adopts the corrected E6c staged configuration.
