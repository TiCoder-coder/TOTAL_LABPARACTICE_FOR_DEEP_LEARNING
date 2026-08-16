# Practice 3 v2.3 — Parameter Usage Audit

**Date:** 2026-08-16

This document lists all configuration parameters configured across historical `experiment_config.json` files and details their true runtime status.

## 1. Minimal Active Parameters (KEEP - ACTIVE)
These parameters actively govern model behavior, optimizer rules, and experimental variations.

- `learning_rate` (Optimizer initial LR)
- `weight_decay` (AdamW L2 Penalty)
- `max_epochs` (Total epochs)
- `batch_size` (Train batch size)
- `eval_batch_size` (Validation batch size)
- `warmup_steps` (Linear scheduler warmup steps)
- `gradient_clipping` (Trainer `max_grad_norm`)
- `label_smoothing_factor` (Loss label smoothing)
- `early_stopping_patience` (Number of epochs to wait for improvement)
- `early_stopping_threshold` (Minimum required delta)
- `seq_classif_dropout` (Canonical dropout kwarg for HF `AutoModel`)
- `attn_implementation` (MPS Execution Compatibility Fix: `eager`)
- `fine_tuning_strategy` (Controls `StagedFinetuningCallback` behavior)

## 2. Safety and Metadata (KEEP - SAFETY)
These ensure the strict protocol and reproducibility of the runs.

- `experiment_id`
- `protocol_version`
- `stage`
- `config_hash`
- `dataset_fingerprint`
- `split_fingerprint`
- `tensorboard_log_dir`
- `library_versions`
- `controlled_variable`
- `best_lr_dependency`
- `status`

## 3. Deprecated and Cleaned Up (REMOVE / NO_EFFECT)
These parameters were found in the historical JSONs but ignored by the runner logic, which hardcoded its own constant requirements. 

- `model`: Was `"distilbert/distilbert-base-uncased"` (Now strictly hardcoded)
- `tokenizer`: Was `"distilbert-base-uncased"` (Now strictly hardcoded)
- `optimizer`: Was `"AdamW"` / `"adamw_torch"` (Now hardcoded directly to `"adamw_torch"` in `TrainingArguments`)
- `scheduler`: Was `"linear"` (Now hardcoded to `"linear"` in `TrainingArguments`)
- `evaluation_strategy`: Was `"epoch"` (Hardcoded)
- `logging_strategy`: Was `"epoch"` (Hardcoded)
- `save_strategy`: Was `"epoch"` (Hardcoded)
- `load_best_model_at_end`: Was `true` (Hardcoded)
- `selection_metric`: Was `"validation_loss"` (Mapped directly to `"eval_loss"`)
- `greater_is_better`: Was `false` (Hardcoded)
- `ranking_tolerance`: (Never actually consumed during Train, only during Ranking)
- `classifier_dropout`: (Dead alias. Replaced universally by `seq_classif_dropout`)

## 4. Staged Fine-Tuning Execution
The `E6c` staged implementation was proven to safely unfreeze parameters without altering optimizer/scheduler constraints. The strategy parameters (layer limits, transition epochs) are baked directly into `StagedFinetuningCallback`, avoiding dynamic runtime issues.

## 5. Historical Integrity Check
**No historical JSONs were touched or mutated.** The `experiment_runner_v2_3.py` script was updated to ignore the `NO_EFFECT` parameters so that new configs do not have to define them. Historical artifacts remain 100% reproducible and cryptographically identical.
