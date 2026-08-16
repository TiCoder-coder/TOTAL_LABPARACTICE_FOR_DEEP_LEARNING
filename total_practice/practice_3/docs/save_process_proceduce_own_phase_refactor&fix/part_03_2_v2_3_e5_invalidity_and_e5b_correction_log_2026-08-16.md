# Practice 3 v2.3 — E5 Invalidity Finding & E5b Correction Log

**Date:** 2026-08-16

## Finding: E5_classifier_dropout_0.20 is an Invalid Controlled Experiment

### Root Cause (Two Independent Failures)

**Failure 1 — Baseline Already at Target Value:**
The pretrained `distilbert-base-uncased` model ships with `seq_classif_dropout = 0.20` hardcoded in its `config.json`. This is the standard HuggingFace DistilBERT default. The E5 experiment was designed to test this same value (0.20), meaning _zero change was introduced_ relative to the baseline. The experiment had no hypothesis to test.

**Failure 2 — Runner Did Not Apply Config:**
Even if E5 had targeted a different value, `experiment_runner_v2_3.py` at the time did not read `classifier_dropout` from the experiment config and did not pass `seq_classif_dropout` as a keyword argument to `AutoModelForSequenceClassification.from_pretrained()`. The config field existed only in the JSON file but was never transmitted to the model at training time.

### Runtime Evidence
Verified by inference-only model inspection:
```
E1 model.config.seq_classif_dropout = 0.20  (default, no override)
E4 model.config.seq_classif_dropout = 0.20  (default, no override)
E5 model.config.seq_classif_dropout = 0.20  (passed 0.20 = no change)
```

This explains why E5 result (val_loss=0.41570) is virtually identical to E1 (val_loss=0.41563): they ran the exact same model.

### Disposition of E5
E5 is **preserved as execution evidence** with its status set to `COMPLETED_BUT_INVALID_CONTROLLED_EXPERIMENT`. It is excluded from the final valid hyperparameter ranking but is retained for full auditability and to document this finding.

---

## Correction: E5b_classifier_dropout_0.40

### Design
- **Baseline reference:** E1_lr_1e-5 (Stage A winner)
- **Controlled variable:** `seq_classif_dropout`
- **Baseline value:** 0.20 (DistilBERT default)
- **E5b value:** 0.40 (doubling the baseline)
- **Why 0.40:** Large enough to create measurable regularization without preventing the classifier head from learning (which would occur at values approaching 1.0). The step is clearly distinguishable from noise and is a standard exploratory choice.

All other parameters remain identical to E1: `learning_rate=1e-5`, `weight_decay=0.01`, `max_epochs=15`, `patience=4`, `warmup_steps=720`, `label_smoothing=0.1`, `max_grad_norm=1.0`, same split, same tokenizer, same seed=42.

### Runner Fix
`experiment_runner_v2_3.py` was updated to:
1. Read `seq_classif_dropout` from the experiment config if present.
2. Pass it as a kwarg to `from_pretrained()`, overriding the pretrained default.
3. Assert at runtime that `model.config.seq_classif_dropout == config["seq_classif_dropout"]`, failing closed if there is any mismatch.

This fix is **backward-compatible**: E1/E2/E3/E4 configs do not contain `seq_classif_dropout`, so the runner continues to use the pretrained default (0.20) for them — no retroactive change to their model behavior.

### Zero-Training Validation (ZTV)
E5b underwent a successful inference-only forward pass:
- `model.config.seq_classif_dropout = 0.40` ✓
- `model.dropout.p = 0.40` ✓
- Loss finite, logits shape `(16, 2)` ✓
- `Trainer.train()` not called ✓
