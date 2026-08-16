# PART 3.2.E — Root-cause Correction Diagnosis Plan

Date: 2026-08-15  
Purpose: isolate the root-cause category before any `practice_3_v2.1` proposal.

## Objective

Compare a minimal direct PyTorch update loop with Hugging Face Trainer using an identical, known-easy, Train-only classification fixture and identical model initialization/configuration. Determine whether the observed no-learning behavior belongs primarily to Trainer integration, the shared model/data/loss path, the installed environment stack, or remains unresolved.

## Safety boundary

- Do not invoke the official v2 runner.
- Do not read/write the real registry or any official experiment directory.
- Do not access v2 Validation, Holdout, or official Test.
- Do not modify frozen protocol/configuration.
- Place all runtime output beneath an auto-removed `/private/tmp` directory.
- Persist only this plan and the final analysis document.

## Known-easy fixture

Create 64 deterministic synthetic sentences, balanced 32/32. Negative texts repeatedly contain an explicit `terrible negative bad` signal; positive texts repeatedly contain an explicit `excellent positive good` signal. Small deterministic suffix variations preserve unique records without weakening the label signal. Validate counts, text uniqueness, labels `{0,1}`, tokenized labels, and fixture SHA-256.

The fixture is synthetic by design and is not an official metric. Its only purpose is to prove that the training mechanisms can memorize an obvious binary signal.

## Controlled configuration

- CPU only
- `distilbert/distilbert-base-uncased`
- fresh deterministic initialization, seed 42
- max length 80, dynamic padding
- frozen/eval DistilBERT backbone; train `pre_classifier` and `classifier`
- full-fixture batch size 64
- AdamW, constant LR `1e-3`, no weight decay
- 100 update steps
- no scheduler decay, evaluation, checkpoint, logging backend, or mixed precision

Freezing the backbone makes this a focused training-mechanism test: pretrained features and the new classification head remain common to both branches while runtime is bounded. Both branches must start with matching fingerprints and initial predictions/loss.

## Direct PyTorch branch

For each step: `model.train()` while keeping the frozen backbone in eval mode, forward, loss, `zero_grad`, backward, record gradient norms, clip gradients, and `optimizer.step`. Record initial/final loss, accuracy, class counts, logits statistics, head gradients and parameter deltas.

## Hugging Face Trainer branch

Use the exact same fixture, seed, initialization, trainable parameters, LR, batch size and 100 steps. Use a small Trainer subclass/callback only to keep the frozen backbone in eval mode and force deterministic sequential full-batch sampling. Do not change loss or optimizer semantics. Record the same measurements.

## Equivalence and interpretation

- Verify initial model fingerprints and metrics match.
- Success threshold: final Train accuracy >= 90%.
- Direct succeeds / Trainer fails: Trainer integration issue.
- Both fail: shared model/data/loss or environment-stack issue.
- Both succeed: basic stack and Trainer work; official Rotten Tomatoes data/configuration interaction is the likely category.

## Environment audit

Record Python, torch, transformers, datasets, accelerate and tokenizers versions. Compare them against official compatibility metadata/release documentation without changing the environment. Recommend a stable environment revision only if evidence justifies it.

## Output

Create `docs/plan-doc/analysis_error/part_03_2_root_cause_correction_result_2026-08-15.md` with fixture validation, both branches, gradient/logit/parameter evidence, category decision, compatibility notes and integrity checks.

## Completion criteria

The controlled A/B completes or records an exact blocker; root-cause category is selected only when evidence supports it; official state is hash/status verified unchanged; no final correction is implemented.
