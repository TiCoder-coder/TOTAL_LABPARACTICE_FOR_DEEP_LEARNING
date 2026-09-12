# Step 17 — POST_HOC_V2_BENCHMARK Pre-Process Plan

## 1. Status and label

This is a preparation contract only. `BENCHMARK_AUTHORIZED = NO` and
`TEST_ACCESS_AUTHORIZED = NO`. A future Human-authorized execution must be
labelled **POST_HOC_V2_BENCHMARK**. The old Test set was already observed in
the V1 lineage, so the result must never be described as an unbiased unseen
Test evaluation.

## 2. Immutable source lock

The sole model source is
`artifacts/model_improvement_v2/final_model_lock/v2_final_model_lock.json`.
Execution is allowed only after its SHA-256, internal lock fingerprint,
`LOCKED` status, and `step17_ready=true` are verified. The benchmark uses only
the three Step 16 final-refit checkpoints for seeds 42, 123, and 2026. Each
checkpoint path, SHA-256, run ID, seed, epoch 16, config fingerprint, and
scaler SHA-256 must match the lock before Test is opened.

## 3. Prediction policy

The locked policy is:

`prediction = mean(seed42, seed123, seed2026)`

with immutable weights `[1/3, 1/3, 1/3]`. Per-seed results are reported, but
best-seed selection is forbidden. No model, seed, epoch, alpha, feature,
scaler, hyperparameter, or ensemble weight may change after Test access.

## 4. Data and scaler contract

- Feature variant: `FS2_TF1`, exact canonical order, 33 features.
- Lookback/horizon: 72/1.
- Boundary: `WB0_CONTEXT_CARRY_OVER`.
- The exact locked Step 16 X/Y scaler JSON artifacts are transform-only.
- No scaler fitting, partial fitting, reconstruction, or Test statistics.
- The old Test rows are opened once by the V2 Step 17 loader and retained in
  memory for all three seeds, the ensemble, and fixed persistence.
- Every output must share identical ordered target IDs, timestamps, targets,
  and population fingerprint.

## 5. Evaluation

Each checkpoint is loaded strictly, placed on the selected device, switched
to `eval()`, and evaluated under `torch.inference_mode()`. Metrics in raw Wh
are RMSE, MAE, and R² for each seed and the equal-weight ensemble. The fixed
`PERSISTENCE_LAST_VALUE` predictor is evaluated on the same targets without
tuning. Parameter count and measured inference runtime are reported. Peak
memory is not claimed because the canonical engine does not provide a
portable locked measurement contract.

The V1 `final_test_summary.json` may be copied as a read-only historical
reference only after explicit Step 17 authorization. It cannot select or
modify the V2 model and is labelled historical evidence.

## 6. Test-access gate

Preparation and preflight must report zero Test files opened and zero
checkpoint payloads loaded. Benchmark mode requires both explicit CLI flags:

- `--authorize-post-hoc-benchmark`
- `--acknowledge-old-test-not-unseen`

Before opening Test, the runner writes an access event. If an access event or
completed result already exists, execution refuses a second access and
requires a read-only recovery audit.

## 7. Output namespace

All new files are confined to:

`artifacts/model_improvement_v2/post_hoc_v2_benchmark/`

Expected authorized outputs are the access event, one prediction CSV per
seed, ensemble and persistence prediction CSVs, per-model metrics, benchmark
summary, provenance/checksum manifest, and an immutable signoff. No V1 or
Step 16 artifact is modified.

## 8. Hard stops

Stop before Test access on any lock/checkpoint/config/scaler mismatch, wrong
feature order, wrong ensemble policy, existing access event, output namespace
collision, or unresolved authorization. Stop after Test access on any target
population mismatch, non-finite prediction, or inference error. Do not
automatically rerun after Test was opened.

## 9. Definition of ready

Ready for the separate Human benchmark gate only when static validation,
focused tests, V2 regression tests, and Step 17 preflight pass without Test
access, inference, checkpoint-payload loading, or training.
