# Phase 14 — Save & Reload Implementation Log

**Date:** 2026-08-14  
**Scope:** Phase 14 only  
**Final status:** PASS

## Authoritative input

- Checkpoint: `checkpoint-1068`
- Selected epoch: 2
- Selected step: 1068
- Source weight SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`
- Phase 13 custom input count: 4
- Locked preprocessing: maximum length 80, truncation and dynamic padding
- Phase 11 Test evaluation count before Phase 14: 1

## Implementation

Created `processing_own_phase/phase_14_save_reload.py` with:

- authoritative Phase 9/11 checkpoint verification;
- Phase 13 input-signature and preprocessing-contract verification;
- save-once `model.save_pretrained()` and `tokenizer.save_pretrained()` flow;
- temporary-directory save and atomic final-directory publication;
- package file size/SHA-256 inventory;
- explicit exclusion of optimizer, scheduler, RNG, Trainer state and training arguments;
- local-only model/tokenizer reload;
- source/reloaded tokenizer and dynamic-padding equivalence;
- exact state-dictionary tensor comparison;
- logits, probability, label and confidence equivalence checks;
- package guard that verifies and loads without resaving or overwriting.

Updated `notebook_practice_3/practice_3.ipynb` with a thin Phase 14 call and presentation cells only.

## Saved package

Path:

`docs/result/phase_14_saved_model/`

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 762 | `2c9cd5b950a56f19713c5f24c82ab5c77d72b2399940465160d47067020b3cdf` |
| `model.safetensors` | 267,832,560 | `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660` |
| `tokenizer.json` | 711,396 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| `tokenizer_config.json` | 349 | `04c4c3f8c445a1415ccbb7578053de0d66aeeb2f34b2bee6073e599cf39a0516` |

`special_tokens_map.json` was not emitted by the installed tokenizer implementation. Special-token names and IDs were verified before/after local reload through the saved tokenizer artifacts; no file was fabricated.

No training-state file exists in the package.

## Save/reload equivalence

- DistilBERT sequence-classification configuration: PASS
- `num_labels=2`: PASS
- `0 -> NEGATIVE`, `1 -> POSITIVE`: PASS
- Source/reloaded tokenizer IDs: exact equal
- Source/reloaded attention masks: exact equal
- Special tokens and IDs: exact equal
- Dynamic padding and maximum length 80: PASS
- State-dict keys/order/shapes/dtypes: PASS
- Exact-equal state tensors: 104 / 104
- Predicted labels: exact equal for all four inputs
- Maximum logits absolute difference: 0.0 (`<=1e-6`)
- Maximum probability absolute difference: 0.0 (`<=1e-7`)
- Maximum confidence absolute difference: 0.0 (`<=1e-7`)
- Source checkpoint fingerprint unchanged after save/reload: PASS

## Guard issue and resolution

The initial package creation and equivalence run passed. The immediate guard call then produced a false integrity failure because JSON converts integer `id2label` keys to strings. Package hashes, weights and semantic configuration were valid.

The guard comparison now normalizes JSON label keys before semantic comparison. Full evidence is saved in:

`docs/plan-doc/analysis_error/phase_14_guard_json_label_key_mismatch_2026-08-14.md`

Re-verification returned:

`PASS loaded_verified_phase_14_package_no_resave`

The package was not deleted, overwritten or resaved during this correction.

## Result artifacts

- `docs/result/phase_14_saved_model/package_manifest.json`
- `docs/result/phase_14_save_reload_verification.json`

Both contain real executed package/integrity/equivalence evidence.

## Notebook Run All

Executed offline using the repository `.venv` Python 3.11 kernelspec.

- 15 code cells executed continuously with counts 1–15;
- zero error outputs;
- Phase 9 guard loaded artifacts without retraining;
- Phase 11 guard loaded artifacts without Test reevaluation;
- Phase 13 loaded its verified inference artifact;
- Phase 14 guard verified/loaded the package without resaving;
- Phase 11 and Phase 14 both report `test_evaluation_count=1`;
- source checkpoint remains unchanged;
- no Phase 0–13 regression was detected;
- Phase 15 was not started.

## Isolation

- `training_performed=false`
- `backward_called=false`
- `optimizer_step_performed=false`
- `scheduler_step_performed=false`
- `test_accessed=false`
- `test_evaluated=false`
- `test_evaluation_count=1`
- `checkpoint_selected_or_changed=false`
- `phase_15_started=false`

## Conclusion

`PHASE 14: PASS — READY FOR PHASE 15 PLANNING`
