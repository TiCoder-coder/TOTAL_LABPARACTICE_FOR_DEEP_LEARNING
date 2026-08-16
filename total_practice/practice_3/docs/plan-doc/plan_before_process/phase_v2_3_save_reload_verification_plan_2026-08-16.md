# Phase v2.3 Save / Reload Verification Plan

**Date:** 2026-08-16
**Protocol:** practice_3_v2.3

## 1. Purpose
The purpose of this phase is to strictly verify the reproducibility of the final locked winner model. By saving the `E4_weight_decay_0.05` model and its tokenizer to disk, entirely destroying them from memory, reloading them from the save directory, and re-running the exact same 9 custom inference sentences, we prove that the model state is successfully persisted and can be deterministically reconstructed for future usage.

## 2. Original Winner & Target Save Destination
- **Winner Checkpoint:** `E4_weight_decay_0.05` (`d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191`)
- **Save Destination:** `docs/result/practice_3_v2_3/final_saved_model/`
- Older `phase_14_saved_model/` artifacts will be audited and preserved as historical, ensuring no silent overwrites.

## 3. Expected Files & Hashing
The `save_pretrained` function is expected to produce:
- **Model:** `model.safetensors`, `config.json`
- **Tokenizer:** `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`, `vocab.txt`
All exported files will be hashed and recorded in `final_saved_model_manifest.json`.

## 4. Reload Procedure & Verification
- The original model and tokenizer will be completely deleted from the Python process.
- The exported model and tokenizer will be loaded from the final save destination using `local_files_only=True`.
- Model state dictionaries (parameters) will be compared for exact shape and tensor values.
- Tokenizer vocab size and token mappings will be explicitly checked.

## 5. Inference Comparison & Tolerance Policy
- The 9 custom sentences from the previous step will be inferred again.
- **Labels:** Must match exactly 100% (9/9).
- **Probabilities & Confidence:** An absolute difference threshold of `1e-6` will be applied. If any prediction or token sequence changes unexpectedly, the verification will FAIL CLOSED.

## 6. Strict Rules & Data Protection
- **No Training:** The model parameters and hyperparameters will not be changed.
- **Holdout / Test Protection:** Neither the Holdout split nor the Official Test split will be loaded or inferred. `evaluation_count` will remain exactly 1.
