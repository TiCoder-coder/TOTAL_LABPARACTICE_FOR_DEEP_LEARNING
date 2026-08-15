# Phase 13 — New-Sentence Inference Implementation Log

**Date:** 2026-08-14  
**Scope:** Phase 13 only  
**Final status:** PASS

## Approved input

- Phase 0–12 were treated as frozen PASS prerequisites.
- Authoritative model: Phase 9 epoch 2 / step 1068 / `checkpoint-1068`.
- Authoritative Phase 11 weight SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.
- Tokenizer contract: `distilbert/distilbert-base-uncased`, maximum length 80, truncation and dynamic batch padding.

## Implementation

Created `processing_own_phase/phase_13_new_sentence_inference.py` with:

- strict single-string/list input validation;
- Phase 9/11 checkpoint identity and fingerprint verification;
- local-only checkpoint/model/tokenizer loading;
- MPS/CUDA/CPU-compatible device selection;
- `model.eval()` and `torch.inference_mode()` forward passes;
- logits `[B, 2]`, softmax probabilities, binary label and confidence output;
- finite/range/sum/count/order/confidence/determinism assertions;
- checkpoint fingerprint recheck after inference;
- guarded artifact reuse keyed by input signature, checkpoint fingerprint and preprocessing contract;
- explicit isolation evidence: no training, backward, optimizer/scheduler step, Test access/evaluation, checkpoint change or Phase 14 start.

Updated `notebook_practice_3/practice_3.ipynb` with a thin Phase 13 call and presentation. No inference processing logic was duplicated in the notebook.

## Executed custom inputs

Four newly authored sentences were used for the fixed notebook demonstration:

1. clearly positive;
2. clearly negative;
3. negation;
4. mixed/contrast sentiment.

They were not copied from the Test split and no Test dataset/prediction artifact was used as an inference input.

## Result evidence

- Result: `docs/result/phase_13_inference_examples.json`
- Status: PASS
- Device: CPU
- Input/output count: 4 / 4
- Logits shape: `[4, 2]`
- Checkpoint fingerprint: exact Phase 11 match
- Finite logits/probabilities: PASS
- Probability range and row sum: PASS
- Predicted-class confidence: PASS
- Same-input deterministic inference: PASS
- Checkpoint unchanged: PASS

Observed demo predictions:

| Input | Prediction | Confidence |
|---|---:|---:|
| Warm, convincing and moving performances | POSITIVE | 0.993701 |
| Tedious, predictable and slow story | NEGATIVE | 0.987616 |
| “It is not a bad movie at all.” | POSITIVE | 0.925192 |
| Excellent acting but shallow plot | NEGATIVE | 0.977097 |

These values are presentation examples, not Test metrics and not evidence for checkpoint selection.

## Guard verification

A second Phase 13 call returned:

`loaded_verified_phase_13_artifact_no_model_inference`

This confirms the matching default-demo artifact can be reused without loading the model or performing another forward pass.

## Notebook Run All

The first notebook attempt used the host `python3` kernelspec, which pointed to Anaconda Python 3.13 and failed at Phase 1 due to an incompatible Torch binary. No Phase result failed; the wrong kernel was the cause.

A temporary kernelspec was then created under `/private/tmp/practice3-jupyter` pointing to the repository `.venv` Python 3.11. The notebook was executed offline with that kernel.

Final verification:

- all 14 code cells executed continuously with execution counts 1–14;
- no code cell contains an error output;
- Phase 9 guard: `loaded_verified_artifacts_no_retraining`;
- Phase 11 guard: `loaded_verified_phase_11_artifacts_no_test_reevaluation`;
- Phase 13 guard: verified artifact load on notebook Run All;
- Phase 11 `test_evaluation_count` remains 1;
- Phase 0–12 outputs completed without regression;
- Phase 14 was not started.

## Isolation statement

- `training_performed=false`
- `backward_called=false`
- `optimizer_step_performed=false`
- `scheduler_step_performed=false`
- `test_accessed=false`
- `test_evaluated=false`
- `test_evaluation_count=1`
- `checkpoint_changed=false`
- `phase_14_started=false`

## Conclusion

`PHASE 13: PASS — READY FOR PHASE 14 PLANNING`
