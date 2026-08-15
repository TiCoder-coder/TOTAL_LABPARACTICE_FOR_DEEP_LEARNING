# Phase 09 Fine-Tuning Process Log — 2026-08-12

## Scope

Implemented and executed only Phase 9 under the approved plan. Test was not
accessed and Phase 10 was not started.

## Pre-run Safety Check

`docs/result/phase_09_training/` was absent before retry. No partial artifact
or checkpoint was deleted or overwritten.

## Debug Gate

- Dataset source: existing offline Rotten Tomatoes cache.
- Train/Validation subset counts: 1,000/200.
- Epochs: 1.
- Global/optimizer steps: 63.
- Training loss: 0.10204486241416326.
- Validation loss: 0.007877531461417675.
- Validation Accuracy/Precision/Recall/F1: 1.0/1.0/1.0/1.0.
- Checkpoint: `debug_checkpoints/checkpoint-63`.
- Status: PASS.
- Debug result was not treated as a final experiment.

## Fresh Full Baseline

After debug PASS, seed 42 was reset and a new generic pretrained DistilBERT
classifier was constructed. Debug weights/checkpoint were not reused.

- Train/Validation counts: 8,530/1,066.
- Epochs: 3.
- Batch sizes Train/Validation: 16/32.
- Learning rate: 2e-5.
- Weight decay: 0.01.
- Global steps: 1,602.
- Overall training loss: 0.27031363917051926.
- Runtime: approximately 226.3 seconds on MPS.

## Per-epoch Validation

| Epoch | Train loss | Eval loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4214 | 0.4152307808 | 0.8189493433 | 0.7583586626 | 0.9362101313 | 0.8379513014 |
| 2 | 0.2416 | 0.3904653192 | 0.8564727955 | 0.8492647059 | 0.8667917448 | 0.8579387187 |
| 3 | 0.1480 | 0.4926925004 | 0.8545966229 | 0.8399280576 | 0.8761726079 | 0.8576675849 |

## Checkpoint Policy

Real candidates were checkpoints 534, 1068 and 1602. The authoritative
selection was `checkpoint-1068` (epoch 2) because its Validation loss
0.3904653192 was the minimum. No loss tie occurred, so the F1/earlier-epoch
tie-breakers were not needed. Trainer and explicit policy agreed. The policy
checkpoint was explicitly reloaded.

The reloaded checkpoint also passed a two-sample Validation-only forward
sanity check with logits shape `[2,2]`, finite logits and finite loss.

## Artifacts and Guard

All approved JSON artifacts, Trainer state and four real checkpoint directories
(one debug plus three full) were created under
`docs/result/phase_09_training/`. The notebook guard validates the manifest,
artifact set, sample/record counts and selected checkpoint before loading it;
later Run All executions do not retrain by default.

## Isolation

- Test accessed: false.
- Test used for checkpoint selection: false.
- Hyperparameter search: not performed.
- Phase 10 started: false.

## Result

Phase 9 PASS and ready for Phase 10 planning only.
