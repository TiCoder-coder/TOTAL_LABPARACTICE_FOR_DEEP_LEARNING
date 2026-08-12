# Phase 9 — Selection and Checkpoint Verification

## Locked selection

[`hyperparameter_selection_locked.json`](../../outputs/hyperparameter_selection_locked.json)
locks the winner before Test access:

| Field | Value |
|---|---|
| Run | `E2_resnet18_partial_2b5b94de` |
| Training mode | `partial_finetune` |
| Head / backbone LR | `0.001 / 0.0001` |
| Hidden layers | `[]` |
| Primary metric | minimum Validation loss |
| Tie-breaker | maximum Validation accuracy |
| Best epoch | 24 |
| Validation loss | `0.4303597612` |
| Accuracy at selected checkpoint | `94.42%` |
| Test used for selection | No |

The selected file is
[`best_val_loss.pt`](../../runs/E2_resnet18_partial_2b5b94de/best_val_loss.pt).
Its SHA256 is
`a906600b717f94aa4cf40ca8504f1b82f6618b80cc6bf9b1a7a63813cfd3223b`.
SHA256 detects file changes; it is not described as an independent digital
signature or absolute anti-forgery system.

## Verification gate

Before Test is constructed, [`final_evaluate.py`](../../processing_own_phase/final_evaluate.py):

1. checks the locked checkpoint path and SHA256;
2. rebuilds the original 5,000-image Validation subset;
3. reloads the saved model architecture and weights;
4. rebuilds the original label-smoothed Cross Entropy criterion;
5. re-evaluates Validation;
6. requires both loss and accuracy to match within tolerance.

Verification result:

| Check | Result |
|---|---:|
| Recorded accuracy | 94.42% |
| Reloaded accuracy | 94.42% |
| Accuracy delta | 0 |
| Recorded loss | 0.4303597612 |
| Reloaded loss | 0.4303597626 |
| Loss delta | `1.34e-9` |
| Status | **PASS** |
| Test accessed by verification | No |

Only this PASS unlocks Phase 10. The notebook repeats Validation verification
but never invokes Final Test inference.
