# Phase 9 Results — Locked Selection and Verification

Notebook cells 18–21 present the ranking winner and reload verification.

| Evidence | Result | Artifact |
|---|---|---|
| Locked winner | `E2_resnet18_partial_2b5b94de` | [selection JSON](../../outputs/hyperparameter_selection_locked.json) |
| Selected checkpoint | epoch 24 `best_val_loss.pt` | [checkpoint](../../runs/E2_resnet18_partial_2b5b94de/best_val_loss.pt) |
| SHA256 | `a906600b...fd3223b` | [manifest](../../runs/E2_resnet18_partial_2b5b94de/checkpoint_manifest.json) |
| Accuracy reload | 94.42% → 94.42%, delta 0 | [summary](../../outputs/summary.json) |
| Loss reload | delta `1.34e-9` | [locked selection](../../outputs/hyperparameter_selection_locked.json) |
| Verification | PASS before Test access | [verification implementation](../../processing_own_phase/final_evaluate.py) |
| Final Test receipt | one evaluation | [receipt](../../outputs/final_test_receipt_a906600b717f.json) |

SHA256 provides change detection. The documentation does not claim an
independently signed, absolute anti-forgery guarantee.
