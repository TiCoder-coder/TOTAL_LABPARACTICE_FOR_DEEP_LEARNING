# Migration and verification scripts

- `switch_notebook_authority.py`: one-time deterministic Phase 6 notebook setup rewrite.
- `validate_active_authority.py`: reloads E2 on Validation only from active resources; it never constructs Test.
- `run_accuracy_refactor.py`: runs authorized Train/Validation repeated-seed experiments and Validation-only soft-voting/TTA selection.
- `build_accuracy_refactor_notebook.py`: applies the factual E2 report corrections and deterministically rebuilds the development notebook.
