# Practice 2.2 tests — Phase 3 copy

These tests preserve the old assertions while using the staged `practice_2_2.*` namespace. Run from the project root with:

```bash
PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
```

The suite does not train on project data or run Final Test. The accuracy-pipeline integration check trains only a small synthetic dataset created inside the test temporary directory.

`test_paths_resources.py` verifies cwd-independent discovery, the relative-path registry, canonical hashes and the immutable-artifact fallback adapter.

`test_accuracy_pipeline.py` verifies manifest authorization, Train/Validation leakage guards, Test-split rejection, domain-safe transforms, staged freezing, exact epoch metrics, two-stage training, offline checkpoint reload and Validation-only TTA using synthetic data only.
