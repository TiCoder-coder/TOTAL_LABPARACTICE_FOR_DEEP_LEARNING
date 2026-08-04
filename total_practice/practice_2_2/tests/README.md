# Practice 2.2 tests — Phase 3 copy

These tests preserve the old assertions while using the staged `practice_2_2.*` namespace. Run from the project root with:

```bash
PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
```

The suite does not train a model or run Final Test.

`test_paths_resources.py` verifies cwd-independent discovery, the relative-path registry, canonical hashes and the immutable-artifact fallback adapter.
