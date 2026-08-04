# Configs

`canonical_registry.json` is the Phase 6 active canonical authority. It resolves
the new dataset, manifest, checkpoints, artifacts, notebook and HTML report with
machine-independent paths relative to the Practice 2.2 root.

`canonical_registry_legacy_compat.json` is an explicit read-only compatibility
fallback for pre-migration locations. It is never loaded by default.

`canonical_registry_phase5_draft.json` is retained as an inactive migration
record and must not be used as authority.
