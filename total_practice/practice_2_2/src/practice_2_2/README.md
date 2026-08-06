# Practice 2.2 source — Phase 6 canonical package

`practice_2_2.*` is the recommended/default import namespace. Resource access
uses the active registry and resolves the new Practice 2.2 layout by default.
Practice-specific copied algorithms remain unchanged; shared modules use thin
compatibility wrappers where required.

Use `PYTHONPATH=src` for the new `practice_2_2.*` import namespace. Existing `processing_own_phase.*` imports remain supported.

The old `processing_own_phase.*` namespace remains available for regression
compatibility. It is not the canonical resource authority.

`paths.py` and `resources.py` provide cwd-independent discovery, active
canonical validation and explicit read-only fallback for pre-migration paths.

`accuracy_pipeline.py` provides the fail-closed Train/Validation accuracy workflow: manifest authorization, domain-safe transforms, staged transfer learning, exact epoch loss, repeated-seed selection, offline checkpoint reload, soft voting and safe TTA. Its public data APIs reject the Test split.
