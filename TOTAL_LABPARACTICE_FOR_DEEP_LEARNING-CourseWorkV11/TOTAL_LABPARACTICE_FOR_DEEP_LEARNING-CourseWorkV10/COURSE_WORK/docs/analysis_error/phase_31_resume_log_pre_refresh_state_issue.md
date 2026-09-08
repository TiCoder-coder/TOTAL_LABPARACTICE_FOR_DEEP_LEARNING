# Phase 31 Resume Log Pre-Refresh State Issue

## Scope

The corrected Phase 31 routing generated a structurally valid processing log, but the persisted visible status remained `LOG_STALE` while the post-write inspector state was `VALID_REUSABLE`.

## Root cause

The selective-resume log was built before it was saved. Its visible state therefore described the old log being replaced. Saving that document repaired the checksum state, but the document still contained the pre-refresh status.

## Impact

- Scientific artifacts remain valid.
- Canonical Phase 31 state is `VALID_REUSABLE`.
- The notebook-facing HTML can display a stale status badge and stale action even though no repair or training remains.

## Required correction

Materialize a selective-resume log in the reporting layer. Save the initial document, reinspect the resulting state, and rebuild once only when the initial state was `LOG_MISSING` or `LOG_STALE`. Persist and render the post-write document. Use the same materialization path from the terminal runner and notebook-facing renderer.
