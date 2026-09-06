# Phase 31 Processing Log Refresh Route Issue

## Scope

The Phase 31 terminal runner completed WD0 and WD2, reused WD1, finalized the canonical sweep artifacts and then failed while refreshing the notebook-facing processing log.

## Observed failure

`scripts/run_all_pending.py` routed every phase through `build_phase_processing_log()`.

`build_phase_processing_log()` is the presentation path for Phase 0 through Phase 30. Phase 31 and Phase 32 use the selective-resume presentation path implemented by `build_phase_resume_log()` and `save_phase_resume_log()`.

The mismatch raised `ValueError: Unsupported phase_id: 31` after canonical finalization.

## Impact

- Phase 31 training results are complete.
- The Phase 31 signoff, manifest, results, reference update and winner are present and valid.
- WD0 and WD2 must not be trained again.
- The existing Phase 31 processing log remains stale because its refresh step did not complete.
- Phase 32 must remain unstarted until the Phase 31 presentation log is repaired and the complete Phase 31 state is verified.

## Root cause

The terminal recovery runner did not distinguish between the standard processing-log contract used through Phase 30 and the selective-resume log contract used by Phase 31 and Phase 32.

## Required correction

Route Phase 31 and Phase 32 log refreshes through the selective-resume reporting API. Preserve the standard processing-log route for Phase 0 through Phase 30. Verify the repaired log, canonical Phase 31 state, experiment registry, Test-access prohibition and notebook checksum before Phase 32 execution.
