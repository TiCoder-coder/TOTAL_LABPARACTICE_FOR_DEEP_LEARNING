# Phase 31 Resume Log Post-Write State Corrective Plan

## Objective

Make the persisted Phase 31 and Phase 32 selective-resume logs represent their verified post-write state.

## Execution sequence

1. Add one reporting-layer materialization function for selective-resume logs.
2. Build and save the initial log.
3. Rebuild and save once when the initial state is `LOG_MISSING` or `LOG_STALE`.
4. Return the final persisted log and path.
5. Route the notebook-facing selective renderer through this function.
6. Route the terminal recovery runner through the same function.
7. Add unit coverage for the single-pass blocked path and two-pass stale-log repair path.
8. Refresh only Phase 31 and verify that both the inspector state and visible log status are `VALID_REUSABLE`.

## Constraints

- Do not retrain any condition.
- Do not modify canonical sweep artifacts.
- Do not modify the notebook.
- Do not create a refresh loop.
- Do not weaken checksum validation.

## Acceptance criteria

- The reporting layer owns selective-log materialization.
- A stale or missing derived log requires at most two atomic writes.
- The final persisted status equals the verified post-write state.
- Phase 31 remains `VALID_REUSABLE`.
