# EXPERIMENTS-v1

EXPERIMENTS-v1 is the canonical local registry contract for all runs from Phase 14 onward.

The JSONL file is the detailed machine-readable run source. CSV files are derived inspection views.

Every run is registered before execution with a unique run ID, canonical config fingerprint and complete upstream lineage.

Completed configs are immutable. A rerun receives a new run ID and requires a canonical rerun reason when its config fingerprint already exists.

Artifacts and metrics are linked by run ID. Missing, orphaned or checksum-invalid records fail registry validation.

Development runs cannot register Test metrics. Final Test registration requires FINAL_TEST family, FINAL_TEST execution type, an explicit model lock ID and authorization.

Phase 13 creates no real run, model, checkpoint, prediction or metric result. Synthetic lifecycle tests run only in a temporary directory.
