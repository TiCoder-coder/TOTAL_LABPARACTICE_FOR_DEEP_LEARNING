# Phase 14 Guard — JSON Label-Key Mismatch

**Date:** 2026-08-14  
**Phase:** 14 — Save & Reload  
**Final status:** FIXED AND VERIFIED

## Error

The first save/reload/equivalence run completed with PASS, but the immediate second `run_or_load_phase_14()` call rejected the otherwise valid package:

```text
RuntimeError: Phase 14 package/artifact exists but failed integrity;
refusing overwrite or repair
```

## Evidence

The package audit showed:

- manifest hash: PASS;
- all package file sizes and SHA-256 values: PASS;
- authoritative source checkpoint fingerprint: PASS;
- Phase 13 input signature: PASS;
- state-dict and prediction evidence: PASS;
- tokenizer summary: PASS;
- only `model_loaded_from_package` metadata comparison failed.

## Root Cause

At runtime, the normalized `id2label` dictionary uses integer keys (`0`, `1`). JSON object keys are strings after artifact serialization and reload (`"0"`, `"1"`). The guard compared these representations directly, causing a false integrity failure. Model weights, configuration meaning, labels and package files were unchanged.

## Fix

Added `_model_summary_equal()` in `processing_own_phase/phase_14_save_reload.py`. The helper normalizes JSON-round-tripped `id2label` keys back to integers and `label2id` values to integers before comparing semantic model summaries.

No package, model weight, tokenizer file, checkpoint, Test artifact or earlier Phase was changed.

## Re-verification

The subsequent guard call returned:

```text
PASS loaded_verified_phase_14_package_no_resave True
```

This confirms the package was verified and loaded without invoking `save_pretrained()` again.
