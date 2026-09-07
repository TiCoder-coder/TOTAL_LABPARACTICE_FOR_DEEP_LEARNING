# Phase 46 / Phase 47 Lineage Drift Audit — 2026-09-06

**Document version:** `phase_46_47_lineage_drift_audit_2026_09_06.md`
**Created:** 2026-09-06
**Author:** Read-only audit (Part 2C of the corrective workflow)
**Classification:** Root-cause / error analysis — NOT a corrective execution record
**Scope:** Phase 45 → Phase 46 → Phase 47 lineage, current branch `CourseWorkV9`, HEAD `81feda942b0af87f1e93e5d158a51b7210e784d7`

---

## 1. TL;DR

The current Phase 45 final model lock is **valid and consistent**. The lineage between Phase 45 and the currently-active Phase 46 / Phase 47 artifacts is **NOT reconstructable** from the active evidence. Corrective execution has **NOT** happened yet. The corrective path is:

```
Phase 46 corrective re-materialization (under current Phase 45 lock)
  → 17-gate pre-Test gate verification (PASS required)
  → Human approval gate
  → Phase 47 corrective Test re-evaluation (documented invalidation of prior Test access)
```

This document captures the root cause and the verifiable evidence trail. It does **not** authorize any execution.

---

## 2. Phase 45 lock is valid

The active Phase 45 lock is consistent across all its artifacts:

| Field | Value |
|---|---|
| `candidate_id` | `TR_C2_ALT_LOOKBACK` |
| `lookback_steps` | `72` |
| `final_refit_epochs` | `30` (median of RO inner-best epochs 30, 37, 19) |
| `seeds` | `[42, 123, 2026]` |
| `config_fingerprint` | `585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24` |
| `final_lock_sha256` | `81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec` |
| `recipe_sha256` | `857dbaf7903792cdba3e126a50d8919992f02e0fff838cba86180026c9e0220c` |
| `population_sha256` | `552e6895dbc6f7a93b0ea9a2ab4de77abd58d8780ad4bc85b697a485ccbc31a9` |
| `lineage_sha256` | `9ba532539d548156e1d5d933abe8c1463617cdc790741e48d5249b452995d9fe` |
| `feature_sha256` | `fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee` |
| `feature_variant` | `FS2_TF1` |
| `target_scaling` | `YS1` |
| `boundary_protocol` | `WB0_CONTEXT_CARRY_OVER` |
| `status` | `PASS` |
| `ready_for_phase46` | `true` |
| `test_status` | `NOT_ACCESSED` |

The Phase 45 handoff to Phase 46 (`phase46_three_seed_handoff.json`) carries the same `candidate_id`, `config_fingerprint`, `final_refit_epochs`, and `seeds`. **No drift at Phase 45 itself.**

---

## 3. Phase 46 active artifacts are stale / inconsistent with Phase 45

The current active Phase 46 artifacts reference a different model universe:

| Field | Phase 45 lock | Phase 46 active signoff / manifest / verification |
|---|---|---|
| `candidate_id` | `TR_C2_ALT_LOOKBACK` | `TR_C0_PRIMARY` |
| `lookback_steps` | `72` | `36` |
| `final_refit_epochs` | `30` | `50` |
| `config_sha256` (signoff) | `585c5e79...` | `a711a9b8...` |
| `final_lock_sha256` (signoff) | `81fb87c4...` | `a711a9b8...` (= config_sha, not a true lock) |
| `recipe_sha256` | `857dbaf7...` | `46d5ab12...` |
| `population_sha256` | `552e6895...` | `a40ded88...` |
| `lineage_sha256` | `9ba53253...` | `376ae5fc...` |
| `average_rmse_wh` | `47.8888` (diagnostic) | `0.0` (placeholder) |

The Phase 46 per-seed checkpoint metadata files each carry a **different** per-seed `config_sha256` (`b15a19dc...`, `dd82d743...`, `59a50add...`), which internally contradicts the `phase_46_signoff.json` claim `all_same_config: true` and `all_same_epochs: true`.

**Diagnosis:** the active Phase 46 artifacts were produced against an earlier (pre-corrective) configuration and were never re-emitted against the current Phase 45 lock. This is documented as `INVALIDATED_BY_CONTRACT_DEVIATION` in the in-progress corrective plan (`phase_46_corrective_reimplementation_plan.md` §12.2).

---

## 4. Phase 46 checkpoint binaries are missing from the current repo

The Phase 46 active `official_checkpoints/seed_*/` directories contain only `_metadata.json` sidecars. The actual `.pt` checkpoint binaries are **not present** in the repository:

```
COURSE_WORK/artifacts/three_seed_final_runs/official_checkpoints/seed_42/seed_42_FINAL_REFIT.pt       (missing)
COURSE_WORK/artifacts/three_seed_final_runs/official_checkpoints/seed_123/seed_123_FINAL_REFIT.pt     (missing)
COURSE_WORK/artifacts/three_seed_final_runs/official_checkpoints/seed_2026/seed_2026_FINAL_REFIT.pt   (missing)
```

The experiment registry contains records for `RUN_TR_FSD_0151_199F1315` and `RUN_TR_FSD_0153_B15A19DC` with `lookback_steps: 36`, `max_epochs: 50`, `target_access_mode: VALIDATION`, and `best_validation_rmse_wh: 41.71...` for seed 42 — i.e. the historical runs were TRAIN-only, lookback 36, 50 epochs. These match the stale Phase 46 active signoff values and **do not** match the current Phase 45 lock.

**Diagnosis:** the active Phase 46 metadata describes a model universe that was historically trained with a different protocol (TRAIN-only, lookback 36, 50 epochs). It was never the configuration that the current Phase 45 lock mandates.

---

## 5. Phase 47 references checkpoint SHAs / run_ids not present in active Phase 46

The current active Phase 47 signoff, summary, evaluation contract, evaluation manifest, release verification, prediction checksums, and phase-48-52 handoffs reference per-seed checkpoint SHA256s and run_ids that:

- Do **not** appear in the experiment registry (`COURSE_WORK/artifacts/experiments/experiment_registry.jsonl`).
- Do **not** match the Phase 46 active official checkpoint metadata (`2a99eab49...`, `4d1469be6b...`, `30064f562...`).
- Do **not** exist as binary `.pt` files in the repository.
- Do **not** correspond to any training log, run directory, or status record.

| Seed | Phase 47 evidence | Phase 46 active metadata |
|---|---|---|
| 42 | `RUN_TR_FSD_0254_2B11AC68` / `c3cfad116aa91d47...` | `RUN_TR_FSD_0153_B15A19DC` / `2a99eab49911c1f47...` |
| 123 | `RUN_TR_FSD_0254_3858DDA9` / `8a134fec517be0df...` | `RUN_TR_FSD_0154_DD82D743` / `4d1469be6b0ca78ba...` |
| 2026 | `RUN_TR_FSD_0255_C7E123FB` / `8753800539f7a617...` | `RUN_TR_FSD_0155_59A50ADD` / `30064f562e0dc9ec...` |

The Phase 47 release-verification references `phase47_test_release.json`, which does **not** exist in the repository.

The prediction bundle CSV files referenced by `prediction_checksums.json` do **not** exist in the repository:

```
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed42.csv       (missing)
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed123.csv     (missing)
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_seed2026.csv    (missing)
COURSE_WORK/artifacts/final_test/predictions/final_test_predictions_persistence.csv (missing)
```

**Diagnosis:** Phase 47 lineage to Phase 46 checkpoints is **not reconstructable** from the active evidence.

---

## 6. Phase 47 used `config_fingerprint` as `final_lock_sha256`

The current active `phase_47_signoff.json`, `final_test_summary.json`, `final_test_evaluation_contract.json`, `final_test_evaluation_manifest.json`, `final_test_release_verification.json`, and `final_test_access_event.json` all store `final_lock_sha256 = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"`, which is the **Phase 45 config_fingerprint** — not the Phase 45 final_lock_sha256 (`81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec`).

This conflation is also documented by `phase58_active_revision_audit_checkpoint.json`:

```
FINAL_LOCK_SHA_CORRECT: false
FINAL_LOCK_SHA_ACTUAL: 585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24
FINAL_LOCK_SHA_CANONICAL: 81fb87c44b6af31b6f65eff13956d9dc94a38dc75731a2c0af088228dd1bd4ec
FINAL_LOCK_SHA_DISCREPANCY: Phase58 conflated the config fingerprint with the canonical model lock.
```

At source level, `src/course_work/final_test_evaluation/__init__.py:52` defines:

```python
LOCKED_CONFIG_FP = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
```

and `src/course_work/scripts/p47_final_test_evaluation.py:191` writes that constant into `final_test_evaluation_contract.json.final_lock_sha256` — i.e. the same `config_fingerprint` value is used in the `final_lock_sha256` field. This is the **source-level root cause** of the documented Phase 47 lock_sha vs config_fingerprint conflation.

**Diagnosis:** `config_fingerprint` and `final_lock_sha256` are two different identifiers, each with its own scope. They must be stored and compared separately. The current source-level constant should be split.

---

## 7. Existing Phase 47 metrics are numerically consistent but NOT lineage-valid

The current active Phase 47 per-seed metrics match the expected values numerically:

| Seed | MAE | RMSE | R² |
|---|---|---|---|
| 42 | 29.52865 | 64.94275 | 0.48927 |
| 123 | 27.11487 | 61.98609 | 0.53472 |
| 2026 | 28.94229 | 64.56014 | 0.49527 |
| **mean ± SD** | 28.52860 ± 1.25894 | 63.82966 ± 1.60800 | 0.50642 ± 0.02469 |

Test population SHA matches: `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`. Test N = 2961.

However, **numerical consistency is not lineage validity**. The same numerical values could in principle have been produced by any model that produces equivalent outputs. The missing prediction bundle CSVs and missing checkpoint binaries mean that the SHA-based lineage chain `final_lock_sha256 → config_fingerprint → checkpoint_sha256 → prediction_bundle_sha256` cannot be reconstructed from the active evidence.

**Diagnosis:** the prior Phase 47 Test access must be classified `INVALIDATED_BY_PHASE46_LINEAGE_DRIFT`. The numerical values are preserved as historical audit evidence but are **not** authoritative.

---

## 8. Model selection remains frozen at Phase 45

No corrective execution has occurred. The model selection (`TR_C2_ALT_LOOKBACK`, lookback 72, 30 epochs, seeds [42, 123, 2026]) is unchanged. The corrective path does **not** introduce any new model selection — it only re-materializes the already-locked model under correct Phase 46 protocol and re-evaluates it on Test under correct Phase 47 protocol.

Per `rule_code.md` Rule #77 (Scientific Protocol Preservation), no scientific decision may be allowed to depend on prior Test results. The corrective execution is therefore governed by:

1. Corrective Phase 46 re-materialization against the current Phase 45 lock.
2. 17-gate pre-Test gate verification.
3. Explicit human approval before any Test access.
4. Corrective Phase 47 Test re-evaluation with documented invalidation of prior access.
5. Prior Phase 47 evidence preserved under `artifacts/final_test/_history/PHASE47_PRE_CORRECTIVE_<UTC>/`.

---

## 9. Corrective path (NOT YET EXECUTED)

```
   [PART 2D — Plan only]                        [PART 2E — Docs only]
                                                   This document.
                                                   + phase_46_corrective_post_audit_reimplementation_plan.md
                                                   + phase_47_corrective_post_audit_test_re_evaluation_plan.md
            |
            v
   [HUMAN REVIEW of 3 docs]
            |
            v
   [HUMAN APPROVAL #1 — authorize creation of code changes per Part 2D §9]
            |
            v
   [IMPLEMENT CODE CHANGES — see Part 2D §9]
     - Split LOCKED_CONFIG_FP constant
     - Remove TR_C0_PRIMARY fallback from p46 runner
     - Source final_refit_epochs strictly from Phase 45
     - Distinguish config_fingerprint vs final_lock_sha256
     - Persist actual .pt checkpoint binaries
     - Persist actual prediction bundle CSVs
     - Require identical config_fingerprint across 3 seeds
     - Generate phase47_test_release.json only after all gates pass
     - Preserve old Phase 46/47 evidence before corrected run
            |
            v
   [HUMAN APPROVAL #2 — authorize corrective Phase 46 training]
            |
            v
   [CORRECTIVE PHASE 46 EXECUTION]
            |
            v
   [17-GATE PRE-TEST GATE — PASS required]
            |
            v
   [HUMAN APPROVAL #3 — authorize corrective Phase 47 Test re-evaluation]
            |
            v
   [CORRECTIVE PHASE 47 EXECUTION — documented invalidation of prior Test access]
            |
            v
   [PHASE 47 PASS SIGN-OFF — only then authoritative]
```

---

## 10. Status

```
ROOT_CAUSE_DOCUMENTED: YES
CORRECTIVE_EXECUTION_OCCURRED: NO
MODEL_SELECTION_FROZEN_AT_PHASE_45: YES
PRIOR_PHASE_47_LINEAGE_STATUS: INVALIDATED_BY_PHASE46_LINEAGE_DRIFT
PRIOR_PHASE_47_NUMERICS_STATUS: PRESERVED_AS_HISTORICAL_EVIDENCE
AUTHORITATIVE_PHASE_47_RESULTS: NOT_YET_PRODUCED
```

**This document is a root-cause audit only. It does not authorize any execution.**
