# PHASE 40 STRICT-BEST CORRECTIVE PLAN

## Metadata

| Field | Value |
|-------|-------|
| Plan ID | `CW-PHASE40-CORRECTIVE-001` |
| Phase | 40 (S18 RevIN sweep) |
| Status | DRAFT |
| Author | Cursor Main Agent |
| Date | 2026-08-24 |
| Purpose | Resolve Phase 40 strict-BEST sign-off inconsistency (declared absolute tolerance violated by declared RMSE/MAE deltas) without changing scientific results, without retraining RN1 or RN0, without accessing Test, and without starting Phase 41 |
| Scope | verification methodology, sign-off correction, documentation |
| Plan file | `COURSE_WORK/docs/plan-doc/plan_before_process/phase_40_s18_revin_strict_best_corrective_plan.md` |

---

## 1. Current status

```text
PHASE 40 = BLOCKED
RN0 reference       : RUN_TR_S14_0023_A711A9B8  Validation RMSE = 57.69679988114431
RN1 run             : RUN_TR_S18_0031_A711A9B8
RN1 stored RMSE     : 62.28695816170348
RN1 stored MAE      : 28.58192738256379
RN1 stored R²       : 0.5440007380016099
RN1 recomputed RMSE : 62.286963535407715 (CPU)
RMSE delta          : 5.373704233591070e-06
MAE  delta          : 7.290684507665901e-06
R²   delta          : 7.868116669307312e-08
sign-off tolerance  : 1e-06
sign-off status     : PASS  ← INCONSISTENT with declared tolerance under absolute-only reading
recorded device     : mps
verification device : cpu (silent fallback; no Phase 40 corrective authorizing CPU fallback exists)
```

## 2. Blocked issue (read-only audit findings)

The Phase 40 strict-BEST verifier at `scripts/phase40_strict_best_rn1.py` implements:

```python
tolerance = 1e-6
rmse_ok = math.isclose(recomputed.rmse_wh, stored_rmse, rel_tol=tolerance, abs_tol=tolerance)
mae_ok  = math.isclose(recomputed.mae_wh,  stored_mae,  rel_tol=tolerance, abs_tol=tolerance)
r2_ok   = math.isclose(recomputed.r2,     stored_r2,   rel_tol=tolerance, abs_tol=tolerance)
```

Python semantics for `math.isclose(a, b, rel_tol=R, abs_tol=A)`:

```text
|a - b| <= max(R * max(|a|, |b|), A)
```

For RN1:

| Metric | Stored | Recomputed | Delta | abs check (delta <= 1e-6) | rel check (delta <= 1e-6 * |stored|) | isclose |
|--------|--------|-----------|-------|---------------------------|--------------------------------------|---------|
| RMSE | 62.28695816170348 | 62.286963535407715 | 5.37e-06 | FAIL | delta <= 6.23e-05 → PASS | PASS (rel arm) |
| MAE | 28.58192738256379 | 28.5819346732483 | 7.29e-06 | FAIL | delta <= 2.86e-05 → PASS | PASS (rel arm) |
| R² | 0.5440007380016099 | 0.5440006593204432 | 7.87e-08 | PASS | PASS | PASS |

The sign-off records `tolerance: 1e-06` and `status: PASS`. Under absolute-only reading the contract fails on RMSE and MAE. Under the actual isclose evaluation the contract passes (via the relative arm). The sign-off is therefore **internally inconsistent** if read literally.

## 3. What must NOT change

These are inviolable Phase 40 / Phase 39 invariants per the Phase 40 Detail and the Phase 39 sign-off chain:

| Invariant | Value |
|-----------|-------|
| RN0 reference run | `RUN_TR_S14_0023_A711A9B8` |
| RN0 reference Validation RMSE | `57.69679988114431` |
| RN1 run | `RUN_TR_S18_0031_A711A9B8` |
| RN1 stored Validation RMSE | `62.28695816170348` |
| RN1 stored Validation MAE | `28.58192738256379` |
| RN1 stored Validation R² | `0.5440007380016099` |
| RN1 selection metric | `validation_rmse_wh` |
| RN1 tie rule | `RN0_ON_EXACT_RMSE_TIE` |
| RN1 selection direction | `MIN` |
| RN1 frozen configuration | Phase 39 S17 GC1 winner fields |
| RN1 epoch cap | `50` |
| RN1 epochs executed | `18` |
| RN1 best epoch | `8` |
| Population fingerprint | `a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987` |
| Metric contract fingerprint | `4509825a7be2ee75220f88bedf31da6d062e6fee5168af45b0d54ec1db137198` |
| All Phase 9-39 hyperparameters | Frozen |
| RevIN parameter delta vs RN0 | `+56` (2 * 28) |
| RevIN channel scope | 28 RevIN / 5 passthrough / 1 historical Appliances |
| All upstream artifacts (`RUN_TR_S14_0023_*`, `RUN_TR_S18_0031_*`) | Untouched |
| Test access | FORBIDDEN |

## 4. Root-cause status

### 4.1 No canonical Phase 40 strict-BEST contract is defined

The Phase 40 Detail (`docs/plan-doc/plan_detail_for_each_phase/Phase_40_S18_RevIN_sweep.md`) is silent on:

- Numerical tolerance (no mention of `1e-6`, `1e-9`, or any tolerance)
- Verification device policy (no mention of MPS-required or CPU-allowed)
- Comparison operator (no `math.isclose` vs raw `abs()` choice)
- Stored predictions vs recompute trade-off

`src/course_work/sweeps/revin.py` defines `verify_phase_40_preflight` (preflight gate) but does NOT define a canonical `verify_phase_40_best()` function. There is no architecture-level verifier module for Phase 40, unlike Phase 35 (`sweeps/layers.py:verify_phase_35_n1_best`), Phase 36 (`sweeps/ffn.py:verify_phase_36_fnn_best`), or Phase 37 (`sweeps/loss.py:verify_phase_37_huber_best`).

This means `scripts/phase40_strict_best_rn1.py` is an ad-hoc verifier that I authored, not a canonical sweep-level implementation. The 1e-6 tolerance and CPU-fallback logic are my ad-hoc choices, not project canon.

### 4.2 The current sign-off is internally inconsistent

`artifacts/sweeps/S18_revin/phase_40_signoff.json` reports:

```json
"strict_best_verification": {
  ...
  "tolerance": 1e-06,
  "verification_device": "cpu (MPS unavailable on this host; deterministic fallback)",
  "verification_device_warning": "Recorded device mps unavailable; CPU fallback used. ..."
  ...
}
```

The sign-off declares `tolerance: 1e-06` and PASS while recording:

- RMSE delta `5.37e-6` (> 1e-6, fails the literal declared tolerance)
- MAE delta `7.29e-6` (> 1e-6, fails the literal declared tolerance)
- Verification device `cpu` with warning about MPS being unavailable

Two independent inconsistencies exist:

1. Numeric: `delta > tolerance` while `status = PASS` (only true under isclose's relative arm).
2. Device: silent CPU fallback without a Phase 40 corrective plan authorizing it.

Per Phase 23 precedent (`plan_before_process/phase_23_stale_signoff_resume_precedence_corrective_plan.md`) "Training may start only when the current runtime exposes CUDA or MPS. A CPU fallback is not authorized." And per Phase 18 ("do not silently enable CPU fallback env behavior").

Per `working_rule.md` rule 25 (No Silent Test Weakening) and `rule_code.md` rule 107 (No Silent Test Weakening): "không tăng tolerance chỉ để pass".

### 4.3 Phase precedents are split

| Phase | Tolerance | Comparison | Device | Notes |
|-------|-----------|------------|--------|-------|
| Phase 35 (N1) | `1e-6` | `math.isclose(rel=1e-6, abs=1e-6)` | `device_type="cpu"` | PASS with delta ~3.5e-7 |
| Phase 36 (F64/F256) | `1e-6` | `math.isclose(rel=1e-6, abs=1e-6)` | `device_type="cpu"` | PASS for F256 with delta ~8e-7 |
| Phase 37 (Huber) | `1e-9` | `abs(stored - rec) <= 1e-9` (raw abs) | MPS required, raise on CPU fallback | Initial FAIL delta 5.75e-7; fixed via `phase_37_device_consistent_fix.md` (use MPS device) |
| Phase 39 (GC0) | `1e-6` | `math.isclose(rel=1e-6, abs=1e-6)` | `device_type="cpu"` | PASS delta 9.12e-08 |
| Phase 40 (RN1) current | `1e-6` (ad hoc) | `math.isclose(rel=1e-6, abs=1e-6)` | silent CPU fallback | delta 5.37e-06 (5x larger than Phase 39 GC0); sign-off inconsistent |

Phase 37 is the only phase that (a) uses raw absolute tolerance `1e-9`, (b) requires MPS, and (c) explicitly forbids silent CPU fallback. Phase 35/36/39 (and 40 as I implemented) all use `math.isclose(rel=abs=1e-6)` with CPU verification — and all PASSED because observed deltas were well under `1e-6` in absolute terms. Phase 40 RN1 is the first to use the same pattern with delta > 1e-6 in absolute terms.

## 5. Candidate evaluation

### Candidate 1 — Same-device MPS strict recomputation

Use `recorded_device = config["runtime"]["device_type"]`. Raise `RuntimeError` if recorded device is unavailable (Phase 37 pattern).

- **Scientific validity:** HIGH. Eliminates device-boundary numerical drift; reproduces training-time evaluation provenance exactly.
- **Contract impact:** Mirrors Phase 37. If MPS is unavailable on this host, raises RuntimeError → Phase 40 strict BEST = `NOT_VERIFIABLE_ON_CURRENT_DEVICE`. No PASS verdict under that condition.
- **Retraining required:** NO.
- **Test access required:** NO.
- **Architecture/rule amendment required:** NO. Phase 37 corrective (`docs/plan-doc/plan_to_refactor&fix/phase_37_device_consistent_fix.md`) already established this policy and is an approved precedent.
- **Reproducibility:** HIGH. Strict recomputation uses MPS → must match training-time stored metrics within `1e-9`.
- **Risks:**
  - MPS may not be available on the current verification host. The script would raise RuntimeError, returning NOT_VERIFIABLE. Then a Human-approved device policy is required to define what the host should do.
  - This is the **strictest** and most defensible policy but it converts a "PASS" claim into a "host-blocked" verdict, which is not a hidden fix.
- **Does it change scientific results:** NO.
- **Verdict:** Scientifically correct but currently unrunnable on a CPU-only host. Either requires a host with MPS, or requires a Human-approved device policy amendment.

### Candidate 2 — Stored `best_validation_predictions.csv` provenance verification

Use the training-time computed predictions stored at `artifacts/runs/<run_id>/predictions/best_validation_predictions.csv` as canonical evidence.

- **Scientific validity:** HIGH if the CSV exists and is float64.
- **Contract impact:** Treats stored training-time predictions as authoritative. Equivalent to "Phase 37 Candidate 2" in `phase_37_strict_best_corrective_plan.md`.
- **Retraining required:** YES for RN1. The current `RUN_TR_S18_0031_A711A9B8` artifact set contains only:
  - `config.json`
  - `metrics/best_validation_metrics.json`
  - `status.json`

  It does NOT contain `predictions/best_validation_predictions.csv` nor `checkpoints/best_checkpoint.pt`. Same for `RUN_TR_S14_0023_A711A9B8` (RN0 reference). The project pattern (Phase 35/36/39) recomputes from checkpoints, not from stored predictions.
- **Test access required:** NO.
- **Architecture amendment required:** YES. Would require adding `predictions/best_validation_predictions.csv` to every future sweep run's mandatory artifact set, which is outside Phase 40 scope.
- **Reproducibility:** HIGH if CSV is float64 and present.
- **Risks:**
  - **NOT VIABLE without retraining RN1 to regenerate predictions**, which is forbidden by the user's instruction ("DO NOT retrain RN1") and by the Phase 40 Detail.
  - Even if predictions existed, the strict verifier pattern in `sweeps/ffn.py` (Phase 36) explicitly `pd.read_csv(ROOT / f'artifacts/runs/{RUN_ID}/predictions/best_validation_predictions.csv')` — so the project does NOT currently store predictions in `predictions/`. The artifact set must be extended project-wide.
- **Does it change scientific results:** NO if predictions existed; YES (requires new training) in the current state.
- **Verdict:** **REJECTED** under the current constraints. Cannot be implemented without retraining RN1 or extending the project-wide artifact schema outside Phase 40 scope.

### Candidate 3 — Explicit Phase 40 tolerance/device policy amendment

Author a Phase 40 corrective plan that explicitly codifies: "tolerance = 1e-6 (abs + rel via math.isclose); device = recorded MPS preferred, deterministic CPU fallback allowed with documented warning; relative arm is the operative rule for large-stored-magnitude metrics".

- **Scientific validity:** MEDIUM. Codifies the existing ad-hoc behavior into a Human-approved contract. The relative arm is legitimate when stored metric magnitude >> device-boundary noise (~1e-6 in absolute units).
- **Contract impact:** Adds an explicit Phase 40 tolerance/device policy where none currently exists. This is a Phase 40 Detail amendment, not an architecture-rule amendment (the device policy is sweep-specific).
- **Retraining required:** NO.
- **Test access required:** NO.
- **Architecture/rule amendment required:** YES — Phase 40 plan amendment (`docs/plan-doc/plan_before_process/...`) with Human approval per `rule_code.md` rule 11-12, then a code change in `scripts/phase40_strict_best_rn1.py` to make the relative-arm rule explicit, then a sign-off regeneration.
- **Reproducibility:** MEDIUM. The relative tolerance is sensitive to the magnitude of the metric; in practice RMSE ~ 60 Wh gives `1e-6 * 60 = 6e-5` of room, which is well above any plausible device-boundary noise (`~1e-6` from float32 MPS→CPU rounding).
- **Risks:**
  - Per `rule_code.md` rule 107 ("No Silent Test Weakening") the tolerance change must be justified by numerical evidence, not "to make current numbers pass". Here the justification is: the device-boundary noise is bounded by float32 hardware rounding ~1e-6 absolute; the relative arm is the project's de facto policy from Phase 35/36/39; observed delta 5.37e-6 is within 1 order of magnitude of typical device noise and within 2 orders of magnitude of relative tolerance. This is justifiable.
  - However, it still does not authorize CPU fallback contractually for Phase 40. Phase 23 ("A CPU fallback is not authorized") and Phase 18 ("do not silently enable CPU fallback") apply.
- **Does it change scientific results:** NO. RN0 RMSE 57.70 vs RN1 RMSE 62.29 — RN0 still wins; winner decision is unchanged.
- **Verdict:** ACCEPTABLE IF combined with explicit CPU-fallback device policy and corrected sign-off. This is the **most aligned with project precedent** and the **least invasive**.

### Candidate 4 (additional) — Document the existing isclose+CPU pattern as canonical for Phase 40

This is a specialization of Candidate 3 that does not invent new policy but rather:

1. Confirms the project's de facto strict-BEST pattern from Phase 35/36/39/40 is `math.isclose(rel=abs=1e-6)` with CPU verification.
2. Adds an explicit Phase 40 sweep-level clause documenting that pattern as the Phase 40 contract.
3. Corrects the sign-off to declare both `abs_tolerance: 1e-6` and `rel_tolerance: 1e-6` (matching isclose semantics) rather than just `tolerance: 1e-6`.
4. Adds explicit CPU-fallback authorization for Phase 40 with `verification_device_warning` mandatory.

- **Scientific validity:** HIGH. Same as Phase 35/36/39 which were accepted.
- **Contract impact:** Codifies existing precedent.
- **Retraining required:** NO.
- **Test access required:** NO.
- **Architecture amendment required:** YES — Phase 40 plan amendment only.
- **Reproducibility:** HIGH.
- **Verdict:** ACCEPTABLE and is functionally identical to Candidate 3 with clearer wording.

## 6. Recommended repair

**Recommended candidate: Hybrid of Candidate 1 + Candidate 4**

Tier 1 (primary): Re-run strict BEST with **recorded-device MPS** via Candidate 1. This is the most defensible contract. If the host has MPS available, the recomputation will produce delta < 1e-9 and PASS exactly. If MPS is unavailable, the verifier raises RuntimeError per Phase 37 pattern and Phase 40 strict BEST is recorded as `NOT_VERIFIABLE_ON_CURRENT_DEVICE` (not PASS, not FAIL). This is honest, evidence-based, and Phase-37-aligned.

Tier 2 (fallback for CPU-only hosts): If and only if a Human explicitly approves the Phase 40 plan amendment, also implement Candidate 4 (document the existing 1e-6 isclose+CPU pattern as the Phase 40 contract) as a separate, explicitly approved code path. **This requires a Phase 40 plan amendment with USER_APPROVAL_GATE** because it is a tolerance/device-policy decision that did not exist before.

Tier 1 must be tried first and reported with full evidence before any Tier 2 amendment is considered.

**Reasons for this recommendation:**

1. **Phase 37 is the closest precedent** for tolerance `1e-9` absolute and MPS-required device policy. It is Human-approved (`docs/plan-doc/plan_to_refactor&fix/phase_37_device_consistent_fix.md`) and demonstrates the project explicitly rejected CPU fallback as a silent fix.
2. **Phase 35/36/39 used `math.isclose(rel=1e-6, abs=1e-6)` with CPU** but only because their observed deltas were < 1e-6 absolute. Phase 40 RN1 is the first to exceed the absolute arm by 5x, so the project's de facto policy is being strained.
3. **The sign-off is already internally inconsistent** (`tolerance=1e-6` while `delta>1e-6` reports PASS). At minimum, the sign-off must be corrected to either (a) explicitly declare the relative arm and document the CPU fallback, or (b) record `NOT_VERIFIABLE_ON_CURRENT_DEVICE` if Tier 1 cannot be executed.
4. **Candidate 2 is not viable** without retraining RN1 (forbidden) or extending the project-wide artifact schema (out of scope).
5. **Candidate 3 alone (without Tier 1)** would silently lock in the ad-hoc behavior without addressing the underlying device-boundary question. This violates `rule_code.md` rule 103 (No Hidden Fallback) and rule 107 (No Silent Test Weakening).

## 7. Source/artifact scope

### 7.1 Files ABSOLUTELY NOT to modify

- `artifacts/runs/RUN_TR_S14_0023_A711A9B8/` (RN0 reference, all files)
- `artifacts/runs/RUN_TR_S18_0031_A711A9B8/` (RN1, all files)
- All upstream Phase 0-39 scientific artifacts
- `notebook_course_work/CourseWork.ipynb`
- `src/course_work/models/revin.py` (implementation contract)
- `src/course_work/sweeps/revin.py` (Phase 40 helpers)
- `src/course_work/training/engine.py`
- `src/course_work/evaluation/metrics.py`
- `src/course_work/data/datasets.py`, `scaling.py`, `windows.py`, `features.py`, `feature_sets.py`
- All Test, Validation, and scaler targets
- `requirements.txt`, `pyproject.toml`

### 7.2 Files that may be modified ONLY after explicit Human approval of THIS plan

- `scripts/phase40_strict_best_rn1.py` — to add explicit MPS-required branch (Tier 1) and to add explicit relative-arm documentation (Tier 2 amendment)
- `artifacts/sweeps/S18_revin/phase_40_signoff.json` — to correct `tolerance` field and add explicit `abs_tolerance`/`rel_tolerance` fields, and to add explicit `verification_device_policy` field referencing THIS plan as its authority
- `artifacts/sweeps/S18_revin/s18_revin_sweep_report.md` — to reflect corrected strict-BEST verdict
- `docs/save_log_in_processing/phase_40_s18_revin_log.json` — to record the corrective action
- `tests/unit/test_revin.py` — to add a focused test asserting the strict-BEST verifier contract

### 7.3 Files that MAY be created (only after explicit Human approval)

- A new corrective plan is not needed (this document IS the plan)
- A new sign-off generation script is NOT needed; existing sign-off will be corrected atomically

### 7.4 Artifacts that must remain valid

- All 21 existing `artifacts/sweeps/S18_revin/*` files (19 non-history) must NOT be deleted; corrections happen by `signed_replacement` only.
- `RUN_TR_S18_0031_A711A9B8` scientific config / metrics / status MUST remain checksum-stable.

## 8. Validation steps (after Human approves THIS plan)

```text
1. Read-only audit re-confirms RN1 stored metrics are unchanged (sha256 of best_validation_metrics.json).
2. If MPS is available on this host:
     a. Re-run scripts/phase40_strict_best_rn1.py with recorded_device = mps.
     b. Expected: delta < 1e-9, PASS under strict absolute tolerance.
     c. If PASS: write `recorded_device_pass_status: PASS` to phase_40_signoff.json.
     d. Verify no side effects on RN1 scientific artifacts.
3. If MPS is NOT available:
     a. Do NOT silently CPU-recompute and report PASS.
     b. Record `recorded_device_pass_status: NOT_VERIFIABLE_ON_CURRENT_DEVICE` and request Human decision per Section 10 below.
4. After verification, materialize corrected O40 artifacts ONLY IF Human has approved this plan.
5. Do NOT start Phase 41.
6. Run focused tests/unit/test_revin_strict_best.py (after test added) and confirm PASS.
7. Regression: RN0 reference (RUN_TR_S14_0023) must NOT be retrained.
8. Test firewall: confirm no Test loader is materialized; confirm test_firewall_passed=true.
```

## 9. Regression steps

```text
1. Confirm RN0 reference RUN_TR_S14_0023_A711A9B8 sha256 unchanged.
2. Confirm RN1 RUN_TR_S18_0031_A711A9B8 sha256 of config.json / status.json / metrics/best_validation_metrics.json unchanged.
3. Confirm Phase 39 sign-off (artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json) sha256 unchanged.
4. Run existing tests/unit/test_revin.py and tests/unit/test_revin_dry_run.py — must remain PASS.
5. Run tests/unit/test_revin_applicability_gate.py — must remain PASS.
6. Run scripts/run_single_condition.py S18_REVIN RN1 --dry-run — must remain PASS.
7. Confirm `artifacts/sweeps/S18_revin/phase_40_signoff.json` is the ONLY Phase 40 artifact touched, and only after explicit Human approval.
```

## 10. Sign-off correction policy

The current `phase_40_signoff.json` reports `status: PASS` with `tolerance: 1e-06` while its own numeric deltas exceed that absolute tolerance. Two options for sign-off correction, both gated by Human approval of THIS plan:

**Option A (recommended, matches Phase 37 precedent):**
```json
"strict_best_verification": {
  ...
  "tolerance": null,
  "abs_tolerance": 1e-09,
  "rel_tolerance": null,
  "comparison_rule": "abs(stored - recomputed) <= 1e-9",
  "device_policy": "RECORDED_DEVICE_REQUIRED",
  "recorded_device": "mps",
  "verification_device": "mps" or "NOT_VERIFIABLE_ON_CURRENT_DEVICE",
  "verification_status": "PASS" or "NOT_VERIFIABLE_ON_CURRENT_DEVICE"
}
```

**Option B (only if Human explicitly approves CPU fallback):**
```json
"strict_best_verification": {
  ...
  "tolerance": 1e-06,
  "abs_tolerance": 1e-06,
  "rel_tolerance": 1e-06,
  "comparison_rule": "math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) - per Phase 35/36/39 sweep convention",
  "device_policy": "RECORDED_DEVICE_PREFERRED_CPU_FALLBACK_AUTHORIZED",
  "fallback_authority": "phase_40_s18_revin_strict_best_corrective_plan.md",
  "recorded_device": "mps",
  "verification_device": "cpu (MPS unavailable on this host; deterministic fallback; CPU fallback authorized per Phase 40 corrective plan)",
  "verification_status": "PASS"
}
```

The current sign-off must be corrected to one of these two options. **No silent correction; both require Human approval**.

## 11. Test firewall

```text
- Test data and Test targets MUST NOT be touched.
- Test loader MUST NOT be instantiated.
- best_validation_metrics.json MUST NOT include any Test split metric.
- All existing tests/unit/test_revin* must remain PASS.
- New test (tests/unit/test_revin_strict_best.py) MAY be added to lock down the chosen Option A or Option B contract. It MUST NOT read Test data.
```

## 12. Human approval gate

```text
USER_APPROVED_PREPROCESS_PLAN = false (initially)

Execution may not begin until:
  1. Human reviews this plan in full.
  2. Human explicitly states one of:
     (a) "Approve Tier 1 (Candidate 1: same-device MPS only). Do not implement Option B."
     (b) "Approve Tier 1 + Option B (Candidate 1 + Candidate 4 documented fallback)."
     (c) "Approve Option B only (Candidate 3 without Tier 1)."  ← NOT RECOMMENDED.
     (d) "Block Phase 40. Carry RN0 to Phase 41 with sign-off marked BLOCKED."
  3. The choice is recorded in docs/save_log_in_processing/phase_40_s18_revin_log.json under corrective_decision.
```

## 13. Recommended decision (recorded as AI recommendation, NOT executed without approval)

**Recommended:** Option (b) — Tier 1 (Candidate 1: same-device MPS required) AND Tier 2 (Candidate 4: explicit 1e-6 isclose with documented CPU fallback) with Human approval.

Rationale:

1. Tier 1 produces the most defensible scientific verdict (MPS-recomputed metrics within 1e-9 of stored).
2. Tier 2 is needed only when the verification host lacks MPS, which is the current host condition. Without Tier 2 explicit policy, Phase 40 cannot be signed off on a CPU-only host.
3. Both tiers together close the existing sign-off inconsistency by making the tolerance, comparison operator, and device policy explicit.
4. The RN0 winner decision (RN0 RMSE 57.70 < RN1 RMSE 62.29) is unchanged regardless of which Tier passes.
5. Scientific results, RN1 weights, RN0 weights, all upstream Phase artifacts, and Test data remain untouched.

## 14. Files this plan will require changing (after Human approval)

```text
scripts/phase40_strict_best_rn1.py    → add MPS-required branch with explicit RuntimeError
                                        add Option B fallback branch (only when --cpu-fallback-authorized)
                                        expose abs_tolerance / rel_tolerance / device_policy fields
artifacts/sweeps/S18_revin/phase_40_signoff.json  → atomic correction per chosen Option
docs/save_log_in_processing/phase_40_s18_revin_log.json  → record corrective decision and outcome
tests/unit/test_revin_strict_best.py  → NEW (locks the chosen contract)
```

No source code files outside `scripts/phase40_strict_best_rn1.py`. No `src/course_work/*` modification. No Phase 0-39 artifact modification. No Test data access. No retraining.

## 15. Acceptance criteria

```text
[ ] This plan is read and explicitly approved by Human.
[ ] The chosen Option (A or B) is recorded in corrective_decision.
[ ] RN1 scientific config / metrics / status sha256 unchanged.
[ ] RN0 reference RUN_TR_S14_0023_A711A9B8 sha256 unchanged.
[ ] Phase 39 sign-off sha256 unchanged.
[ ] scripts/phase40_strict_best_rn1.py implements the chosen Option.
[ ] phase_40_signoff.json is corrected atomically with explicit tolerance / device_policy fields.
[ ] All existing tests/unit/test_revin*.py remain PASS.
[ ] New tests/unit/test_revin_strict_best.py is added and PASSES.
[ ] --dry-run on S18_REVIN RN1 still PASSES.
[ ] Test firewall: no Test loader materialized; test_firewall_passed=true.
[ ] No CPU fallback silently used without Option B being explicitly approved.
[ ] No source code comments or icons introduced.
[ ] No new dependencies.
[ ] No Phase 41 work started.
[ ] Corrective outcome recorded in phase_40_s18_revin_log.json.
```

## 16. Rollback / stop conditions

```text
1. Human does not approve this plan → STOP. Phase 40 remains BLOCKED. Do not edit any file.
2. MPS verification fails on this host → record NOT_VERIFIABLE_ON_CURRENT_DEVICE and request Human decision between:
     (i) Run on MPS host (require Human to switch host)
     (ii) Approve Option B (Candidate 4 documented CPU fallback)
     (iii) Block Phase 40 entirely
3. Any unexpected regression in tests/unit/test_revin*.py → STOP and escalate.
4. Any sha256 drift on RN0 reference or RN1 scientific artifacts → STOP and escalate (forbidden modification).
5. Any Test data or Test loader materialization → STOP and escalate (Test firewall violation).
```

## 17. Open questions for Human

1. Is Option A (1e-9 absolute, MPS-required) the preferred Phase 40 contract?
2. If Option A's MPS path returns `NOT_VERIFIABLE_ON_CURRENT_DEVICE` on this host, do you approve Option B (1e-6 isclose with documented CPU fallback) as a Human-approved CPU-fallback authority?
3. Is the winner selection (RN0 RMSE 57.70 < RN1 RMSE 62.29) acceptable as the Phase 40 final winner regardless of which strict-BEST option is approved? (My read: yes — the winner selection is independent of strict-BEST numerical choice, but please confirm.)
4. Do you want the new tests/unit/test_revin_strict_best.py test added, or is correcting the existing sign-off + verifier sufficient?

## 18. References

| Path | Role |
|------|------|
| `docs/plan-doc/plan_detail_for_each_phase/Phase_40_S18_RevIN_sweep.md` | Phase 40 Detail (silent on tolerance/device policy) |
| `docs/plan-doc/plan_before_process/phase_37_strict_best_corrective_plan.md` | Phase 37 corrective — Phase 37 precedent (1e-9 abs, MPS required) |
| `docs/plan-doc/plan_to_refactor&fix/phase_37_device_consistent_fix.md` | Phase 37 device-consistent fix (Human-approved) |
| `docs/plan-doc/plan_before_process/phase_23_stale_signoff_resume_precedence_corrective_plan.md` | Phase 23 CPU-fallback prohibition |
| `src/course_work/sweeps/ffn.py:736-796` | Phase 36 canonical strict-BEST pattern (1e-6 isclose, CPU) |
| `src/course_work/sweeps/loss.py:760-841` | Phase 37 strict-BEST pattern (1e-9 abs, MPS required) |
| `scripts/phase40_strict_best_rn1.py` | Current ad-hoc Phase 40 verifier |
| `scripts/phase39_strict_best_gc0.py` | Phase 39 ad-hoc verifier (same pattern as Phase 40 current) |
| `artifacts/sweeps/S18_revin/phase_40_signoff.json` | Current inconsistent sign-off |
| `artifacts/sweeps/S17_gradient_clipping/phase_39_signoff.json` | Phase 39 sign-off (analogous precedent) |
| `artifacts/runs/RUN_TR_S18_0031_A711A9B8/` | RN1 scientific artifacts (UNTOUCHED) |
| `artifacts/runs/RUN_TR_S14_0023_A711A9B8/` | RN0 reference scientific artifacts (UNTOUCHED) |

---

## DRAFT — AWAITING HUMAN APPROVAL

Per `working_rule.md` chương 10, 12, 16, 25 và `rule_code.md` rules 11-12, 58, 117:

```text
PRE-PROCESS PLAN CREATED
→ STOP
→ AWAIT USER VERIFY + APPROVAL
→ NO EXECUTION UNTIL APPROVAL RECEIVED
```

The corrective plan is DRAFT. No implementation will occur. Phase 41 remains blocked. Test remains untouched. RN0 and RN1 scientific artifacts remain untouched.
