# R6 Domain-Safe Transform Ablation

## Scope

R6 audits preprocessing alternatives for the cosmetic-product classification domain. It compares T0 through T4 with one controlled configuration change per child recipe, applies every recipe to all 2,034 candidate Train assets, generates transformed contact sheets, and measures crop retention, horizontal flips, possible text reversal, Random Erasing, black-border artifacts, and ImageNet normalization validity.

R6 does not train a model, evaluate Validation, evaluate Test, mutate source images, or execute R7. R5 and the candidate manifest remain blocked, so R6 cannot select or advance a recipe.

## Content authority

- Authorized content split: Train
- Candidate Train assets: 2,034
- Held-out manifest assets: 862
- Train assets opened: 2,034
- Validation content access: 0
- Test content access: 0
- Invalid or held-out asset IDs in the access log: 0

Every image-opening path verifies that its manifest row belongs to Train. The final content access manifest exactly equals the Train asset set.

## Controlled recipes

| Recipe | Parent | Only changed variable |
|---|---|---|
| T0 | none | Existing baseline |
| T1 | T0 | Replace Resize 256 plus CenterCrop 224 with direct Resize 224 |
| T2 | T1 | Raise RandomResizedCrop scale floor from 0.70 to 0.90 |
| T3 | T2 | Set horizontal-flip probability from 0.50 to 0.00 |
| T4 | T2 | Set Random Erasing probability from 0.10 to 0.00 |

All recipes retain light ColorJitter with brightness, contrast, and saturation at 0.20 and hue at 0.05. All model tensors use ImageNet mean and standard deviation. Random operation seeds are derived independently from seed 42, asset ID, and operation name. Shared operations therefore use the same random draw across parent and child recipes.

T3 and T4 are independent children of T2. T3 isolates flip removal while retaining erasing. T4 isolates erasing removal while retaining flips. R6 does not combine both changes because that would no longer be a one-variable comparison against T2.

## Full Train audit

R6 performs 10,170 recipe-asset transformations.

| Recipe | Mean Train area retained | Minimum retained | Evaluation area retained | Flips | Erasing |
|---|---:|---:|---:|---:|---:|
| T0 | 80.12% | 70.02% | 76.56% | 1,005 | 224 |
| T1 | 80.12% | 70.02% | 100.00% | 1,005 | 224 |
| T2 | 94.29% | 90.01% | 100.00% | 1,005 | 224 |
| T3 | 94.29% | 90.01% | 100.00% | 0 | 224 |
| T4 | 94.29% | 90.01% | 100.00% | 1,005 | 0 |

T1 preserves the complete evaluation image instead of discarding approximately 23.44% of source area. T2 preserves substantially more Train area than T1. T3 removes all possible text reversal caused by horizontal flips. T4 removes all Random Erasing events.

The mean erasing area across all Train assets is approximately 0.64% for T0 through T3 because only 224 of 2,034 deterministic draws activate erasing. T4 has zero erased area.

## Black-border audit

The five recipes use only in-bounds crop and resize operations. They do not use affine transformation, rotation, perspective projection, padding, or out-of-bounds sampling. The audit records zero out-of-bounds fill pixels and zero introduced black-border artifacts for every recipe.

Near-black border fractions remain descriptive because some source products and layouts contain genuinely black content at image edges. An earlier provisional detector based only on increasing near-black edge pixels was rejected after it confused cropped black products with generated padding. The final artifact criterion is tied to actual out-of-bounds fill generation.

## Normalization audit

Every transformed Train and evaluation image is converted to a three-channel float tensor and normalized with:

- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

All 10,170 recipe-asset checks produce finite normalized tensors. Non-finite count is zero.

## Transformed contact sheets

R6 creates 50 deterministic contact sheets, one for every recipe-class pair. Each sheet contains 12 Train assets and has three columns in fixed order:

1. Original image.
2. Train transform output.
3. Evaluation transform output.

Visual inspection confirms the expected controlled differences. T0 can crop products and promotional text more aggressively in both Train and evaluation views. T2 preserves more source area but still reverses text when a flip draw activates. T3 preserves the lighter crop while removing text reversal. No inspected sheet contains transform-generated black padding.

These observations are diagnostic only. The 50 recipe-class review records remain pending for dominant-product clipping, visible black borders, harmful text reversal, and erasing of discriminative cues.

## Validation decision boundary

The R6 exit gate requires an explicit decision on horizontal flips from Validation evidence. Validation is unavailable because the predecessor gates have not passed. R6 therefore records:

- Validation evidence used: false
- Selected recipe: null
- Advanced recipes: 0
- Automatic selection performed: false

T3 appears qualitatively safer for text-bearing images, but this observation is not promoted to a recipe decision. Selecting it now would violate the declared gate.

## Artifacts

- `recipe_specs.json`: immutable T0-T4 parent and configuration definitions.
- `transform_audit.json`: per-asset metrics for all five recipes.
- `transform_summary.json`: aggregate crop, flip, erasing, border, and normalization results.
- `transformed_contact_sheets/`: 50 recipe-class comparison sheets.
- `contact_sheet_index.json`: exact Train asset IDs and column order for every sheet.
- `transform_review_template.json`: pending manual review fields for every recipe-class sheet.
- `content_access_manifest.json`: exact Train-only access evidence.
- `transform_ablation_report.json`: R6 gate result.
- `artifact_manifest.json`: size and SHA-256 for every R6 output.
- `docs/R6_VERIFICATION.json`: machine-readable R6 verification report.

## Gate result

R6 remains blocked because:

1. R5 has not passed.
2. The candidate Train manifest is not authorized for model use.
3. All 50 transformed contact-sheet reviews are pending.
4. Valid Validation evidence for the text-reversal decision is unavailable.
5. No single recipe has been authorized to advance.

Train-only isolation passes, black-border audit passes, and normalization audit passes. These checks do not override the predecessor, manual-review, and Validation-selection gates.

Two consecutive full R6 executions produced identical SHA-256 values for recipe specs, the complete transform audit, transform summary, representative PNG contact sheets, and the R6 verification report.
