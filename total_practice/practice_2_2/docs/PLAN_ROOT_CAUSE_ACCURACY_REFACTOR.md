# Practice 2.2 Accuracy Root-Cause Audit and Refactor Plan

## 1. Document status

- Status: `PROPOSED_FOR_REVIEW`
- Scope: analysis and implementation planning only
- Implementation authorized: `NO`
- Canonical notebook mutation authorized: `NO`
- Final Test re-evaluation authorized: `NO`
- Audited notebook: `notebooks/04_canonical_report.ipynb`
- Frozen canonical lineage: `canonical_26dc4625_52aaf974_s42_v1`
- Newer visual-safe lineage: `v2_visual_group_stratified_s42`

This document diagnoses why Validation Accuracy is approximately 78.31% and
Final Test Accuracy is approximately 76.36%. It defines a gated refactor plan,
but does not authorize editing, retraining, model selection, or Test access.

## 2. Executive conclusion

The evidence does not support the hypothesis that Accuracy is low because a
split is missing labels. Every class is represented in Train, Validation, and
Test, class imbalance is mild, predictions do not collapse to a subset of
classes, and Validation/Test Accuracy are statistically consistent.

The strongest root causes are:

1. Dataset contamination and ambiguous image-label semantics.
2. Missing product-level provenance and incomplete visual grouping.
3. Too little effective diversity for the amount of trainable model capacity.
4. Product-domain transforms that can remove or corrupt useful text and shape.
5. A confirmed weighted epoch-loss aggregation bug in the canonical path.

The model is not simply too weak. E2 reaches 98.31% Train Accuracy, and the
visual-safe E2 reaches 98.27% Train Accuracy while Validation remains 69.12%.
This is direct evidence of memorization and weak generalization.

The frozen 78.31%/76.36% result must remain historical. A refactor must create a
new lineage and must not overwrite or repeatedly evaluate the old Final Test.

## 3. Verification limitations

The active checkout currently has no `data/` directory, canonical manifest,
canonical checkpoints, prediction CSV, visual-leakage candidate CSV, or V2
manifest. Therefore, the audit can verify stored notebook outputs, reports,
source contracts, history JSON, confusion data embedded in the notebook, and
archived figures, but it cannot inspect all 3,202 source images or recompute the
split from current files.

Restoring the exact dataset and manifests is a mandatory gate before any
implementation. Missing resources must never be regenerated and represented as
the old canonical lineage.

## 4. Phase-by-phase findings

### Phase 1 - Problem Definition

Status: `PARTIALLY_WELL_DEFINED`

The ten labels are valid as business categories, but several are not reliably
separable from appearance alone. Bottles and tubes for `body_wash`, `shampoo`,
`facial_cleanser`, `toner`, `serum`, `moisturizer`, and `sunscreen` can be
visually similar. Product text is often the strongest class signal.

The current objective lacks explicit rules for:

- multi-product bundles;
- lifestyle or before/after images without a dominant product;
- text-only marketing banners;
- products serving more than one category;
- non-cosmetic query matches;
- whether classification targets the dominant product, listing category, or
  image content.

Without an annotation contract, some errors are unresolvable label ambiguity
rather than model failures.

### Phase 2 - Environment Setup

Status: `NOT_REPRODUCIBLE_IN_CURRENT_CHECKOUT`

The notebook is correctly artifact-only and Test-safe by design. However, its
artifact gate currently fails because the active registry resources are absent.
The environment dependency specification is also incomplete.

### Phase 3 - Data Loading and Split

Status: `CLASS_COVERAGE_PASS; VISUAL_GROUPING_NOT_PROVEN`

Canonical class counts are:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| body_wash | 207 | 49 | 47 |
| face_mask | 202 | 46 | 49 |
| facial_cleanser | 230 | 46 | 45 |
| lipstick | 200 | 42 | 44 |
| moisturizer | 157 | 38 | 33 |
| perfume | 199 | 44 | 44 |
| serum | 207 | 45 | 48 |
| shampoo | 194 | 41 | 39 |
| sunscreen | 223 | 47 | 48 |
| toner | 197 | 40 | 43 |

Findings:

- All ten classes occur in all three splits.
- The Train max/min count ratio is approximately 1.46, which is mild.
- Per-class split ratios remain close to 70/15/15.
- Final predictions cover all ten classes, with predicted counts from 34 to 52.
- Missing labels and class collapse are rejected as root causes.

The split groups exact hashes, generated families, and narrowly defined
perceptual duplicates. It cannot recover original Tiki product identity because
the crawler does not persist `product_id` or image URL and names files only with
a sequential image counter.

V2.5A later found strong cross-split visual near-duplicates despite metadata
isolation. V2.5B built a visual-safe split, and E2 Validation Accuracy dropped
to 69.12%. This does not prove that every V2 leakage pair crosses the frozen
canonical split, because the canonical manifest is unavailable for re-audit.
It does prove that the old grouping features are not sufficient evidence of
product-level independence.

### Phase 4 - Exploratory Data Analysis

Status: `INSUFFICIENT_FOR_DATA_READINESS`

The canonical notebook shows class counts but does not audit semantic label
quality, product diversity, source-template bias, or visual near-duplicates.

Later train-only EDA reports:

- only about 2,027 Train images;
- 729 heuristic review candidates;
- very bright images with RGB mean near `[0.780, 0.761, 0.750]`;
- effective-group diversity ratios of approximately 84.6% to 94.2%;
- stronger duplicate concentration in `body_wash` and `shampoo`.

Archived qualitative figures reveal material data problems:

- a `toner` sample shows flower-food packaging rather than cosmetic toner;
- some samples are before/after or lifestyle images without a clear product;
- many images are promotional collages dominated by text and template layout;
- repeated Tiki Trading banners provide shortcut features unrelated to class;
- several errors are visually or semantically ambiguous even to a human.

These figures are historical and cannot quantify total label-noise prevalence,
but they are sufficient to reject a `DATA_READY` conclusion without a manual
semantic audit.

### Phase 5 - Data Preprocessing

Status: `TECHNICALLY_VALID; DOMAIN_ABLATION_REQUIRED`

ImageNet normalization is appropriate as the default for ImageNet-pretrained
weights and is not a leading root cause.

Potentially harmful domain interactions are:

- source files were already resized/cropped to 224x224, then evaluation applies
  `Resize(256) + CenterCrop(224)`, which crops the visible area again;
- `RandomResizedCrop(scale=(0.70, 1.00))` can remove product labels or edges;
- horizontal flips reverse product text;
- Random Erasing can remove the small discriminative label region;
- E4 rotation/affine produces black borders and stronger visual distortion.

E4 reduced the generalization gap but also reduced Validation Accuracy from
78.31% to 72.37%, so simply increasing generic augmentation is rejected.

### Phase 6 - Model Building

Status: `CAPACITY_MISMATCH; NOT MODEL_WEAKNESS`

E1 trains only 5,130 parameters and underfits at approximately 59.82% Validation
Accuracy. E2 trains approximately 8.4 million parameters and reaches 98.31%
Train Accuracy. E3 reduces trainable depth and falls to 70.32% Validation.

ResNet18 therefore has enough capacity to memorize the current Train set. A
larger network by itself is not a justified fix. Better pretrained features,
higher input resolution, or OCR-aware features may help only after the data
contract and split are repaired.

### Phase 7 - Model Training

Status: `OVERFITTING_CONFIRMED; LOSS_REPORTING_BUG_CONFIRMED`

E2 reaches its best Validation Accuracy at epoch 14:

- Train Accuracy: 98.31%
- Validation Accuracy: 78.31%
- gap: 20.00 percentage points

On the newer visual-safe split, E2 reaches:

- Train Accuracy: 98.27%
- Validation Accuracy: 69.12%
- gap: 29.15 percentage points

The canonical weighted CrossEntropy path aggregates epoch loss using
`loss.item() * batch_size`. Weighted CrossEntropy mean reduction is normalized
by the sum of target class weights, not by batch size. Consequently:

- Train and Validation epoch loss are mathematically incorrect;
- ReduceLROnPlateau receives an incorrectly aggregated signal;
- early stopping receives an incorrectly aggregated signal;
- the reported Final Test Loss is also incorrectly aggregated.

Per-batch gradients, predictions, Accuracy, confusion matrix, and F1 are not
directly changed by this aggregation bug. E2 also trained all 15 epochs, so this
bug is a correctness defect but is unlikely to explain the full Accuracy gap.

Class weighting is not strongly justified because imbalance is mild. Canonical
weights range from approximately 0.877 to 1.284. An unweighted baseline must be
run before retaining weighted loss.

The current E2 also fine-tunes layer4 immediately while the classifier is newly
initialized. A head-warmup stage followed by low-LR gradual unfreezing is a
better controlled transfer-learning protocol for a small dataset.

### Phase 8 - Controlled Experiments

Status: `USEFUL_BUT_INCOMPLETE`

E1-E4 establish that head-only underfits, full layer4 overfits, reduced depth
loses adaptation, and stronger generic augmentation hurts. They do not test:

- corrected unweighted loss versus weighted loss;
- head warmup followed by gradual unfreezing;
- domain-safe transforms without text reversal and double cropping;
- multiple seeds;
- a stronger pretrained feature extractor under the same data contract;
- higher-resolution or OCR-assisted classification.

E4 changes several transform operations together, so it cannot attribute which
operation caused the loss in Accuracy.

### Phase 9 - Model Selection

Status: `TEST_SAFE; HIGH_VARIANCE`

Selection is correctly Validation-only. However, one split and one seed are
used. With 438 Validation samples, one prediction changes Accuracy by about
0.228 percentage points. The 95% Wilson interval for 343/438 correct is
approximately 74.21% to 81.92%.

Future selection must use repeated seeds and report mean, standard deviation,
and per-class metrics. Test must remain unavailable until one configuration is
frozen.

### Phase 10 - Final Test Evaluation

Status: `CONSISTENT_WITH_VALIDATION; FROZEN_HISTORICAL_RESULT`

The 95% Wilson interval for 336/440 correct is approximately 72.18% to 80.09%.
The Validation/Test difference is 1.95 percentage points, with an approximate
two-proportion p-value of 0.49. There is no evidence that Test is unexpectedly
lower than Validation or that a simple Validation/Test distribution mismatch
caused the result.

The old Final Test has been viewed and analyzed. It cannot be repeatedly reused
to guide and validate a refactored model while retaining an unbiased `Final
Test` claim.

### Phase 11 - Error Analysis

Status: `SUPPORTS_DATA_AND_ONTOLOGY_ROOT_CAUSE`

The largest confusion is `body_wash -> shampoo` with 10 of 47 body-wash Test
images. Serum is confused with sunscreen, toner, and moisturizer. These classes
share bottle/tube shape and differ heavily through packaging text or product
function.

Several high-confidence errors occur in consecutive filenames and repeated
marketing layouts. This supports source-product correlation and shortcut
learning. Macro F1 and weighted F1 are both approximately 0.762, so global class
imbalance is not the dominant failure mode.

### Phase 12 - Reproducibility and Conclusion

Status: `HISTORICALLY_RECORDED; NOT CURRENTLY_EXECUTABLE`

Stored hashes and the one-time guard document the old run, but the active
resources needed to verify them are absent. The notebook should remain a frozen
historical report, not become the mutable workspace for a new lineage.

## 5. Root-cause ranking

| Rank | Candidate cause | Confidence | Effect |
|---:|---|---|---|
| 1 | Semantic contamination, weak label contract, ambiguous images | High | Lowers achievable generalization and produces systematic confusions |
| 2 | Missing product provenance and incomplete visual grouping | High | Inflates old estimates and reduces effective independent diversity |
| 3 | Small effective dataset versus 8.4M trainable parameters | High | Produces 20-29 point Train/Validation gaps |
| 4 | Product-inappropriate crop/flip/erase transforms | Medium-high | Removes text/product cues and adds artifacts |
| 5 | Weighted epoch-loss aggregation bug | Confirmed | Corrupts loss, scheduler, early stopping, and Test-loss reporting |
| 6 | Immediate layer4 fine-tuning from a random classifier head | Medium | Can destabilize pretrained features and accelerate memorization |
| 7 | ResNet18 feature ceiling at 224px | Medium | May limit fine-grained packaging/text recognition after data repair |
| 8 | Mild class imbalance | Low | Not supported by Macro/weighted metrics or prediction distribution |
| 9 | Missing labels in a split | Rejected | Every class is present in all splits |
| 10 | Validation/Test mismatch | Rejected | Accuracy difference is statistically consistent with sampling noise |
| 11 | Too few epochs | Rejected | Model already nearly memorizes Train and peaks at epoch 14 |

## 6. Refactor principles

1. Data correctness before model tuning.
2. Preserve the frozen canonical notebook and artifacts as historical evidence.
3. Create a new lineage; never overwrite a fingerprinted artifact.
4. One controlled variable per experiment whenever practical.
5. No Test predictions, per-class Test metrics, or Test error inspection before
   model and threshold freeze.
6. Record product-level provenance and group all images from the same product.
7. Report uncertainty across seeds, not only the best single run.
8. Do not promise a target Accuracy before measuring post-cleaning label quality.

## 7. Proposed implementation plan

### R0 - Restore and freeze authority

Objective: make the evidence executable without mutating it.

Actions:

- restore the exact 3,202-file dataset, canonical manifest, V2 manifests,
  checkpoints, and JSON/CSV artifacts from a verified backup;
- verify every existing SHA-256 against registry and migration records;
- classify resources as `frozen_historical`, `v2_experimental`, or `new_work`;
- make `04_canonical_report.ipynb` read-only in project policy;
- refuse implementation if restored hashes differ from recorded hashes.

Exit gate:

- canonical resource verifier passes;
- no training or Final Test DataLoader is constructed;
- a read-only inventory report records every restored resource.

### R1 - Define the label and image-content contract

Objective: make each target label operationally unambiguous.

Actions:

- document positive, negative, and borderline examples for each class;
- define handling for bundles, multi-label products, before/after images,
  lifestyle images, text-only banners, and non-cosmetic query matches;
- require one dominant target product per accepted image;
- create `accept`, `quarantine`, and `relabel` decisions with reason codes;
- prohibit automatic deletion during audit.

Exit gate:

- all ten classes have signed annotation rules;
- two reviewers agree on a pilot sample with a predeclared agreement threshold;
- ambiguous policy cases are resolved before full review.

### R2 - Rebuild provenance and semantic-quality audit

Objective: measure actual independent product diversity and label purity.

Actions:

- preserve raw images immutably;
- for future crawling, persist product ID, listing URL, image URL, query,
  category ID, image index, timestamp, and raw SHA-256;
- use Tiki category constraints and negative keywords, not query text alone;
- prioritize product packshots and reject unrelated detail-gallery images;
- manually review all 729 historical heuristic candidates;
- oversample manual review for the main confusion groups;
- review every cross-label visual/embedding neighbor;
- record label-noise rate and accepted unique-product count per class.

Exit gate:

- zero accepted images violate the annotation contract in the reviewed set;
- every accepted image has a stable semantic decision and audit trail;
- each class meets a declared minimum of unique product groups;
- classes below the minimum receive newly collected products, not offline image
  derivatives.

### R3 - Build product and visual groups

Objective: prevent same-product and near-duplicate leakage.

Actions:

- group by product ID when available;
- union product groups with exact-byte and decoded-pixel duplicates;
- add pHash, dHash, SSIM, and pretrained embedding neighbors;
- manually confirm uncertain edges near split boundaries;
- quarantine confirmed cross-label conflicts;
- store a deterministic connected-component ID for every accepted image.

Exit gate:

- product ID, exact hash, decoded hash, source group, and confirmed visual group
  each have zero cross-split violations;
- all remaining review-only cross-split pairs are documented;
- the grouping algorithm is deterministic under seed 42.

### R4 - Create a new stratified group split

Objective: create the new model-development lineage.

Actions:

- split connected product/visual groups, not individual images;
- target 70/15/15 while optimizing class counts and unique-product counts;
- guarantee every class appears in every split;
- keep generated derivatives excluded;
- seal the new Test split before any model training;
- create manifest, summary, fingerprints, leakage report, and Test guard.

Preferred lineage name:

`v3_product_visual_group_s42_v1`

Exit gate:

- all leakage assertions pass;
- class and unique-product distribution deviations are within declared limits;
- Test access count is zero;
- no old canonical metric is presented as directly comparable on the new split.

### R5 - Replace count-only EDA with train-only data readiness

Objective: prove the new Train set is learnable before model experiments.

Actions:

- report images and unique products per class;
- render deterministic per-class contact sheets;
- audit label purity, dominant-product visibility, text ratio, brightness,
  contrast, blur, crop loss, watermarks, banners, bundles, and people-only images;
- report duplicate-component size distribution;
- inspect embedding clusters and class-overlap neighborhoods;
- quantify source/vendor/template concentration;
- derive all pixel statistics from Train only.

Exit gate:

- no unresolved critical label candidates;
- no dominant source template explains a class;
- data-readiness report is generated from Train only;
- Validation and Test remain unused for content-driven tuning.

### R6 - Refactor preprocessing with one-variable ablations

Objective: preserve product and text cues while retaining useful invariance.

Baseline transform candidate:

- aspect-ratio-preserving resize plus padding or direct 224 resize;
- no second center crop when source is already 224x224;
- light crop with scale floor 0.90 if crop is retained;
- light ColorJitter;
- no horizontal flip in the first domain-safe baseline;
- no Random Erasing in the first domain-safe baseline;
- ImageNet normalization.

Required ablations:

| ID | Single change |
|---|---|
| T0 | Existing base transform on the new clean split |
| T1 | Remove second Resize/CenterCrop and preserve full product |
| T2 | T1 plus crop floor 0.90 instead of 0.70 |
| T3 | T2 plus horizontal flip on/off comparison |
| T4 | T2 plus light Random Erasing on/off comparison |

Exit gate:

- transformed contact sheets show no clipped dominant product;
- no black-border artifacts;
- text reversal is explicitly accepted or rejected from Validation evidence;
- only one transform recipe advances.

### R7 - Correct training mathematics and checkpoint loading

Objective: establish a trustworthy baseline implementation.

Actions:

- replace weighted epoch aggregation with a tested normalizer or use
  `reduction='sum'` plus the exact denominator;
- apply the same aggregator to Train, Validation, and any future Test pass;
- start with unweighted CrossEntropy because imbalance is mild;
- test class weighting and label smoothing only as separate experiments;
- construct checkpoint models with `weights=None` before loading a complete
  state dict, avoiding an unnecessary network/cache dependency;
- log head and backbone learning rates separately;
- keep frozen BatchNorm statistics in evaluation mode;
- add unit tests for weighted/unweighted loss and offline checkpoint reload.

Exit gate:

- batch-partition-invariant epoch loss tests pass;
- scheduler and early stopping consume the corrected Validation loss;
- offline save/load predictions are identical;
- no Test loader is imported by the training entry point.

### R8 - Use staged transfer learning

Objective: reduce early backbone drift and memorization.

Protocol candidate:

1. Initialize one shared pretrained ResNet18 state.
2. Train the classifier head for 3 to 5 epochs.
3. Restore the best head-warmup checkpoint.
4. Unfreeze layer4 only.
5. Use head LR around `3e-4` to `1e-3` and backbone LR around `1e-5` to `3e-5`.
6. Use corrected early stopping and a predeclared scheduler.

Controlled experiments:

| ID | Change from clean ResNet18 baseline |
|---|---|
| M0 | Existing immediate layer4 fine-tuning |
| M1 | Head warmup then layer4 unfreeze |
| M2 | M1 with dropout 0.35 |
| M3 | M1 with label smoothing 0.05 |

Each experiment must start from the same initial pretrained tensor hash and use
the selected transform from R6.

Exit gate:

- candidate improves mean Validation metric across seeds, not only one peak;
- generalization gap decreases without material loss of Validation performance;
- no result is selected from Test evidence.

### R9 - Evaluate feature-extractor ceiling

Objective: test whether ResNet18 is the remaining bottleneck after data repair.

Actions:

- compare one stronger, efficient pretrained backbone such as EfficientNet-B0
  or ConvNeXt-Tiny under the same split, transform, seeds, and staged protocol;
- optionally test a higher input resolution only as a separate experiment;
- consider OCR-assisted or image-text features only after the image-only
  baseline is stable and the project scope permits it.

Exit gate:

- parameter count, latency, memory, and Accuracy/Macro F1 tradeoffs reported;
- architecture advances only on repeated-seed Validation evidence;
- a larger model is rejected if it only increases Train Accuracy or gap.

### R10 - Robust model selection

Objective: reduce single-seed and single-epoch selection noise.

Actions:

- use at least seeds 42, 123, and 2026 for finalists;
- predeclare primary metric as mean Validation Macro F1 and secondary metric as
  mean Validation Accuracy, or reverse them before experiments begin;
- report mean, standard deviation, Wilson intervals, per-class recall, and
  confusion stability;
- select one configuration, then train/freeze the final checkpoint according to
  a predeclared rule.

Exit gate:

- selected configuration wins consistently across seeds;
- selection record contains no Test-derived field;
- model/config/checkpoint hashes are frozen before Test authorization.

### R11 - One-time evaluation on the new Test lineage

Objective: obtain an unbiased final estimate for the refactored project.

Actions:

- prefer a newly collected, product-disjoint holdout;
- if using the untested V2/V3 Test split, prove it has never been used for model
  performance, threshold, or error-driven tuning;
- run exactly one final evaluation after freeze;
- report Accuracy, Macro F1, per-class metrics, confidence interval, confusion,
  calibration, and evaluation count;
- mark the old 76.36% result as historical and not directly comparable.

Exit gate:

- Test evaluation count equals one;
- model state is unchanged before/after inference;
- all output hashes and guard fields pass;
- no post-Test reselection or retraining occurs.

### R12 - Create a new report notebook

Objective: preserve historical evidence and present the new lineage clearly.

Actions:

- keep `04_canonical_report.ipynb` unchanged;
- create a new artifact-only notebook, for example
  `05_product_visualsafe_report.ipynb`;
- make each phase load only persisted new-lineage artifacts;
- include data-quality decisions, unique-product counts, leakage proof,
  repeated-seed results, uncertainty, and limitations;
- ensure Run All performs no training and no repeat Test evaluation.

Exit gate:

- all code cells execute from restored persisted resources;
- zero error outputs;
- notebook/HTML hashes recorded;
- all local links and registry paths pass.

## 8. Experiment ordering

The required order is:

1. Restore resources.
2. Fix ontology and data quality.
3. Build product/visual groups and new split.
4. Re-run the same ResNet18 baseline on the new clean split.
5. Fix training math and establish unweighted baseline.
6. Select domain-safe preprocessing.
7. Test staged fine-tuning.
8. Test one stronger backbone only if ResNet18 remains the bottleneck.
9. Repeat finalists across seeds.
10. Freeze once and evaluate the new Test once.

This order is mandatory for causal attribution. Model tuning before data repair
would optimize against an unreliable split and noisy labels.

## 9. Success criteria

Data integrity criteria:

- all accepted samples comply with the label contract;
- all accepted images have provenance or a documented legacy limitation;
- zero confirmed product/visual group crosses a split;
- unique-product counts and class distributions meet declared thresholds;
- no offline-generated derivative is used for model selection or evaluation.

Training correctness criteria:

- mathematically correct epoch loss;
- reproducible split and initialization hashes;
- offline checkpoint reload;
- frozen BatchNorm behavior verified;
- Train/Validation/Test transform isolation verified.

Model criteria:

- no fixed Accuracy target is guaranteed before data cleaning;
- finalist must improve repeated-seed Validation evidence;
- improvement must not be explained only by higher Train Accuracy;
- Macro F1 and weak-class recall must improve or remain acceptable;
- Test is evaluated once only after all decisions are frozen.

## 10. Files expected to change only after approval

Potential new or modified implementation surfaces are:

- crawl metadata and category filtering modules;
- non-destructive data-audit and annotation manifests;
- product/visual grouping and split modules;
- transform configuration and transform tests;
- canonical loss aggregation and checkpoint loading utilities;
- staged training entry point and experiment registry;
- resource registry for the new lineage;
- new artifact-only report notebook and documentation;
- targeted unit/integration tests.

The frozen canonical notebook, old checkpoint hashes, old Final Test artifacts,
and old registry evidence must not be rewritten.

## 11. Approval checklist

Before implementation, the reviewer must explicitly approve:

- [ ] preserving `04_canonical_report.ipynb` as frozen historical evidence;
- [ ] restoring exact missing resources before any refactor;
- [ ] creating a new lineage rather than overwriting canonical artifacts;
- [ ] semantic label and image-content audit before model tuning;
- [ ] product/visual group split as the new split authority;
- [ ] unweighted CE as the first corrected baseline;
- [ ] domain-safe transform ablations;
- [ ] staged head-warmup and layer4 fine-tuning;
- [ ] repeated-seed Validation selection;
- [ ] exactly one new-lineage Final Test evaluation after freeze.

No implementation phase may begin until this plan is approved.
