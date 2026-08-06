# R5 Train-Only Data Readiness

## Scope

R5 evaluates whether the candidate Train partition is ready for model development. It performs image-quality analysis, pixel statistics, component-diversity analysis, deterministic contact sheets, pretrained-embedding clustering, class-overlap neighborhoods, and semantic-review preparation.

R5 does not train a model, evaluate Validation, evaluate Test, alter source images, quarantine assets automatically, or execute R6. Because R4 is blocked, this run is a provisional Train-only audit and cannot authorize model development.

## Content authority

- Authorized content split: Train
- Candidate Train assets: 2,034
- Held-out manifest assets: 862
- Unique Train image assets opened: 2,034
- Validation image content opened: 0
- Test image content opened: 0
- Invalid or held-out asset IDs in the content access log: 0

The R4 manifest is loaded to select Train IDs and prove disjointness. Every image-opening function rejects rows whose split is not Train. The content access manifest exactly matches the 2,034 Train asset IDs.

The R3 embedding file was created before splitting for visual grouping. R5 builds an asset-index map, then materializes only the 2,034 rows selected by the Train manifest. Validation and Test embeddings do not participate in clustering, nearest-neighbor calculations, thresholds, or reports.

## Train distribution

| Class | Images | Components | Component diversity |
|---|---:|---:|---:|
| body_wash | 213 | 172 | 80.75% |
| face_mask | 209 | 175 | 83.73% |
| facial_cleanser | 226 | 192 | 84.96% |
| lipstick | 201 | 172 | 85.57% |
| moisturizer | 160 | 134 | 83.75% |
| perfume | 201 | 171 | 85.07% |
| serum | 211 | 183 | 86.73% |
| shampoo | 192 | 159 | 82.81% |
| sunscreen | 224 | 197 | 87.95% |
| toner | 197 | 170 | 86.29% |

All classes exceed the declared 80% provisional component-diversity threshold. The largest component contains four images and no class has a component accounting for more than 2.5% of its Train images.

Product IDs and vendor metadata are unavailable. Unique-product and vendor concentration therefore remain unknown and cannot be replaced by image count, source filename, or component count.

## Pixel statistics

All 102,057,984 pixels used for statistics come from Train images.

- RGB mean: `[0.77639, 0.76001, 0.74998]`
- RGB standard deviation: `[0.27712, 0.26809, 0.28430]`
- Mean grayscale brightness: `0.76213`
- Mean grayscale contrast: `0.19729`
- Image dimensions: exactly 224 by 224 for all Train assets

The high channel means confirm the bright-background characteristic already suggested by earlier EDA. These values are descriptive evidence only; R5 does not select normalization or preprocessing. That decision belongs to later gated phases.

## Automated image audit

R5 measures edge density, border-edge density, text-like connected regions, banner-band structure, brightness, contrast, Laplacian variance, and color-content ratio. These signals prioritize manual review but never make semantic decisions.

The Train review reasons are:

| Automated reason | Images |
|---|---:|
| Extremely bright | 701 |
| Border crop risk | 188 |
| High text-like area | 174 |
| Banner-band candidate | 97 |
| High-similarity cross-label neighbor | 21 |
| Extremely dark | 11 |
| Low color content | 4 |
| Low contrast | 2 |
| Potentially blurry | 1 |

Brightness and text heuristics are not rejection rules. Bright packshots and text-bearing product packaging may be valid. Border, text, and banner flags require image-level confirmation.

## Manual review contract

The review queue covers all 2,034 Train assets:

- P1: 1,039
- P2: 995
- Completed: 0
- Automatic accept, relabel, or quarantine decisions: 0

For every image, a reviewer must record semantic decision, proposed label, reason code, dominant-product visibility, text role, crop loss, watermark, banner, bundle status, bundle-policy compliance, people-only status, reviewer ID, and timestamp.

This review is required because label purity, dominant-product visibility, bundles, watermarks, banners, and people-only content cannot be established reliably from the automated metrics alone.

## Embedding clusters

R5 applies deterministic cosine k-means with seed 42 to the Train-only ResNet18 embeddings. It creates 20 clusters and converges in 33 iterations. Each cluster stores class composition, entropy, dominant-label share, assignments, and representative Train asset IDs.

The largest per-class cluster concentration is 38.97% for `body_wash`, exceeding the declared 35% review threshold. Visual inspection of the representative sheet shows repeated bottle forms and promotional layouts. This is a concentration signal, not proof that the cluster is an invalid source template. It remains a blocker until provenance and manual template review can distinguish legitimate class morphology from source-layout shortcuts.

## Class-overlap neighborhoods

For each Train asset, R5 finds its five nearest Train neighbors by cosine similarity.

- Top-1 neighbor has a different label for 440 images.
- Top-1 cross-label ratio: 21.63%.
- Unique cross-label neighbor pairs: 3,878.
- Cross-label pairs above the strict 0.95 review threshold contribute 21 Train review flags.

Prominent overlap counts include `facial_cleanser` with `sunscreen`, `body_wash` with `shampoo`, `serum` with `toner`, and `perfume` with `toner`. The pair contact sheet shows that highly similar promotional layouts can cross class labels. These results support manual semantic and template review before model experiments.

## Visual outputs

R5 generates 31 deterministic PNG files:

- 10 per-class contact sheets with 20 Train images each.
- 20 embedding-cluster representative sheets.
- 1 cross-label neighbor-pair sheet containing 12 highest-similarity Train-only pairs.

The images contain no model predictions and no Validation or Test content. Repeated executions use the same seed-derived asset ordering.

Two consecutive full R5 executions produced identical SHA-256 values for pixel statistics, class readiness, embedding clusters, embedding neighborhoods, the review queue, representative contact sheets, and the R5 verification report.

## Artifacts

- `train_image_audit.json`: Train image metrics and automated review reasons.
- `train_pixel_statistics.json`: Train-only RGB and image statistics.
- `class_readiness.json`: class, component, source-group, product, vendor, and cluster concentration summary.
- `embedding_cluster_report.json`: Train-only clusters and assignments.
- `embedding_neighbor_report.json`: Train-only neighborhoods and overlap counts.
- `train_review_queue.json`: full manual semantic-review template.
- `contact_sheet_index.json`: exact asset IDs used in every figure.
- `content_access_manifest.json`: complete Train content-access evidence.
- `data_readiness_report.json`: R5 gate result.
- `artifact_manifest.json`: size and SHA-256 for every R5 JSON and PNG artifact.
- `docs/R5_VERIFICATION.json`: machine-readable R5 verification report.

## Gate result

R5 remains blocked because:

1. R4 has not passed.
2. The candidate manifest is not authorized for model use.
3. Product provenance coverage is zero.
4. All 2,034 semantic reviews are pending.
5. There are 1,039 unresolved P1 review candidates.
6. Source and vendor provenance is unavailable.
7. The `body_wash` embedding-cluster concentration exceeds the review threshold.

The Train-only isolation gate passes, component diversity passes, and no dominant exact source group or duplicate component is detected. Those passes do not override the unresolved semantic and provenance gates, so `data_ready` remains false.
