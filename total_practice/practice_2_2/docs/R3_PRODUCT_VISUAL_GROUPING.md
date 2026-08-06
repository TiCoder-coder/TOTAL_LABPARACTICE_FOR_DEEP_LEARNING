# R3 Product and Visual Grouping

## Scope

R3 builds deterministic connected components for the original image assets inventoried by R2. It does not create Train, Validation, or Test assignments, does not train a model, does not evaluate Test, and does not modify or delete source images.

Because R2 semantic review is incomplete, every R3 member is marked as a provisional original asset pending R2 acceptance. The grouping artifacts must not be treated as an approved training dataset until the predecessor gates pass.

## Input authority

- Dataset: `data/final/data_clean_balanced`
- R2 inventory: `artifacts/new_work/r2_provenance_semantic_audit_v1/legacy_inventory.json`
- R2 verification: `docs/R2_VERIFICATION.json`
- Grouping seed: `42`
- Included assets: 2,896 original, decodable images
- Excluded assets: 306 generated derivatives
- Dataset SHA-256: `fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5`

## Identity evidence

R3 checks four deterministic identity keys:

1. Product ID when provenance is available.
2. Exact raw-file SHA-256.
3. Exact decoded-pixel SHA-256.
4. R2 source group.

An identity edge is automatically unioned only when both assets have the same class label. A cross-label identity edge is never unioned and is placed in the manual review queue.

The current legacy inventory has zero product-ID coverage. Product-level independence is therefore not proven, even though every provisional original receives a component ID.

## Visual evidence

Each image is represented by four complementary signals:

- pHash captures low-frequency perceptual structure.
- dHash captures relative grayscale transitions.
- SSIM compares local luminance, contrast, and structure at a fixed 96 by 96 resolution.
- A 512-dimensional ResNet18 ImageNet embedding captures pretrained visual semantics.

The embedding checkpoint is torchvision `ResNet18_Weights.IMAGENET1K_V1`. Its full SHA-256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Embeddings are L2-normalized and extracted on CPU in sorted asset-ID order without shuffling.

Candidate pairs enter detailed comparison when they are among the eight nearest embedding neighbors with cosine similarity at least 0.94, have pHash distance at most 8, or have dHash distance at most 5.

A same-label visual pair is automatically confirmed only when all four strict conditions hold:

- pHash distance at most 4;
- dHash distance at most 4;
- SSIM at least 0.95;
- embedding cosine similarity at least 0.98.

All other candidates remain review-only. Every cross-label candidate remains review-only regardless of score. R3 never converts a visual score directly into a cross-label merge or quarantine decision.

## Connected components

Confirmed same-label identity and visual edges are unioned with a deterministic union-find implementation. Each component ID is derived from the SHA-256 of its sorted member asset IDs and begins with `r3g_`. The ID therefore does not depend on traversal order.

Current output:

- Components: 2,462
- Singleton components: 2,042
- Largest component: 4 images
- Confirmed identity or visual edges: 448
- Cross-label components: 0
- Assets assigned exactly once: 2,896 of 2,896

Two consecutive grouping runs produced identical SHA-256 values for `group_manifest.json`, `visual_edges.json`, `accepted_edges.json`, `review_queue.json`, `group_report.json`, and `docs/R3_VERIFICATION.json`.

## Manual review queue

R3 documents 2,325 unresolved candidate edges:

- Cross-label edges: 1,023
- Same-label uncertain edges: 1,302

Each pair has a deterministic `edge_id`, both asset IDs and paths, both labels, all available visual scores, review status, decision, reviewer, timestamp, and notes fields. Reviewers must use one of the decisions declared in `configs/r3_grouping_policy.json`.

A confirmed cross-label conflict may be quarantined only after a human decision is recorded. The current confirmed conflict count is zero and R3 performed no automatic quarantine.

## Artifacts

- `pretrained_embeddings.npz`: normalized feature matrix and ordered asset IDs.
- `embedding_metadata.json`: model, weight checksum, shape, and execution metadata.
- `phashes.json`: pHash value for every included asset.
- `visual_edges.json`: all candidates scored by the four visual signals.
- `accepted_edges.json`: automatically confirmed same-label edges used for union.
- `review_queue.json`: all unresolved same-label and cross-label edges.
- `group_manifest.json`: deterministic connected components.
- `group_report.json`: gate result and aggregate statistics.
- `artifact_manifest.json`: size and SHA-256 for every persisted R3 output.
- `docs/R3_VERIFICATION.json`: human-readable location for the machine gate report.

## Gate result

R3 is blocked for four explicit reasons:

1. R2 has not passed.
2. Product provenance coverage is zero.
3. Cross-label visual review is incomplete.
4. Same-label uncertain-edge review is incomplete.

This blocked state is required. It prevents provisional components from being presented as leakage-safe groups. R4 may begin only after the unresolved reviews and predecessor gates are complete or after a separately approved exception is documented.
