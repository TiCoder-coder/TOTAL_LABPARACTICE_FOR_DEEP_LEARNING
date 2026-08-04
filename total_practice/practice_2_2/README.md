# Practice 2.2 — Canonical Cosmetic Product Classification

Practice 2.2 is a 10-class cosmetic-product image classification project using ImageNet-pretrained ResNet18. The submission is an educational, reproducible experiment from the persisted cleaned dataset; it is **not claimed to be production-ready**.

The official presentation is [`notebooks/04_canonical_report.ipynb`](notebooks/04_canonical_report.ipynb). Its `Run All` flow is read-only: it loads persisted artifacts through the active registry and never trains a model, creates a Test DataLoader, or repeats Final Test.

## Canonical result

| Item | Canonical value |
|---|---|
| Run ID | `canonical_26dc4625_52aaf974_s42_v1` |
| Winner | `E2_partial_finetune` |
| Architecture | ResNet18, full `layer4` + classifier trainable |
| Validation Accuracy | 78.31% |
| Validation Macro F1 | 0.7821 |
| Final Test Accuracy | **76.36%** |
| Final Test Macro F1 | **0.7617** |
| Final Test evaluations | **1** |

Canonical artifacts live only under:

```text
artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/
```

Any result outside this lineage is historical, ablation-only, or legacy and must not be reported as the final result. In particular, stale metrics such as 78.57% or 81.80% are non-canonical.

## Objective and dataset provenance

The objective is supervised single-label classification of cosmetic product images into:

```text
body_wash, face_mask, facial_cleanser, lipstick, moisturizer,
perfume, serum, shampoo, sunscreen, toner
```

- Source: Tiki.vn internal search API.
- Raw images collected: approximately 2,922.
- Final balanced directory: 3,202 files.
- Offline-generated derivatives: 306.
- Cross-class suspicious files quarantined: 2.
- Exact decoded-pixel duplicates remaining: 0 groups.
- Per-image source URLs were not persisted.
- Raw-data reconstruction is therefore not fully reproducible.

The original crawl/clean process removed corrupt or small files, exact duplicates and low-quality content, then resized images to 224×224. Offline derivatives were created historically for balancing, but the canonical model-use policy excludes every generated derivative.

## Leakage-safe canonical split

The canonical split is generated once with seed 42 and stored in:

```text
data/manifests/canonical_split/split_manifest.csv
```

| Split | Model-use originals |
|---|---:|
| Train | 2,016 |
| Validation | 438 |
| Test | 440 |

Exact hashes, generated families and perceptual duplicate clusters cannot cross split boundaries. Cross-class suspicious near-duplicates are quarantined. Test remained locked throughout model selection.

## Model and training protocol

- ResNet18 with `ResNet18_Weights.DEFAULT`.
- Image size 224 and ImageNet normalization.
- AdamW; head LR `1e-3`, backbone LR `1e-4`.
- Weight decay `2e-4`.
- Batch size 32, maximum 15 epochs.
- Class-weighted CrossEntropyLoss with label smoothing `0.05`.
- Dropout `0.20`, gradient clipping `1.0`.
- ReduceLROnPlateau; early stopping on Validation loss with patience 3.
- Best checkpoint selected only by Validation Accuracy.
- Seed 42 and deterministic behavior where supported.

## Controlled experiments

| Experiment | Strategy | Train Acc | Val Acc | Val Loss | Val Macro F1 | Gap | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| E1 | Head only | 59.87% | 59.82% | 1.3976 | 0.5975 | 0.05 | Underfit |
| E2 | Full layer4 + head | 98.31% | **78.31%** | **1.0218** | **0.7821** | 20.00 | Selected |
| E3 | layer4.1 + head | 90.08% | 70.32% | 1.2799 | 0.7017 | 19.76 | Rejected |
| E4 | Stronger online augmentation | 85.76% | 72.37% | 1.2491 | 0.7229 | 13.39 | Rejected |

E1 underfit. E3 reduced capacity but lost domain adaptation. E4 reduced memorization but also reduced Validation performance materially. E2 was frozen as winner entirely from Validation evidence; only E2 was evaluated on Final Test.

## Final Test

Final Test was evaluated exactly once after `final_selection.json` froze E2.

| Metric | Result |
|---|---:|
| Test Loss | 1.0187 |
| Test Accuracy | **76.36%** |
| Macro Precision | 0.7654 |
| Macro Recall | 0.7655 |
| Macro F1 | 0.7617 |
| Weighted F1 | 0.7622 |
| Correct / Incorrect | 336 / 104 |
| Samples | 440 |

Validation Accuracy was 78.31% and Test Accuracy was 76.36%, a difference of 1.95 percentage points. No retraining or model reselection occurred after Test results were viewed.

The five lowest-recall Test classes were serum, body_wash, facial_cleanser, sunscreen and moisturizer. The largest confusion pairs were body_wash→shampoo (10), serum→sunscreen (5), serum→toner (5), sunscreen→serum (5), and serum→moisturizer (4). This error analysis is descriptive only.

## Reproducibility and integrity

```text
Seed:                42
Dataset fingerprint: 26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6
Split fingerprint:   52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d
Checkpoint SHA-256:  4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3
```

Environment recorded by the canonical snapshot: Python 3.11.9, PyTorch 2.12.1, TorchVision 0.27.1, NumPy 1.26.4 and pandas 2.2.2. Exact package information and transform representations are stored in `config_snapshot.json`.

The completion guard is:

```text
final_test/FINAL_TEST_COMPLETED.json
FINAL_TEST_COMPLETED = true
final_test_evaluation_count = 1
repeat_evaluation_allowed = false
```

Experiment reproduction is supported from the current cleaned dataset and canonical manifest. Full raw-data reconstruction is only partial because source URLs and historical offline-generation scripts were not preserved.

## Inspect results safely

```bash
cd total_practice/practice_2_2
jupyter notebook notebooks/04_canonical_report.ipynb
```

`Run All` only reads the manifest, histories, comparisons and Final Test files. If required artifacts are missing, it raises a clear `FileNotFoundError`; it does not regenerate them.

Run automated tests without training or Test evaluation:

```bash
cd total_practice/practice_2_2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
```

## Project structure

```text
practice_2_2/
├── configs/                      # Active relative-path authority registry
├── data/final/                   # Canonical dataset authority
├── data/manifests/               # Canonical split and fingerprints
├── notebooks/                    # Official read-only report
├── src/practice_2_2/             # Recommended canonical import package
├── artifacts/canonical/          # E1/E2 lineage and locked Final Test artifacts
├── artifacts/ablations/          # E3/E4 controlled experiments
├── reports/html/                 # Static canonical report
└── tests/                        # Automated contracts and guards
```

Pre-migration resources inside this project are archived under
`archive/phase_7/pre_migration_practice_2_2/` as read-only compatibility
locations. Practice 2-owned duplicate notebooks, outputs, reports, source and
tests were moved to `archive/phase_7/`; the duplicate cosmetic dataset under
`practice_2/data/` was removed after byte-for-byte verification. Only documented
thin `processing_own_phase.*` forwarding wrappers remain in Practice 2.

## Limitations

- Per-image Tiki source URLs and product IDs are unavailable.
- Historical raw reconstruction and offline augmentation cannot be reproduced exactly.
- Some visually ambiguous or weakly informative samples remain.
- Several cosmetic categories overlap strongly in packaging and shape.
- Evaluation uses one canonical split and one seed, not cross-validation.
- The project has no deployment, monitoring, calibration or production-provenance layer.

See [`LEGACY_ARTIFACTS.md`](LEGACY_ARTIFACTS.md) for non-canonical output treatment and [`SUBMISSION_AUDIT.md`](SUBMISSION_AUDIT.md) for the final consistency audit.
