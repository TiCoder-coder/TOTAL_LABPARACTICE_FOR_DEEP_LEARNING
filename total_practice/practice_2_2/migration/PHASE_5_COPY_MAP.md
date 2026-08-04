# Phase 5 — Pre-Copy Inventory and Copy Map

This inventory was completed before any Phase 5 canonical resource copy.

The deterministic directory digest is SHA-256 over sorted lines in the form:

```text
<file_sha256> <size_bytes> <relative_path>\n
```

## Source → destination map

| Group | Source | Destination | Files | Bytes | Source digest |
|---|---|---|---:|---:|---|
| A. Canonical dataset | `data_clean_balanced/` | `data/final/data_clean_balanced/` | 3,202 | 48,502,262 | `fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5` |
| B. Canonical split | `outputs/practice_2_2/canonical_split/` | `data/manifests/canonical_split/` | 3 | 1,067,486 | `8016ca618fd203955f1597e2fbd308b6bb5bdae578428fdc59d9b6d8f9a9cd1c` |
| C. Canonical E1/E2 checkpoints | `runs/practice_2_2/canonical_26dc4625_52aaf974_s42_v1/` | `artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/checkpoints/` | 4 | 313,702,500 | `171e418b986aee6b1c25bd85d4ec177da3196d3c8739536f76e66ec6aa657d04` |
| D. Canonical output lineage | `outputs/practice_2_2/canonical_26dc4625_52aaf974_s42_v1/` | `artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/outputs/` | 16 | 335,691 | `1805e4bff094dc512f57ef82bd9e8e3f2c1d4e3dacb9a27aef5710b5b4334184` |
| E1. E3 run | `runs/practice_2_2/canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1/` | `artifacts/ablations/E3/<run-id>/checkpoints/` | 2 | 165,229,782 | `e9a87b489493fdb42667828cab06b014750e99d5da01b5e43fb402bdddd5df22` |
| E2. E3 outputs | `outputs/practice_2_2/canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1/` | `artifacts/ablations/E3/<run-id>/outputs/` | 4 | 7,360 | `36daf35df1e9bf2551d437cc78227d32b2821e48229dfdd96f22367cbd84eaaf` |
| F1. E4 run | `runs/practice_2_2/canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1/` | `artifacts/ablations/E4/<run-id>/checkpoints/` | 2 | 224,014,206 | `f22fe7ee7afdeb59af00807b175a6d7bf3ef8e993377ddd71c305f39a8b556d2` |
| F2. E4 outputs | `outputs/practice_2_2/canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1/` | `artifacts/ablations/E4/<run-id>/outputs/` | 4 | 10,219 | `70c3f90d1da9e69b573def0fdc9f2041c6a3322963c7a9a3c1b48b7873739794` |
| G. Quarantine record | `outputs/practice_2_2/canonical_split/quarantined_images.csv` | `data/quarantine/quarantined_images.csv` | 1 | 914 | `be08d24f3e2b105bbd02728bb276811f4a1fec304ffb51cc8e3b7af15086cef1` |

## Pre-copy canonical identity

- Dataset fingerprint: `26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6`
- Split fingerprint: `52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d`
- E2 best checkpoint: `4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3`

No source is removed, renamed or overwritten by this map.
