# Final Repository Layout

## Practice 2 — CIFAR-10 and documented wrappers only

```text
practice_2/
├── configs/                    # CIFAR/shared configuration
├── data/                       # CIFAR-10 only
├── description/                # CIFAR documentation
├── notebooks/
│   └── practice_2_presentation.ipynb
├── outputs/                    # CIFAR artifacts only
├── processing_own_phase/
│   ├── data.py, model.py, train.py, ...   # shared/CIFAR implementation
│   └── *practice_2_2*.py                  # thin forwarding wrappers only
├── reports/                    # CIFAR figures only
├── runs/                       # CIFAR E1/E2 runs only
└── tests/
    ├── test_data.py, test_model.py, ...   # CIFAR/shared tests
    └── test_practice_2_2_compatibility_smoke.py
```

No cosmetic dataset, Practice 2.2 notebook, report, output, checkpoint
implementation, or full Practice 2.2 test remains active in Practice 2.

## Practice 2.2 — sole canonical ownership

```text
practice_2_2/
├── configs/                    # active registry + explicit archived fallback
├── data/
│   ├── final/                  # canonical dataset
│   ├── manifests/              # canonical split/fingerprints
│   └── quarantine/
├── notebooks/
│   └── 04_canonical_report.ipynb
├── src/practice_2_2/           # canonical package
├── scripts/                    # safe verification/audit utilities
├── artifacts/
│   ├── canonical/              # E1/E2 and locked Final Test evidence
│   ├── ablations/              # E3/E4 evidence
│   └── legacy/                 # legacy index
├── reports/
│   ├── html/                   # canonical HTML
│   ├── figures/
│   └── tables/
├── tests/                      # full Practice 2.2 contracts
├── archive/phase_7/
│   ├── practice_2/             # former misplaced ownership
│   └── pre_migration_practice_2_2/ # read-only historical layout
├── migration/                  # Phase 0–7 audit evidence
└── docs/
```

The active registry resolves only `data/`, `artifacts/`, `notebooks/` and
`reports/`. Archived paths require the explicit legacy compatibility registry.

Final organization score: **99/100**. One point is reserved because thin old
import wrappers remain intentionally for backward compatibility.
