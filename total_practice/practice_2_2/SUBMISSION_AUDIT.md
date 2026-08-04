# Practice 2.2 — Final Submission Audit

| Area | Status | Evidence |
|---|---|---|
| Dataset provenance | PARTIAL | Tiki.vn and raw count documented; per-image URLs unavailable |
| Cleaning pipeline | PARTIAL | Current files auditable; historical balance scripts unavailable |
| Leakage protection | PASS | Duplicate-aware canonical manifest and split assertions |
| Training pipeline | PASS | Frozen configs, histories, checkpoint hashes and reload verification |
| Validation-only selection | PASS | E2 frozen before Test; E3/E4 rejected on Validation |
| Final Test integrity | PASS | 440 samples, one evaluation, immutable completion guard |
| Artifact lineage | PASS | Canonical, ablation and legacy outputs explicitly separated |
| Reproducibility | PARTIAL | Canonical cleaned-data run reproducible; raw reconstruction partial |
| Notebook presentation | PASS | Official notebook is artifact-only and safe under Run All |
| Automated tests | PASS | Full test suite passes after cleanup |

## Canonical identity

```text
Run:        canonical_26dc4625_52aaf974_s42_v1
Winner:     E2_partial_finetune
Test Acc:   76.36%
Macro F1:   0.7617
Checkpoint: 4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3
```

## Verdict

The submission is internally consistent and suitable as a learning-project report. Provenance and raw reconstruction limitations prevent a production-readiness claim.

Final audit score: **92/100**.
