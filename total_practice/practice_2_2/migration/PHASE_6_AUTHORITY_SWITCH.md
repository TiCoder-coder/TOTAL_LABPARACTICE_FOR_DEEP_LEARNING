# Phase 6 — Canonical Authority Switch

Switch timestamp: `2026-08-03T16:38:00Z`

The verified Phase 5 copies are now the active Practice 2.2 authority. No old
resource was moved, deleted or archived.

| Resource | Previous compatibility location | Active authority |
|---|---|---|
| Dataset | `data_clean_balanced/` | `data/final/data_clean_balanced/` |
| Split | `outputs/practice_2_2/canonical_split/` | `data/manifests/canonical_split/` |
| E2 checkpoint | `runs/practice_2_2/<run>/E2_partial_finetune/best.pt` | `artifacts/canonical/<run>/checkpoints/E2_partial_finetune/best.pt` |
| Canonical outputs | `outputs/practice_2_2/<run>/` | `artifacts/canonical/<run>/outputs/` |
| E3/E4 | old run/output trees | `artifacts/ablations/E3/` and `artifacts/ablations/E4/` |
| Notebook | `practice_2/notebooks/practice_2_2.ipynb` | `notebooks/04_canonical_report.ipynb` |
| HTML | old notebook export | `reports/html/practice_2_2_canonical_report.html` |
| Import | `processing_own_phase.*` | `practice_2_2.*` |

`configs/canonical_registry.json` has `authority_status = active` and contains
only root-relative new-layout paths. The exact previous active registry is
preserved as `migration/canonical_registry_phase4_active_backup.json`.
Pre-migration paths are available only through
`configs/canonical_registry_legacy_compat.json` and the explicit
`load_legacy_compat_registry()` / `resolve_legacy_resource()` APIs.

The Phase 5 draft remains inactive as a migration record. Immutable lineage
JSON was not rewritten; historical paths inside it are resolved at runtime by
the read-only registry adapter.
