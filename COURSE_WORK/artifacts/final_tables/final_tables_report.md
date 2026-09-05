# PHASE 58 — FINAL TABLES REPORT
_Version: FINAL_TABLES-v2  •  Date: 2026-09-05  •  Seeds: 42 / 123 / 2026_

## 1. Objective
Phase 58 produces the final reporting tables from frozen upstream Phase 44-57 artifacts.

## 2. Upstream evidence freeze
All 14 upstream phases PASS / PASS_WITH_WARNING (see phase58_preflight_audit.csv).

## 3. Evidence-class separation
Each table tags with evidence_class: FROZEN_CONFIG / HELD_OUT_TEST / DEVELOPMENT / POST_TEST_DIAGNOSTIC.

## 4. Final table inventory
- Main: FT01..FT10 (10 tables)
- Appendix: FA01..FA12 (12 tables)

## 5. Source-of-truth ledger
See `final_table_source_ledger.csv` for cell-by-cell traceability.

## 6. Final model label and metric contracts
Locked; see `final_tables_contract.json`.

## 7. Rounding and unit contracts
Locked; see `final_table_render_config.json`.

## FT01 — Final Experimental and Model Configuration
- Evidence class: `METHOD / LOCKED_CONFIG`
- Rows: 53
- CSV: `tables/csv/FT01_rows.csv`
- Markdown: `tables/markdown/FT01_rows.md`
- LaTeX: `tables/latex/FT01_rows.tex`

## FT02 — Final Held-Out Test Performance
- Evidence class: `HELD_OUT_TEST_EVIDENCE`
- Rows: 6
- CSV: `tables/csv/FT02_rows.csv`
- Markdown: `tables/markdown/FT02_rows.md`
- LaTeX: `tables/latex/FT02_rows.tex`

## FT03 — Rolling-Origin Temporal Robustness (DEVELOPMENT_EVIDENCE)
- Evidence class: `DEVELOPMENT_EVIDENCE`
- Rows: 5
- CSV: `tables/csv/FT03_rows.csv`
- Markdown: `tables/markdown/FT03_rows.md`
- LaTeX: `tables/latex/FT03_rows.tex`

## FT04 — Final Prediction and Residual Diagnostics
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 30
- CSV: `tables/csv/FT04_rows.csv`
- Markdown: `tables/markdown/FT04_rows.md`
- LaTeX: `tables/latex/FT04_rows.tex`

## FT05 — Error-by-Regime and Worst-Error Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 165
- CSV: `tables/csv/FT05_rows.csv`
- Markdown: `tables/markdown/FT05_rows.md`
- LaTeX: `tables/latex/FT05_rows.tex`

## FT06 — Last-Query Temporal Attention Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 6
- CSV: `tables/csv/FT06_rows.csv`
- Markdown: `tables/markdown/FT06_rows.md`
- LaTeX: `tables/latex/FT06_rows.tex`

## FT07 — Within-Seed Head Comparison Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 6
- CSV: `tables/csv/FT07_rows.csv`
- Markdown: `tables/markdown/FT07_rows.md`
- LaTeX: `tables/latex/FT07_rows.tex`

## FT08 — Error-Conditioned Attention Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 108
- CSV: `tables/csv/FT08_rows.csv`
- Markdown: `tables/markdown/FT08_rows.md`
- LaTeX: `tables/latex/FT08_rows.tex`

## FT09 — Seed-Stability Attention Summary (S57-A + S57-B + S57-C)
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 80
- CSV: `tables/csv/FT09_rows.csv`
- Markdown: `tables/markdown/FT09_rows.md`
- LaTeX: `tables/latex/FT09_rows.tex`

## FT10 — Evidence and Limitation Summary
- Evidence class: `EVIDENCE_AND_LIMITATION`
- Rows: 10
- CSV: `tables/csv/FT10_rows.csv`
- Markdown: `tables/markdown/FT10_rows.md`
- LaTeX: `tables/latex/FT10_rows.tex`

## 18. Appendix package
FA01..FA12 built under `tables/csv/`, `tables/markdown/`, `tables/latex/`.

## 19. Cross-table consistency audits
- final_table_cross_consistency_audit.csv (5 rows)
- final_table_population_audit.csv (19 rows)
- final_table_model_lock_audit.csv (10 rows)
- final_table_seed_audit.csv (8 rows)
- final_table_unit_audit.csv (33 rows)
- final_table_rounding_audit.csv (428781 rows)

## 20. Warnings/caveats
See `final_tables_discrepancies.json` and FA12.

## 21. Phase59 claim handoff
See `phase59_final_conclusions_handoff.json`.

## 22. Definition of Done
- FT01..FT10 built ✓
- FA01..FA12 built ✓
- CSV / Markdown / LaTeX emitted ✓
- Source ledger built ✓
- All cross-table audits emitted ✓
- Phase 59 handoff emitted ✓
