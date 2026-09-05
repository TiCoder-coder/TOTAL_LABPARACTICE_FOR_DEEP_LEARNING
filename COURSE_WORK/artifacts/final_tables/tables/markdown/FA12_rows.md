## FA12 - Upstream Warnings and Reporting Caveats

_Upstream caveats propagated to final-report tables._

| caveat_title | caveat_description | propagated_to_table | source_phase |
|---|---|---|---|
| Single-house dataset | No multi-house generalization possible | FT10 | Phase 0 |
| Three-seed limit | Only 3 final seeds (42 / 123 / 2026); SD is descriptive only | FT10 | Phase 47 |
| Last-query pooling caveat | Last-step pooling: last-query corresponds to newest encoded token | FT06 | Phase 45 |
| Lookback 12h truncation | Recent 24h mass truncated because lookback=72 steps=12h | FT06 | Phase 45 |
| Attention NOT raw-feature importance | Temporal allocation only; no causal attribution | FT06-FT09 | Phase 54-57 |
| Attention NOT causal explanation | Descriptive diagnostics only | FT06-FT09 | Phase 54-57 |
| Same-index head NOT semantically aligned | Frozen canonical JSD matching by Phase 57 | FT09 | Phase 57 |
| Phase 56 HIGH/LOW are diagnostic cohorts only | NOT deployment regimes | FT08 | Phase 56 |
| Pooled vs mean fold RMSE | FT03 pooled RMSE comes from Phase 44 authoritative output | FT03 | Phase 44 |
| Three-seed summary is NOT an ensemble | Mean ± sample SD of seed-level metrics | FT02 | Phase 47 |
| R² not clamped | Negative R² retained as valid evidence | FT02 | Phase 47 |
| Dense-case coverage limited to Phase 51 worst-case set | Selection-conditioned supplementary analysis | FT09 | Phase 51/57 |
| Cycle consistency Layer 0 = 1/4 | Pairwise optimal head identities not fully cycle-consistent | FT09 | Phase 57 |

> Caveats propagated from upstream phases.
