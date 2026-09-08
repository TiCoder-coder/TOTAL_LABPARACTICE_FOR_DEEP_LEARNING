# FEATURESETS-v1

## Definitions

FS0 contains the 25 raw exogenous channels and excludes historical Appliances and random controls.

FS1 adds historical Appliances to FS0.

FS2 adds rv1 and rv2 to FS1 as random-control channels.

TF0 excludes all engineered calendar channels.

TF1 adds hour_sin, hour_cos, dow_sin, dow_cos and weekend.

## Registered variants

| Variant | Feature count | Baseline reference | Selection status |
|---|---:|---|---|
| FS0_TF0 | 25 | No | UNTESTED |
| FS0_TF1 | 30 | No | UNTESTED |
| FS1_TF0 | 26 | No | UNTESTED |
| FS1_TF1 | 31 | Yes | UNTESTED |
| FS2_TF0 | 28 | No | UNTESTED |
| FS2_TF1 | 33 | No | UNTESTED |

## Feature-order rule

Every downstream phase must load the ordered list and fingerprint from feature_set_registry.json. Manual column lists and dtype-based automatic selection are not allowed.

## Leakage rule

Metadata is excluded from every model variant. Appliances is a historical channel only in FS1 and FS2. rv1 and rv2 appear only in FS2. No variant contains a future target or future observation.

## Ablation rule

FS0 to FS1 changes only historical target availability. FS1 to FS2 changes only random-control availability. TF0 to TF1 changes only the five engineered calendar channels. Phase 7 does not select a winner.
