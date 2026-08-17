# WINDOWS-v1

WINDOWS-v1 defines sequence-to-one geometry for lookbacks 36, 72 and 144 with horizon 1 at a 10-minute cadence.

For target position j, input end is j minus H and input start is input end minus L plus one. The target row never enters X.

Samples are assigned to TRAIN, VALIDATION or TEST by target timestamp. WB0 permits continuous historical context from prior periods. WB1 eligibility requires every input row to share the target split.

WINDOWPOP-v1 is the ordered intersection of native valid targets for L36, L72 and L144. Controlled experiments use this fixed target population.

Feature timelines are transformed with frozen SCALING-v1 bundles and materialized lazily as float32 arrays shaped L by F. Full three-dimensional window tensors are not stored.

Historical Appliances values through input end are valid for FS1 and FS2 under the one-step observed-history assumption.

Test window structure is available, but Test target values and outcome analysis remain locked until Phase 47.
