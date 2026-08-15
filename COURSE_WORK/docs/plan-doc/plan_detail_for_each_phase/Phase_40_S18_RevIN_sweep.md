# PHASE 40 — S18 RevIN SWEEP

## Controlled plan cho Reversible Instance Normalization trong Transformer Regression

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting:** Sequence-to-One, One-Step-Ahead  
**Upstream:** `SWEEP_S17_GRADIENTCLIP-v1`  
**Sweep ID:** `S18_REVIN`  
**Version:** `SWEEP_S18_REVIN-v1`  
**File trước:** `Phase_39_S17_Gradient-clipping_sweep.md`

---

# 1. Mục tiêu của Phase 40

Phase 40 kiểm tra đúng một yếu tố:

```text
RN0 = RevIN OFF
RN1 = RevIN ON
```

Câu hỏi khoa học:

> Sau khi toàn bộ feature set, scaling, lookback, pooling, activation, batch, learning rate, weight decay, dropout, d_model, heads, layers, FFN, loss, epoch cap, patience và gradient clipping đã được khóa từ các phase trước, việc thêm Reversible Instance Normalization có cải thiện khả năng dự báo `Appliances_{t+1}` hay không?

Nguyên tắc:

```text
One RevIN Factor
+ Same Information
+ Same Samples
+ Same Backbone
+ Same Optimizer/Training
+ Validation RMSE Selection
+ No Test
```

---

# 2. Vị trí trong chuỗi sweep

```text
S1  Feature-set
S2  Time-feature
S3  Target-scaling
S4  Lookback
S5  Pooling
S6  Activation
S7  Batch
S8  Learning-rate
S9  Weight-decay
S10 Dropout
S11 d_model
S12 Heads
S13 Layers
S14 FFN
S15 Loss
S16 Epoch-cap
S17 Gradient-clipping
S18 RevIN        ← Phase 40
S19 Boundary protocol
```

Phase 40 không được thay lại bất kỳ winner nào từ S1–S17.

---

# 3. Current reference phải load từ Phase 39

Bắt buộc đọc:

```text
s17_gradient_clipping_winner.json
s17_reference_update.json
phase_39_signoff.json
```

Resolve runtime:

```text
FV*      selected feature variant
YS*      selected target scaling
L*       selected lookback
P*       selected pooling
A*       selected activation
B*       selected batch
LR*      selected learning rate
WD*      selected weight decay
DR*      selected dropout
D*       selected d_model
H*       selected heads
HD*      selected head_dim
N*       selected layers
F*       selected FFN width
LOSS*    selected training loss
EPOCHS*  selected max_epochs
GC*      selected clipping policy
```

Không hard-code `FS1`, `YS1`, `L144`, `D64`, `H4`, `N2`, `MSE`, `E50`, `GC1`.

---

# 4. RevIN contract của project

RN1 được khóa như sau:

```text
revin_enabled      = true
eps                = 1e-5
affine             = true
affine_weight_init = 1
affine_bias_init   = 0
centering          = mean
variance           = population variance
unbiased           = false
statistics_detached= true
running_stats      = false
subtract_last      = false
```

Statistics được tính:

```text
per sample
per feature/channel
across sequence-time dimension only
```

Input có shape:

```text
[B, L*, F]
```

Với channel `c` của sample `b`:

\[
\mu_{b,c} = rac{1}{L}\sum_t x_{b,t,c}
\]

\[
\sigma_{b,c}
=
\sqrt{
rac{1}{L}\sum_t (x_{b,t,c}-\mu_{b,c})^2
+\epsilon
}
\]

Sau đó:

\[
z_{b,t,c}
=
rac{x_{b,t,c}-\mu_{b,c}}{\sigma_{b,c}}
\gamma_c+eta_c
\]

`mean` và `stdev` phải detached.

---

# 5. Điểm đặc biệt của project: output chỉ có 1 target

Canonical RevIN thường được trình bày theo dạng input/output có các feature tương ứng.

Project này:

```text
input  = multivariate [B,L,F]
output = Appliances only [B,1]
```

Do đó S18 phải dùng **target-selective reversible adapter**.

Flow RN1:

```text
x_model [B,L,F]
↓
normalize eligible input signal channels bằng RevIN
↓
merge lại đúng original feature order
↓
Transformer backbone
↓
regression head [B,1]
   (normalized target coordinate)
↓
denormalize chỉ bằng statistics của historical Appliances channel
↓
convert target từ X-target coordinate sang y_model coordinate
↓
return [B,1] y_model
```

Training Engine bên ngoài vẫn nhận:

```text
prediction [B,1] y_model
target     [B,1] y_model
```

Vì vậy loss/early stopping/metrics vẫn dùng cùng code path.

---

# 6. Applicability gate — bắt buộc phải có historical Appliances

RN1 chỉ hợp lệ nếu current selected input feature set chứa:

```text
historical Appliances
```

Lý do: prediction output là future Appliances và reversible denormalization phải dùng statistics của corresponding observed target-history channel.

Runtime gate:

```text
matches = exact feature name "Appliances"

if count == 1:
    RN1_APPLICABLE = true

if count == 0:
    RN1_APPLICABLE = false

if count > 1:
    FAIL configuration/schema
```

---

# 7. Trường hợp FS0 thắng ở S1

FS0 không có historical Appliances.

Khi đó tuyệt đối không được:

```text
thêm Appliances chỉ để tính RevIN
dùng target-history như hidden side information
dùng statistics của channel khác để denorm target
dùng future target
```

Cách xử lý đúng:

```text
overall_status = SKIPPED_NOT_APPLICABLE
reason         = HISTORICAL_TARGET_CHANNEL_ABSENT
RN1 run        = không tạo
RN1 metrics    = không tạo
selected/carry = RN0
selection_basis= APPLICABILITY_CONSTRAINT
approved_for_phase41 = true
```

Quan trọng:

```text
không được báo RN0 "thắng" RN1
```

vì không hề có empirical comparison.

---

# 8. Trường hợp FS1 hoặc FS2

Nếu selected FV* chứa historical Appliances:

```text
RN1_APPLICABLE = true
```

và chạy S18 bình thường:

```text
RN0 → reuse S17 winner
RN1 → one fresh seed42 run
```

---

# 9. RevIN channel scope

RN1 chỉ normalize các **continuous forecasting signal channels**.

Scope policy:

```text
G1 historical Appliances
G2 raw continuous exogenous signals
G3 rv1/rv2 nếu current feature set có
```

Pass-through:

```text
G4 deterministic time features
```

Nếu TF1 đang active:

```text
hour_sin
hour_cos
dow_sin
dow_cos
weekend
```

phải giữ nguyên.

Hard check:

```text
x_before[..., time_indices]
==
x_after[..., time_indices]
```

---

# 10. Không normalize time features

Lý do kỹ thuật:

```text
sin/cos đã encode periodic phase
weekend là binary calendar indicator
```

Per-window instance normalization sẽ phá semantic scale của các covariates này.

Do đó S18 contract:

```text
RevIN signal channels → normalize
time features         → pass-through
```

---

# 11. Global scaling từ Phase 9 vẫn giữ nguyên

RN1 không thay thế `SCALING-v1`.

Pipeline:

```text
raw
→ train-only global X scaling
→ window builder
→ x_model
→ RN1 local per-window normalization
→ Transformer
```

RN0:

```text
global scaling only
```

RN1:

```text
same global scaling + RevIN
```

Không refit scaler.
Không bỏ scaler.
Không fit Validation/Test.

---

# 12. Target coordinate bridge — phần bắt buộc

Historical Appliances trong `x_model` được transform theo X-side scaling.

Future target `y_model` lại theo selected YS*.

Hai coordinate có thể khác nhau.

Do đó sau target-channel RevIN denorm, prediction phải đi:

```text
X-target coordinate
→ inverse frozen X-target transform
→ raw Appliances Wh
→ frozen selected Y target transform
→ y_model
```

Nếu:

```text
YS*=YS0
```

thì bước cuối là identity.

Nếu:

```text
YS*=YS1
```

thì dùng frozen train-only target StandardScaler.

Không dùng Validation statistics.

---

# 13. Không được tính loss trước coordinate bridge

Sai:

```text
normalized RN1 prediction
→ criterion
```

Đúng:

```text
normalized prediction
→ target-channel RevIN denorm
→ X-target to raw
→ raw to y_model
→ criterion
```

Nếu S15 chọn Huber:

```text
Huber delta = 1.0 y_model-space
```

vẫn giữ đúng semantics cũ.

---

# 14. No future leakage

RevIN statistics chỉ được lấy từ:

```text
X_{t-L+1:t}
```

Không được sử dụng:

```text
Appliances_{t+1}
future horizon
Validation label
Test label
outside-window future rows
```

Hard leakage unit test:

```text
same input window
different future y

→ normalized input identical
→ RevIN context identical
```

---

# 15. WB0 vẫn giữ nguyên

Phase 40:

```text
boundary protocol = WB0
```

Phase41 mới kiểm WB0 vs WB1.

RevIN không được thay split/window eligibility.

---

# 16. RevIN scope được resolve theo feature registry

Không hard-code index.

Phải tạo:

```text
revin_channel_names
revin_channel_indices

passthrough_channel_names
passthrough_channel_indices

target_channel_name = Appliances
target_original_index
target_revin_subset_index
```

Store toàn bộ vào provenance.

---

# 17. Feature order không được thay

Sau gather → norm → scatter:

```text
feature order RN1 == feature order RN0
```

Không reorder model inputs.

---

# 18. Learnable affine parameters

Với:

```text
C_R = số channels nằm trong RevIN scope
```

RN1 có:

```text
gamma [C_R]
beta  [C_R]
```

Expected trainable parameter increase:

\[
\Delta P = 2C_R
\]

Hard:

```text
Params_RN1 - Params_RN0 = 2*C_R
```

Runtime count là source of truth.

---

# 19. Backbone phải giữ nguyên

Các phần này phải identical:

```text
input projection
positional encoding
MHA geometry
LayerNorm
FFN
activation
dropout
num layers
pooling
regression-head shape
```

RN1 chỉ thêm reversible input/output adapter.

---

# 20. State-dict delta expectation

Expected:

```text
RN1 state_dict
=
RN0 backbone state_dict keys
+
RevIN affine keys
```

Không được xuất hiện shape drift ở Transformer backbone.

Transient instance:

```text
mean
stdev
```

không được trở thành learned global checkpoint statistics.

---

# 21. RevIN context design

Khuyến nghị adapter API:

```text
x_norm, context = normalize(x)

pred_norm = backbone(x_norm)

pred_model = denormalize_target(
    pred_norm,
    context
)
```

`context` chứa đúng batch hiện tại:

```text
mean
stdev
target local index
```

Không reuse context giữa batches.

---

# 22. Không dùng hidden mutable context thiếu kiểm soát

Rủi ro:

```text
batch A norm
batch B norm
batch A denorm bằng stats batch B
```

Do đó preferred:

```text
explicit forward-local context
```

không global cache.

---

# 23. Round-trip tests

Bắt buộc:

```text
x
→ norm
→ denorm
≈ x
```

cho eligible channels.

Test cases:

```text
random values
constant channel
low variance
B=1
B>1
selected L*
synthetic shorter L for unit tests
```

---

# 24. Target-slice round-trip

Riêng target channel:

```text
historical Appliances X-coordinate
→ target RevIN norm
→ target RevIN denorm
≈ original target X-coordinate
```

---

# 25. X/Y coordinate bridge tests

Test:

```text
X-target coordinate
→ raw Wh
→ y_model
→ inverse y transform
→ raw Wh
→ X-target coordinate
```

Round-trip phải allclose.

---

# 26. Low-variance safety

Hard:

```text
eps=1e-5
unbiased=False
```

Nếu window gần constant:

```text
stdev finite >0
normalized finite
denorm finite
```

Không remove sample.

---

# 27. Batch independence

RevIN là per-instance.

Take sample A:

```text
normalize([A])
```

và:

```text
normalize([A,B,C])
```

Expected result/context của A:

```text
allclose.
```

Nếu khác → đang dùng batch statistics → FAIL.

---

# 28. Neighbor sample leakage test

Thay sample B bằng giá trị cực lớn.

Sample A RevIN output phải không đổi.

---

# 29. Batch permutation test

Permute batch order.

Output/context phải chỉ permute tương ứng, không đổi values.

---

# 30. Statistics detach test

Bắt buộc verify:

```text
mean detached
stdev detached
```

nhưng:

```text
normalized x path remains differentiable
prediction path remains differentiable
gamma/beta remain learnable.
```

---

# 31. Gradient-flow tests

Synthetic backward phải cho:

```text
backbone grads finite
gamma grad finite/non-None
beta grad finite/non-None
```

nơi mathematically active.

Không detach whole RevIN transform.

---

# 32. Optimizer coverage

RN1 affine params phải vào optimizer đúng một lần.

Audit:

```text
gamma group
beta group
LR
WD
policy rule
```

Không tạo RevIN-specific LR.

Không tạo RevIN-specific WD ngoài existing parameter-group policy.

---

# 33. Initialization fairness

RN1 có thêm affine params nên whole state không thể giống RN0.

Fair comparison target:

```text
shared Transformer backbone initial state
```

should ideally match.

RN1 affine:

```text
gamma=1
beta=0
```

Nếu historical RN0 initial fingerprint không còn:

```text
NOT_VERIFIABLE
```

không retrain RN0.

---

# 34. Sample-order fairness

Same:

```text
Train population
shuffle policy
DataLoader generator policy
worker policy
batch size B*
```

Nếu RN0 reference order fingerprints có:

```text
compare
```

nếu không:

```text
NOT_VERIFIABLE.
```

---

# 35. External model contract phải identical

RN0:

```text
forward(x) → [B,1] y_model
```

RN1:

```text
forward(x) → [B,1] y_model
```

Training Engine không cần biết internal RevIN coordinate.

---

# 36. Standard/attention-inspection equivalence

RN1 eval mode:

```text
model(x)
```

và:

```text
forward_with_attention(x).prediction
```

phải allclose sau full target denorm/bridge.

Attention tensors vẫn:

```text
[B,H*,L*,L*]
```

per Encoder layer.

S18 không dùng attention để chọn winner.

---

# 37. Official training settings giữ nguyên

Hard:

```text
B*
AdamW
LR*
WD*
DR*
LOSS*
EPOCHS*
patience=10
min_delta=0
GC*
scheduler=None
warmup=None
accumulation=1
seed=42
TRAINING_ENGINE-v1
METRICS-v1
```

---

# 38. RN0 reference reuse

RN0:

```text
reuse exact S17 winner
```

nếu full config match.

Không retrain RN0.

---

# 39. RN1 fresh run

Nếu applicable:

```text
register RN1
reseed42
fresh DataLoaders
fresh Transformer backbone
fresh RevIN affine params
fresh criterion
fresh AdamW
selected GC*
train via same Training Engine
```

Không warm-start từ RN0.

Không optimizer-state reuse.

---

# 40. Primary metric

Winner metric:

```text
verified BEST Validation RMSE Wh
```

Không dùng:

```text
Train loss
instance stats
affine gamma
runtime
parameter count
attention plots
```

để override RMSE.

---

# 41. Winner rule

Nếu RN1 applicable:

\[
winner
=
rg\min(RMSE_{RN0},RMSE_{RN1})
\]

full precision.

Exact tie:

```text
prefer RN0
```

vì:

```text
simpler
fewer parameters
no local reversible adapter
```

Chỉ exact tie.

---

# 42. Not-applicable không phải tie

Nếu target history absent:

```text
RN1 not run
```

thì:

```text
no empirical winner
RN0 carried by applicability constraint
```

---

# 43. RMSE/R² consistency

Same Validation targets:

```text
RMSE ranking
and
R² ranking
```

phải consistent.

Nếu conflict:

```text
STOP/INVESTIGATE
```

đặc biệt kiểm:

```text
y_model bridge
inverse target transform
population IDs
metric implementation.
```

MAE có thể legitimately disagree.

---

# 44. RevIN mechanism diagnostics

Secondary diagnostics:

```text
C_R
target-channel index
per-window target means
per-window target stdevs
per-feature mean/std distributions
low-variance incidence
gamma/beta at INIT/BEST
round-trip error
bridge error
runtime overhead
optional memory overhead.
```

---

# 45. Không dùng gamma làm feature importance

Sai:

```text
large gamma = important feature
```

Gamma chỉ là learned affine scaling parameter trong normalization mechanism.

---

# 46. Instance-stat probe

Nếu muốn plot statistics, chọn fixed probe/sample IDs trước khi nhìn kết quả.

Không cherry-pick.

No Test.

---

# 47. Hypotheses

## H-S18-01

```text
RN1 có thể cải thiện RMSE bằng cách giảm local level/scale variation của signal windows.
```

## H-S18-02

```text
RN0 có thể tốt hơn nếu absolute local level/scale mang predictive information và local normalization gây khó cho representation.
```

## H-S18-03

```text
RN1 có thể redundant nếu global train-only scaling + selected Transformer đã xử lý đủ distribution variation.
```

## H-S18-04

```text
RN1 benefit có thể phụ thuộc mạnh vào selected lookback/feature set/target scaling.
```

Tất cả ban đầu:

```text
UNTESTED.
```

---

# 48. Interaction limitations bắt buộc báo cáo

S18 là conditional experiment.

Không kiểm:

```text
RevIN × Feature-set grid
RevIN × Target-scaling grid
RevIN × Lookback grid
RevIN × Loss grid
RevIN × Clipping grid
RevIN × d_model grid
RevIN × Dropout grid
```

Chỉ current winners.

Ngoài ra chỉ test:

```text
affine=True
eps=1e-5
mean centering
full-lookback stats
registered scope
```

Không kết luận về variant khác.

---

# 49. Preconditions

Phase40 chỉ bắt đầu khi:

```text
Phase39 PASS
or
PASS_WITH_WARNING
```

và:

```text
approved_for_phase40=true.
```

Critical unresolved upstream issue → STOP.

---

# 50. Required upstream contracts

Verify:

```text
ENV-v1
FEATURES-v1
FEATURESETS-v1
SPLIT-v1
SCALING-v1
WINDOWS-v1
WINDOWPOP-v1
DATALOADERS-v1
METRICS-v1
EXPERIMENTS-v1
TRANSFORMER_IMPL-v1
ATTENTION_VERIFY-v1
TRAINING_ENGINE-v1
SWEEP_S1_FEATURESET-v1
SWEEP_S2_TIMEFEATURE-v1
SWEEP_S3_TARGETSCALING-v1
SWEEP_S4_LOOKBACK-v1
SWEEP_S5_POOLING-v1
SWEEP_S6_ACTIVATION-v1
SWEEP_S7_BATCH-v1
SWEEP_S8_LEARNINGRATE-v1
SWEEP_S9_WEIGHTDECAY-v1
SWEEP_S10_DROPOUT-v1
SWEEP_S11_DMODEL-v1
SWEEP_S12_HEADS-v1
SWEEP_S13_LAYERS-v1
SWEEP_S14_FFN-v1
SWEEP_S15_LOSS-v1
SWEEP_S16_EPOCHCAP-v1
SWEEP_S17_GRADIENTCLIP-v1
```

---

# 51. Fail-fast preflight sequence

Trước khi train RN1:

```text
1. Verify Phase39 signoff
2. Load exact S17 winner
3. Freeze all S1-S17 selections
4. Load exact feature order/fingerprint
5. Find historical Appliances
6. Applicability gate
7. Resolve RevIN eligible channels
8. Verify time-feature passthrough
9. Resolve X-target transform
10. Resolve Y-target transform
11. Build X→raw→Y bridge
12. Verify stats axes
13. Full RevIN round-trip tests
14. Target-slice denorm tests
15. Coordinate bridge tests
16. Constant/low-variance tests
17. Batch-independence tests
18. Future-target leakage tests
19. Neighbor-sample leakage tests
20. Gradient-flow/stats-detach tests
21. Backbone invariance audit
22. State-dict delta audit
23. Parameter delta audit
24. Optimizer coverage audit
25. Common population audit
26. Training config delta audit
27. RN0 reuse gate
28. Test firewall
29. Registry readiness
```

Any critical fail → không train.

---

# 52. Required artifacts directory

```text
artifacts/
└── sweeps/
    └── S18_revin/
        ├── s18_revin_sweep_manifest.json
        ├── s18_revin_sweep_contract.json
        ├── s18_revin_preflight_audit.csv
        ├── s18_revin_applicability_audit.json
        ├── s18_run_matrix.csv
        ├── s18_revin_definition_audit.csv
        ├── s18_revin_scope_audit.csv
        ├── s18_feature_order_audit.csv
        ├── s18_target_channel_audit.csv
        ├── s18_scaler_bridge_audit.json
        ├── s18_statistics_axis_audit.csv
        ├── s18_roundtrip_tests.csv
        ├── s18_target_denorm_tests.csv
        ├── s18_coordinate_bridge_tests.csv
        ├── s18_batch_independence_tests.csv
        ├── s18_leakage_tests.csv
        ├── s18_gradient_flow_tests.csv
        ├── s18_low_variance_tests.csv
        ├── s18_architecture_invariance_audit.csv
        ├── s18_state_dict_delta_audit.csv
        ├── s18_parameter_count_audit.csv
        ├── s18_optimizer_coverage_audit.csv
        ├── s18_common_data_audit.csv
        ├── s18_training_config_delta_audit.csv
        ├── s18_initialization_audit.csv
        ├── s18_sample_order_audit.csv
        ├── s18_instance_stats_diagnostics.csv
        ├── s18_affine_diagnostics.csv
        ├── s18_optimizer_budget_audit.csv
        ├── s18_revin_run_provenance.csv
        ├── s18_revin_metrics.csv
        ├── s18_revin_effect.csv
        ├── s18_optimization_diagnostics.csv
        ├── s18_convergence_diagnostics.csv
        ├── s18_runtime_diagnostics.csv
        ├── s18_generalization_diagnostics.csv
        ├── s18_hypothesis_outcomes.csv
        ├── s18_revin_findings.csv
        ├── s18_revin_winner.json
        ├── s18_reference_update.json
        ├── s18_revin_sweep_tests.csv
        ├── s18_revin_discrepancies.json
        ├── s18_revin_sweep_summary.json
        ├── s18_revin_sweep_report.md
        ├── figures/
        ├── README_S18_REVIN_SWEEP.md
        └── phase_40_signoff.json
```

New scientific RN1 run remains:

```text
artifacts/runs/<run_id>/
```

Không duplicate checkpoint tree.

---

# 53. Required output list

```text
O40.1  Manifest
O40.2  Contract
O40.3  Preflight
O40.4  Applicability audit
O40.5  Run/skip matrix
O40.6  RevIN definition audit
O40.7  Scope audit
O40.8  Feature-order audit
O40.9  Target-channel audit
O40.10 Scaler bridge audit
O40.11 Statistics-axis audit
O40.12 Round-trip tests
O40.13 Target-denorm tests
O40.14 Coordinate bridge tests
O40.15 Batch-independence tests
O40.16 Leakage tests
O40.17 Gradient-flow tests
O40.18 Low-variance tests
O40.19 Architecture invariance
O40.20 State-dict delta
O40.21 Parameter count
O40.22 Optimizer coverage
O40.23 Common data audit
O40.24 Training-config delta
O40.25 Initialization audit
O40.26 Sample-order audit
O40.27 Instance-stat diagnostics
O40.28 Affine diagnostics
O40.29 Optimizer budget
O40.30 Run provenance
O40.31 RN0 reference
O40.32 RN1 verified run if applicable
O40.33 Metrics if applicable
O40.34 Effect table if applicable
O40.35 Optimization diagnostics
O40.36 Convergence diagnostics
O40.37 Runtime diagnostics
O40.38 Optional generalization diagnostics
O40.39 Hypothesis outcomes
O40.40 Findings
O40.41 Winner or skip outcome
O40.42 Phase41 reference update
O40.43 Figures if applicable
O40.44 Tests
O40.45 Discrepancies
O40.46 Summary
O40.47 Report
O40.48 README
O40.49 Sign-off
```

---

# 54. Manifest schema

`s18_revin_sweep_manifest.json` minimum:

```text
sweep_id
sweep_version
source_s17_winner_run_id
candidate_revin_ids = [RN0,RN1]
rn1_mode = target_selective
rn1_applicable
eps = 1e-5
affine = true
centering = mean
unbiased = false
stats_detached = true
scope_policy
time_feature_policy = passthrough
target_denorm_channel = Appliances
feature_variant_id
target_scaling_id
lookback_id
pooling_id
activation_id
batch_id
learning_rate
weight_decay
dropout_probability
d_model
num_heads
head_dim
num_layers
ffn_dim
loss_id
max_epochs
patience
gradient_clip_id
new_runs_required
reused_runs
swept_field = revin
expected_parameter_delta
primary_metric = validation_rmse_wh
selection_direction = MIN
tie_rule = RN0_ON_EXACT_RMSE_TIE
not_applicable_policy = CARRY_RN0_WITHOUT_EMPIRICAL_COMPARISON
population_fingerprint
metric_version
training_engine_version
seed = 42
test_access = forbidden
status
```

---

# 55. Applicability artifact

`s18_revin_applicability_audit.json`:

```text
selected_feature_variant
feature_fingerprint
feature_names
historical_target_required
historical_target_name
historical_target_match_count
historical_target_present
target_index
x_target_transform_available
y_target_transform_available
scope_resolvable
rn1_applicable
reason_if_not
hidden_target_injection_forbidden
recommended_status
approved_for_phase41_if_skipped
```

---

# 56. Scope audit schema

`s18_revin_scope_audit.csv`:

```text
feature_name
feature_index
feature_group
selected
revin_eligible
revin_applied
passthrough
reason
status
```

---

# 57. Target-channel audit schema

```text
target_name
match_count
original_feature_index
revin_subset_index
x_scaler_mapping
y_scaler_mapping
stats_source
future_target_used
status
```

Expected:

```text
future_target_used=false
```

---

# 58. Parameter audit schema

```text
revin_id
total_trainable_params
backbone_trainable_params
revin_channel_count
expected_affine_params
observed_affine_params
delta_vs_rn0
expected_delta
delta_valid
status
```

---

# 59. Metrics schema

If applicable:

```text
revin_id
run_id
source_type
revin_channel_count
revin_parameter_count
total_trainable_parameters
epochs_completed
stop_reason
best_epoch
validation_mae_wh
validation_rmse_wh
validation_r2
rmse_rank
is_winner
population_fingerprint
metric_version
status
```

---

# 60. Effect table

```text
rn0_rmse_wh
rn1_rmse_wh
delta_rmse_rn0_to_rn1
rmse_improvement_pct
rn0_mae_wh
rn1_mae_wh
mae_delta
rn0_r2
rn1_r2
r2_delta
metric_ranking_divergence
rmse_r2_consistent
rn0_params
rn1_params
parameter_delta
winner
```

---

# 61. Findings codes

Possible:

```text
RN0_GAIN
RN1_GAIN
REVIN_EXACT_TIE
RN1_NOT_APPLICABLE
HISTORICAL_TARGET_CHANNEL_ABSENT
TARGET_CHANNEL_MAPPING_VERIFIED
REVIN_SCOPE_VERIFIED
TIME_FEATURE_PASSTHROUGH_VERIFIED
ROUNDTRIP_VERIFIED
TARGET_DENORM_VERIFIED
XY_BRIDGE_VERIFIED
BATCH_INDEPENDENCE_VERIFIED
NO_FUTURE_TARGET_LEAKAGE
PARAMETER_DELTA_VERIFIED
BACKBONE_INVARIANCE_VERIFIED
BACKBONE_INIT_MATCH_VERIFIED
BACKBONE_INIT_NOT_VERIFIABLE
SAMPLE_ORDER_MATCH_VERIFIED
SAMPLE_ORDER_NOT_VERIFIABLE
LOW_VARIANCE_WINDOWS_PRESENT
AFFINE_PARAMETER_SHIFT
RUNTIME_OVERHEAD
METRIC_RANKING_DIVERGENCE
RMSE_R2_RANKING_INCONSISTENCY
INHERITED_WARNING
```

---

# 62. Winner artifact

Empirical path:

```text
sweep_id
empirical_comparison_performed=true
winner_revin_id
winner_run_id
winner_rmse_wh
winner_mae_wh
winner_r2
runner_up_revin_id
runner_up_rmse_wh
rmse_margin_wh
revin_channel_count
parameter_delta
selection_metric
tie_rule
population_fingerprint
test_status
status
```

Skip path:

```text
empirical_comparison_performed=false
rn1_applicable=false
reason=HISTORICAL_TARGET_CHANNEL_ABSENT
selected_revin_id=RN0
selection_basis=APPLICABILITY_CONSTRAINT
rn0_run_id=<S17 winner>
rn1_run_id=null
rmse_comparison=null
status=SKIPPED_NOT_APPLICABLE
```

---

# 63. Phase41 handoff

`s18_reference_update.json` phải carry:

```text
all current S1-S17 winners
selected_revin_id
selected_revin_enabled
revin_config_if_enabled
rn1_applicable
selection_basis
winner_or_carried_run_id
current_boundary_protocol = WB0
population_fingerprint
approved_for_phase41
```

Phase41 sau đó kiểm:

```text
WB0 vs WB1
```

và giữ RevIN setting từ S18.

---

# 64. Phase41 nuance nếu RN1 thắng

WB1 phải tự xây đúng strict-isolation windows rồi RevIN chỉ tính stats trên chính những window đó.

Không được:

```text
borrow WB0 history merely for RevIN stats.
```

---

# 65. Recommended figures

Nếu empirical comparison có:

```text
S18_01_validation_rmse_by_epoch.png
S18_02_validation_mae_by_epoch.png
S18_03_train_criterion_by_epoch.png
S18_04_instance_target_mean_distribution.png
S18_05_instance_target_std_distribution.png
S18_06_revin_affine_parameters.png
S18_07_best_validation_metrics.png
S18_08_parameter_count_vs_rmse.png
S18_09_runtime_vs_rmse.png
S18_10_generalization_gap_optional.png
```

Nếu RN1 not applicable:

```text
không tạo fake comparison plots.
```

---

# 66. Safe reporting

Nếu RN1 thắng:

> Under the frozen selected feature, scaling, model and training configuration, the registered target-selective RevIN setup achieved lower Validation RMSE than RN0.

Nếu RN0 thắng:

> Adding the registered target-selective RevIN setup did not improve Validation RMSE under the current frozen configuration, so RN0 was retained.

Nếu exact tie:

> RN0 and RN1 produced exactly equal full-precision Validation RMSE; RN0 was retained by the predefined parsimony rule.

Nếu not applicable:

> RN1 was not evaluated because the locked selected feature set did not contain historical Appliances, which is required by the registered target-selective reversible denormalization design. RN0 was carried forward by applicability constraint, not by empirical superiority.

---

# 67. Reporting prohibitions

Không được nói:

```text
RevIN luôn xử lý được distribution shift
RevIN thay thế global scaling
RevIN thay thế LayerNorm
gamma lớn = feature quan trọng
RN1 dùng future statistics
RN0 thắng RN1 khi RN1 không chạy
FS0 được sửa để bật RevIN
RevIN attention đẹp hơn nên thắng
```

---

# 68. Discrepancy taxonomy

```text
S17_REFERENCE_MISSING
S17_WINNER_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
POOLING_DRIFT
ACTIVATION_DRIFT
BATCH_DRIFT
LR_DRIFT
WD_DRIFT
DROPOUT_DRIFT
DMODEL_DRIFT
HEAD_DRIFT
LAYER_DRIFT
FFN_DRIFT
LOSS_DRIFT
EPOCH_CAP_DRIFT
GRADIENT_CLIP_DRIFT
REVIN_DEFINITION_MISMATCH
TARGET_CHANNEL_ABSENT
TARGET_CHANNEL_DUPLICATE
REVIN_SCOPE_MISMATCH
TIME_FEATURE_NORMALIZED
BATCH_STATS_USED
RUNNING_STATS_USED
SUBTRACT_LAST_USED
EPS_MISMATCH
AFFINE_MISMATCH
STATS_NOT_DETACHED
WRONG_STATS_AXIS
CONTEXT_REUSED
TARGET_INDEX_MISMATCH
TARGET_AFFINE_INDEX_MISMATCH
ROUNDTRIP_FAILURE
TARGET_DENORM_FAILURE
XY_BRIDGE_FAILURE
X_SCALER_MISMATCH
Y_SCALER_MISMATCH
FUTURE_TARGET_LEAKAGE
NEIGHBOR_SAMPLE_LEAKAGE
FEATURE_ORDER_DRIFT
HIDDEN_TARGET_HISTORY_INJECTION
GLOBAL_SCALING_REMOVED
BACKBONE_DRIFT
UNEXPECTED_PARAMETER_DELTA
UNEXPECTED_STATE_DICT_DELTA
AFFINE_OPTIMIZER_MISSING
AFFINE_OPTIMIZER_DUPLICATE
REVIN_SPECIFIC_LR
WARM_START_USED
OPTIMIZER_STATE_REUSE
POPULATION_MISMATCH
TRAINING_ENGINE_MISMATCH
METRIC_VERSION_MISMATCH
RUN_FAILURE
NUMERICAL_INSTABILITY
CHECKPOINT_VERIFICATION_FAILURE
RANKING_ERROR
RMSE_R2_RANKING_INCONSISTENCY
TEST_FIREWALL_VIOLATION
HIDDEN_RERUN
OTHER
```

---

# 69. Status model

```text
PASS
PASS_WITH_WARNING
SKIPPED_NOT_APPLICABLE
FAIL
```

`SKIPPED_NOT_APPLICABLE` chỉ hợp lệ khi locked FV* không có historical Appliances.

---

# 70. Common mistakes

```text
1. Bật RevIN nhưng không lock channel scope.
2. Normalize time features.
3. Dùng mean/std toàn batch.
4. Dùng future target trong stats.
5. FS0 nhưng lén lấy historical target.
6. Bỏ global scaler vì có RevIN.
7. Denorm target nhưng quên X→Y coordinate bridge.
8. Tính Huber/MSE trước RN1 denorm.
9. Apply RevIN sau input projection.
10. Dùng running stats kiểu BatchNorm.
11. Dùng subtract-last dù protocol mean-based.
12. affine=False để match parameter count.
13. Dùng mutable stats của batch trước.
14. RN1 warm-start từ RN0 BEST.
15. Chọn RN0 vì ít params hơn dù RMSE cao hơn.
16. Report RN0 thắng khi RN1 not applicable.
17. Dùng Test.
```

---

# 71. Execution pseudocode

```text
verify_phase39()
s17 = load_s17_winner()
freeze_all_prior_fields()

features = load_exact_feature_order(FV*)
matches = find_exact("Appliances")

if len(matches) == 0:
    write_not_applicable()
    carry_RN0_to_phase41()
    signoff(SKIPPED_NOT_APPLICABLE)
    STOP

assert len(matches) == 1
target_idx = matches[0]

revin_indices = resolve_groups(G1,G2,G3)
passthrough_indices = resolve_group(G4)
assert target_idx in revin_indices

x_target_transform = load_frozen_X_target_transform()
y_transform = load_frozen_Y_transform(YS*)
bridge = build_X_to_raw_to_Y_bridge()

run_stats_axis_tests()
run_roundtrip_tests()
run_target_denorm_tests()
run_coordinate_bridge_tests()
run_low_variance_tests()
run_batch_independence_tests()
run_leakage_tests()
run_gradient_flow_tests()
audit_time_feature_passthrough()
audit_feature_order()
audit_backbone_schema()
audit_parameter_delta(expected=2*len(revin_indices))
audit_optimizer_coverage()

rn0 = resolve_exact_S17_reference()

register_RN1()
reseed(42)

loaders = build_fresh_loaders(
    FV*, YS*, L*, B*, WINDOWPOP-v1
)

backbone = build_fresh_selected_transformer()

adapter = build_revin_adapter(
    revin_indices,
    passthrough_indices,
    target_idx,
    eps=1e-5,
    affine=True,
    bridge=bridge
)

model = RN1Model(adapter, backbone)

criterion = build_selected_loss(LOSS*)

optimizer = build_fresh_AdamW(
    model,
    LR*,
    WD*,
    frozen_group_policy
)

rn1 = TRAINING_ENGINE_v1.fit(
    model,
    criterion,
    optimizer,
    max_epochs=EPOCHS*,
    patience=10,
    gradient_clip=GC*
)

verify_RN1_BEST(rn1)

metrics = compare_same_validation_population(rn0, rn1)

winner = min_full_precision_RMSE(
    RN0=rn0,
    RN1=rn1,
    exact_tie="RN0"
)

write_diagnostics()
write_winner()
write_phase41_reference()
write_summary_report_readme_signoff()
```

---

# 72. Acceptance checklist

```text
[ ] Phase39 valid.
[ ] S17 winner loaded exactly.
[ ] All S1-S17 winners frozen.
[ ] RN0/RN1 registered.
[ ] RN1 eps=1e-5.
[ ] RN1 affine=True.
[ ] gamma init=1.
[ ] beta init=0.
[ ] mean centering.
[ ] unbiased=False.
[ ] stats detached.
[ ] no running stats.
[ ] no subtract-last.
[ ] exact feature order loaded.
[ ] historical Appliances audited.
[ ] target uniqueness audited.
[ ] FS0 skip policy implemented.
[ ] no hidden target injection.
[ ] RevIN scope resolved.
[ ] time features pass through.
[ ] global X scaler unchanged.
[ ] YS* unchanged.
[ ] X-target mapping valid.
[ ] Y-target mapping valid.
[ ] X→raw→Y bridge passes.
[ ] stats axes correct.
[ ] no batch stats.
[ ] round-trip passes.
[ ] target denorm passes.
[ ] low-variance passes.
[ ] batch independence passes.
[ ] neighbor leakage test passes.
[ ] future-target leakage test passes.
[ ] stats detach passes.
[ ] backbone gradient path passes.
[ ] gamma/beta gradients pass.
[ ] feature order unchanged.
[ ] external output [B,1] y_model.
[ ] loss applied after bridge.
[ ] Huber semantics unchanged if active.
[ ] early stop uses RMSE Wh.
[ ] BEST uses RMSE Wh.
[ ] backbone shapes identical.
[ ] state-dict delta only RevIN affine.
[ ] RN1 parameter delta=2*C_R.
[ ] affine params optimizer-covered once.
[ ] no RevIN-specific LR.
[ ] same selected clipping.
[ ] same selected epoch cap.
[ ] same batch/sample population.
[ ] same Train IDs.
[ ] same Validation IDs.
[ ] RN0 exact S17 reference reused.
[ ] RN0 not retrained.
[ ] RN1 official run fresh.
[ ] no warm-start.
[ ] no optimizer-state reuse.
[ ] RN1 BEST verified.
[ ] Wh metrics verified.
[ ] full-precision RMSE used.
[ ] RMSE/R² consistent.
[ ] exact tie=RN0.
[ ] not-applicable not reported as win.
[ ] no runtime/param override.
[ ] single-seed limitation reported.
[ ] Validation-only limitation reported.
[ ] interaction limitations reported.
[ ] Phase41 reference written.
[ ] Test untouched.
```

---

# 73. Acceptance criteria

Empirical PASS requires:

```text
historical Appliances present
RN0 valid reused reference
RN1 exact registered implementation
no information leakage
same data/population/backbone/training
correct signal/time-feature scope
correct X→Y target bridge
all reversible/unit tests pass
parameter delta exactly 2*C_R
RN1 fresh seed42
RN1 BEST verified
winner by full-precision Validation RMSE Wh
Phase41 handoff generated
Test untouched
```

Non-applicable completion requires:

```text
historical target absent
RN1 not run
no hidden target injection
RN0 carried without empirical superiority claim
SKIPPED_NOT_APPLICABLE
Phase41 handoff generated
```

---

# 74. Failure conditions

FAIL nếu:

```text
future-target leakage
hidden historical target injection
batch stats
wrong target index
time-feature normalization
broken target coordinate bridge
global scaling removed/refit
RevIN config drift
backbone architecture drift
unexpected parameter delta
affine params missing from optimizer
different LR/loss/clip/budget
population mismatch
RN0 favorable rerun
RN1 warm-start
score-based rerun
Test access
```

---

# 75. Phase41 handoff contract

Phase41 nhận:

```text
current run id
selected RevIN ID
selected RevIN config if RN1
all S1-S17 selected fields
current boundary protocol = WB0
population fingerprint
metric version
```

Phase41 chỉ được thay:

```text
WB0 vs WB1
```

Không thay RevIN.

---

# 76. Definition of Done

Empirical path:

```text
Fixed S1-S17 Configuration
+ RN0/RN1
+ Historical Target Applicability
+ Leakage-safe Per-instance Stats
+ Time Features Pass-through
+ Correct Target Denormalization
+ Correct X→Y Coordinate Bridge
+ Exact 2*C_R Parameter Delta
+ One Fresh RN1
+ One Reused RN0
+ Verified BEST Metrics
+ S18 Winner
+ Phase41 Reference
+ No Test
```

Non-applicable path:

```text
Historical Target Absent
+ No Hidden Information Injection
+ RN1 Not Run
+ RN0 Carried Transparently
+ Phase41 Reference
```

---

# 77. Final execution summary

Correct empirical path:

```text
Load S17 winner
→ freeze everything
→ locate Appliances
→ resolve RevIN scope
→ preserve time features
→ preserve global scaling
→ build target scaler bridge
→ run reversible/leakage tests
→ reuse RN0
→ fresh RN1
→ verify RN1 BEST
→ compare Validation RMSE
→ exact tie RN0
→ Phase41
```

Correct not-applicable path:

```text
no historical Appliances
→ RN1 not scientifically valid
→ do not add hidden target history
→ no RN1 run
→ RN0 carry-forward
→ SKIPPED_NOT_APPLICABLE
→ Phase41
```

Incorrect:

```text
FS0
→ secretly add Appliances for RevIN
→ normalize time features
→ use future y in stats
→ remove global scaler
→ denorm in wrong coordinate
→ choose by parameter count
→ inspect Test
```

Chỉ sau khi `SWEEP_S18_REVIN-v1` kết thúc ở:

```text
PASS
PASS_WITH_WARNING
SKIPPED_NOT_APPLICABLE
```

mới chuyển sang **PHASE 41 — S19 Boundary protocol check**.

---

# 78. Method reference note

RN1 contract được grounded trên RevIN method và official PyTorch implementation của Kim et al., trong đó statistics được tính riêng theo từng feature của từng sequence, sau đó normalization được đảo ngược ở output. Project này bổ sung một target-selective adapter vì coursework model có multivariate input nhưng chỉ dự báo một output `Appliances`.

Các quyết định sau là project-specific và phải giữ nguyên như controlled protocol của Phase40:

```text
historical-target applicability gate
continuous-signal-only RevIN scope
time-feature pass-through
target-only denormalization
X-target → raw Wh → y_model bridge
FS0 SKIPPED_NOT_APPLICABLE policy
```
