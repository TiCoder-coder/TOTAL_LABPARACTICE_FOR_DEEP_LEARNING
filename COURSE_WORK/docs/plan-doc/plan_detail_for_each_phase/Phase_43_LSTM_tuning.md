# PHASE 43 — LSTM TUNING

## Kế hoạch tuning có kiểm soát cho LSTM baseline trước Rolling-Origin Robustness

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** Sequence-to-One, One-Step-Ahead  
**Upstream:** `CANDIDATE_SYNTHESIS-v1`  
**Phase ID:** `PHASE_43_LSTM_TUNING`  
**Output version:** `LSTM_TUNING-v1`  
**Phase trước:** `Phase_42_Candidate_synthesis.md`  
**Phase sau:** `Phase_44_Rolling-origin_robustness.md`

---

# 1. Vai trò của Phase 43

Phase 43 tạo một **LSTM baseline đã được tuning hợp lý và có kiểm soát** để việc so sánh Transformer với LSTM ở Phase44 và các phase cuối không bị lệch vì LSTM chỉ dùng một cấu hình mặc định chưa tối ưu.

Mục tiêu:

```text
1. Giữ nguyên forecasting task và data contract đã khóa.
2. Dùng cùng current selected feature/data context từ Phase42.
3. Xây một LSTM reference trên final shared data context.
4. Tune một số hyperparameters có ý nghĩa trực tiếp với LSTM.
5. Dùng sequential one-factor controlled tuning.
6. Giới hạn compute budget.
7. Không mở full Cartesian grid.
8. Chọn tuned LSTM bằng verified Validation RMSE Wh.
9. Freeze tuned LSTM trước Rolling-Origin.
10. Không sử dụng Test.
```

Nguyên tắc trung tâm:

\[
\boxed{
Same\ Task/Data
+
Standard\ LSTM\ Family
+
Bounded\ One\text{-}Factor\ Tuning
+
Validation\ RMSE\ Selection
+
No\ Test
}
\]

---

# 2. Vì sao Phase43 cần tồn tại

Phase20 đã chạy:

```text
LSTM_B0
```

như official learned baseline ban đầu.

Tuy nhiên từ Phase23 đến Phase42, Transformer đã được:

```text
feature/data selection
+
architecture tuning
+
optimizer tuning
+
robustness preparation.
```

Nếu final comparison chỉ dùng LSTM_B0 ban đầu thì kết luận:

```text
Transformer vs LSTM
```

có thể bị ảnh hưởng bởi chênh lệch tuning effort.

Phase43 giảm rủi ro đó bằng cách cho LSTM một tuning budget riêng nhưng được giới hạn trước.

---

# 3. Phase43 không cố biến LSTM thành một second main model

Mục tiêu vẫn là:

```text
strong but interpretable baseline
```

không phải exhaustive LSTM AutoML.

Không được:

```text
Bayesian optimization
Optuna unrestricted
random search hàng trăm run
full Cartesian grid
architecture search
bidirectional LSTM
attention-LSTM
CNN-LSTM
encoder-decoder LSTM
stacked residual LSTM
```

trừ khi có Protocol Amendment riêng.

---

# 4. Model family bị khóa

Phase43 chỉ dùng standard unidirectional LSTM từ:

```text
LSTM_IMPL-v1
```

Architecture core:

```text
[B,L,F]
→ nn.LSTM(batch_first=True, unidirectional)
→ last sequence output
→ Linear(hidden_size,1)
→ [B,1]
```

Không output activation.

Không state carry-over giữa windows.

---

# 5. Shared fairness contract từ Phase42

Phase43 phải load:

```text
phase43_lstm_tuning_handoff.json
phase_42_signoff.json
transformer_candidate_shortlist.json
```

Hard shared task/data fields:

```text
forecast horizon = H1
selected feature variant = FV*
selected target scaling = YS*
selected lookback = L*
boundary protocol = WB0
split version = SPLIT-v1
population policy = WINDOWPOP-v1
train-only scaler artifacts
Validation metric = RMSE Wh
Test locked
```

---

# 6. Những gì LSTM phải giống Transformer để comparison công bằng

Hard:

```text
same target definition
same forecast horizon
same feature information
same time features
same historical-target inclusion/exclusion
same selected target scaling
same selected lookback
same Train target IDs
same Validation target IDs
same boundary protocol WB0
same chronological split
same scaler provenance
same metric implementation
same Test firewall.
```

---

# 7. Những gì không cần giống Transformer

LSTM là model family khác nên các fields sau không cần copy:

```text
d_model
num_heads
head_dim
Transformer layers
FFN width
GELU/ReLU sweep
Transformer pooling
positional encoding
RevIN adapter
attention-aware encoder.
```

Phase43 tune LSTM-specific settings riêng.

---

# 8. RevIN policy cho LSTM tuning

Phase43 dùng:

```text
standard LSTM baseline family
RevIN = OFF
```

kể cả khi final Transformer candidate dùng RN1.

Lý do:

```text
Phase43 mục tiêu là tune standard LSTM baseline,
không xây thêm một LSTM+RevIN model family.
```

Report phải nói rõ đây là model-family limitation.

Không được âm thầm thêm RevIN vào LSTM.

---

# 9. LSTM tuning reference

Tạo/resolve:

```text
LSTM_T0_REF
```

Data:

```text
FV*
YS*
L*
H1
WB0
WINDOWPOP-v1
```

Architecture:

```text
hidden_size = 64
num_layers = 2
dropout_arg = 0.1
bidirectional = false
batch_first = true
projection = none
last-step output
Linear(H,1)
```

Training reference:

```text
batch = B*
AdamW
lr = 3e-4
weight_decay = 1e-4
loss = MSE
max_epochs = 50
patience = 10
min_delta = 0
gradient_clip = global norm 1.0
scheduler = None
warmup = None
accumulation = 1
AMP = false
seed = 42
```

---

# 10. Vì sao LSTM tuning reference không copy toàn bộ final Transformer training choices

Phase43 tune một baseline model family độc lập.

Reference training contract giữ gần `LSTM_B0`:

```text
MSE
E50
P10
GC1
LR3e-4
WD1e-4
```

while data/task context được cập nhật về final shared context.

Điều này tạo một standard LSTM baseline hợp lý, dễ giải thích và không phụ thuộc vào Transformer-specific training decisions.

---

# 11. Batch size policy

Reference uses:

```text
batch = B*
```

từ final shared context nếu Phase42 handoff cung cấp exact current batch.

Nếu Phase42 handoff không explicitly carry batch:

```text
resolve current selected B* from final selected config
```

và record.

Phase43 không mở batch sweep.

---

# 12. Reference reuse từ Phase20

`LSTM_B0` Phase20 chỉ được reuse làm `LSTM_T0_REF` nếu exact match trên:

```text
FV*
YS*
L*
WB0
WINDOWPOP-v1
B*
H64
N2
dropout0.1
AdamW
LR3e-4
WD1e-4
MSE
E50
P10
clip1
seed42
Training Engine
Metric version.
```

Nếu bất kỳ field nào khác:

```text
không reuse
→ run fresh LSTM_T0_REF.
```

---

# 13. Tuning strategy

Phase43 dùng 5 controlled stages:

```text
LT1 — Hidden-size sweep
LT2 — Layer-count sweep
LT3 — LSTM inter-layer dropout sweep
LT4 — Learning-rate sweep
LT5 — Weight-decay sweep
```

Mỗi stage:

```text
winner của stage trước
→ reference cho stage sau.
```

Không Cartesian product.

---

# 14. Registered tuning space

## LT1 Hidden size

```text
LH32  = hidden_size 32
LH64  = hidden_size 64
LH128 = hidden_size 128
```

## LT2 Layers

```text
LN1 = num_layers 1
LN2 = num_layers 2
```

## LT3 Inter-layer dropout

Applicable only when selected:

```text
num_layers >= 2.
```

Options:

```text
LD0  = 0.0
LD1  = 0.1
LD2  = 0.2
```

## LT4 Learning rate

```text
LLR1 = 1e-4
LLR2 = 3e-4
LLR3 = 1e-3
```

## LT5 Weight decay

```text
LWD0 = 0
LWD1 = 1e-4
LWD2 = 1e-3
```

Không candidate ngoài registry này.

---

# 15. Maximum tuning budget

Normal maximum fresh runs:

```text
Reference:
0 or 1 fresh depending reuse

LT1:
2 fresh + 1 reused/reference

LT2:
1 fresh + 1 reused/reference

LT3:
2 fresh + 1 reused/reference
or skipped if N1

LT4:
2 fresh + 1 reused/reference

LT5:
2 fresh + 1 reused/reference
```

Maximum:

```text
10 fresh scientific runs
```

including a fresh reference when Phase20 cannot be reused.

No hidden extra runs.

---

# 16. Why tuning budget is bounded

Phase43 phải đủ mạnh để baseline không bị under-tuned nhưng không được trở thành:

```text
an unlimited second research project.
```

The budget is:

```text
predeclared
small
interpretable
one-factor.
```

---

# 17. Primary selection metric

At every LT stage:

```text
verified BEST Validation RMSE Wh
```

Winner:

\[
\arg\min RMSE_{Validation,Wh}
\]

MAE/R² secondary.

---

# 18. Full precision only

Do not rank using rounded report values.

---

# 19. Stage tie rules

Use predeclared parsimony.

## LT1 hidden-size exact tie

Prefer:

```text
smaller hidden size
32 < 64 < 128.
```

## LT2 exact tie

Prefer:

```text
1 layer.
```

## LT3 exact tie

Prefer:

```text
lower dropout
0 < 0.1 < 0.2.
```

## LT4 exact tie

Prefer:

```text
lower learning rate
1e-4 < 3e-4 < 1e-3.
```

## LT5 exact tie

Prefer:

```text
lower weight decay
0 < 1e-4 < 1e-3.
```

Only exact full-precision RMSE ties.

---

# 20. No arbitrary gain threshold

Do not require:

```text
1%
1 Wh
statistical significance
```

to pick a stage winner.

Strict lower RMSE wins.

---

# 21. Reuse policy across tuning stages

If current reference setting is one candidate in the next sweep:

```text
reuse exact completed current winner
```

instead of retraining.

Examples:

```text
LT1 winner LH64
→ LT2 LN2 reference may reuse LT1 winner

LT4 reference LLR2=3e-4
→ reuse current model if exact.
```

---

# 22. No warm-start between candidates

Every fresh candidate:

```text
seed42
fresh DataLoaders
fresh LSTM
fresh optimizer
fresh early-stop state.
```

Never:

```text
load winner checkpoint and continue with changed hyperparameter.
```

---

# 23. Same initialization policy

Within a stage, candidates should use:

```text
seed42
```

and same model-construction ordering.

Whole-state equality is not expected when dimensions differ.

For same-shape candidates:

```text
initial-state fingerprint match strongly preferred.
```

---

# 24. Same sample order

Fresh candidates:

```text
same Train sample IDs
same loader seed policy
same batch size B*
```

and ideally same per-epoch sample-order fingerprints when candidate batch is unchanged.

All LT stages keep B* fixed, so this is feasible.

---

# 25. Same data across all LSTM tuning runs

Hard:

```text
same FV*
same YS*
same L*
same WB0
same WINDOWPOP-v1
same Train IDs
same Val IDs.
```

No data sweep in Phase43.

---

# 26. Same scaler artifacts

Hard:

```text
X scaler checksums
Y scaler checksum/identity
```

same across all LSTM candidates.

---

# 27. Same target units for selection

All stage comparisons use:

```text
Validation RMSE Wh
```

after inverse target transform.

---

# 28. LT1 — Hidden-size sweep

Reference state:

```text
num_layers=2
dropout_arg=0.1
LR3e-4
WD1e-4
MSE
E50
P10
GC1
```

Compare:

```text
H32
H64
H128.
```

Only:

```text
hidden_size
```

changes.

---

# 29. Hidden-size parameter impact

For standard LSTM, parameter count changes materially with hidden size.

This is expected.

Do not parameter-match.

Record:

```text
trainable parameters
runtime
memory
```

as secondary engineering diagnostics.

---

# 30. Hidden-size candidate shape contract

For each:

```text
LSTM output sequence [B,L,H]
last output         [B,H]
head                [B,1].
```

No output shape change.

---

# 31. LT2 — Layer-count sweep

Use LT1 winner hidden size.

Compare:

```text
N1
N2.
```

Keep raw `dropout_arg=0.1`.

PyTorch LSTM semantics:

```text
dropout is applied between recurrent layers only,
not after the last layer.
```

Therefore with:

```text
num_layers=1
```

effective inter-layer dropout is:

```text
0
```

even if raw argument remains 0.1.

This is a semantic consequence of changing depth, not a second manually tuned factor.

---

# 32. Layer-depth audit

Record:

```text
raw_dropout_arg
effective_inter_layer_dropout
num_inter_layer_dropout_sites.
```

Expected:

```text
N1 → 0 sites
N2 → 1 site.
```

---

# 33. No external dropout added to compensate N1

Do not add:

```text
Dropout after LSTM
Dropout before head
```

when N1.

That changes architecture family.

---

# 34. LT3 — Dropout sweep applicability

If LT2 winner:

```text
num_layers = 2
```

run:

```text
D0
D0.1
D0.2.
```

Only raw LSTM inter-layer dropout changes.

If LT2 winner:

```text
num_layers = 1
```

then:

```text
LT3 = SKIPPED_NOT_APPLICABLE
```

because built-in LSTM dropout has no effective inter-layer location.

Carry:

```text
effective dropout = 0.
```

---

# 35. Do not force a dropout sweep when N1 wins

No external dropout module is introduced.

---

# 36. LT4 — Learning-rate sweep

Use winner after LT3 or LT2 skip.

Compare:

```text
1e-4
3e-4
1e-3.
```

Only AdamW LR changes.

Keep:

```text
WD=1e-4
MSE
E50
P10
clip1
```

fixed.

---

# 37. No LR scheduler

All candidates:

```text
scheduler=None
warmup=None.
```

Constant LR.

---

# 38. No LR scaling by hidden size/layers

Do not automatically rescale LR.

The sweep directly tests fixed values.

---

# 39. LT5 — Weight-decay sweep

Use LT4 winner LR.

Compare:

```text
0
1e-4
1e-3.
```

Optimizer remains:

```text
AdamW
```

even for:

```text
WD=0.
```

Do not switch to Adam.

---

# 40. Weight decay is decoupled AdamW weight decay

No explicit L2 term added to loss.

---

# 41. Optimizer parameter-group policy

Use the same LSTM optimizer-group policy across LT1–LT5.

Recommended baseline:

```text
single AdamW group over all trainable LSTM + head parameters
```

unless Phase20 implementation already froze a different valid grouping.

Phase43 must inspect and preserve the actual LSTM baseline policy.

Do not invent bias/LayerNorm exclusion mid-tuning.

---

# 42. Loss remains MSE

Hard:

```text
MSELoss(reduction="mean")
```

throughout Phase43.

No MSE-vs-Huber tuning.

Reason:

```text
keep LSTM as standard interpretable baseline
and control tuning budget.
```

---

# 43. Epoch cap remains 50

Hard:

```text
max_epochs=50
patience=10
min_delta=0.
```

No LSTM epoch-cap sweep.

---

# 44. Gradient clipping remains max_norm 1.0

Hard:

```text
global norm clipping
max_norm=1.0
norm_type=2
```

with non-finite guard.

No clipping sweep in Phase43.

---

# 45. Training Engine

Use:

```text
TRAINING_ENGINE-v1
```

same mechanics as Transformer runs:

```text
sample-weighted Train loss
full Validation every epoch
BEST by Validation RMSE Wh
early stopping
atomic BEST/LAST checkpoints
fresh BEST reload verification.
```

---

# 46. No stateful LSTM training

Each window independent.

Initialize recurrent hidden/cell states via default zeros per forward.

No carry hidden state across batches/windows.

---

# 47. No PackedSequence

All windows fixed length.

No.

---

# 48. No teacher forcing

No decoder.

---

# 49. No bidirectional LSTM

Hard:

```text
bidirectional=false.
```

Bidirectional within an already-past lookback would not necessarily cause future-target leakage, but it changes baseline architecture and is outside registered tuning scope.

---

# 50. No projection

Hard:

```text
proj_size=0.
```

---

# 51. No attention layer

No.

---

# 52. No CNN front-end

No.

---

# 53. No residual recurrent stack

No.

---

# 54. No extra dense MLP head

Head remains:

```text
Linear(hidden_size,1).
```

---

# 55. Same feature tensor order

Input feature order identical to Phase42 primary data contract.

---

# 56. LSTM input size dynamic

Runtime:

```text
input_size = F_current
```

from final selected feature variant.

No hard-coded 31.

---

# 57. Historical Appliances handling

If final selected feature variant includes past Appliances:

```text
LSTM receives it exactly like Transformer data contract.
```

If excluded:

```text
LSTM does not receive it.
```

No hidden extra autoregressive input.

---

# 58. Time features handling

Use exact selected time-feature state.

No LSTM-specific extra calendar features.

---

# 59. Target scaling

Use exact selected YS*.

Training predictions and targets remain:

```text
y_model.
```

Evaluation:

```text
inverse to Wh
→ METRICS-v1.
```

---

# 60. Validation population

Same target IDs as current primary WB0 configuration.

Hard.

---

# 61. Test firewall

Forbidden:

```text
Test DataLoader iteration
Test predictions
Test metrics
Test residuals
Test-based tuning.
```

---

# 62. Reference run verification

If Phase20 reusable:

```text
verify exact config
verify BEST
verify population
verify metric version
verify no Test contamination.
```

If not:

```text
register LSTM_T0_REF
train fresh
verify BEST.
```

---

# 63. Reference metrics

Record:

```text
Validation MAE Wh
Validation RMSE Wh
Validation R²
best epoch
epochs completed
stop reason
parameter count
runtime.
```

This is the baseline against which tuned LSTM improvement can be contextualized.

---

# 64. Tuned LSTM winner

Final LT5 winner becomes:

```text
LSTM_TUNED
```

If LT3 skipped, lineage remains valid.

Final winner configuration must include:

```text
hidden_size*
num_layers*
dropout*
learning_rate*
weight_decay*
```

plus fixed training/data settings.

---

# 65. Final winner is existing completed run

Because LT5 is final stage and reference/candidates are actual runs:

```text
LSTM_TUNED must map to a completed registered run.
```

No synthesized unevaluated LSTM winner.

---

# 66. Comparison against LSTM_T0_REF

Compute:

\[
\Delta RMSE_{tuning}
=
RMSE_{REF}
-
RMSE_{TUNED}
\]

Positive:

```text
tuning improved.
```

Even if tuned result is worse due sequential decisions, final winner across phase should never be worse than the current reference at each stage because reference is always included as a candidate.

Hard audit:

```text
Final tuned RMSE <= reference RMSE
```

within exact stage lineage.

---

# 67. Negative tuning improvement indicates a bug

If:

```text
RMSE_TUNED > RMSE_REF
```

investigate:

```text
reference omitted from stage
ranking error
full precision issue
wrong run ID
population mismatch.
```

---

# 68. Stage winner lineage

Create:

```text
LT0_REF
→ LT1 winner
→ LT2 winner
→ LT3 winner/skip
→ LT4 winner
→ LT5 winner.
```

Every next stage reference must equal prior winner.

---

# 69. No backtracking combinations

Suppose:

```text
LT1 H128 wins
LT2 N1 wins
LT4 LR1e-4 wins
```

do not later test:

```text
H64 + N1 + LR1e-4
```

unless it is exactly a reference already created.

No combinatorial backtracking.

---

# 70. Sequential tuning limitation

Mandatory reporting:

```text
The final LSTM configuration is a greedy sequential tuning result, not a global optimum over the Cartesian hyperparameter space.
```

---

# 71. Single-seed limitation

All tuning:

```text
seed42.
```

No seed robustness here.

Phase46 final multi-seed is for final locked model(s); Phase44 gives temporal robustness.

---

# 72. LSTM-vs-Transformer comparison in Phase43

Phase43 may provide contextual Validation table with:

```text
Persistence
LSTM_T0_REF
LSTM_TUNED
TR_C0_PRIMARY
```

only when all rows are on exact same Validation target population.

This is descriptive.

Do not use Transformer score to tune LSTM.

---

# 73. Transformer score must not influence LSTM candidate decisions

Winner at each stage:

```text
min LSTM candidate Validation RMSE
```

independent of whether it beats Transformer.

---

# 74. Persistence score must not influence tuning

Same.

---

# 75. Methodological pass does not require LSTM to beat Transformer

Phase43 PASS if tuning procedure valid.

---

# 76. Methodological pass does not require LSTM to beat Persistence

Also not required.

A weaker baseline remains scientifically informative.

---

# 77. Parameter count formulas are runtime-verified

Do not hard-code expected counts except for sanity/reference examples.

For each run:

```text
runtime_trainable_params = source of truth.
```

---

# 78. Architecture complexity diagnostics

Record:

```text
hidden size
layers
effective recurrent dropout
parameter count
checkpoint size
optional peak memory
runtime/epoch.
```

Secondary only.

---

# 79. Convergence diagnostics

For each candidate:

```text
best epoch
stop epoch
steps to best
Train loss trajectory
Validation RMSE trajectory
best-to-last gap
gradient norm/clipping fraction
non-finite events.
```

---

# 80. Numerical stability

Any:

```text
NaN
Inf
non-finite gradients
```

is failure.

Do not silently skip batches.

---

# 81. Gradient clipping telemetry

Since clip1 fixed, record:

```text
mean preclip norm
max preclip norm
clipping fraction.
```

Secondary.

---

# 82. No rescue policy

If candidate unstable:

```text
do not lower LR
do not increase clipping
do not reduce hidden size mid-run.
```

Run is failed under registered config.

---

# 83. Technical rerun policy

Allowed only for infrastructure failures:

```text
process interruption
hardware error
artifact-write error
corrupt checkpoint.
```

Not for poor RMSE.

---

# 84. Score-based rerun forbidden

No.

---

# 85. Stage incomplete on candidate failure

If one registered candidate fails numerically due valid configuration:

```text
stage status = INCOMPLETE/FAIL
```

until project failure policy determines treatment.

Do not silently rank remaining candidates unless protocol explicitly permits it.

---

# 86. Run registration

Recommended family:

```text
LSTM_TUNING
```

Subfamilies/tags:

```text
LT0_REFERENCE
LT1_HIDDEN_SIZE
LT2_LAYERS
LT3_DROPOUT
LT4_LEARNING_RATE
LT5_WEIGHT_DECAY.
```

---

# 87. Run ID/provenance

Every fresh run stores:

```text
run_id
stage_id
candidate_id
source_reference_run_id
config fingerprint
data fingerprints
BEST checksum
metrics artifact
status.
```

---

# 88. Output directory

```text
artifacts/
└── lstm_tuning/
    ├── lstm_tuning_manifest.json
    ├── lstm_tuning_contract.json
    ├── phase43_preflight_audit.csv
    ├── lstm_shared_data_contract.json
    ├── lstm_reference_resolution.json
    ├── lstm_reference_audit.csv
    ├── lstm_tuning_space.json
    ├── lstm_run_matrix.csv
    ├── lstm_stage_lineage.csv
    ├── lt1_hidden_size_metrics.csv
    ├── lt1_hidden_size_winner.json
    ├── lt2_layers_metrics.csv
    ├── lt2_layers_winner.json
    ├── lt3_dropout_applicability.json
    ├── lt3_dropout_metrics.csv
    ├── lt3_dropout_winner.json
    ├── lt4_learning_rate_metrics.csv
    ├── lt4_learning_rate_winner.json
    ├── lt5_weight_decay_metrics.csv
    ├── lt5_weight_decay_winner.json
    ├── lstm_architecture_audit.csv
    ├── lstm_training_config_delta_audit.csv
    ├── lstm_common_data_audit.csv
    ├── lstm_initialization_audit.csv
    ├── lstm_sample_order_audit.csv
    ├── lstm_optimizer_group_audit.csv
    ├── lstm_optimizer_budget_audit.csv
    ├── lstm_gradient_diagnostics.csv
    ├── lstm_convergence_diagnostics.csv
    ├── lstm_runtime_diagnostics.csv
    ├── lstm_run_provenance.csv
    ├── lstm_tuning_effect.csv
    ├── lstm_contextual_baseline_comparison.csv
    ├── lstm_tuned_winner.json
    ├── phase44_rolling_origin_lstm_handoff.json
    ├── lstm_tuning_findings.csv
    ├── lstm_tuning_tests.csv
    ├── lstm_tuning_discrepancies.json
    ├── lstm_tuning_summary.json
    ├── lstm_tuning_report.md
    ├── figures/
    │   ├── LSTM_43_01_hidden_size_rmse.png
    │   ├── LSTM_43_02_layers_rmse.png
    │   ├── LSTM_43_03_dropout_rmse.png
    │   ├── LSTM_43_04_learning_rate_rmse.png
    │   ├── LSTM_43_05_weight_decay_rmse.png
    │   ├── LSTM_43_06_validation_learning_curves.png
    │   ├── LSTM_43_07_parameter_count_vs_rmse.png
    │   ├── LSTM_43_08_runtime_vs_rmse.png
    │   └── LSTM_43_09_reference_vs_tuned.png
    ├── README_LSTM_TUNING.md
    └── phase_43_signoff.json
```

---

# 89. Required outputs

```text
O43.1  Tuning manifest
O43.2  Tuning contract
O43.3  Preflight audit
O43.4  Shared data contract
O43.5  Reference resolution
O43.6  Reference audit
O43.7  Tuning space
O43.8  Run matrix
O43.9  Stage lineage
O43.10 LT1 metrics/winner
O43.11 LT2 metrics/winner
O43.12 LT3 applicability/metrics/winner
O43.13 LT4 metrics/winner
O43.14 LT5 metrics/winner
O43.15 Architecture audit
O43.16 Training delta audit
O43.17 Common data audit
O43.18 Initialization audit
O43.19 Sample-order audit
O43.20 Optimizer-group audit
O43.21 Optimizer-budget audit
O43.22 Gradient diagnostics
O43.23 Convergence diagnostics
O43.24 Runtime diagnostics
O43.25 Run provenance
O43.26 Tuning effect
O43.27 Contextual baseline comparison
O43.28 Final tuned LSTM winner
O43.29 Phase44 handoff
O43.30 Findings
O43.31 Tests
O43.32 Discrepancies
O43.33 Summary
O43.34 Report
O43.35 Figures
O43.36 README
O43.37 Sign-off
```

---

# 90. Tuning manifest

`lstm_tuning_manifest.json`:

```text
version = LSTM_TUNING-v1
phase = 43
source_phase42_signoff
source_phase43_handoff
model_family = STANDARD_UNIDIRECTIONAL_LSTM
reference_id = LSTM_T0_REF
stages = [LT1,LT2,LT3,LT4,LT5]
max_fresh_runs = 10
selection_metric = validation_rmse_wh
selection_direction = MIN
seed = 42
test_access = forbidden
sequential_tuning = true
cartesian_search = false
status
created_at
```

---

# 91. Shared data contract artifact

`lstm_shared_data_contract.json`:

```text
forecast_task
target = Appliances
horizon = 1
feature_variant_id
feature_names
feature_fingerprint
target_scaling_id
lookback_id
lookback_steps
boundary_protocol = WB0
window_population_version
train_target_ids_fingerprint
validation_target_ids_fingerprint
x_scaler_checksums
y_scaler_checksum_or_identity
split_version
metric_version
test_locked = true
```

---

# 92. Reference resolution artifact

`lstm_reference_resolution.json`:

```text
phase20_run_id
phase20_exact_match
mismatch_fields_if_any
reference_source = REUSED_PHASE20 | FRESH_PHASE43
reference_run_id
reference_config
status
```

---

# 93. Tuning space artifact

`lstm_tuning_space.json`:

```text
LT1:
hidden_size = [32,64,128]

LT2:
num_layers = [1,2]

LT3:
dropout = [0.0,0.1,0.2]
applicable_if_num_layers_gte_2 = true

LT4:
learning_rate = [1e-4,3e-4,1e-3]

LT5:
weight_decay = [0,1e-4,1e-3]

fixed:
MSE
E50
P10
GC1
scheduler none
warmup none
B*
seed42
```

---

# 94. Run matrix

`lstm_run_matrix.csv` fields:

```text
stage_id
candidate_id
source_type
source_run_id
requires_new_training
feature_variant_id
target_scaling_id
lookback_id
boundary_protocol
batch_size
hidden_size
num_layers
dropout_arg
effective_inter_layer_dropout
learning_rate
weight_decay
loss
max_epochs
patience
gradient_clip
seed
train_population_fingerprint
validation_population_fingerprint
model_config_fingerprint
training_config_fingerprint
status
```

---

# 95. Stage lineage artifact

`lstm_stage_lineage.csv`:

```text
stage
reference_run_id
candidate_run_ids
winner_run_id
winner_value
winner_rmse_wh
next_stage_reference
status
```

Hard:

```text
next_stage_reference == current winner.
```

---

# 96. Stage metrics schema

For every LT stage:

```text
candidate_id
run_id
candidate_value
source_type
epochs_completed
stop_reason
best_epoch
trainable_parameters
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

# 97. LT3 applicability artifact

`lt3_dropout_applicability.json`:

If selected layers=1:

```text
applicable=false
reason=NO_INTER_LAYER_DROPOUT_SITE
selected_effective_dropout=0
stage_status=SKIPPED_NOT_APPLICABLE
```

If layers=2:

```text
applicable=true.
```

---

# 98. Architecture audit

`lstm_architecture_audit.csv`:

```text
run_id
input_size
hidden_size
num_layers
dropout_arg
effective_dropout
bidirectional
proj_size
batch_first
pooling_semantics
head_in_features
head_out_features
output_shape_verified
parameter_count
status
```

---

# 99. Training config delta audit

For each stage candidate vs stage reference:

```text
field
reference_value
candidate_value
allowed_to_differ
derived_difference
status
```

Only stage factor may differ.

---

# 100. Common data audit

`lstm_common_data_audit.csv`:

```text
run_id
train_count
validation_count
train_ids_match
validation_ids_match
feature_fingerprint_match
x_scaler_match
y_scaler_match
lookback_match
boundary_protocol_match
status
```

---

# 101. Initialization audit

`lstm_initialization_audit.csv`:

```text
stage
candidate_id
seed
shape_compatible_with_reference
initial_state_fingerprint
reference_fingerprint_available
same_shape_match_expected
match
status
```

When hidden/layers differ:

```text
whole-state match not expected.
```

---

# 102. Sample-order audit

Since batch fixed:

```text
stage
epoch_or_probe
candidate_id
sample_order_fingerprint
reference_order_fingerprint
same_order
status
```

---

# 103. Optimizer-group audit

`lstm_optimizer_group_audit.csv`:

```text
run_id
group_id
parameter_count
parameter_names_fingerprint
learning_rate
weight_decay
policy_id
status
```

Across WD sweep:

```text
topology/membership same
only WD changes.
```

---

# 104. Optimizer-budget audit

```text
run_id
train_samples_per_epoch
batch_size
steps_per_epoch
epochs_completed
total_optimizer_steps
best_epoch
steps_to_best
status
```

---

# 105. Gradient diagnostics

```text
run_id
epoch
mean_preclip_grad_norm
max_preclip_grad_norm
clipped_batches
total_batches
clipping_fraction
nonfinite_events
status
```

---

# 106. Convergence diagnostics

```text
run_id
stage
first_epoch_rmse
best_epoch
best_rmse
last_epoch
last_rmse
early_stopped
cap_reached
epochs_after_best
status
```

---

# 107. Runtime diagnostics

```text
run_id
device
parameter_count
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
time_to_best_optional
peak_memory_optional
status
```

Secondary only.

---

# 108. Tuning effect

`lstm_tuning_effect.csv`:

```text
reference_run_id
tuned_run_id
reference_rmse_wh
tuned_rmse_wh
rmse_improvement_wh
rmse_improvement_pct
reference_mae_wh
tuned_mae_wh
mae_improvement_wh
reference_r2
tuned_r2
r2_delta
reference_params
tuned_params
status
```

---

# 109. Contextual baseline comparison

Only if exact same Validation population:

```text
model
run_id
role
sample_count
mae_wh
rmse_wh
r2
population_fingerprint
comparable
notes
```

Rows may include:

```text
Persistence
LSTM_T0_REF
LSTM_TUNED
TR_C0_PRIMARY.
```

---

# 110. Final winner artifact

`lstm_tuned_winner.json` minimum:

```text
version = LSTM_TUNING-v1
model_family = LSTM
winner_id = LSTM_TUNED
winner_run_id
feature_variant_id
target_scaling_id
lookback_id
boundary_protocol = WB0
batch_size
hidden_size
num_layers
dropout_arg
effective_inter_layer_dropout
learning_rate
weight_decay
loss = MSE
max_epochs = 50
patience = 10
gradient_clip = 1.0
bidirectional = false
proj_size = 0
pooling = LAST_SEQUENCE_OUTPUT
seed = 42
validation_mae_wh
validation_rmse_wh
validation_r2
parameter_count
population_fingerprint
metric_version
test_status = NOT_ACCESSED
status
```

---

# 111. Phase44 LSTM handoff

`phase44_rolling_origin_lstm_handoff.json`:

```text
lstm_tuning_version
winner_run_id
winner_config
winner_config_fingerprint
shared_feature_variant
shared_target_scaling
shared_lookback
boundary_protocol = WB0
window_population_policy
loss=MSE
max_epochs=50
patience=10
gradient_clip=1.0
metric_version
seed_policy_for_fold_training
test_locked=true
source_phase42_transformer_shortlist_fingerprint
ready_for_phase44=true
```

---

# 112. Phase44 fairness contract

Phase44 receives:

```text
frozen Transformer candidate shortlist
tuned LSTM
Persistence baseline
```

All must use the same fold-level forecasting task and evaluation timestamps according to Phase44.

Phase43 does not define rolling folds.

---

# 113. No Phase44 fold peeking

Phase43 must not inspect future Phase44 fold results.

---

# 114. Findings codes

`lstm_tuning_findings.csv` possible:

```text
PHASE20_REFERENCE_REUSED
FRESH_FINAL_DATA_REFERENCE_REQUIRED
H32_GAIN
H64_GAIN
H128_GAIN
N1_GAIN
N2_GAIN
DROPOUT_STAGE_SKIPPED_N1
D0_GAIN
D01_GAIN
D02_GAIN
LR1_GAIN
LR2_GAIN
LR3_GAIN
WD0_GAIN
WD1_GAIN
WD2_GAIN
EXACT_TIE_PARSIMONY
METRIC_RANKING_DIVERGENCE
RMSE_R2_RANKING_INCONSISTENCY
NUMERICAL_INSTABILITY
REFERENCE_TO_TUNED_IMPROVEMENT
REFERENCE_TO_TUNED_EXACT_TIE
TUNED_LSTM_BEATS_PERSISTENCE_CONTEXTUAL
TUNED_LSTM_BELOW_PERSISTENCE_CONTEXTUAL
TUNED_LSTM_BEATS_TRANSFORMER_CONTEXTUAL
TUNED_LSTM_BELOW_TRANSFORMER_CONTEXTUAL
SAMPLE_ORDER_NOT_VERIFIABLE
INITIALIZATION_NOT_VERIFIABLE
TEST_FIREWALL_PRESERVED
INHERITED_WARNING
```

Contextual comparison findings do not alter winner.

---

# 115. Discrepancy taxonomy

`lstm_tuning_discrepancies.json`:

```text
PHASE42_NOT_APPROVED
HANDOFF_MISSING
SHARED_DATA_CONTRACT_MISMATCH
FEATURE_VARIANT_DRIFT
TARGET_SCALING_DRIFT
LOOKBACK_DRIFT
BOUNDARY_PROTOCOL_DRIFT
TRAIN_POPULATION_MISMATCH
VALIDATION_POPULATION_MISMATCH
SCALER_MISMATCH
REFERENCE_REUSE_MISMATCH
LSTM_ARCHITECTURE_DRIFT
BIDIRECTIONAL_ENABLED
PROJECTION_ENABLED
STATEFUL_TRAINING_ENABLED
PACKED_SEQUENCE_UNEXPECTED
EXTRA_HEAD_LAYER_ADDED
LOSS_DRIFT
EPOCH_CAP_DRIFT
PATIENCE_DRIFT
GRADIENT_CLIP_DRIFT
BATCH_DRIFT
SCHEDULER_ADDED
WARMUP_ADDED
CARTESIAN_SEARCH_ATTEMPT
UNREGISTERED_HYPERPARAMETER
STAGE_REFERENCE_NOT_PRIOR_WINNER
MULTI_FACTOR_STAGE_CHANGE
DROPOUT_STAGE_RUN_WHEN_N1
WARM_START_USED
OPTIMIZER_STATE_REUSE
SCORE_BASED_RERUN
FAILED_CANDIDATE_OMITTED
ROUNDING_RANK_ERROR
RMSE_R2_RANKING_INCONSISTENCY
TEST_LABEL_ACCESSED
TEST_PREDICTION_GENERATED
TEST_METRIC_COMPUTED
OTHER
```

---

# 116. Status model

## PASS

```text
shared data/task valid
reference valid
all applicable LT stages complete
stage lineage valid
winner verified
Phase44 handoff ready
Test untouched.
```

## PASS_WITH_WARNING

Possible:

```text
LT3 skipped because N1 selected
small tuning margins
metric ranking divergence
initialization fingerprints unavailable
sample-order fingerprints unavailable
inherited warning.
```

## FAIL

Examples:

```text
data fairness broken
hidden Cartesian search
multi-factor change
invalid candidate silently omitted
Test access
winner not verified.
```

---

# 117. Stage execution order

Exact:

```text
Resolve LSTM_T0_REF
↓
LT1 Hidden Size
↓
LT2 Layers
↓
LT3 Dropout if applicable
↓
LT4 Learning Rate
↓
LT5 Weight Decay
↓
LSTM_TUNED
```

No reordering after seeing results.

---

# 118. Preflight sequence

Before reference/fresh training:

```text
1. Verify Phase42 signoff.
2. Load Phase43 LSTM handoff.
3. Freeze shared data/task.
4. Resolve B*.
5. Verify WINDOWPOP-v1 target IDs.
6. Verify scaler checksums.
7. Verify LSTM_IMPL-v1.
8. Lock standard LSTM family.
9. Lock tuning space.
10. Lock tuning budget.
11. Lock MSE/E50/P10/GC1.
12. Lock seed42.
13. Audit Phase20 reuse eligibility.
14. Lock Test firewall.
15. Verify Experiment Registry.
```

---

# 119. LT1 execution

```text
reference H64
candidate H32
candidate H128
```

Fresh only for candidates not already exact valid run.

Each fresh:

```text
reseed42
fresh loaders
fresh model
fresh optimizer
train
verify BEST
```

Build LT1 metrics.

Pick minimum RMSE.

---

# 120. LT2 execution

Take exact LT1 winner config.

Compare:

```text
N1
N2.
```

Reference candidate reused if matching current winner.

Only layer count changes.

Select by RMSE.

---

# 121. LT3 execution

If:

```text
N*=1
```

write:

```text
SKIPPED_NOT_APPLICABLE
```

and continue.

If:

```text
N*=2
```

compare:

```text
dropout 0
0.1
0.2.
```

Select by RMSE.

---

# 122. LT4 execution

Use current winner.

Compare constant LR:

```text
1e-4
3e-4
1e-3.
```

Select by RMSE.

---

# 123. LT5 execution

Use LT4 winner.

Compare AdamW WD:

```text
0
1e-4
1e-3.
```

Select by RMSE.

Final winner = LSTM_TUNED.

---

# 124. Winner verification after LT5

Fresh-load:

```text
exact LSTM_TUNED config
BEST checkpoint
```

Evaluate full ordered Validation.

Verify:

```text
sample IDs
prediction shape
finite values
MAE/RMSE/R²
recorded BEST metrics.
```

---

# 125. No Test in final winner verification

Hard.

---

# 126. RMSE/R² consistency guard

Same Validation targets.

If:

```text
RMSE ranking != R² ranking
```

investigate.

MAE can differ.

---

# 127. Stage comparison must include reference

Every stage candidate set includes current reference.

This guarantees greedy tuning cannot intentionally move to worse RMSE.

---

# 128. No stage winner by runtime

No.

---

# 129. No stage winner by parameter count except exact tie

No.

---

# 130. No stage winner by train loss

No.

---

# 131. No stage winner by clipping fraction

No.

---

# 132. No stage winner by convergence speed

No.

---

# 133. No Test-based early selection

No.

---

# 134. Contextual comparison with Transformer

After LSTM_TUNED frozen, optional Validation context:

```text
TR_C0_PRIMARY
vs
LSTM_TUNED
```

must use same current Validation target population.

This comparison is **not final robustness evidence**.

Phase44 decides temporal robustness.

---

# 135. Contextual comparison with Persistence

Same rule.

---

# 136. Report tuning effort transparently

Include:

```text
number of fresh runs
number of reused runs
number of failed runs
number of skipped stages
total registered candidate count.
```

Do not hide tuning budget.

---

# 137. Comparison fairness caveat

Transformer received broader tuning than LSTM.

Phase43 should state:

```text
LSTM tuning is intentionally bounded and architecture-appropriate;
it does not claim exhaustive optimization parity with the Transformer search space.
```

This is honest and academically safer.

---

# 138. Why standard LSTM uses MSE fixed

Reasoning:

```text
1. baseline remains recognizable
2. controls tuning budget
3. Phase20 baseline already used MSE
4. metric selection remains RMSE Wh
5. avoids turning LSTM tuning into another S1–S18-scale search.
```

---

# 139. Why batch is fixed

Avoid extra optimizer-step confound and compute expansion.

---

# 140. Why lookback is not retuned for LSTM

Lookback is part of shared forecasting/data contract from Phase42.

Retuning it separately would make model-family comparison less controlled.

---

# 141. Why feature set is not retuned for LSTM

Same reason.

Both model families should receive the same selected information set for main comparison.

---

# 142. Why target scaling is not retuned for LSTM

Same target representation ensures fairer training/evaluation context.

---

# 143. Why boundary protocol remains WB0

WB0 is primary project protocol after S19 sensitivity review.

WB1 remains separate sensitivity evidence.

---

# 144. Why no RevIN for LSTM

RevIN was evaluated as part of Transformer-side design chain.

Applying it to LSTM would create:

```text
LSTM+RevIN
```

which is not the standard baseline family registered in Phase15/20.

This limitation must be reported if Transformer final uses RN1.

---

# 145. Potential limitation if RN1 strongly benefits Transformer

If final Transformer uses RN1 and LSTM does not:

```text
comparison reflects complete selected Transformer pipeline
vs tuned standard LSTM baseline.
```

Do not claim pure backbone-only causality.

This nuance belongs in final discussion.

---

# 146. Phase44 will strengthen fairness

Rolling-origin evaluates:

```text
same temporal folds
same forecast target timestamps per fold
same metrics
```

for Transformer/LSTM/Persistence.

Therefore Phase43 only needs to deliver a frozen tuned LSTM spec.

---

# 147. No final Test conclusion in Phase43

No.

---

# 148. Recommended report structure

`lstm_tuning_report.md`:

```text
1. Objective
2. Fairness contract from Phase42
3. Why LSTM tuning is needed
4. Standard LSTM family definition
5. LSTM_T0_REF resolution
6. Tuning budget and registered space
7. LT1 Hidden-size results
8. LT2 Layer-count results
9. LT3 Dropout results/applicability
10. LT4 Learning-rate results
11. LT5 Weight-decay results
12. Stage lineage
13. Tuned LSTM configuration
14. Reference-to-tuned improvement
15. Convergence/gradient/runtime diagnostics
16. Contextual Persistence/Transformer comparison
17. Sequential/single-seed limitations
18. RevIN/model-family limitation
19. Phase44 handoff
20. Test firewall
```

---

# 149. README requirements

`README_LSTM_TUNING.md` explains:

```text
what Phase43 tunes
what it does not tune
shared data fairness
reference reuse logic
5 tuning stages
dropout N1 applicability
winner rules
compute budget
fresh-run rules
no warm-start
why MSE/E50/GC1 fixed
why no RevIN
Phase44 handoff
No Test.
```

---

# 150. Figures

Recommended:

```text
LSTM_43_01_hidden_size_rmse.png
LSTM_43_02_layers_rmse.png
LSTM_43_03_dropout_rmse.png
LSTM_43_04_learning_rate_rmse.png
LSTM_43_05_weight_decay_rmse.png
LSTM_43_06_validation_learning_curves.png
LSTM_43_07_parameter_count_vs_rmse.png
LSTM_43_08_runtime_vs_rmse.png
LSTM_43_09_reference_vs_tuned.png
```

If LT3 skipped:

```text
do not generate fake dropout comparison figure.
```

---

# 151. Phase43 sign-off

`phase_43_signoff.json` minimum:

```text
phase = 43
phase_name = LSTM tuning
version = LSTM_TUNING-v1
source_phase42_signoff
shared_feature_variant
shared_target_scaling
shared_lookback
boundary_protocol = WB0
population_fingerprint
reference_run_id
reference_source
lt1_winner_run_id
lt1_hidden_size
lt2_winner_run_id
lt2_num_layers
lt3_status
lt3_winner_run_id_if_any
lt3_dropout
lt4_winner_run_id
lt4_learning_rate
lt5_winner_run_id
lt5_weight_decay
tuned_lstm_run_id
tuned_validation_rmse_wh
tuned_validation_mae_wh
tuned_validation_r2
reference_validation_rmse_wh
tuning_improvement_wh
fresh_run_count
reused_run_count
failed_run_count
test_status = NOT_ACCESSED
phase44_handoff_ready
overall_status
created_at
```

---

# 152. Acceptance checklist — shared data

```text
[ ] Phase42 PASS/PASS_WITH_WARNING.
[ ] Phase43 handoff exists.
[ ] FV* loaded.
[ ] YS* loaded.
[ ] L* loaded.
[ ] WB0 fixed.
[ ] H1 fixed.
[ ] WINDOWPOP-v1 fixed.
[ ] Same Train target IDs.
[ ] Same Validation target IDs.
[ ] Feature order fixed.
[ ] X scaler checksums fixed.
[ ] Y scaler checksum/identity fixed.
[ ] Test locked.
```

---

# 153. Acceptance checklist — LSTM family

```text
[ ] LSTM_IMPL-v1 used.
[ ] batch_first=true.
[ ] unidirectional.
[ ] proj_size=0.
[ ] stateless windows.
[ ] no PackedSequence.
[ ] no attention.
[ ] no CNN.
[ ] no decoder.
[ ] last sequence output used.
[ ] Linear(H,1) head.
[ ] output [B,1].
[ ] no output activation.
[ ] RevIN off.
```

---

# 154. Acceptance checklist — tuning protocol

```text
[ ] LT1 exact candidates 32/64/128.
[ ] LT2 exact candidates 1/2.
[ ] LT3 exact candidates 0/.1/.2 if applicable.
[ ] LT3 skipped if N1.
[ ] LT4 exact candidates 1e-4/3e-4/1e-3.
[ ] LT5 exact candidates 0/1e-4/1e-3.
[ ] No extra values.
[ ] No Cartesian product.
[ ] Max fresh runs <=10.
[ ] Every stage includes current reference.
[ ] Every stage changes one factor only.
[ ] Winner becomes next reference.
[ ] No backtracking combinations.
```

---

# 155. Acceptance checklist — training

```text
[ ] B* fixed.
[ ] AdamW fixed.
[ ] MSE fixed.
[ ] max_epochs=50.
[ ] patience=10.
[ ] min_delta=0.
[ ] clip1 fixed.
[ ] scheduler=None.
[ ] warmup=None.
[ ] accumulation=1.
[ ] AMP=false.
[ ] seed42.
[ ] fresh model for fresh runs.
[ ] fresh optimizer.
[ ] fresh loaders.
[ ] no warm-start.
[ ] no optimizer-state reuse.
[ ] no score-based rerun.
[ ] nonfinite events fail.
```

---

# 156. Acceptance checklist — metrics

```text
[ ] full Validation each epoch.
[ ] RMSE Wh selects BEST.
[ ] RMSE Wh selects stage winner.
[ ] full precision ranking.
[ ] MAE secondary.
[ ] R² secondary.
[ ] RMSE/R² consistency checked.
[ ] exact stage tie rules applied.
[ ] reference-to-tuned effect computed.
[ ] tuned BEST reloaded/verified.
```

---

# 157. Acceptance checklist — provenance

```text
[ ] Phase20 reuse exact-match audited.
[ ] All fresh runs registered.
[ ] Config fingerprints generated.
[ ] Data fingerprints generated.
[ ] BEST checksums generated.
[ ] Stage lineage complete.
[ ] Failed runs retained.
[ ] Fresh/reused counts reported.
[ ] No hidden runs.
```

---

# 158. Acceptance checklist — handoff

```text
[ ] LSTM_TUNED winner JSON generated.
[ ] Phase44 LSTM handoff generated.
[ ] Transformer shortlist fingerprint referenced.
[ ] Persistence anchor reference available.
[ ] Test status NOT_ACCESSED.
[ ] Phase43 signoff generated.
```

---

# 159. Acceptance criteria

Phase43 PASS only when:

```text
The final shared forecasting/data contract from Phase42 is preserved.

A valid final-context LSTM reference exists.

The standard unidirectional LSTM family is preserved.

Only the registered LT1–LT5 tuning space is used.

The tuning budget is not exceeded.

All applicable stages follow sequential one-factor tuning.

Every stage includes the current reference.

Fresh candidates start from seed42 with fresh model/optimizer/loaders.

No warm-start or optimizer-state reuse occurs.

MSE/E50/P10/GC1 remain fixed.

All candidates use the same Train/Validation populations.

Validation RMSE Wh selects stage winners.

Exact tie rules are respected.

Final tuned LSTM BEST is verified.

Tuning effect vs LSTM_T0_REF is reported.

Phase44 handoff is generated.

Test remains untouched.
```

---

# 160. Failure conditions

Phase43 FAIL if:

```text
data context differs from Phase42

lookback or feature set is retuned

RevIN is added to LSTM

bidirectional/projection/attention variant is introduced

Cartesian grid is run

unregistered candidate value is added

more than tuning budget is used

stage changes multiple factors

stage winner is not next reference

reference candidate is omitted

failed candidate is silently dropped

candidate warm-starts from previous winner

optimizer state is reused

MSE/E50/P10/GC1 contract changes

Validation population differs across candidates

winner selected by rounded metric/runtime/train loss

score-based reruns occur

Test is accessed.
```

---

# 161. Common mistakes

## 161.1 Dùng Phase20 LSTM_B0 dù final data context đã đổi

Sai nếu config/data không exact-match.

## 161.2 Tune feature set riêng cho LSTM

Sai shared-data fairness.

## 161.3 Tune lookback riêng cho LSTM

Sai current Phase43 contract.

## 161.4 Chạy full H×N×D×LR×WD grid

Sai.

## 161.5 N1 thắng nhưng vẫn sweep built-in dropout

Không có effective inter-layer site.

## 161.6 N1 rồi thêm external dropout để “công bằng”

Sai architecture change.

## 161.7 LR cao bị NaN rồi tự bật clip mạnh hơn

Sai registered config.

## 161.8 Load previous BEST rồi đổi LR/WD train tiếp

Sai warm-start.

## 161.9 Chọn LSTM theo việc có đánh bại Transformer không

Sai.

## 161.10 Dùng Test để chọn hidden size

Forbidden.

---

# 162. Recommended execution pseudocode

```text
assert phase42.signoff_valid
handoff = load_phase43_lstm_tuning_handoff()

freeze_shared_data_contract(handoff)

assert WB == "WB0"
assert H == 1
assert test_locked

reference_cfg = {
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.1,
    batch: B*,
    lr: 3e-4,
    wd: 1e-4,
    loss: MSE,
    max_epochs: 50,
    patience: 10,
    clip: 1.0,
    seed: 42
}

reference = try_resolve_phase20_exact_match(reference_cfg)

if not reference.exact_match:
    reference = train_fresh(reference_cfg)

verify_best(reference)

current = reference

# LT1
lt1_candidates = [
    current with hidden=32,
    current with hidden=64,
    current with hidden=128
]
lt1_runs = materialize_with_reference_reuse(lt1_candidates)
current = select_min_rmse(lt1_runs, exact_tie="smaller_hidden")

# LT2
lt2_candidates = [
    current with layers=1,
    current with layers=2
]
lt2_runs = materialize_with_reference_reuse(lt2_candidates)
current = select_min_rmse(lt2_runs, exact_tie="fewer_layers")

# LT3
if current.num_layers >= 2:
    lt3_candidates = [
        current with dropout=0.0,
        current with dropout=0.1,
        current with dropout=0.2
    ]
    lt3_runs = materialize_with_reference_reuse(lt3_candidates)
    current = select_min_rmse(lt3_runs, exact_tie="lower_dropout")
else:
    write_lt3_skipped_not_applicable()

# LT4
lt4_candidates = [
    current with lr=1e-4,
    current with lr=3e-4,
    current with lr=1e-3
]
lt4_runs = materialize_with_reference_reuse(lt4_candidates)
current = select_min_rmse(lt4_runs, exact_tie="lower_lr")

# LT5
lt5_candidates = [
    current with wd=0,
    current with wd=1e-4,
    current with wd=1e-3
]
lt5_runs = materialize_with_reference_reuse(lt5_candidates)
current = select_min_rmse(lt5_runs, exact_tie="lower_wd")

LSTM_TUNED = current

verify_best_checkpoint(LSTM_TUNED)
assert same_validation_population(reference, LSTM_TUNED)
assert LSTM_TUNED.rmse <= reference.rmse

write_tuning_effect(reference, LSTM_TUNED)
write_phase44_lstm_handoff(LSTM_TUNED)
write_summary_report_readme_signoff()

assert test_not_accessed
```

---

# 163. Definition of Done

\[
\boxed{
Same\ Final\ Data/Task
+
Standard\ LSTM
+
Bounded\ LT1\text{-}LT5\ Tuning
+
One\text{-}Factor\ Sequential\ Selection
+
Verified\ Validation\ RMSE
+
Frozen\ Tuned\ LSTM
+
No\ Test
+
Phase44\ Handoff
}
\]

---

# 164. Final status contract

```text
PHASE 43 tunes the standard LSTM baseline.

Shared with Transformer:
same target
same H1
same selected features
same YS*
same L*
same WB0
same Train/Validation target IDs
same scaler provenance
same RMSE Wh metrics.

LSTM reference:
H64
N2
dropout .1
B*
LR3e-4
WD1e-4
MSE
E50
P10
GC1
seed42.

Stages:
LT1 hidden: 32/64/128
LT2 layers: 1/2
LT3 dropout: 0/.1/.2 if layers>=2
LT4 LR: 1e-4/3e-4/1e-3
LT5 WD: 0/1e-4/1e-3.

No:
feature retuning
lookback retuning
RevIN
bidirectional
attention
projection
Cartesian grid
warm-start
optimizer-state reuse
Test.

Selection:
verified BEST Validation RMSE Wh.

Final:
LSTM_TUNED
→ frozen
→ Phase44 Rolling-Origin Robustness.
```

---

# 165. Final check

Correct:

```text
Phase42 handoff
→ final shared data
→ resolve/fresh LSTM reference
→ hidden sweep
→ layer sweep
→ dropout sweep if applicable
→ LR sweep
→ WD sweep
→ verify tuned BEST
→ freeze LSTM_TUNED
→ Phase44 handoff
```

Incorrect:

```text
retune data
→ full Cartesian LSTM grid
→ add bidirectional/attention
→ choose model because it beats Transformer
→ inspect Test
```

Chỉ sau khi:

```text
LSTM_TUNING-v1 = PASS / PASS_WITH_WARNING
```

và:

```text
phase44_handoff_ready = true
```

mới chuyển sang **PHASE 44 — Rolling-Origin Robustness**.
