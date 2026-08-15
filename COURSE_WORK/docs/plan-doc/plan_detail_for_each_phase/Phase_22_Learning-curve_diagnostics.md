# PHASE 22 — LEARNING-CURVE DIAGNOSTICS

## Kế hoạch chẩn đoán learning dynamics của LSTM B0 và Transformer B0

**Project:** UCI Appliances Energy Prediction  
**Task:** Multivariate Time-Series Regression  
**Forecasting contract:** sequence-to-one, one-step-ahead  
**Phase trước:** `Phase_21_Transformer_B0_run.md`  
**Output version:** `LEARNING_DIAGNOSTICS-v1`

---

# 1. Vai trò của Phase 22

Phase 22 là cầu nối giữa hai official baseline runs và chuỗi controlled experiments tiếp theo.

```text
Phase 20
→ Official LSTM baseline

Phase 21
→ Official Transformer B0

Phase 22
→ Learning-curve diagnostics

Phase 23–41
→ Controlled Transformer experiments
```

Phase này không train thêm model. Nhiệm vụ là đọc các source artifacts bất biến từ Phase 20 và 21, kiểm toán tính toàn vẹn của training history, trực quan hóa learning dynamics, chẩn đoán các pattern như underfitting-like, overfitting-like, plateau, instability hoặc gradient stress, sau đó chuyển các quan sát đó thành hypothesis có thể kiểm chứng trong những phase đã pre-register.

Nguyên tắc trung tâm:

\[
\boxed{
Audit
+
Visualize
+
Diagnose
+
Compare
+
Hypothesize
+
Do\ Not\ Tune
}
\]

---

# 2. Câu hỏi Phase 22 phải trả lời

Phase 22 cần trả lời tối thiểu:

1. LSTM B0 và Transformer B0 có training history đầy đủ, liên tục và nhất quán với best checkpoint hay không?
2. Train loss thay đổi như thế nào theo epoch?
3. Validation loss thay đổi như thế nào?
4. Validation RMSE Wh, MAE Wh và R² thay đổi ra sao?
5. Best epoch nằm sớm, giữa hay gần cuối quá trình train?
6. Early stopping có kích hoạt đúng contract không?
7. Có dấu hiệu Validation deterioration sau best epoch không?
8. Có dấu hiệu underfitting-like hoặc plateau không?
9. Gradient norm có ổn định không?
10. Gradient clipping có được kích hoạt thường xuyên không?
11. Learning rate có giữ đúng `3e-4` như baseline hay không?
12. Runtime mỗi epoch có bất thường không?
13. LSTM và Transformer có learning dynamics khác nhau ra sao?
14. Những quan sát nào nên được kiểm chứng bằng Phase 23–41?
15. Có lỗi integrity nào nghiêm trọng đến mức phải dừng các sweep và quay lại audit upstream hay không?

---

# 3. Phase 22 không làm gì

Không thực hiện:

```text
Không train model mới.

Không resume baseline để train thêm.

Không đổi checkpoint.

Không chọn checkpoint bằng metric mới.

Không tune feature set.

Không tune lookback.

Không tune pooling.

Không tune activation.

Không tune batch size.

Không tune learning rate.

Không tune weight decay.

Không tune dropout.

Không tune d_model.

Không tune heads.

Không tune layers.

Không tune FFN.

Không đổi MSE sang Huber.

Không đổi epoch cap.

Không đổi gradient clipping.

Không bật RevIN.

Không đổi WB0 sang WB1.

Không chạy seed mới.

Không mở Test.

Không phân tích attention map.
```

---

# 4. Preconditions

Phase 22 chỉ được bắt đầu khi:

```text
Phase 20 = PASS
Phase 21 = PASS
```

và hai official source runs đều:

```text
status = COMPLETED
best checkpoint = verified
Validation metrics = verified
Test = LOCKED
```

---

# 5. Source artifacts bắt buộc

## 5.1 LSTM B0

Tối thiểu:

```text
config.json
status.json
training_history.csv
runtime_summary.json
best_validation_metrics.json
best_validation_predictions.csv
lstm_gradient_summary.json
lstm_training_stability_summary.json
best checkpoint verification artifact
```

## 5.2 Transformer B0

Tối thiểu:

```text
config.json
status.json
training_history.csv
runtime_summary.json
best_validation_metrics.json
best_validation_predictions.csv
transformer_gradient_summary.json
transformer_training_stability_summary.json
transformer_parameter_summary.json
best checkpoint verification artifact
```

## 5.3 Context baselines

Có thể đọc:

```text
PERSISTENCE-v1 Validation metrics
```

chỉ để đặt best learned-model metrics vào context. Persistence không có learning curve.

---

# 6. Source-of-truth hierarchy

Ưu tiên dữ liệu theo thứ tự:

```text
1. training_history.csv

2. best_validation_metrics.json

3. checkpoint verification metadata

4. runtime / gradient summaries

5. Experiment Registry records

6. figures được generate từ source artifacts

7. README/log chỉ dùng hỗ trợ.
```

Không lấy số chính thức từ screenshot, console text hoặc giá trị gõ tay.

---

# 7. Immutable-source rule

`training_history.csv` của Phase 20/21 là source artifact bất biến.

Phase 22 chỉ được:

```text
read
validate
derive
visualize
summarize
```

Không được:

```text
edit
drop epochs
fill missing rows
repair metrics locally
remove spikes
round source values
```

Nếu source sai, phải quay về phase tạo source để audit.

---

# 8. Fairness gate trước khi so learning curves

LSTM B0 và Transformer B0 chỉ được comparative-diagnose khi các field sau giống nhau:

```text
FS1_TF1
L144
H1
YS1
WB0
WINDOWPOP-v1
B64
seed 42
AdamW
LR 3e-4
WD 1e-4
MSE
max_epochs 50
patience 10
gradient clip max_norm 1.0
TRAINING_ENGINE-v1
METRICS-v1
```

Architecture khác nhau là chủ đích.

---

# 9. Population gate

Hard assertion:

```text
LSTM population_fingerprint
==
Transformer population_fingerprint
```

Nếu không bằng nhau:

```text
FAIL comparative diagnostics.
```

---

# 10. Metric gate

Hard:

```text
same METRICS-v1
same target unit = Wh
```

---

# 11. Training Engine gate

Hard:

```text
same TRAINING_ENGINE-v1
```

Nếu Training Engine khác nhau, learning dynamics comparison bị confounded.

---

# 12. Diagnostic principle quan trọng về train loss và Validation loss

Train loss thường được đo khi:

```text
model.train()
dropout active
weights thay đổi batch-by-batch
```

Validation loss được đo khi:

```text
model.eval()
dropout inactive
weights cố định tại cuối epoch
```

Vì vậy:

```text
Validation loss - Train loss
```

chỉ là diagnostic gap, không phải generalization gap tuyệt đối.

Đặc biệt, Validation loss nhỏ hơn Train loss không tự động chứng minh leakage.

---

# 13. Primary learning metric

Primary trajectory:

```text
Validation RMSE Wh
```

vì đây là metric đã khóa để chọn best checkpoint.

---

# 14. Secondary diagnostic signals

```text
Validation MAE Wh
Validation R²
Train MSE loss model-space
Validation MSE loss model-space
Mean gradient norm preclip
Max gradient norm preclip
Fraction batches clipped
Learning rate
Epoch duration
Best epoch
Stop reason
```

---

# 15. Diagnostic taxonomy

Mỗi finding nên được gán một code:

```text
D0  HEALTHY_LEARNING
D1  UNDERFITTING_LIKE
D2  OVERFITTING_LIKE
D3  PLATEAU
D4  OPTIMIZATION_INSTABILITY
D5  GRADIENT_STRESS
D6  EARLY_BEST_PATHOLOGY
D7  LATE_CONVERGENCE
D8  METRIC_DIVERGENCE
D9  RUNTIME_ANOMALY
D10 PIPELINE_INTEGRITY_ANOMALY
D11 INCONCLUSIVE
```

---

# 16. D0 — HEALTHY_LEARNING

Pattern có thể phù hợp với healthy learning khi:

```text
Train loss giảm.

Validation RMSE nhìn chung giảm.

Không có nonfinite event.

Gradient finite.

Best epoch hợp lý.

Không có severe oscillation kéo dài.
```

Không cần mọi epoch đều monotonic.

---

# 17. D1 — UNDERFITTING_LIKE

Evidence có thể gồm:

```text
Train loss vẫn cao hoặc plateau sớm.

Validation metric cũng yếu.

Best epoch gần epoch cap.

Validation vẫn cải thiện chậm ở cuối run.

Train/Validation không cho thấy deterioration rõ.
```

Chỉ kết luận:

```text
consistent with underfitting-like behavior
```

không khẳng định nguyên nhân.

---

# 18. D2 — OVERFITTING_LIKE

Evidence mạnh hơn khi đồng thời có:

```text
Train loss tiếp tục giảm.

Validation RMSE đạt minimum rồi tăng.

Best epoch sớm hơn last epoch đáng kể.

Early stopping xảy ra sau chuỗi non-improving epochs.
```

---

# 19. D3 — PLATEAU

Pattern:

```text
Validation RMSE thay đổi rất ít trong một khoảng epoch kéo dài.
```

Không sử dụng một threshold phổ quát kiểu:

```text
<1% trong 5 epoch = plateau
```

như quy luật khoa học cứng.

---

# 20. D4 — OPTIMIZATION_INSTABILITY

Possible evidence:

```text
Train loss spikes.

Validation RMSE oscillates mạnh.

Gradient norm spikes.

Epoch-to-epoch metric changes lớn và không có xu hướng ổn định.
```

Không tự kết luận LR sai. Phase 30 mới kiểm chứng.

---

# 21. D5 — GRADIENT_STRESS

Possible evidence:

```text
Gradient norms lớn.

Fraction batches clipped cao.

Clipping diễn ra ở nhiều epoch.

Gradient spikes đi cùng loss instability.
```

Phase 39 mới kiểm clipping causally.

---

# 22. D6 — EARLY_BEST_PATHOLOGY

Possible pattern:

```text
best epoch = 1 hoặc rất sớm

sau đó Validation không hồi phục.
```

Có thể liên quan optimization, regularization, capacity hoặc data; cần audit và controlled experiments.

---

# 23. D7 — LATE_CONVERGENCE

Possible pattern:

```text
best epoch ở gần epoch 50

Validation vẫn đang cải thiện khi run kết thúc.
```

Điều này tạo hypothesis cho:

```text
Phase 38 — epoch cap 50 vs 100
```

chứ không cho phép tự train thêm trong Phase 22.

---

# 24. D8 — METRIC_DIVERGENCE

Ví dụ:

```text
RMSE tốt hơn nhưng MAE xấu hơn

hoặc RMSE xấu đi nhiều hơn MAE.
```

Có thể cho thấy large errors đang ảnh hưởng mạnh.

Mapping:

```text
Phase 37 — Huber
Phase 50 — error by regime
Phase 51 — worst errors
```

---

# 25. D9 — RUNTIME_ANOMALY

Ví dụ:

```text
epoch duration tăng đột ngột

batch count thay đổi

Validation time biến động bất thường.
```

Đây là engineering finding, không phải model-quality finding.

---

# 26. D10 — PIPELINE_INTEGRITY_ANOMALY

Ví dụ:

```text
missing epoch
duplicate epoch
best epoch mismatch
metric mismatch
LR tự đổi dù scheduler=None
sample count drift
population mismatch
nonfinite metric
```

D10 là critical. Không được tiếp tục sweep cho tới khi resolve.

---

# 27. D11 — INCONCLUSIVE

Dùng khi evidence không đủ.

Không ép mọi run phải bị gắn nhãn overfit hoặc underfit.

---

# 28. Confidence level

Mỗi finding có:

```text
HIGH
MEDIUM
LOW
```

Ví dụ:

```text
HIGH:
Train loss giảm lâu sau best epoch trong khi Validation RMSE deteriorate liên tục.

LOW:
Một RMSE spike đơn lẻ.
```

---

# 29. Severity level

```text
CRITICAL
MAJOR
MODERATE
MINOR
INFO
```

`CRITICAL` dành cho integrity issues có thể làm baseline invalid.

---

# 30. Action type

```text
NO_ACTION
MONITOR
TEST_PRE_REGISTERED_FACTOR
AUDIT_UPSTREAM
PROTOCOL_AMENDMENT_REQUIRED
```

---

# 31. History integrity audit

Cho từng model, kiểm:

```text
epoch bắt đầu từ 1
epoch unique
epoch contiguous
row count = epochs_completed
last row epoch = last_completed_epoch
best_epoch tồn tại
best_epoch = argmin(raw Validation RMSE)
best RMSE = min(raw Validation RMSE)
stop reason consistent
early-stop fields consistent
```

---

# 32. Best-epoch tie rule

Do Training Engine dùng strict improvement:

```text
equal RMSE does not replace earlier best
```

khi recompute best epoch phải giữ earliest strict minimum.

---

# 33. Full precision rule

Recompute bằng raw full-precision values.

Không dùng display-rounded RMSE.

---

# 34. Best checkpoint identity

Verify:

```text
best checkpoint epoch
==
recomputed best epoch
```

---

# 35. Best metric identity

Verify:

```text
best_validation_metrics.json RMSE
==
history minimum RMSE
```

trong tolerance contract.

---

# 36. Stop-reason audit

Nếu:

```text
EARLY_STOPPING
```

thì final history phải tương thích với patience state.

Nếu:

```text
MAX_EPOCHS
```

B0 phải kết thúc ở epoch 50.

---

# 37. Learning-rate audit

Do baseline không scheduler:

```text
LR phải = 3e-4 ở mọi epoch.
```

Bất kỳ drift nào là integrity anomaly.

---

# 38. Sample-count audit

Mỗi epoch:

```text
Train sample count constant
Validation sample count constant
```

---

# 39. Metric finite audit

Hard:

```text
RMSE finite
MAE finite
```

R² phải follow `METRICS-v1` status.

---

# 40. Gradient audit

All recorded:

```text
mean_grad_norm_preclip
max_grad_norm_preclip
fraction_batches_clipped
```

phải finite và có range hợp lệ.

---

# 41. Clipping fraction range

Hard:

\[
0 \le fraction\_batches\_clipped \le 1.
\]

---

# 42. Epoch-duration audit

```text
finite
>=0
```

---

# 43. Canonical derived epoch table

Tạo:

```text
learning_curve_epoch_summary.csv
```

Một row cho mỗi:

```text
model × epoch.
```

Fields:

```text
run_id
model
epoch
train_loss_model_space
validation_loss_model_space
validation_mae_wh
validation_rmse_wh
validation_r2
learning_rate
mean_grad_norm_preclip
max_grad_norm_preclip
fraction_batches_clipped
epoch_duration_seconds
is_best
bad_epochs_after_epoch
early_stop_triggered
validation_rmse_delta_prev_epoch
```

---

# 44. Long-format table

Tạo:

```text
learning_curve_long.csv
```

Fields:

```text
run_id
model
epoch
metric_name
metric_value
metric_unit
is_best_epoch
source_artifact
diagnostic_version
```

---

# 45. Epoch-to-epoch RMSE delta

\[
\Delta RMSE_t
=
RMSE_t-RMSE_{t-1}
\]

Interpretation:

```text
negative = improvement
positive = deterioration
```

---

# 46. Initial-to-best improvement

\[
Improvement_{initial\rightarrow best}
=
RMSE_{epoch1}-RMSE_{best}
\]

---

# 47. Best-to-last change

\[
\Delta_{best\rightarrow last}
=
RMSE_{last}-RMSE_{best}
\]

Positive:

```text
last epoch worse than best.
```

---

# 48. Best-to-last relative change

\[
100
\times
\frac{RMSE_{last}-RMSE_{best}}
{RMSE_{best}}
\]

Diagnostic only.

---

# 49. Best epoch fraction

\[
best\_epoch\_fraction
=
\frac{best\_epoch}
{epochs\_completed}
\]

Diagnostic only.

---

# 50. Epochs after best

\[
epochs\_after\_best
=
epochs\_completed-best\_epoch
\]

---

# 51. Initial diagnostic window

Set:

```text
K_initial = min(5, epochs_completed)
```

để summarize early learning.

---

# 52. Tail diagnostic window

Set:

```text
K_tail = min(5, epochs_completed)
```

để summarize late trend.

Đây là descriptive window, không phải scientific threshold.

---

# 53. Initial diagnostics table

Tạo:

```text
learning_curve_initial_diagnostics.csv
```

Fields:

```text
model
initial_k
rmse_epoch1
rmse_initial_end
rmse_change
num_improvement_steps
train_loss_change
largest_rmse_spike
status
```

---

# 54. Tail diagnostics table

Tạo:

```text
learning_curve_tail_diagnostics.csv
```

Fields:

```text
model
tail_k
tail_start_epoch
tail_end_epoch
rmse_start_wh
rmse_end_wh
rmse_change_wh
rmse_change_pct
num_tail_improvements
best_epoch_in_tail
train_loss_change
status
```

---

# 55. Best epoch comparison

Tạo:

```text
best_epoch_comparison.csv
```

Fields:

```text
model
epochs_completed
best_epoch
epochs_after_best
best_epoch_fraction
stop_reason
best_rmse_wh
last_rmse_wh
best_to_last_change_wh
best_to_last_change_pct
```

---

# 56. Early stopping diagnostics

Tạo:

```text
early_stopping_diagnostics.csv
```

Fields:

```text
model
patience
min_delta
stop_reason
best_epoch
last_epoch
epochs_after_best
final_bad_epochs
early_stop_triggered
history_consistent
status
```

---

# 57. Gradient diagnostics

Tạo:

```text
gradient_diagnostics.csv
```

Fields:

```text
model
epochs_completed
global_max_grad_norm_preclip
mean_epoch_mean_grad_norm
median_epoch_mean_grad_norm
mean_fraction_batches_clipped
max_fraction_batches_clipped
epochs_with_any_clipping
nonfinite_gradient_events
status
```

---

# 58. Gradient interpretation

Gradient summary dùng để trả lời:

```text
có gradient stress không
clipping có thường xuyên không
có spike không
```

Không dùng để quyết định ngay:

```text
clip off/on
LR bao nhiêu
```

---

# 59. Runtime diagnostics

Tạo:

```text
runtime_diagnostics.csv
```

Fields:

```text
model
device
epochs_completed
total_runtime_seconds
mean_epoch_seconds
median_epoch_seconds
min_epoch_seconds
max_epoch_seconds
time_to_best_seconds_optional
status
```

---

# 60. Runtime caveat

First epoch có thể chậm hơn do:

```text
backend setup
cache initialization
kernel warm-up
```

Không over-interpret.

---

# 61. Cross-model comparison table

Tạo:

```text
learning_curve_model_comparison.csv
```

Rows:

```text
LSTM_B0
TRANSFORMER_B0
```

Fields:

```text
run_id
epochs_completed
best_epoch
best_epoch_fraction
stop_reason
initial_rmse_wh
best_rmse_wh
last_rmse_wh
initial_to_best_improvement_pct
best_to_last_degradation_pct
best_mae_wh
best_r2
mean_grad_norm_preclip
max_grad_norm_preclip
mean_fraction_batches_clipped
mean_epoch_duration_seconds
trainable_parameters
```

---

# 62. Persistence context table

Tạo:

```text
best_checkpoint_baseline_context.csv
```

Rows:

```text
Persistence
LSTM_B0
Transformer_B0
```

Columns:

```text
model
run_id
mae_wh
rmse_wh
r2
population_fingerprint
metric_version
```

Persistence không có epoch fields.

---

# 63. Không biến Phase 22 thành ranking phase

Có thể report:

```text
best B0 Validation RMSE among Persistence/LSTM/Transformer
```

nhưng không gọi đó là:

```text
final winner.
```

---

# 64. Core figures

Tối thiểu tạo:

```text
LC01_lstm_train_validation_loss.png

LC02_transformer_train_validation_loss.png

LC03_validation_rmse_comparison.png

LC04_validation_mae_comparison.png

LC05_validation_r2_comparison.png

LC06_gradient_norm_comparison.png

LC07_gradient_clipping_fraction.png

LC08_epoch_duration_comparison.png
```

---

# 65. Figure LC01

LSTM:

```text
epoch
vs
Train model-space MSE
and
Validation model-space MSE
```

Label rõ:

```text
Standardized target-space MSE
```

vì baseline dùng YS1.

---

# 66. Figure LC02

Transformer tương tự.

---

# 67. Figure LC03 — primary figure

Overlay:

```text
LSTM Validation RMSE Wh

Transformer Validation RMSE Wh
```

Mark:

```text
best epoch của từng model.
```

---

# 68. Persistence reference on LC03

Optional:

```text
horizontal Persistence Validation RMSE line
```

sau khi population/metric guard PASS.

---

# 69. Figure LC04

Validation MAE Wh trajectories.

---

# 70. Figure LC05

Validation R² trajectories.

---

# 71. Figure LC06

Gradient norm trajectories:

```text
mean_grad_norm_preclip
max_grad_norm_preclip
```

Nếu quá clutter, tách model figures.

---

# 72. Figure LC07

Fraction batches clipped per epoch.

---

# 73. Figure LC08

Epoch duration comparison.

Runtime only.

---

# 74. Optional figures

```text
LC09_best_to_last_zoom.png

LC10_best_checkpoint_baseline_context.png
```

---

# 75. Raw-curve rule

Raw values phải là source of truth.

---

# 76. Smoothing rule

Default:

```text
smoothing_enabled = False
```

Nếu dùng smoothing:

```text
visualization only
raw curve vẫn hiển thị
method ghi vào manifest
không được dùng smooth minimum để chọn checkpoint
```

---

# 77. Không xóa spikes

Spikes là evidence cần giữ.

---

# 78. Không extrapolate curve

Không dự đoán:

> Nếu train thêm 20 epoch RMSE sẽ còn giảm X Wh.

Phase 38 kiểm trực tiếp epoch cap.

---

# 79. Không confidence interval từ epochs

Epochs không phải independent repeated runs.

---

# 80. Không significance test giữa curves

Một seed/run mỗi architecture không phù hợp để làm hypothesis test bằng cách coi epochs là samples.

---

# 81. Single-seed limitation

Bắt buộc ghi:

```text
Both B0 runs use seed 42 only.
```

Differences may include initialization/training stochasticity.

---

# 82. Validation-only limitation

Bắt buộc ghi:

```text
All diagnostics are based on Train/Validation development behavior.
No Test evidence is used.
```

---

# 83. No causal attribution

Không viết:

```text
Self-attention causes faster convergence.
```

Có thể viết:

```text
Transformer B0 reached its best Validation checkpoint in fewer/more epochs than LSTM B0 in these seed-42 runs.
```

---

# 84. MAE/RMSE divergence interpretation

Nếu:

```text
RMSE deteriorates strongly
MAE changes little
```

có thể ghi:

```text
consistent with a subset of larger errors affecting squared-error-sensitive RMSE.
```

Mapping:

```text
Phase 37
Phase 50
Phase 51
```

---

# 85. No residual mining in Phase 22

Dù best predictions đã có, không làm deep residual analysis ở đây.

Roadmap đã dành:

```text
Phase 48 — Prediction analysis
Phase 49 — Residual analysis
Phase 50 — Error by regime
Phase 51 — Worst-error analysis
```

---

# 86. No attention analysis

Roadmap:

```text
Phase 52–57.
```

---

# 87. Diagnostic findings artifact

Tạo:

```text
learning_diagnostic_findings.csv
```

Fields:

```text
finding_id
model
diagnostic_code
title
epoch_range
evidence_metric_1
evidence_value_1
evidence_metric_2
evidence_value_2
interpretation
confidence
severity
action_type
mapped_future_phase
status
```

---

# 88. Finding wording rule

Good:

> Transformer B0 shows a sustained rise in Validation RMSE after the best epoch while Train loss continues to fall; this pattern is consistent with overfitting-like behavior.

Bad:

> Transformer overfits because dropout is too low.

Nguyên nhân dropout chưa được kiểm chứng.

---

# 89. Every finding must have evidence

Không có evidence fields:

```text
finding invalid.
```

---

# 90. Critical findings

Nếu category:

```text
D10 PIPELINE_INTEGRITY_ANOMALY
```

severity thường:

```text
CRITICAL.
```

Then:

```text
do not proceed to Phase 23
until resolved.
```

---

# 91. Hypothesis registry

Tạo:

```text
learning_curve_hypothesis_registry.csv
```

Fields:

```text
hypothesis_id
source_model
source_finding_id
hypothesis_statement
evidence_summary
confidence
pre_registered_phase
factor
candidate_options
expected_observation_if_supported
expected_observation_if_not_supported
status
```

---

# 92. Hypothesis initial status

Always:

```text
UNTESTED.
```

---

# 93. Hypothesis must be falsifiable

Mỗi hypothesis cần chỉ rõ:

```text
what would support it
what would fail to support it.
```

---

# 94. Mapping to Phase 23 — Feature set

Poor signal/representation hypothesis:

```text
Phase 23
FS0_TF1
FS1_TF1
FS2_TF1
```

---

# 95. Mapping to Phase 24 — Time features

Calendar contribution hypothesis:

```text
TF0 vs TF1.
```

---

# 96. Mapping to Phase 25 — Target scaling

Optimization-scale hypothesis:

```text
YS0 vs YS1.
```

---

# 97. Mapping to Phase 26 — Lookback

Temporal-context hypothesis:

```text
L36
L72
L144.
```

---

# 98. Mapping to Phase 27 — Pooling

Representation-readout hypothesis:

```text
LAST_STEP
MEAN.
```

---

# 99. Mapping to Phase 28 — Activation

```text
ReLU
GELU.
```

---

# 100. Mapping to Phase 29 — Batch size

Optimization-noise hypothesis:

```text
B32
B64.
```

---

# 101. Mapping to Phase 30 — Learning rate

Instability/slow optimization hypothesis:

```text
1e-4
3e-4
1e-3.
```

Không pre-declare winner.

---

# 102. Mapping to Phase 31 — Weight decay

Overfitting-like hypothesis:

```text
0
1e-4
1e-3.
```

---

# 103. Mapping to Phase 32 — Dropout

Overfitting-like hypothesis:

```text
0.1
0.2
0.3.
```

---

# 104. Mapping to Phase 33 — d_model

Capacity hypothesis:

```text
32
64.
```

---

# 105. Mapping to Phase 34 — Heads

```text
2
4.
```

---

# 106. Mapping to Phase 35 — Layers

```text
1
2.
```

---

# 107. Mapping to Phase 36 — FFN

```text
64
128
256.
```

---

# 108. Mapping to Phase 37 — Loss

Large-error sensitivity hypothesis:

```text
MSE
Huber.
```

---

# 109. Mapping to Phase 38 — Epoch cap

Late-convergence hypothesis:

```text
50
100.
```

---

# 110. Mapping to Phase 39 — Gradient clipping

Gradient stress hypothesis:

```text
OFF
ON max_norm=1.
```

---

# 111. Mapping to Phase 40 — RevIN

Distribution-shift/normalization hypothesis:

```text
OFF
ON.
```

---

# 112. Mapping to Phase 41 — Boundary

WB sensitivity remains pre-registered:

```text
WB0
WB1.
```

Learning curve alone usually không quyết định boundary protocol.

---

# 113. Mapping LSTM findings

LSTM-specific improvement:

```text
Phase 43 — LSTM tuning.
```

Không dùng Transformer sweep để sửa LSTM.

---

# 114. No ad hoc values

Không invent:

```text
dropout 0.17
LR 7e-5
d_model 48
```

nếu chưa có Protocol Amendment.

---

# 115. Do not reorder master plan casually

Diagnostic context có thể giúp interpret, nhưng master sweep schedule vẫn authoritative.

---

# 116. Learning curve integrity audit artifact

Tạo:

```text
learning_curve_integrity_audit.csv
```

Checks:

```text
run_id_match
history_checksum_valid
epochs_contiguous
epochs_unique
history_rows_match_epochs
best_epoch_recomputed
best_metric_recomputed
checkpoint_best_epoch_match
stop_reason_match
LR_constant
sample_counts_constant
metric_finite
gradient_finite
test_locked
status
```

---

# 117. Cross-model fairness audit

Tạo:

```text
learning_curve_comparison_audit.csv
```

Fields:

```text
field
lstm_value
transformer_value
must_match
matches
status
```

---

# 118. Fields bắt buộc so

```text
feature_variant
lookback
horizon
target_scaling
boundary_protocol
population_fingerprint
batch_size
seed
optimizer
learning_rate
weight_decay
criterion
max_epochs
patience
gradient_clip
training_engine_version
metric_version
```

---

# 119. Figure manifest

Tạo:

```text
learning_curve_figure_manifest.csv
```

Fields:

```text
figure_id
filename
models
metrics
source_artifacts
smoothing_used
smoothing_method
best_epoch_marker
purpose
status
```

---

# 120. Plot source rule

Mỗi figure phải trace được về:

```text
learning_curve_epoch_summary.csv
```

hoặc source history.

---

# 121. Diagnostic config

Tạo:

```text
learning_curve_diagnostic_config.json
```

Minimum:

```text
diagnostic_version = LEARNING_DIAGNOSTICS-v1
primary_metric = validation_rmse_wh
secondary_metrics
initial_k_rule = min(5,N)
tail_k_rule = min(5,N)
smoothing_enabled = false
taxonomy_version
confidence_levels
test_access = forbidden
```

---

# 122. Source manifest

Tạo:

```text
learning_curve_source_manifest.json
```

Minimum:

```text
diagnostic_version
lstm_run_id
lstm_history_path
lstm_history_sha256
lstm_best_checkpoint_sha256
lstm_config_fingerprint
transformer_run_id
transformer_history_path
transformer_history_sha256
transformer_best_checkpoint_sha256
transformer_config_fingerprint
training_engine_version
metric_version
population_version
population_fingerprint
test_access = forbidden
created_at
```

---

# 123. Diagnostic summary JSON

Tạo:

```text
learning_curve_diagnostic_summary.json
```

Structure:

```text
diagnostic_version

integrity_status

source_runs

lstm:
  epochs_completed
  best_epoch
  stop_reason
  initial_rmse
  best_rmse
  last_rmse
  best_to_last_change
  gradient_summary
  diagnostic_codes

transformer:
  ...

comparison:
  best_rmse_difference
  epoch_to_best_difference
  runtime_context

critical_findings

hypotheses_created

test_status

overall_status
```

---

# 124. Human-readable report

Tạo:

```text
learning_curve_diagnostic_report.md
```

Sections:

```text
1. Scope
2. Source artifacts
3. Integrity audit
4. LSTM learning dynamics
5. Transformer learning dynamics
6. Cross-model comparison
7. Gradient diagnostics
8. Early-stopping diagnostics
9. Runtime diagnostics
10. Diagnostic findings
11. Hypothesis mapping
12. Limitations
13. Phase 23 handoff
```

---

# 125. Report language

Use:

```text
suggests
consistent with
may indicate
requires controlled verification
```

Avoid unsupported:

```text
proves
causes
definitely due to
```

---

# 126. Diagnostic tests artifact

Tạo:

```text
learning_curve_diagnostic_tests.csv
```

Categories:

```text
SOURCE
HISTORY
BEST_CHECKPOINT
EARLY_STOPPING
METRIC
GRADIENT
RUNTIME
FAIRNESS
DERIVATION
FIGURE
HYPOTHESIS
TEST_FIREWALL
```

---

# 127. Required test suite

Recommended tests:

```text
LCD22-001 LSTM source run COMPLETED
LCD22-002 Transformer source run COMPLETED
LCD22-003 LSTM history checksum valid
LCD22-004 Transformer history checksum valid
LCD22-005 LSTM epochs contiguous
LCD22-006 Transformer epochs contiguous
LCD22-007 LSTM no duplicate epochs
LCD22-008 Transformer no duplicate epochs
LCD22-009 LSTM history rows = completed epochs
LCD22-010 Transformer history rows = completed epochs
LCD22-011 LSTM best epoch recomputed correctly
LCD22-012 Transformer best epoch recomputed correctly
LCD22-013 LSTM best RMSE matches metrics artifact
LCD22-014 Transformer best RMSE matches metrics artifact
LCD22-015 LSTM checkpoint epoch matches history
LCD22-016 Transformer checkpoint epoch matches history
LCD22-017 LSTM stop reason consistent
LCD22-018 Transformer stop reason consistent
LCD22-019 LSTM LR constant
LCD22-020 Transformer LR constant
LCD22-021 Train counts constant
LCD22-022 Validation counts constant
LCD22-023 RMSE finite
LCD22-024 MAE finite
LCD22-025 gradient diagnostics finite
LCD22-026 clipping fraction valid
LCD22-027 epoch duration valid
LCD22-028 FS1_TF1 match
LCD22-029 L144 match
LCD22-030 H1 match
LCD22-031 YS1 match
LCD22-032 WB0 match
LCD22-033 population fingerprint match
LCD22-034 B64 match
LCD22-035 seed 42 match
LCD22-036 optimizer match
LCD22-037 LR match
LCD22-038 WD match
LCD22-039 MSE match
LCD22-040 epoch cap match
LCD22-041 patience match
LCD22-042 gradient clipping match
LCD22-043 Training Engine match
LCD22-044 metric version match
LCD22-045 best-to-last calculation correct
LCD22-046 initial-to-best calculation correct
LCD22-047 best epoch fraction correct
LCD22-048 RMSE deltas correct
LCD22-049 tail diagnostics correct
LCD22-050 gradient summary correct
LCD22-051 comparison table source-derived
LCD22-052 figures source-derived
LCD22-053 no smoothed metric used for selection
LCD22-054 no model training executed
LCD22-055 no hyperparameter changed
LCD22-056 hypotheses map to registered phases
LCD22-057 hypotheses marked UNTESTED
LCD22-058 single-seed limitation documented
LCD22-059 Validation-only limitation documented
LCD22-060 no Test artifact accessed
LCD22-061 critical finding blocks sweep handoff
LCD22-062 phase sign-off generated
```

---

# 128. Discrepancy taxonomy

```text
SOURCE_ARTIFACT_MISSING
SOURCE_CHECKSUM_MISMATCH
RUN_ID_MISMATCH
HISTORY_EPOCH_GAP
HISTORY_DUPLICATE_EPOCH
BEST_EPOCH_MISMATCH
BEST_METRIC_MISMATCH
CHECKPOINT_HISTORY_MISMATCH
STOP_REASON_MISMATCH
LR_PROTOCOL_VIOLATION
SAMPLE_COUNT_DRIFT
NONFINITE_METRIC
NONFINITE_GRADIENT
POPULATION_MISMATCH
METRIC_VERSION_MISMATCH
TRAINING_ENGINE_MISMATCH
CONFIG_FAIRNESS_MISMATCH
TEST_FIREWALL_VIOLATION
PLOT_SOURCE_MISMATCH
OTHER
```

---

# 129. Discrepancy log

Tạo:

```text
learning_curve_discrepancies.json
```

Fields:

```text
id
severity
category
model_or_run
expected
actual
impact
recommended_action
resolved
resolution_notes
```

---

# 130. Status model

## PASS

```text
source histories valid
fairness gates pass
no critical integrity issue
diagnostics generated
hypotheses mapped
Test untouched
```

## PASS_WITH_WARNING

Ví dụ:

```text
strong instability observed
but source artifacts are valid
```

hoặc runtime anomaly không ảnh hưởng metric integrity.

## FAIL

Ví dụ:

```text
best checkpoint/history mismatch
missing epochs
population mismatch
metric version mismatch
LR changed unexpectedly
Test accessed
```

---

# 131. Critical anomaly protocol

Nếu có D10/CRITICAL:

```text
1. Stop Phase 22 interpretation.

2. Do not start Phase 23.

3. Trace affected source to Phase 19/20/21.

4. Fix/version according to protocol.

5. Reproduce affected official run if needed.

6. Re-run Phase 22.
```

Không sửa source history tại Phase 22.

---

# 132. Source-code organization

Recommended:

```text
src/
└── diagnostics/
    ├── learning_curves.py
    ├── training_history.py
    ├── gradient_diagnostics.py
    ├── runtime_diagnostics.py
    ├── diagnostic_findings.py
    └── plots.py
```

---

# 133. Function design

Recommended:

```text
load_training_history()

validate_history_integrity()

recompute_best_epoch()

compute_epoch_deltas()

compute_initial_diagnostics()

compute_tail_diagnostics()

compute_best_to_last_diagnostics()

compute_gradient_diagnostics()

compute_runtime_diagnostics()

compare_baseline_configs()

build_learning_curve_long_table()

build_model_comparison_table()

classify_diagnostic_findings()

build_hypothesis_registry()

write_diagnostic_manifest()
```

---

# 134. Calculation/plot separation

Numeric calculations không được phụ thuộc plotting code.

Plots nhận:

```text
clean derived tables
```

làm input.

---

# 135. Phase 22 compute profile

Phase 22 nên:

```text
low compute
CPU-friendly
no model training
no GPU requirement
```

Model checkpoint loading thường không cần ngoài metadata integrity.

---

# 136. Output directory

```text
artifacts/
└── learning_diagnostics/
    ├── learning_curve_source_manifest.json
    ├── learning_curve_diagnostic_config.json
    ├── learning_curve_integrity_audit.csv
    ├── learning_curve_comparison_audit.csv
    ├── learning_curve_epoch_summary.csv
    ├── learning_curve_long.csv
    ├── learning_curve_model_comparison.csv
    ├── best_checkpoint_baseline_context.csv
    ├── best_epoch_comparison.csv
    ├── early_stopping_diagnostics.csv
    ├── gradient_diagnostics.csv
    ├── runtime_diagnostics.csv
    ├── learning_curve_initial_diagnostics.csv
    ├── learning_curve_tail_diagnostics.csv
    ├── learning_diagnostic_findings.csv
    ├── learning_curve_hypothesis_registry.csv
    ├── learning_curve_figure_manifest.csv
    ├── learning_curve_diagnostic_tests.csv
    ├── learning_curve_discrepancies.json
    ├── learning_curve_diagnostic_summary.json
    ├── learning_curve_diagnostic_report.md
    ├── figures/
    │   ├── LC01_lstm_train_validation_loss.png
    │   ├── LC02_transformer_train_validation_loss.png
    │   ├── LC03_validation_rmse_comparison.png
    │   ├── LC04_validation_mae_comparison.png
    │   ├── LC05_validation_r2_comparison.png
    │   ├── LC06_gradient_norm_comparison.png
    │   ├── LC07_gradient_clipping_fraction.png
    │   └── LC08_epoch_duration_comparison.png
    ├── README_LEARNING_DIAGNOSTICS.md
    └── phase_22_signoff.json
```

---

# 137. Required outputs

```text
O22.1  learning_curve_source_manifest.json
O22.2  learning_curve_diagnostic_config.json
O22.3  learning_curve_integrity_audit.csv
O22.4  learning_curve_comparison_audit.csv
O22.5  learning_curve_epoch_summary.csv
O22.6  learning_curve_long.csv
O22.7  learning_curve_model_comparison.csv
O22.8  best_checkpoint_baseline_context.csv
O22.9  best_epoch_comparison.csv
O22.10 early_stopping_diagnostics.csv
O22.11 gradient_diagnostics.csv
O22.12 runtime_diagnostics.csv
O22.13 learning_curve_initial_diagnostics.csv
O22.14 learning_curve_tail_diagnostics.csv
O22.15 learning_diagnostic_findings.csv
O22.16 learning_curve_hypothesis_registry.csv
O22.17 core diagnostic figures
O22.18 learning_curve_figure_manifest.csv
O22.19 learning_curve_diagnostic_tests.csv
O22.20 learning_curve_discrepancies.json
O22.21 learning_curve_diagnostic_summary.json
O22.22 learning_curve_diagnostic_report.md
O22.23 README_LEARNING_DIAGNOSTICS.md
O22.24 phase_22_signoff.json
```

---

# 138. Recommended notebook structure

```text
Cell 22.1  Phase title
Cell 22.2  Verify Phase 20/21 sign-offs
Cell 22.3  Declare LEARNING_DIAGNOSTICS-v1
Cell 22.4  Load source manifests/configs
Cell 22.5  Verify source checksums/run IDs
Cell 22.6  Load LSTM history
Cell 22.7  Load Transformer history
Cell 22.8  History integrity audit
Cell 22.9  Recompute best epochs
Cell 22.10 Early stopping audit
Cell 22.11 Baseline fairness audit
Cell 22.12 Build epoch summary
Cell 22.13 Build long-format table
Cell 22.14 Compute epoch deltas
Cell 22.15 Compute initial diagnostics
Cell 22.16 Compute tail diagnostics
Cell 22.17 Compute best-to-last diagnostics
Cell 22.18 Gradient diagnostics
Cell 22.19 Runtime diagnostics
Cell 22.20 Best-epoch comparison
Cell 22.21 Cross-model comparison
Cell 22.22 Persistence context
Cell 22.23 Plot LSTM losses
Cell 22.24 Plot Transformer losses
Cell 22.25 Plot Validation RMSE
Cell 22.26 Plot MAE/R²
Cell 22.27 Plot gradients/clipping
Cell 22.28 Plot epoch duration
Cell 22.29 Classify findings
Cell 22.30 Build hypothesis registry
Cell 22.31 Validate phase mappings
Cell 22.32 Build tests/discrepancies
Cell 22.33 Write diagnostic summary
Cell 22.34 Write diagnostic report
Cell 22.35 Write README/manifest
Cell 22.36 Phase sign-off
```

---

# 139. Quy trình thực thi chuẩn

```text
Verify Phase 20/21
        ↓
Load immutable histories
        ↓
Verify checksums + run IDs
        ↓
History integrity audit
        ↓
Best checkpoint / stop-rule audit
        ↓
LSTM-vs-Transformer fairness audit
        ↓
Build canonical epoch tables
        ↓
Compute deltas / initial / tail / best-last
        ↓
Gradient diagnostics
        ↓
Runtime diagnostics
        ↓
Generate raw learning curves
        ↓
Classify findings
        ↓
Assign confidence + severity
        ↓
Map findings to pre-registered phases
        ↓
Create UNTESTED hypothesis registry
        ↓
Write report + artifacts
        ↓
Verify Test untouched
        ↓
LEARNING_DIAGNOSTICS-v1 sign-off
```

---

# 140. Fail-fast order

Không bắt đầu plotting nếu:

```text
run ID mismatch

history checksum invalid

best checkpoint mismatch

epoch integrity fail

population mismatch

metric version mismatch.
```

Phải audit trước, visualize sau.

---

# 141. Recommended diagnostic workflow cho từng model

```text
A. Verify source history.

B. Identify best epoch.

C. Compare epoch 1 → best.

D. Compare best → last.

E. Inspect raw Train/Validation losses.

F. Inspect Validation RMSE/MAE/R².

G. Inspect gradients/clipping.

H. Inspect early-stop trajectory.

I. Inspect runtime.

J. Assign diagnostic code(s).

K. Create hypothesis only if evidence supports it.
```

---

# 142. Learning-curve diagnostic report quality rule

Mỗi model nên có một concise evidence block:

```text
Observed:
<runtime-derived facts>

Interpretation:
<diagnostic interpretation>

Confidence:
HIGH/MEDIUM/LOW

Next verification:
<pre-registered phase>
```

---

# 143. Example evidence pattern — overfitting-like

```text
Observed:
Train loss decreases from epoch A to B.
Validation RMSE reaches minimum at epoch C.
Validation RMSE then deteriorates until stopping.
Best-to-last RMSE change is positive.

Interpretation:
Pattern is consistent with overfitting-like development after epoch C.

Confidence:
MEDIUM/HIGH depending consistency.

Next verification:
Weight decay/dropout/capacity experiments according to master plan.
```

Không ghi số A/B/C giả trước execution.

---

# 144. Example evidence pattern — late convergence

```text
Observed:
Best epoch occurs at or near final epoch.
Validation RMSE remains on a downward tail trend.

Interpretation:
Current 50-epoch cap may be limiting convergence.

Next verification:
Phase 38 — E50 vs E100.
```

---

# 145. Example evidence pattern — gradient stress

```text
Observed:
Large max gradient norms.
High clipping fraction across multiple epochs.

Interpretation:
Optimization experiences persistent gradient stress.

Next verification:
Phase 30 and Phase 39.
```

Không kết luận ngay clipping should be disabled.

---

# 146. Single-run restraint

Một baseline run không đủ để nói:

```text
Transformer always converges faster than LSTM.
```

Chỉ report run-specific observation.

---

# 147. No hidden model-selection logic

Phase 22 không được tạo rule:

```text
because Transformer B0 is worse than LSTM, skip Transformer sweeps.
```

Transformer is assignment main model; controlled experiments vẫn tiếp tục.

---

# 148. No hidden checkpoint reselection

Không được chọn epoch khác vì:

```text
curve looks smoother
MAE looks better
R² looks better.
```

Best checkpoint remains min Validation RMSE Wh from Phase 20/21.

---

# 149. No score-based rerun

Phase 22 không được trigger baseline rerun chỉ vì result xấu.

Technical/integrity failure mới hợp lệ.

---

# 150. No Test access

Hard status:

```text
test_status = LOCKED
```

trong diagnostic summary/sign-off.

---

# 151. Phase 22 acceptance checklist

```text
[ ] Phase 20 PASS.
[ ] Phase 21 PASS.
[ ] LEARNING_DIAGNOSTICS-v1 declared.
[ ] LSTM run ID recorded.
[ ] Transformer run ID recorded.
[ ] Source history checksums recorded.
[ ] Best checkpoint checksums recorded.
[ ] No Test artifact loaded.
[ ] Histories treated immutable.
[ ] Epochs contiguous.
[ ] Epochs unique.
[ ] Row counts match completed epochs.
[ ] Best epochs recomputed.
[ ] Best RMSE values recomputed.
[ ] Best metrics match source.
[ ] Best checkpoint metadata matches.
[ ] Stop reasons verified.
[ ] Early-stop semantics verified.
[ ] LR constant at baseline value.
[ ] Sample counts constant.
[ ] Metrics finite.
[ ] Gradient diagnostics finite.
[ ] Clipping fractions valid.
[ ] Runtime fields valid.
[ ] FS1_TF1 equality verified.
[ ] L144 equality verified.
[ ] H1 equality verified.
[ ] YS1 equality verified.
[ ] WB0 equality verified.
[ ] Population fingerprints equal.
[ ] B64 equality verified.
[ ] Seed 42 equality verified.
[ ] Optimizer/LR/WD/loss equality verified.
[ ] Epoch cap/patience/clipping equality verified.
[ ] Training Engine equality verified.
[ ] Metric version equality verified.
[ ] Epoch summary created.
[ ] Long table created.
[ ] Initial diagnostics created.
[ ] Tail diagnostics created.
[ ] Best-to-last diagnostics created.
[ ] Early-stopping table created.
[ ] Gradient table created.
[ ] Runtime table created.
[ ] Model comparison created.
[ ] Persistence context created.
[ ] Core figures generated from source.
[ ] Raw curves remain source of truth.
[ ] No smoothing used for selection.
[ ] No new training run.
[ ] No hyperparameter changed.
[ ] No checkpoint reselected.
[ ] Findings evidence-grounded.
[ ] Findings have confidence.
[ ] Findings have severity.
[ ] Critical finding blocks Phase 23.
[ ] Hypotheses mapped to pre-registered phases.
[ ] Hypotheses remain UNTESTED.
[ ] No ad hoc hyperparameter values invented.
[ ] Single-seed limitation documented.
[ ] Validation-only limitation documented.
[ ] No causal claims from curves.
[ ] Summary saved.
[ ] Diagnostic report saved.
[ ] Discrepancy log saved.
[ ] Test remains LOCKED.
[ ] Phase 22 sign-off saved.
```

---

# 152. Acceptance criteria

Phase 22 chỉ PASS khi:

```text
Both baseline histories are valid.

Best checkpoint/history identities are consistent.

LSTM and Transformer comparison fairness is verified.

Learning curves are generated from machine-readable sources.

Gradient, early-stopping and runtime diagnostics are complete.

No source history is edited.

No model is retrained.

No hyperparameter is changed.

No Test data is accessed.

Every diagnostic conclusion is evidence-grounded.

Every proposed explanation remains a hypothesis.

Hypotheses map to controlled future phases.

No unresolved critical integrity anomaly remains.
```

---

# 153. Khi nào Phase 22 FAIL?

```text
History missing or duplicate epochs.

Recorded best epoch is not actual raw RMSE minimum.

Best checkpoint disagrees with history.

LR changes despite scheduler=None.

Sample counts change across epochs.

LSTM/Transformer populations differ.

Metric versions differ.

Histories are manually repaired.

A smoothed curve determines a new best checkpoint.

Model training occurs inside Phase 22.

Hyperparameters are altered based on visual inspection.

Test metrics/predictions are opened.

Artifacts from different run IDs are mixed.

A diagnostic hypothesis is reported as proven cause.
```

---

# 154. Các lỗi thường gặp

## 154.1 Chỉ nhìn Train loss và gọi là overfitting

Sai. Cần Validation trajectory.

## 154.2 Validation loss thấp hơn Train loss và gọi là leakage

Không đủ bằng chứng vì train/eval dropout semantics khác.

## 154.3 Một spike = unstable

Không đủ.

## 154.4 Smooth curve rồi lấy minimum mới

Sai.

## 154.5 Thấy overfit rồi tăng dropout ngay

Phase 32 mới kiểm.

## 154.6 Thấy gradients lớn rồi giảm LR ngay

Phase 30 mới kiểm.

## 154.7 Thấy clipping thường xuyên rồi tắt clipping ngay

Phase 39 mới kiểm.

## 154.8 Best epoch = 50 rồi train thêm ngay

Phase 38 mới kiểm.

## 154.9 Dùng epochs làm independent samples để chạy p-value

Không phù hợp.

## 154.10 Tạo confidence interval từ một training history

Không phù hợp.

## 154.11 Xóa spike vì cho là “outlier epoch”

Không.

## 154.12 Gõ tay số liệu vào plot

Không.

## 154.13 Dùng Phase 19 SANITY history

Chỉ official Phase 20/21.

## 154.14 Bỏ các sweep vì B0 trông tốt

Master plan vẫn authoritative.

## 154.15 Mở Test để xem curve “có giống Validation không”

Forbidden.

---

# 155. Handoff sang Phase 23

Phase 23 nhận:

```text
TRANSFORMER_B0-v1
LEARNING_DIAGNOSTICS-v1
baseline run ID
baseline config fingerprint
learning_curve_hypothesis_registry.csv
```

Phase 23 chỉ thay factor:

```text
Feature set.
```

Reference:

```text
FS1_TF1
```

Comparison:

```text
FS0_TF1
FS1_TF1
FS2_TF1
```

---

# 156. Handoff sang Phase 24–41

Phase 22 findings cung cấp diagnostic context nhưng không thay option registry:

```text
Phase 24 TF0/TF1
Phase 25 YS0/YS1
Phase 26 L36/L72/L144
Phase 27 LAST_STEP/MEAN
Phase 28 ReLU/GELU
Phase 29 B32/B64
Phase 30 LR 1e-4/3e-4/1e-3
Phase 31 WD 0/1e-4/1e-3
Phase 32 dropout .1/.2/.3
Phase 33 D32/D64
Phase 34 H2/H4
Phase 35 N1/N2
Phase 36 FFN64/128/256
Phase 37 MSE/Huber
Phase 38 E50/E100
Phase 39 clipping OFF/ON
Phase 40 RevIN OFF/ON
Phase 41 WB0/WB1
```

---

# 157. Handoff sang Phase 43

LSTM-specific findings được lưu để phục vụ:

```text
Phase 43 — LSTM tuning.
```

Không overwrite LSTM_BASELINE-v1.

---

# 158. Handoff sang Phase 46

Multi-seed phase sau này sẽ kiểm whether learning-curve observations ở seed 42 có ổn định across seeds hay không.

---

# 159. Handoff sang Phase 48–51

Nếu Phase 22 thấy RMSE/MAE divergence, lưu hypothesis để later error analysis kiểm:

```text
Phase 48 Prediction analysis
Phase 49 Residual analysis
Phase 50 Error-by-regime
Phase 51 Worst-error analysis
```

---

# 160. Definition of Done

Phase 22 hoàn thành khi:

\[
\boxed{
Valid\ Histories
+
Transparent\ Learning\ Curves
+
Evidence\text{-}Based\ Diagnosis
+
Gradient/EarlyStop\ Audit
+
Controlled\ Hypotheses
+
No\ Tuning
+
No\ Test
}
\]

Pipeline:

```text
Phase 20/21 official runs
        ↓
history integrity
        ↓
learning curves
        ↓
gradient/early-stop diagnostics
        ↓
evidence-based findings
        ↓
UNTESTED hypotheses
        ↓
pre-registered controlled sweeps
```

---

# 161. Final status contract

```text
Phase 22 analyzes learning.

Phase 22 does not train.

Primary curve:
Validation RMSE Wh.

Secondary:
MAE
R²
model-space losses
gradient norms
clipping fraction
runtime.

Raw history:
source of truth.

Smoothing:
visualization only.

Diagnosis:
evidence-based,
confidence-tagged,
not causal proof.

Hypotheses:
mapped to pre-registered phases,
status UNTESTED.

No checkpoint reselection.
No hidden rerun.
No hyperparameter change.
No Test.

Only after LEARNING_DIAGNOSTICS-v1 PASS
may PHASE 23 — Feature-set Sweep begin.
```

---

# 162. Final check

```text
Integrity audit FIRST.
Plots SECOND.
Interpretation THIRD.
Hypotheses FOURTH.
Experiments LATER.
```

Đây là nguyên tắc quan trọng nhất của Phase 22.

Không được nhìn curve trước rồi thay model ngay.

Không được lấy một pattern từ một seed và biến nó thành causal conclusion.

Không được dùng Test để xác nhận development intuition.

Chỉ sau khi `LEARNING_DIAGNOSTICS-v1` được sign-off mới chuyển sang **PHASE 23 — Feature-set Sweep**.
