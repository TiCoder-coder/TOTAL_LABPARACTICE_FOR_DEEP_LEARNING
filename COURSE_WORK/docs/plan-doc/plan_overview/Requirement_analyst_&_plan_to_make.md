<div align="center">

# KẾ HOẠCH COURSEWORK — TRANSFORMER CHO HỒI QUY CHUỖI THỜI GIAN ĐA BIẾN

## Dự báo mức tiêu thụ năng lượng bằng bộ dữ liệu UCI Appliances Energy Prediction

**Coursework môn Deep Learning — Transformer Encoder cho bài toán hồi quy**

</div>

---

# 1. Đặc tả bài coursework

## 1.1. Đề bài được giao

```text
Multivariate Time-Series Regression

Goal:
Predict energy consumption from past multivariate readings.

Dataset:
UCI Appliances Energy Prediction

Model:
Transformer Encoder for regression

Task Type:
Regression

Extension:
Compare with LSTM baseline
Analyze attention maps
```

Diễn giải học thuật bằng tiếng Việt:

```text
Bài toán:
Hồi quy chuỗi thời gian đa biến

Mục tiêu:
Dự đoán mức tiêu thụ năng lượng từ các phép đo đa biến trong quá khứ.

Bộ dữ liệu:
UCI Appliances Energy Prediction

Mô hình chính:
Transformer Encoder cho hồi quy

Loại tác vụ:
Regression

Phần mở rộng bắt buộc:
So sánh với mô hình cơ sở LSTM
Phân tích các bản đồ attention
```

---

# 2. Đề bài thực sự yêu cầu những gì?

Coursework này bao gồm bốn yêu cầu kỹ thuật cốt lõi:

```text
1. Xây dựng bài toán hồi quy chuỗi thời gian đa biến.

2. Xây dựng mô hình Transformer Encoder cho hồi quy.

3. Xây dựng mô hình cơ sở LSTM và thực hiện so sánh công bằng với Transformer.

4. Trích xuất, trực quan hóa và phân tích các bản đồ attention của Transformer.
```

Do đó, chỉ tải bộ dữ liệu và huấn luyện một mô hình hồi quy truyền thẳng thông thường là chưa đủ.

Tương tự, nếu xem từng dòng dữ liệu là một mẫu độc lập:

```text
Các đặc trưng tại thời điểm t
        ↓
Transformer
        ↓
Appliances tại thời điểm t
```

thì cách làm này chưa khai thác đúng yêu cầu về chuỗi thời gian.

Cách mô hình hóa phù hợp hơn là:

```text
Các quan sát đa biến trong quá khứ
        ↓
Chuỗi thời gian
        ↓
Transformer Encoder
        ↓
Đầu hồi quy
        ↓
Mức tiêu thụ năng lượng trong tương lai
```

---

# 3. Hiểu đúng về bộ dữ liệu

Coursework sử dụng bộ dữ liệu:

```text
UCI Appliances Energy Prediction
```

Đây là bộ dữ liệu hồi quy chuỗi thời gian đa biến chứa các phép đo liên quan đến:

```text
Năng lượng thiết bị gia dụng
Nhiệt độ
Độ ẩm
Năng lượng chiếu sáng
Điều kiện ngoài trời
Dữ liệu thời tiết
Thông tin thời gian
```

Các đặc điểm chính:

```text
Số quan sát        : 19.735
Khoảng lấy mẫu     : 10 phút
Thời gian thu thập : khoảng 4,5 tháng
Loại tác vụ        : Hồi quy
Loại dữ liệu       : Chuỗi thời gian đa biến
Biến mục tiêu      : Appliances
```

Mỗi dòng dữ liệu biểu diễn trạng thái của cùng một hộ gia đình tại một thời điểm cụ thể.

Do đó, các quan sát có phụ thuộc theo thời gian và không nên được xem là các mẫu độc lập ngẫu nhiên nếu mục tiêu của bài toán là dự báo dựa trên dữ liệu quá khứ.

---

# 4. Biến mục tiêu

Biến mục tiêu là:

```text
Appliances
```

Biến này biểu diễn mức tiêu thụ năng lượng của các thiết bị gia dụng, với đơn vị:

```text
Wh
```

Mô hình phải dự đoán một giá trị liên tục:

\[
\hat{y}\in\mathbb{R}
\]

Ví dụ:

```text
Mức tiêu thụ năng lượng dự đoán = 126,4 Wh
```

Đây là bài toán hồi quy.

Không nên tự ý chuyển nó thành các lớp:

```text
THẤP
TRUNG BÌNH
CAO
```

trừ khi đó là một thí nghiệm phụ được bổ sung riêng và có giải thích rõ ràng.

---

# 5. Các nhóm biến trong bộ dữ liệu

Bộ dữ liệu thô chứa nhiều nhóm biến khác nhau.

| Nhóm | Biến | Mô tả |
|---|---|---|
| Thời gian | `date` | Dấu thời gian |
| Mục tiêu | `Appliances` | Mức tiêu thụ năng lượng của thiết bị |
| Chiếu sáng | `lights` | Mức tiêu thụ năng lượng chiếu sáng |
| Nhà bếp | `T1`, `RH_1` | Nhiệt độ và độ ẩm |
| Phòng khách | `T2`, `RH_2` | Nhiệt độ và độ ẩm |
| Phòng giặt | `T3`, `RH_3` | Nhiệt độ và độ ẩm |
| Phòng làm việc | `T4`, `RH_4` | Nhiệt độ và độ ẩm |
| Phòng tắm | `T5`, `RH_5` | Nhiệt độ và độ ẩm |
| Khu vực ngoài trời phía bắc | `T6`, `RH_6` | Nhiệt độ và độ ẩm |
| Phòng ủi | `T7`, `RH_7` | Nhiệt độ và độ ẩm |
| Phòng thiếu niên | `T8`, `RH_8` | Nhiệt độ và độ ẩm |
| Phòng cha mẹ | `T9`, `RH_9` | Nhiệt độ và độ ẩm |
| Thời tiết | `T_out` / `To` | Nhiệt độ ngoài trời |
| Thời tiết | `Press_mm_hg` / `Pressure` | Áp suất khí quyển |
| Thời tiết | `RH_out` | Độ ẩm tương đối ngoài trời |
| Thời tiết | `Windspeed` | Tốc độ gió |
| Thời tiết | `Visibility` | Tầm nhìn |
| Thời tiết | `Tdewpoint` | Nhiệt độ điểm sương |
| Biến đối chứng ngẫu nhiên | `rv1`, `rv2` | Các biến được sinh ngẫu nhiên |

Tên cột chính xác trong file CSV thực tế phải được kiểm tra trực tiếp sau khi tải dữ liệu, không nên giả định hoàn toàn từ tài liệu thứ cấp.

---

# 6. Xử lý `rv1` và `rv2`

Hai biến:

```text
rv1
rv2
```

được đưa vào như các biến đối chứng ngẫu nhiên.

Đối với thí nghiệm chính:

```text
LOẠI rv1
LOẠI rv2
```

Quyết định này phải được ghi rõ trong notebook và báo cáo.

Có thể bổ sung một thí nghiệm kiểm tra:

```text
Mô hình A:
Chỉ sử dụng các đặc trưng có ý nghĩa

Mô hình B:
Các đặc trưng có ý nghĩa + rv1 + rv2
```

Nếu mô hình có biểu hiện phụ thuộc mạnh vào các biến ngẫu nhiên, đây có thể là dấu hiệu của overfitting.

Thí nghiệm này là tùy chọn và không phải thành phần bắt buộc của coursework.

---

# 7. Ý nghĩa của “past multivariate readings”

Đề bài yêu cầu rõ:

> Dự đoán mức tiêu thụ năng lượng từ các phép đo đa biến trong quá khứ.

Do đó, đầu vào phải là một chuỗi các quan sát lịch sử:

\[
X_t=
[x_{t-L+1},x_{t-L+2},\ldots,x_t]
\]

trong đó:

\[
x_t\in\mathbb{R}^{F}
\]

là vector chứa nhiều phép đo tại thời điểm \(t\).

Đầu vào tổng quát của mô hình có dạng:

\[
X\in\mathbb{R}^{B\times L\times F}
\]

trong đó:

```text
B = kích thước batch
L = độ dài cửa sổ lịch sử
F = số lượng đặc trưng đầu vào
```

Biến đích là giá trị tiêu thụ năng lượng ở tương lai.

---

# 8. Định nghĩa bài toán được đề xuất

Đề bài không chỉ định trực tiếp:

```text
Độ dài lookback
Forecast horizon
```

Vì vậy, đây là các quyết định thiết kế thí nghiệm cần được trình bày rõ.

Cấu hình chính được đề xuất:

\[
L=144
\]

Do dữ liệu được lấy mẫu mỗi 10 phút:

\[
144\times10\text{ phút}=24\text{ giờ}
\]

Chân trời dự báo được đề xuất:

\[
H=1
\]

Một bước thời gian tương ứng:

\[
10\text{ phút}
\]

Do đó, bài toán chính trở thành:

\[
\boxed{
X_{t-143:t}
\rightarrow
Appliances_{t+1}
}
\]

Diễn giải:

> Sử dụng 24 giờ dữ liệu đa biến trong quá khứ để dự đoán mức tiêu thụ năng lượng của thiết bị gia dụng trong 10 phút tiếp theo.

---

# 9. Cấu hình dự báo chính

```text
LOẠI BÀI TOÁN
Hồi quy chuỗi thời gian đa biến

MỤC TIÊU
Mức tiêu thụ năng lượng Appliances ở tương lai

LOOKBACK
24 giờ

ĐỘ DÀI CHUỖI
144 thời điểm

CHÂN TRỜI DỰ BÁO
10 phút tiếp theo

HÌNH DẠNG ĐẦU VÀO
[B, 144, F]

HÌNH DẠNG ĐẦU RA
[B, 1]
```

---

# 10. Có nên sử dụng lịch sử của biến mục tiêu?

Các giá trị `Appliances` trong quá khứ là thông tin đã có tại thời điểm suy luận.

Do đó, có thể hợp lệ khi sử dụng:

\[
Appliances_{t-L+1:t}
\]

để dự đoán:

\[
Appliances_{t+1}
\]

Cấu hình đầu vào chính được đề xuất:

```text
Appliances trong quá khứ
+
Nhiệt độ trong quá khứ
+
Độ ẩm trong quá khứ
+
Các biến ngoài trời trong quá khứ
+
Các biến thời tiết trong quá khứ
+
Mức tiêu thụ chiếu sáng trong quá khứ
+
Các đặc trưng thời gian
        ↓
Transformer Encoder
        ↓
Appliances trong tương lai
```

Đây là một mô hình:

```text
Tự hồi quy + biến ngoại sinh
```

Có thể bổ sung ablation study:

```text
Thí nghiệm A:
Chỉ dùng các cảm biến lịch sử

Thí nghiệm B:
Cảm biến lịch sử + Appliances trong quá khứ
```

Đây là phân tích hữu ích nhưng không quan trọng bằng yêu cầu bắt buộc so sánh Transformer với LSTM.

---

# 11. Kỹ thuật tạo đặc trưng thời gian

Không nên đưa chuỗi ký tự `date` trực tiếp vào mô hình.

Nên trích xuất:

```text
hour
day_of_week
weekend
```

Đối với các biến tuần hoàn, nên sử dụng mã hóa sine/cosine.

Với giờ:

\[
hour_{\sin}
=
\sin
\left(
2\pi\frac{hour}{24}
\right)
\]

\[
hour_{\cos}
=
\cos
\left(
2\pi\frac{hour}{24}
\right)
\]

Với thứ trong tuần:

\[
dow_{\sin}
=
\sin
\left(
2\pi\frac{dow}{7}
\right)
\]

\[
dow_{\cos}
=
\cos
\left(
2\pi\frac{dow}{7}
\right)
\]

Cách mã hóa này giữ được bản chất tuần hoàn của thời gian.

---

# 12. Kiểm tra chất lượng dữ liệu

Trước khi xây dựng mô hình, cần kiểm tra:

```text
Kích thước dữ liệu
Tên cột
Kiểu dữ liệu
Giá trị thiếu
Dòng trùng lặp
Timestamp trùng lặp
Thứ tự thời gian
Khoảng cách giữa các timestamp
Phạm vi của biến mục tiêu
Phạm vi của các đặc trưng
```

Ngoài ra phải kiểm tra:

```text
date đã được chuyển sang datetime

timestamp được sắp xếp tăng dần

khoảng lấy mẫu xấp xỉ 10 phút
```

Không nên thay thế các kiểm tra thực tế bằng giả định.

---

# 13. Phân tích dữ liệu khám phá

EDA phải phục vụ trực tiếp cho bài toán dự báo.

Các phân tích được khuyến nghị:

```text
Phân phối của biến mục tiêu
Biến mục tiêu theo thời gian
Mức tiêu thụ trung bình theo giờ
Mẫu tiêu thụ theo ngày
Phân phối các đặc trưng
Tương quan giữa các đặc trưng
Xu hướng nhiệt độ và độ ẩm
Các đỉnh tiêu thụ năng lượng
```

Các hình nên có:

```text
Hình 1
Mức tiêu thụ năng lượng Appliances theo thời gian

Hình 2
Phân phối mức tiêu thụ năng lượng Appliances

Hình 3
Mức tiêu thụ năng lượng trung bình theo giờ trong ngày

Hình 4
Ma trận tương quan giữa các đặc trưng

Hình 5
Hồ sơ tiêu thụ năng lượng đại diện trong 24 giờ
```

---

# 14. Chia dữ liệu theo thứ tự thời gian

Vì đây là bài toán dự báo từ dữ liệu quá khứ, dữ liệu phải được chia theo thời gian.

Đề xuất:

```text
THỜI GIAN ───────────────────────────────────────────→

TRAIN
70%

                    VALIDATION
                    15%

                                 TEST
                                 15%
```

Cách chia này giúp bảo vệ giai đoạn tương lai khỏi rò rỉ vào tập huấn luyện.

---

# 15. Vì sao không được chia ngẫu nhiên?

Chia ngẫu nhiên có thể tạo ra tình huống:

```text
10:00 → train
10:10 → test
10:20 → train
```

Vấn đề còn nghiêm trọng hơn khi tạo sliding window.

Ví dụ:

```text
Cửa sổ train
[t1 ... t144]

Cửa sổ test
[t2 ... t145]
```

Hai cửa sổ này dùng chung:

```text
143 / 144 quan sát
```

Do đó, quy trình sau là sai:

```text
Tạo toàn bộ sliding windows
        ↓
Chia ngẫu nhiên các windows
```

---

# 16. Thứ tự tiền xử lý đúng

Quy trình được khuyến nghị:

```text
Dữ liệu thô
   ↓
Parse timestamp
   ↓
Sắp xếp theo thời gian
   ↓
Kiểm tra chất lượng dữ liệu
   ↓
Feature engineering
   ↓
Chia train/validation/test theo thời gian
   ↓
Fit preprocessing trên TRAIN
   ↓
Biến đổi validation và test
   ↓
Tạo sliding windows
   ↓
Huấn luyện mô hình
```

---

# 17. Chuẩn hóa đặc trưng

Các biến có đơn vị vật lý rất khác nhau.

Ví dụ:

```text
Nhiệt độ
Độ ẩm
Áp suất
Tốc độ gió
Tầm nhìn
Năng lượng
```

Nên chuẩn hóa bằng:

\[
x'
=
\frac{x-\mu_{train}}
{\sigma_{train}}
\]

Scaler chỉ được fit trên dữ liệu train.

\[
\boxed{
\mu,\sigma
\text{ chỉ được ước lượng từ TRAIN}
}
\]

Validation và test phải sử dụng chính scaler đã fit từ train.

---

# 18. Chuẩn hóa biến mục tiêu

Có thể chuẩn hóa biến mục tiêu:

\[
y'
=
\frac{y-\mu_y}
{\sigma_y}
\]

Mô hình dự đoán:

\[
\hat y'
\]

Sau đó inverse transform:

\[
\hat y
=
\hat y'\sigma_y+\mu_y
\]

Các metric cuối cùng phải được báo cáo trên đơn vị gốc:

```text
Wh
```

Ví dụ:

```text
MAE  = 21,4 Wh
RMSE = 42,8 Wh
```

---

# 19. Xây dựng sliding window

Với độ dài lookback \(L\) và horizon \(H\):

\[
X_i=
[x_i,x_{i+1},...,x_{i+L-1}]
\]

\[
y_i=
Appliances_{i+L+H-1}
\]

Ví dụ:

```text
t1 t2 t3 t4 t5 t6 → Appliances(t7)

t2 t3 t4 t5 t6 t7 → Appliances(t8)

t3 t4 t5 t6 t7 t8 → Appliances(t9)
```

Đối với cấu hình chính:

```text
L = 144
H = 1
```

---

# 20. Hình dạng tensor kỳ vọng

DataLoader nên tạo:

\[
X:[B,L,F]
\]

và:

\[
y:[B,1]
\]

Ví dụ:

```text
Batch size       = 64
Sequence length  = 144
Số đặc trưng     = F
```

Khi đó:

\[
X.shape=[64,144,F]
\]

\[
y.shape=[64,1]
\]

Các shape này phải được kiểm tra trước khi huấn luyện.

---

# 21. Kiến trúc Transformer Encoder chính

Đề bài yêu cầu rõ:

```text
Transformer Encoder for regression
```

Kiến trúc baseline được đề xuất:

```text
Đầu vào đa biến
[B, L, F]
        ↓
Chiếu đầu vào
Linear(F → d_model)
        ↓
Positional Encoding
        ↓
Transformer Encoder Layer
        ↓
Transformer Encoder Layer
        ↓
Biểu diễn ngữ cảnh theo thời gian
        ↓
Tổng hợp thông tin thời gian
        ↓
Regression Head
        ↓
Giá trị năng lượng liên tục được dự đoán
```

---

# 22. Phép chiếu đầu vào

Tại mỗi thời điểm:

\[
x_t\in\mathbb{R}^{F}
\]

Transformer cần biểu diễn ẩn:

\[
z_t\in\mathbb{R}^{d_{model}}
\]

Sử dụng:

\[
\boxed{
z_t=W_ex_t+b_e
}
\]

Ví dụ:

```text
F đặc trưng đầu vào
        ↓
Linear(F, 64)
        ↓
Embedding thời gian 64 chiều
```

---

# 23. Positional Encoding

Self-attention không tự biểu diễn thứ tự thời gian.

Do đó:

\[
h_t=z_t+p_t
\]

trong đó:

```text
z_t = biểu diễn đã chiếu của quan sát đa biến
p_t = mã hóa vị trí
```

Sinusoidal positional encoding là lựa chọn phù hợp cho baseline coursework.

---

# 24. Hyperparameter baseline cho Transformer Encoder

Các giá trị khởi đầu được đề xuất:

```text
d_model          = 64
n_heads          = 4
n_layers         = 2
dim_feedforward  = 128 hoặc 256
dropout          = 0.1
activation       = GELU hoặc ReLU
```

Đây chỉ là các giá trị khởi đầu.

Không được mô tả chúng là tối ưu nếu chưa được đánh giá trên validation set.

---

# 25. Có cần causal mask không?

Đối với bài toán sequence-to-one:

\[
X_{t-143:t}
\rightarrow
Appliances_{t+1}
\]

toàn bộ các thời điểm trong đầu vào đều thuộc quá khứ.

Do đó causal mask không bắt buộc.

Transformer Encoder có thể attention trên toàn bộ cửa sổ lịch sử:

```text
Quá khứ t-143
Quá khứ t-142
...
Quá khứ t
        ↓
Dự đoán t+1
```

Không có quan sát tương lai nào được đưa vào input.

---

# 26. Tổng hợp biểu diễn theo thời gian

Transformer trả về:

\[
H\in\mathbb{R}^{B\times L\times d_{model}}
\]

Cần tạo một biểu diễn tóm tắt để thực hiện hồi quy.

Baseline được đề xuất:

\[
h_{summary}
=
H[:,-1,:]
\]

Sau đó:

\[
\hat y
=
W_rh_{summary}+b_r
\]

Phương án khác:

\[
h_{summary}
=
\frac{1}{L}
\sum_{t=1}^{L}H_t
\]

tức mean pooling.

Trong triển khai chính nên chọn một phương pháp và mô tả rõ.

Khuyến nghị ban đầu:

```text
Sử dụng representation tại bước thời gian cuối cùng
```

---

# 27. Regression Head

Đầu ra cuối nên là:

\[
Linear(d_{model},1)
\]

Luồng xử lý:

```text
Biểu diễn từ Transformer
        ↓
Linear Layer
        ↓
Một giá trị liên tục
```

Không sử dụng Softmax.

---

# 28. Hàm mất mát hồi quy

Hàm mất mát baseline:

\[
\boxed{
MSE
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
\]

Sử dụng:

```text
MSELoss
```

Có thể bổ sung:

```text
HuberLoss
```

vì chuỗi năng lượng có thể chứa các đỉnh tiêu thụ lớn.

Tuy nhiên, so sánh loss là phần tùy chọn và có độ ưu tiên thấp hơn:

```text
Transformer vs LSTM
Attention maps
```

---

# 29. Optimizer

Optimizer khởi đầu được đề xuất:

```text
AdamW
```

Có thể bắt đầu tìm kiếm xung quanh:

```text
learning rate ≈ 1e-3
weight decay  ≈ 1e-4
```

Đây chỉ là điểm khởi đầu để tuning, không phải các giá trị cố định tối ưu.

---

# 30. Chiến lược huấn luyện

Khuyến nghị:

```text
Số epoch tối đa       = 50–100
Early stopping        = Có
Theo dõi validation   = Có
Lưu checkpoint tốt nhất = Có
```

Metric được khuyến nghị để chọn checkpoint:

```text
Validation RMSE
```

Tập test cuối cùng không được dùng cho lựa chọn mô hình.

---

# 31. LSTM baseline bắt buộc

Đề bài yêu cầu rõ:

```text
Compare with LSTM baseline
```

Do đó LSTM là thành phần bắt buộc.

Kiến trúc đề xuất:

```text
Input
[B, L, F]
    ↓
LSTM
    ↓
Hidden state cuối
    ↓
Linear Regression Head
    ↓
Một giá trị dự đoán liên tục
```

Thông số khởi đầu:

```text
hidden_size = 64
num_layers  = 2
dropout     = 0.1
```

---

# 32. So sánh Transformer và LSTM một cách công bằng

Cả hai mô hình phải dùng cùng:

```text
Bộ dữ liệu
Biến mục tiêu
Train/validation/test split
Lookback
Forecast horizon
Tập đặc trưng
Scaling
Mẫu huấn luyện
Metric đánh giá
```

Tốt nhất cũng nên đồng nhất:

```text
Loss function
Batch size
Số epoch tối đa
Early stopping protocol
```

Learning rate có thể được tuning riêng nếu cả hai mô hình có ngân sách tuning tương đương.

---

# 33. Bảng so sánh được đề xuất

| Mô hình | MAE | RMSE | \(R^2\) | Số tham số | Thời gian huấn luyện |
|---|---:|---:|---:|---:|---:|
| LSTM | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm |
| Transformer Encoder | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm |

Chỉ điền kết quả sau khi đã chạy thực nghiệm.

Không được tạo ra các số liệu giả.

---

# 34. Các metric hồi quy

## Mean Absolute Error

\[
MAE
=
\frac1N
\sum_{i=1}^{N}
|y_i-\hat y_i|
\]

Diễn giải:

```text
Sai số tuyệt đối trung bình tính theo Wh
```

---

## Root Mean Squared Error

\[
RMSE
=
\sqrt{
\frac1N
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
\]

RMSE phạt các sai số lớn mạnh hơn MAE.

---

## Hệ số xác định

\[
R^2
=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}
\]

Bộ metric cuối được khuyến nghị:

```text
MAE
RMSE
R²
```

---

# 35. Persistence baseline tùy chọn

Một baseline đơn giản nhưng rất hữu ích:

\[
\boxed{
\hat y_{t+1}=y_t
}
\]

Ý nghĩa:

> Giả định mức tiêu thụ năng lượng sau 10 phút bằng đúng mức tiêu thụ hiện tại.

Baseline này giúp xác minh rằng mô hình phức tạp thực sự tạo ra giá trị vượt trội so với một dự báo ngây thơ.

Nếu thời gian coursework hạn chế, ưu tiên vẫn là:

```text
LSTM
Transformer
```

---

# 36. Yêu cầu về attention map

Đề bài yêu cầu:

```text
Analyze attention maps
```

Vì vậy, Transformer phải cho phép truy xuất trọng số self-attention.

Mô hình không nên chỉ trả về prediction.

Cần có khả năng lấy:

```text
attention weights
```

từ một hoặc nhiều encoder layer.

---

# 37. Hình dạng của attention weights

Với multi-head self-attention, attention theo từng head có thể biểu diễn dưới dạng:

\[
[B,H,L,L]
\]

trong đó:

```text
B = batch size
H = số attention heads
L = sequence length
```

Nếu:

```text
H = 4
L = 144
```

thì attention map có dạng:

\[
[B,4,144,144]
\]

---

# 38. Encoder Layer có khả năng lưu attention

Nên xây dựng một custom layer, ví dụ:

```text
EncoderLayerWithAttention
```

Layer này gồm:

```text
Multi-Head Self-Attention
Residual connection
Layer normalization
Feed-Forward Network
Residual connection
Layer normalization
```

Self-attention phải giữ lại attention weights thay vì loại bỏ chúng.

Điều này phục vụ trực tiếp cho phần phân tích attention bắt buộc trong coursework.

---

# 39. Trực quan hóa attention 1 — Heatmap đầy đủ

Với một sequence test:

```text
Trục X:
Các vị trí lịch sử đóng vai trò Key

Trục Y:
Các vị trí lịch sử đóng vai trò Query

Màu:
Attention weight
```

Có thể hiển thị một heatmap cho mỗi attention head được chọn.

Sơ đồ khái niệm:

```text
                 Thời gian lịch sử
             t-143 ............ t
        ┌─────────────────────────┐
 t-143  │                         │
   ...  │       Attention         │
Query   │        Weights          │
   ...  │                         │
     t  │                         │
        └─────────────────────────┘
```

---

# 40. Trực quan hóa attention 2 — Last-query attention

Đối với sequence-to-one forecasting, có thể phân tích:

\[
A[:,h,-1,:]
\]

Câu hỏi:

> Khi tạo representation ngữ cảnh tại thời điểm lịch sử cuối cùng, mô hình tập trung nhiều hơn vào những vị trí lịch sử nào?

Đồ thị được đề xuất:

```text
Trục X:
Số giờ trước thời điểm dự báo

Trục Y:
Attention weight
```

Các mốc:

```text
-24 h
-18 h
-12 h
-6 h
-1 h
-10 phút
```

---

# 41. Trực quan hóa attention 3 — So sánh giữa các head

Nếu dùng bốn attention head:

```text
Head 1
Head 2
Head 3
Head 4
```

Có thể phân tích xem mỗi head có biểu hiện temporal pattern khác nhau hay không.

Các pattern cần quan sát:

```text
Tập trung vào lịch sử gần
Tập trung vào lịch sử xa
Mẫu tuần hoàn
Attention phân tán rộng
```

Không được gán ý nghĩa trước cho từng head.

Mọi diễn giải phải dựa trên kết quả thực nghiệm.

---

# 42. Trực quan hóa attention 4 — Attention trung bình

Trung bình trên các head:

\[
A_{avg}
=
\frac1H
\sum_{h=1}^{H}A_h
\]

Cách này tạo một biểu diễn tổng quan.

Tuy nhiên, vẫn nên giữ ít nhất một trực quan hóa theo từng head vì trung bình có thể che giấu sự khác biệt giữa các head.

---

# 43. Diễn giải attention đúng chuẩn

Nên viết:

> Bản đồ attention cho thấy các vị trí lịch sử nhận được mức trọng số attention cao hơn trong quá trình tính toán của Transformer đối với sequence đang xét.

Không nên viết:

> Timestamp này gây ra prediction.

Attention weights không chứng minh quan hệ nhân quả.

---

# 44. Các câu hỏi phù hợp khi phân tích attention

Có thể khảo sát:

```text
Mô hình có tập trung chủ yếu vào các timestamp gần nhất hay không?

Mô hình có attention tới các vị trí lịch sử xa hay không?

Các attention head có học những temporal pattern khác nhau hay không?

Attention có thay đổi trong những giai đoạn tiêu thụ năng lượng cao hay không?

Attention pattern có ổn định trên nhiều mẫu test hay không?
```

---

# 45. Attention map không thể chứng minh

Attention map một mình không thể chứng minh:

```text
Quan hệ nhân quả
Nguyên nhân vật lý của tiêu thụ năng lượng
Feature importance tuyệt đối
Hành vi thực tế của con người
Khả năng giải thích hoàn chỉnh của mô hình
```

Nên mô tả attention analysis là:

```text
Phân tích hành vi nội tại của mô hình
```

---

# 46. Trực quan hóa Actual vs Predicted

Nên vẽ cùng một đoạn thời gian trên tập test.

```text
Trục X:
Thời gian

Trục Y:
Mức tiêu thụ Appliances (Wh)

Các đường:
Ground Truth
Transformer Prediction
LSTM Prediction
```

Cả hai mô hình phải được so sánh trên cùng đoạn thời gian.

---

# 47. Scatter Actual vs Predicted

Vẽ:

```text
Trục X = Giá trị thực
Trục Y = Giá trị dự đoán
```

Prediction lý tưởng nằm gần:

\[
y=x
\]

Đồ thị này giúp phát hiện:

```text
Xu hướng dự đoán thấp hơn thực tế
Xu hướng dự đoán cao hơn thực tế
Khó khăn khi dự báo các đỉnh năng lượng
```

---

# 48. Phân tích residual

Định nghĩa residual:

\[
e_i=y_i-\hat y_i
\]

Các phân tích nên có:

```text
Phân phối residual
Residual theo giá trị dự đoán
Residual theo thời gian
Các mẫu có sai số tuyệt đối lớn nhất
```

Một câu hỏi quan trọng:

> Mô hình có thường xuyên dự đoán thấp hơn thực tế tại các đỉnh tiêu thụ năng lượng cao hay không?

---

# 49. Phân tích lỗi theo mức tiêu thụ

Chia giá trị thực thành các nhóm dựa trên phân phối dữ liệu thực:

```text
Mức tiêu thụ thấp
Mức tiêu thụ trung bình
Mức tiêu thụ cao
```

Sau đó so sánh MAE hoặc RMSE theo từng nhóm.

Phân tích này có thể phát hiện các vấn đề mà metric tổng thể không thể hiện rõ.

---

# 50. Tập thí nghiệm tối thiểu

Bắt buộc:

```text
Thí nghiệm 1
LSTM baseline

Thí nghiệm 2
Transformer Encoder

Thí nghiệm 3
Phân tích attention map của Transformer

Thí nghiệm 4
LoRA adaptation từ cùng một pretrained Transformer checkpoint

Thí nghiệm 5
Partial fine-tuning từ cùng một pretrained Transformer checkpoint
```

Khuyến nghị:

```text
Thí nghiệm 0
Persistence baseline
```

Tùy chọn:

```text
Head-only fine-tuning
So sánh lookback
MSE vs Huber
Ablation về historical target
```

Các thành phần bắt buộc phải được hoàn thành trước các thí nghiệm tùy chọn.

---

# 51. Các kiến trúc ngoài phạm vi chính

Coursework hiện không yêu cầu triển khai:

```text
Informer
Autoformer
FEDformer
PatchTST
iTransformer
TimesFM
Chronos
Temporal Fusion Transformer
Probabilistic forecasting
Multi-step forecasting
Foundation models
Hệ thống triển khai production
```

Các hướng này có thể được đề cập ở phần related work nhưng không nên thay thế Transformer Encoder được yêu cầu trong đề.

---

# 52. Phạm vi bắt buộc của coursework

Một bài hoàn chỉnh nên bao gồm:

```text
Hiểu bộ dữ liệu
Kiểm tra chất lượng dữ liệu
EDA
Định nghĩa bài toán chuỗi thời gian
Feature engineering
Chia dữ liệu theo thời gian
Scaling chống leakage
Tạo sliding windows
Transformer Encoder cho regression
LSTM baseline
Training protocol
Regression metrics
So sánh công bằng
Actual-vs-predicted analysis
Residual analysis
Attention extraction
Attention visualization
Attention interpretation
Limitations
Conclusion
```

---

# 53. Kiến trúc notebook được đề xuất

```text
PHASE 0
Tổng quan coursework

PHASE 1
Môi trường và khả năng tái lập

PHASE 2
Tải bộ dữ liệu

PHASE 3
Kiểm tra chất lượng dữ liệu

PHASE 4
Phân tích dữ liệu khám phá

PHASE 5
Feature engineering

PHASE 6
Chia dữ liệu theo thời gian

PHASE 7
Scaling chỉ dựa trên train

PHASE 8
Xây dựng sliding windows

PHASE 9
Xây dựng DataLoader

PHASE 10
Persistence baseline

PHASE 11
LSTM baseline

PHASE 12
Xây dựng Transformer Encoder

PHASE 13
Xây dựng Encoder có khả năng lưu attention

PHASE 14
Cấu hình huấn luyện

PHASE 15
Huấn luyện LSTM

PHASE 16
Huấn luyện Transformer

PHASE 17
Đánh giá mô hình

PHASE 18
So sánh mô hình

PHASE 19
Phân tích Actual vs Predicted

PHASE 20
Phân tích residual

PHASE 21
Trích xuất attention map

PHASE 22
Trực quan hóa attention map

PHASE 23
Phân tích attention

PHASE 24
Phân tích lỗi

PHASE 25
Kết luận cuối cùng
```

---

# 54. Pipeline coursework từ đầu đến cuối

```text
UCI Appliances Energy Prediction
                ↓
Tải energydata_complete.csv
                ↓
Parse date
                ↓
Sắp xếp theo thời gian
                ↓
Kiểm tra chất lượng dữ liệu
                ↓
EDA
                ↓
Loại rv1 và rv2
                ↓
Feature engineering cho thời gian
                ↓
Định nghĩa target:
Appliances(t+1)
                ↓
Định nghĩa đầu vào lịch sử
                ↓
Chia thời gian 70/15/15
                ↓
Fit scaler trên TRAIN
                ↓
Transform Train / Validation / Test
                ↓
Sliding windows
L = 144
H = 1
                ↓
DataLoader
                ↓
        ┌───────┴────────┐
        ↓                ↓
      LSTM           Transformer
        │             Encoder
        │                │
        └───────┬────────┘
                ↓
        Chọn mô hình bằng Validation
                ↓
        Best Checkpoint
                ↓
        Test Set chưa bị tác động
                ↓
        MAE / RMSE / R²
                ↓
        So sánh mô hình
                ↓
        Actual vs Predicted
                ↓
        Residual Analysis
                ↓
        Transformer Attention
                ↓
        Attention Maps
                ↓
        Phân tích
                ↓
        Kết luận
```

---

# 55. Cấu hình Transformer khởi đầu được đề xuất

| Hyperparameter | Giá trị khởi đầu |
|---|---:|
| Lookback | 144 |
| Horizon | 1 |
| `d_model` | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| FFN dimension | 128 hoặc 256 |
| Dropout | 0.1 |
| Batch size | 32 hoặc 64 |
| Loss | MSE |
| Optimizer | AdamW |
| Epoch tối đa | 50–100 |
| Early stopping | Có |

Các giá trị này tạo thành một baseline hợp lý và không được xem là cấu hình tối ưu trước khi thực nghiệm.

---

# 56. Cấu hình LSTM khởi đầu được đề xuất

| Hyperparameter | Giá trị khởi đầu |
|---|---:|
| Input size | Cùng số đặc trưng với Transformer |
| Hidden size | 64 |
| Số LSTM layers | 2 |
| Dropout | 0.1 |
| Output size | 1 |
| Loss | MSE |
| Batch size | Giống Transformer |
| Early stopping | Cùng protocol |

---

# 57. Khả năng tái lập

Sử dụng một seed cố định:

```text
SEED = 42
```

Áp dụng cho:

```text
Python random
NumPy
PyTorch
DataLoader generator khi cần
```

Cần ghi nhận:

```text
Phiên bản Python
Phiên bản PyTorch
Thiết bị tính toán
Seed
Nguồn bộ dữ liệu
Kích thước dữ liệu
Lookback
Forecast horizon
Hyperparameter cuối cùng
```

---

# 58. Các sanity check bắt buộc

Trước khi huấn luyện:

```text
Bộ dữ liệu chứa đầy đủ các cột kỳ vọng.

Timestamp được parse thành công.

Timestamp tăng đơn điệu sau khi sắp xếp.

Giá trị thiếu đã được xử lý hoặc xác nhận không tồn tại.

Timestamp trùng lặp đã được kiểm tra.

Quyết định về rv1 và rv2 được ghi rõ.

Thời gian train đứng trước validation.

Thời gian validation đứng trước test.

Scaler chỉ được fit trên train.

X có shape [N, 144, F].

y có shape [N, 1].

Không có thông tin target tương lai trong input.

Transformer output có shape [B, 1].

LSTM output có shape [B, 1].

Loss là hữu hạn.

Gradient là hữu hạn.

Prediction có thể inverse-transform.

Metric được tính trên đơn vị Wh gốc.
```

---

# 59. Cấu trúc phần kết quả được đề xuất

## 59.1. Bộ dữ liệu và thiết lập thí nghiệm

Báo cáo:

```text
Kích thước bộ dữ liệu
Số đặc trưng
Kích thước train/validation/test
Lookback
Forecast horizon
Feature selection
Phương pháp scaling
Phần cứng
Seed
```

---

## 59.2. Hành vi huấn luyện

Trình bày:

```text
Training loss
Validation loss
```

cho:

```text
LSTM
Transformer
```

---

## 59.3. Hiệu năng cuối trên test set

| Mô hình | MAE | RMSE | \(R^2\) |
|---|---:|---:|---:|
| LSTM | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm |
| Transformer | Kết quả thực nghiệm | Kết quả thực nghiệm | Kết quả thực nghiệm |

Có thể bổ sung:

```text
Persistence baseline
```

---

## 59.4. Trực quan hóa dự báo

Hiển thị:

```text
Ground Truth
Transformer Prediction
LSTM Prediction
```

trên cùng đoạn thời gian test.

---

## 59.5. Phân tích lỗi

Phân tích:

```text
Phân phối residual
Các timestamp có sai số lớn nhất
Sai số trong giai đoạn tiêu thụ cao
Systematic bias nếu có
```

---

## 59.6. Phân tích attention

Bao gồm:

```text
Heatmap theo từng attention head
Attention map trung bình
Last-query temporal attention profile
```

Sau đó phân tích temporal behavior được quan sát một cách thận trọng.

---

# 60. Các câu hỏi cốt lõi mà coursework phải trả lời

Sau khi hoàn thành, bài phải có khả năng trả lời:

```text
1. Transformer Encoder có học được biểu diễn thời gian hữu ích
   cho dự báo năng lượng thiết bị hay không?

2. Transformer có vượt LSTM baseline hay không?

3. Mô hình nào đạt MAE và RMSE thấp hơn?

4. Mô hình nào đạt R² tốt hơn?

5. Các mô hình xử lý những đỉnh tiêu thụ năng lượng cao như thế nào?

6. Attention maps của Transformer thể hiện những temporal pattern nào?

7. Các attention head khác nhau có tập trung vào các vị trí lịch sử khác nhau hay không?

8. Độ phức tạp bổ sung của Transformer có mang lại cải thiện định lượng rõ ràng hay không?

9. Những hạn chế nào cản trở việc khái quát hóa kết quả?
```

---

# 61. Hạn chế của bộ dữ liệu

Bộ dữ liệu chỉ đại diện cho dữ liệu được thu thập từ một hộ gia đình cụ thể.

Do đó không được kết luận:

```text
Mô hình dự đoán chính xác mức tiêu thụ năng lượng cho mọi hộ gia đình.
```

Một phát biểu thận trọng hơn:

> Các mô hình được đánh giá trên chuỗi dữ liệu thời gian của hộ gia đình được biểu diễn trong bộ dữ liệu UCI Appliances Energy Prediction.

Khả năng khái quát hóa sang:

```text
Các căn nhà khác
Người sử dụng khác
Khí hậu khác
Cấu hình thiết bị khác
Khu vực địa lý khác
```

vẫn chưa được kiểm chứng.

---

# 62. Hạn chế của attention

Phân tích attention là yêu cầu bắt buộc nhưng phải được diễn giải thận trọng.

Cách viết được khuyến nghị:

> Các bản đồ attention cung cấp một góc nhìn về cách Transformer phân bổ trọng số chú ý lên các vị trí thời gian trong quá khứ, nhưng không nên được xem là bằng chứng nhân quả tuyệt đối cho quyết định dự đoán của mô hình.

---

# 63. Câu hỏi nghiên cứu được đề xuất

Câu hỏi nghiên cứu chính:

> **Transformer Encoder có thể học hiệu quả các phụ thuộc thời gian đa biến phục vụ dự báo ngắn hạn mức tiêu thụ năng lượng thiết bị gia dụng hay không, và hiệu năng dự báo của nó so với LSTM baseline như thế nào?**

Câu hỏi nghiên cứu phụ:

> **Những mẫu phụ thuộc thời gian nào có thể được quan sát từ các bản đồ self-attention của Transformer?**

Hai câu hỏi này bám sát trực tiếp yêu cầu của coursework.

---

# 64. Đặc tả coursework cuối cùng được khuyến nghị

```text
Bài toán:
Dự báo ngắn hạn năng lượng chuỗi thời gian đa biến

Bộ dữ liệu:
UCI Appliances Energy Prediction

Biến mục tiêu:
Appliances

Tần suất lấy mẫu:
10 phút

Lookback:
24 giờ

Độ dài chuỗi:
144 timestamp

Forecast horizon:
10 phút tiếp theo

Loại tác vụ:
Sequence-to-one regression

Đầu vào:
Các phép đo đa biến trong quá khứ
+ Appliances trong quá khứ
+ đặc trưng thời gian

Biến đối chứng ngẫu nhiên:
Loại rv1 và rv2 khỏi thí nghiệm chính

Chia dữ liệu:
Theo thời gian 70% / 15% / 15%

Scaling:
Fit chỉ trên train

Mô hình chính:
Transformer Encoder

Baseline:
LSTM

Baseline tùy chọn:
Persistence

Training loss:
MSE

Optimizer:
AdamW

Metric chính:
MAE
RMSE
R²

Phần mở rộng về khả năng diễn giải:
Per-head attention maps
Average attention map
Last-query temporal attention profile

Phân tích bắt buộc:
Hành vi huấn luyện
So sánh công bằng giữa hai mô hình
Actual vs Predicted
Residual analysis
Attention analysis
Hạn chế của dữ liệu và mô hình
```

---

# 65. Tiêu chí hoàn thành

Coursework chỉ được xem là hoàn chỉnh khi toàn bộ pipeline sau chạy end-to-end:

```text
Dữ liệu UCI thô
      ↓
Chuỗi thời gian đã được kiểm tra
      ↓
Tiền xử lý không gây leakage
      ↓
Sliding windows
      ↓
Huấn luyện LSTM
      ↓
Huấn luyện Transformer
      ↓
Best checkpoint theo Validation
      ↓
Prediction trên Test chưa bị sử dụng
      ↓
MAE / RMSE / R²
      ↓
So sánh LSTM–Transformer công bằng
      ↓
Actual vs Predicted
      ↓
Residual Analysis
      ↓
Attention Maps
      ↓
Phân tích Attention
      ↓
Kết luận có cơ sở thực nghiệm
```

Có thể cô đọng toàn bộ coursework thành:

\[
\boxed{
\text{Định nghĩa đúng bài toán chuỗi thời gian}
+
\text{Transformer Regression}
+
\text{So sánh LSTM công bằng}
+
\text{Phân tích Attention}
}
\]

Bài không nên được hiểu đơn giản là:

```text
Triển khai Transformer rồi báo cáo một metric duy nhất.
```

---

# 66. Bổ sung yêu cầu của giảng viên: LoRA và fine-tuning

Giảng viên định hướng sử dụng và so sánh:

```text
LoRA adaptation
so với
Partial fine-tuning
```

Không sử dụng `full fine-tuning` làm phương án chính.

Yêu cầu này bổ sung một câu hỏi nghiên cứu về hiệu quả tham số:

> LoRA có thể đạt hiệu năng dự báo tương đương hoặc tốt hơn partial fine-tuning trong khi chỉ cập nhật một tỷ lệ nhỏ tham số của Transformer hay không?

LoRA và fine-tuning chỉ có ý nghĩa khi bắt đầu từ một Transformer đã được pretrain. Nếu Transformer được khởi tạo ngẫu nhiên rồi đóng băng để chỉ huấn luyện LoRA adapters, backbone chưa chứa biểu diễn hữu ích và phép so sánh sẽ không hợp lệ.

Do đó, cần phân biệt rõ ba khái niệm:

| Khái niệm | Trạng thái ban đầu | Thành phần được cập nhật |
|---|---|---|
| Training from scratch | Trọng số ngẫu nhiên | Toàn bộ mô hình |
| LoRA adaptation | Pretrained checkpoint | LoRA adapters và regression head |
| Partial fine-tuning | Cùng pretrained checkpoint | Một phần encoder và regression head |

`Transformer trained from scratch` vẫn được giữ lại để đáp ứng và kiểm chứng mô hình chính của đề gốc. Nó không được gọi là fine-tuning và không thay thế thí nghiệm LoRA–partial fine-tuning.

---

# 67. Chiến lược pretraining cho Transformer nền

Phương án chính được chọn cho coursework:

```text
Self-supervised pretraining trên train split của UCI
```

Không sử dụng validation hoặc test target để cập nhật trọng số trong pretraining.

Pretraining pipeline:

```text
Raw training split
        ↓
Preprocessing fit trên train
        ↓
Tạo các chuỗi lịch sử
        ↓
Che ngẫu nhiên một phần timestep hoặc feature
        ↓
Transformer Encoder tái tạo giá trị đã bị che
        ↓
Chọn checkpoint bằng pretraining validation loss
        ↓
Lưu một pretrained base checkpoint duy nhất
```

Pretraining objective chính được đề xuất là masked-value reconstruction:

\[
\mathcal{L}_{pretrain}
=
\frac{1}{|\mathcal{M}|}
\sum_{(t,f)\in\mathcal{M}}
\left(x_{t,f}-\hat{x}_{t,f}\right)^2
\]

trong đó \(\mathcal{M}\) là tập các vị trí bị che.

Các yêu cầu bắt buộc:

```text
Chỉ tính reconstruction loss trên các vị trí bị che
Không pretrain trên test split
Không tạo mask làm thay đổi target forecasting
Lưu chính xác model config và scaler đi kèm checkpoint
Mọi adaptation strategy phải bắt đầu từ cùng checkpoint
```

Self-supervised pretraining nội bộ được chọn vì:

- Giữ đúng kiến trúc Transformer Encoder của coursework.
- Không phụ thuộc vào time-series foundation model bên ngoài.
- Dễ kiểm soát input, attention weights và regression head.
- Cho phép thực hiện LoRA và partial fine-tuning trên cùng backbone.

Hạn chế phải báo cáo:

- Dataset chỉ có 19.735 quan sát nên quy mô pretraining còn nhỏ.
- Pretraining và downstream forecasting dùng cùng nguồn dữ liệu, dù các split vẫn được tách theo thời gian.
- Kết quả không chứng minh khả năng transfer sang hộ gia đình hoặc domain khác.

Nếu giảng viên chỉ định một pretrained time-series model cụ thể sau này, lựa chọn đó phải được đánh giá lại và cập nhật plan trước khi implementation.

---

# 68. Thiết kế LoRA adaptation

Với một linear projection có trọng số pretrained \(W_0\), LoRA giữ \(W_0\) frozen và học một cập nhật low-rank:

\[
W=W_0+\Delta W
\]

\[
\Delta W=BA
\]

trong đó:

\[
A\in\mathbb{R}^{r\times d_{in}},
\qquad
B\in\mathbb{R}^{d_{out}\times r},
\qquad
r\ll\min(d_{in},d_{out})
\]

Vị trí LoRA chính được đề xuất:

```text
Query projection
Value projection
```

Cấu hình khởi đầu:

| Hyperparameter | Giá trị khởi đầu |
|---|---:|
| LoRA rank `r` | 4 hoặc 8 |
| LoRA alpha | 8 hoặc 16 |
| LoRA dropout | 0.05 hoặc 0.1 |
| Frozen modules | Input projection và Transformer backbone gốc |
| Trainable modules | LoRA adapters và regression head |

Các giá trị trên là search space khởi đầu, không được mô tả là tối ưu trước khi validation.

Custom encoder layer nên để các projection attention có cấu trúc rõ ràng. Điều này phục vụ đồng thời:

```text
Gắn LoRA đúng target module
Truy xuất per-head attention weights
Đếm trainable parameters chính xác
```

---

# 69. Thiết kế partial fine-tuning

Phương án so sánh với LoRA được định nghĩa là:

```text
Last-block fine-tuning
```

Với Transformer có \(N\) encoder layers:

```text
Frozen:
- Input projection
- Các encoder layer từ 1 đến N-1

Trainable:
- Encoder layer N
- Final normalization nếu có
- Regression head
```

Đây là `partial fine-tuning`, không phải `full fine-tuning`.

LoRA và partial fine-tuning phải sử dụng:

- Cùng pretrained base checkpoint.
- Cùng downstream train/validation/test windows.
- Cùng feature set, scaler, lookback và horizon.
- Cùng loss, batch size và early-stopping protocol.
- Ngân sách tuning tương đương.
- Cùng tiêu chí lựa chọn checkpoint.

Learning rate có thể được tuning riêng vì số lượng và loại tham số trainable khác nhau, nhưng search budget phải được ghi nhận.

---

# 70. Experiment matrix sau khi bổ sung LoRA

## 70.1. Thí nghiệm bắt buộc

| ID | Mô hình / chiến lược | Khởi tạo | Vai trò |
|---|---|---|---|
| E0 | Persistence | Không có tham số | Sanity baseline |
| E1 | LSTM | Random initialization | Baseline theo đề |
| E2 | Transformer Encoder | Random initialization | Mô hình chính theo đề gốc |
| E3 | Transformer + LoRA | Pretrained base checkpoint | Parameter-efficient adaptation |
| E4 | Transformer + partial fine-tuning | Cùng pretrained checkpoint | Đối chứng với LoRA |
| E5 | Attention analysis | Best Transformer variants | Phân tích temporal attention |

## 70.2. Thí nghiệm khuyến nghị

| ID | Thí nghiệm | Mục đích |
|---|---|---|
| E6 | Head-only fine-tuning | Kiểm tra giá trị của adaptation so với chỉ thay regression head |
| E7 | LoRA rank 4 so với 8 | Kiểm tra trade-off hiệu năng–tham số |
| E8 | Attention trước và sau adaptation | Khảo sát thay đổi temporal behavior |

## 70.3. Thí nghiệm tùy chọn

```text
So sánh lookback
MSE so với Huber loss
Ablation historical Appliances
Ablation rv1 và rv2
Last-token so với mean pooling
```

Các thí nghiệm tùy chọn chỉ được thực hiện sau khi E0–E5 đã hoàn thành.

---

# 71. Metric đánh giá hiệu quả dự báo và hiệu quả tham số

Metric dự báo:

```text
Primary metric   : RMSE trên đơn vị Wh gốc
Secondary metric : MAE và R²
```

RMSE được chọn làm primary metric vì nó phạt mạnh các sai số lớn và phản ánh rõ khó khăn khi dự báo các đỉnh tiêu thụ.

Metric efficiency bắt buộc cho LoRA và partial fine-tuning:

```text
Tổng số tham số
Số trainable parameters
Tỷ lệ trainable parameters
Thời gian huấn luyện
Best validation epoch
Kích thước checkpoint
Peak GPU memory nếu môi trường cho phép đo tin cậy
```

Tỷ lệ tham số được cập nhật:

\[
Trainable\ Percentage
=
\frac{N_{trainable}}{N_{total}}
\times100\%
\]

Bảng kết quả cuối được mở rộng:

| Model / Strategy | MAE | RMSE | \(R^2\) | Trainable params | Trainable % | Train time | Checkpoint size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | Thực nghiệm | Thực nghiệm | Thực nghiệm | 0 | 0% | 0 | 0 |
| LSTM | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm |
| Transformer scratch | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | 100% | Thực nghiệm | Thực nghiệm |
| LoRA | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm |
| Partial fine-tuning | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm | Thực nghiệm |

Không được kết luận LoRA tốt hơn chỉ vì có ít trainable parameters. Kết luận phải đồng thời xét:

```text
Predictive performance
Parameter efficiency
Training cost
Stability
Implementation complexity
```

---

# 72. Leakage control cho pretraining và adaptation

Thứ tự pipeline cập nhật:

```text
Raw chronological data
        ↓
Temporal train / validation / test split
        ↓
Fit preprocessing trên TRAIN
        ↓
Transform train / validation / test
        ↓
Self-supervised pretraining trên TRAIN windows
        ↓
Chọn base checkpoint bằng VALIDATION pretraining loss
        ↓
Khởi tạo mọi adaptation run từ cùng base checkpoint
        ↓
Fine-tune trên downstream TRAIN windows
        ↓
Chọn checkpoint bằng downstream VALIDATION RMSE
        ↓
Đánh giá đúng một lần trên TEST cho báo cáo cuối
```

Các hành vi bị cấm:

- Pretrain backbone trên test windows.
- Fit scaler trên toàn bộ dữ liệu.
- Chọn LoRA rank bằng test metric.
- Chọn số layer partial fine-tuning bằng test metric.
- Dùng các pretrained checkpoint khác nhau cho LoRA và partial fine-tuning.
- Dùng test attention maps để đưa ra quyết định tuning rồi đánh giá lại trên cùng test set mà không công bố.

Validation có thể được dùng cho model selection. Test chỉ dùng cho đánh giá cuối cùng và phân tích sau khi protocol đã được khóa.

---

# 73. Attention analysis sau adaptation

Attention analysis được mở rộng để trả lời:

```text
LoRA có làm thay đổi temporal attention pattern hay không?
Partial fine-tuning có tạo attention tập trung hơn hoặc phân tán hơn không?
Các strategy có khác nhau khi xử lý peak consumption không?
Attention pattern có ổn định trên nhiều test sequences không?
```

Các model được so sánh attention phải:

- Có cùng encoder architecture.
- Bắt đầu từ cùng pretrained checkpoint đối với E3 và E4.
- Dùng cùng test sequences để trực quan hóa.
- Dùng cùng layer, head index và color scale khi so sánh trực tiếp.

Tập sample phân tích nên gồm:

```text
Một sample tiêu thụ bình thường
Một sample peak consumption
Một sample được dự đoán tốt
Một sample có absolute error lớn
```

Attention vẫn chỉ được diễn giải như hành vi nội tại của mô hình, không phải bằng chứng nhân quả hoặc feature importance tuyệt đối.

---

# 74. Reproducibility cho các thí nghiệm adaptation

Ngoài các thông tin ở phần khả năng tái lập, cần lưu:

```text
Base checkpoint identifier
Pretraining objective
Masking ratio
Pretraining epochs và best epoch
Danh sách frozen modules
Danh sách trainable modules
LoRA target modules
LoRA rank, alpha và dropout
Partial fine-tuning layer indices
Trainable parameter count
Learning rate của từng strategy
Early-stopping state
```

Nếu tài nguyên cho phép, chạy ít nhất ba seed:

```text
42
123
2026
```

Kết quả được báo cáo dưới dạng:

\[
mean\pm standard\ deviation
\]

Nếu chỉ chạy một seed, phải ghi rõ đây là limitation và không được diễn giải chênh lệch nhỏ giữa các mô hình như bằng chứng chắc chắn.

---

# 75. Research questions sau khi cập nhật

Câu hỏi nghiên cứu chính:

> Transformer Encoder có học được các phụ thuộc thời gian đa biến hữu ích cho dự báo ngắn hạn mức tiêu thụ năng lượng và đạt hiệu năng như thế nào so với LSTM baseline?

Câu hỏi nghiên cứu về adaptation:

> LoRA có duy trì hoặc cải thiện hiệu năng so với partial fine-tuning trong khi cập nhật ít tham số hơn hay không?

Câu hỏi nghiên cứu về attention:

> Các temporal attention patterns thay đổi như thế nào giữa pretrained Transformer, LoRA adaptation và partial fine-tuning?

Câu hỏi về chi phí:

> Mức giảm trainable parameters và chi phí huấn luyện của LoRA có đủ lớn để bù cho mọi chênh lệch về predictive performance hay không?

---

# 76. Pipeline coursework cập nhật

```text
UCI Appliances Energy Prediction
                ↓
Data validation, EDA, feature engineering
                ↓
Temporal split và train-only scaling
                ↓
Sliding windows: L = 144, H = 1
                ↓
        ┌───────┬───────────┬────────────────────────┐
        ↓       ↓           ↓                        ↓
 Persistence  LSTM  Transformer scratch  Self-supervised pretraining
                                                  ↓
                                    Shared pretrained checkpoint
                                                  ↓
                                     ┌────────────┴────────────┐
                                     ↓                         ↓
                                   LoRA            Partial fine-tuning
                                     └────────────┬────────────┘
                                                  ↓
                 Validation-based model selection
                                ↓
                       Locked test evaluation
                                ↓
                 MAE / RMSE / R² + efficiency
                                ↓
          Actual vs Predicted / Residual / Peak Analysis
                                ↓
             Attention comparison across strategies
                                ↓
              Limitations và evidence-based conclusion
```

Coursework sau khi cập nhật chỉ được xem là hoàn chỉnh khi:

```text
Pipeline dữ liệu không leakage
Persistence và LSTM baseline đã chạy
Transformer scratch đã chạy
Base Transformer đã được pretrain và lưu checkpoint
LoRA và partial fine-tuning dùng cùng base checkpoint
Không sử dụng full fine-tuning làm phương án chính
Các model được chọn bằng validation
Test được giữ độc lập đến đánh giá cuối
Metric dự báo và metric efficiency đều được báo cáo
Attention được trích xuất và so sánh trên cùng test samples
Kết luận dựa trên kết quả thực nghiệm, không dựa trên kỳ vọng
```

Đặc tả cô đọng sau khi bổ sung yêu cầu của giảng viên:

\[
\boxed{
\text{Time-series regression}
+
\text{LSTM baseline}
+
\text{Transformer}
+
\text{LoRA vs partial fine-tuning}
+
\text{Attention analysis}
}
\]

---

<div align="center">

## Trạng thái hiện tại

**Đã hoàn thành phân tích bài toán, xác định phạm vi coursework và bổ sung chiến lược LoRA–partial fine-tuning**

**Bước tiếp theo được khuyến nghị: xây dựng kế hoạch triển khai notebook và kiến trúc code chi tiết**

</div>
