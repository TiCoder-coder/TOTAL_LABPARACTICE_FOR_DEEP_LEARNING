# MAPE Metric Addendum Pre-process Plan

## Mục tiêu

Bổ sung MAPE dưới dạng supplementary reporting metric, kiểm tra lại ML pipeline theo ghi chú đã cung cấp và giữ nguyên toàn bộ scientific selection history.

## Phạm vi được phép

- Version hóa metric contract mới mà không ghi đè `METRICS-v1`.
- Tính MAPE từ prediction artifacts hiện có và đã xác minh.
- Tạo derived addendum artifacts riêng.
- Tạo HTML renderer chỉ đọc.
- Thêm notebook presentation cell chỉ gọi renderer.
- Tạo process compliance audit dựa trên contracts và artifacts hiện hành.

## Phạm vi bị cấm

- Training hoặc fine-tuning.
- Model inference mới trên Test.
- Model selection bằng MAPE.
- Thay đổi Phase 23–41 winners.
- Thay đổi Phase 45 final lock.
- Chọn best seed hoặc ensemble.
- Ghi đè Phase 47–59 canonical artifacts.
- Thay đổi frozen scientific narrative.

## Thứ tự thực thi

### Bước 1

Ghi preservation baseline, source inventory và architecture amendment cho supplementary addendum.

### Bước 2

Xây MAPE implementation tại evaluation layer và unit tests độc lập.

### Bước 3

Xây metric addendum layer để xác minh prediction source, checksum, population, unit và zero-target policy.

### Bước 4

Materialize Validation addendum từ prediction artifacts hiện có. Test addendum chỉ được materialize nếu frozen Test prediction bundles tồn tại và checksum khớp.

### Bước 5

Xây process compliance audit theo các nhóm define problem, data audit, EDA, leakage, preprocessing, feature engineering, baseline, model comparison, tuning, robustness, final lock, Test, error analysis và interpretability.

### Bước 6

Xây reporting renderer chỉ đọc addendum artifacts.

### Bước 7

Thêm notebook section chỉ gọi renderer, bảo toàn tất cả output và metadata hiện có.

### Bước 8

Chạy verification theo tầng, notebook targeted execution và downstream regression tests.

## Tiêu chí nghiệm thu

- Standard MAPE được tính trên original Wh với float64.
- Không epsilon và không silent filtering.
- MAPE không thay đổi selection metric.
- MAE, RMSE và R² giữ nguyên.
- Không Test inference mới.
- Test MAPE bị block rõ ràng nếu frozen prediction bundles thiếu.
- Validation addendum chỉ chứa nguồn có prediction artifacts hợp lệ.
- HTML không chứa widget hoặc script.
- Notebook không chứa công thức hoặc xử lý MAPE.
- Process audit có evidence path và trạng thái cho từng control.
- Canonical Phase 47–59 artifacts giữ nguyên checksum.
