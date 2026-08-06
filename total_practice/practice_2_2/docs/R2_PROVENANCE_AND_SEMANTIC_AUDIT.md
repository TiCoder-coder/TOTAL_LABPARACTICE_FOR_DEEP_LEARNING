# R2 Provenance and Semantic-Quality Audit

## Trạng thái

- Policy: `r2_provenance_policy_v1`
- Audit lineage: `r2_provenance_semantic_audit_v1`
- Trạng thái kỹ thuật: đã chạy trên toàn bộ dataset legacy
- R2 exit gate: `blocked`
- Xóa hoặc sửa ảnh nguồn: không thực hiện
- Training hoặc Final Test: không thực hiện

Machine-readable policy nằm tại `configs/r2_provenance_policy.json`.

## Provenance Contract

Mọi ảnh thu thập mới phải có đủ:

- `asset_id`;
- `source_provider`;
- `product_id`;
- `listing_url`;
- `image_url`;
- `query`;
- `category_id`;
- `category_name`;
- `image_index`;
- `image_role`;
- `crawl_timestamp_utc`;
- `raw_sha256`;
- `class_name`.

Raw asset và metadata sidecar là immutable. Ghi lại cùng asset với cùng nội dung
được phép. Ghi cùng identity nhưng nội dung hoặc metadata khác phải fail.

## Collection Contract

Collector mới bắt buộc truyền `category_id` và `category_name`. Query không thể
được dùng một mình. Mỗi lớp có required terms và negative terms riêng.

Mỗi product chỉ lấy tối đa hai ảnh có vai trò `primary_thumbnail` hoặc
`primary_gallery`. Detail gallery không được sử dụng. Collector không xóa thư
mục cũ, không đổi tên asset đã tồn tại và không ghi đè raw bytes.

Các ảnh bị product/category filter từ chối không được tải xuống. Việc một ảnh
vượt qua filter chỉ cho phép nó đi vào semantic review, không tự động biến nó
thành ground truth.

## Legacy Inventory

Kết quả read-only audit:

| Thuộc tính | Giá trị |
|---|---:|
| Tổng file | 3.202 |
| Ảnh gốc | 2.896 |
| Generated derivative bị loại | 306 |
| Ảnh không giải mã được | 0 |
| Provenance đầy đủ | 0 |
| Provenance coverage | 0% |
| Cross-label visual pairs | 325 |
| Cross-label exact pixel pairs | 0 |

Dataset directory digest vẫn là
`fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5`.

## Quality Heuristics

| Flag | Số ảnh gốc |
|---|---:|
| `EXTREMELY_BRIGHT` | 1.025 |
| `EXTREMELY_DARK` | 14 |
| `LOW_COLOR_CONTENT` | 5 |
| `LOW_CONTRAST` | 4 |
| `POTENTIALLY_BLURRY` | 1 |

Các flag này chỉ sắp xếp ưu tiên review. Nền trắng sáng là đặc điểm phổ biến của
ảnh thương mại điện tử, nên `EXTREMELY_BRIGHT` không phải bằng chứng label noise
và không được dùng để xóa hoặc quarantine tự động.

## Semantic Review Queue

Queue chứa toàn bộ 2.896 ảnh gốc. Mỗi record giữ asset identity, SHA-256, nhãn
hiện tại, provenance status, priority, review reasons và các trường quyết định
đang để trống.

Priority hiện tại:

| Priority | Số ảnh |
|---|---:|
| `P1` | 251 |
| `P2` | 964 |
| `P3` | 1.681 |

`P1` gồm R1 pilot và asset tham gia cross-label visual neighbor. `P2` gồm quality
flags hoặc confusion-group oversampling. `P3` là phần còn lại của full semantic
audit. Không có `P0` vì không phát hiện ảnh hỏng hoặc exact-pixel duplicate khác
lớp.

Review decision phải tuân thủ `r1_label_contract_v1`. Mỗi ảnh cần decision,
reason code, reviewer identity, timestamp và `policy_case_resolved = true`.

## Unique Product Gate

Ngưỡng đã khai báo là ít nhất 100 product identity cho mỗi lớp. Dataset legacy
không có `product_id`, vì vậy source stem hoặc decoded hash không được báo cáo
thay thế như unique product. Hiện unique-product count của cả mười lớp là
`unknown`.

Ảnh mới phải được thu thập theo provenance contract cho đến khi từng lớp đạt
ngưỡng. Offline augmentation không được tính là product mới.

## Blocked Reasons

- `R0_AUTHORITY_NOT_READY`
- `R1_CONTRACT_REVIEW_NOT_PASSED`
- `HISTORICAL_729_CANDIDATES_NOT_RESTORED`
- `PRODUCT_PROVENANCE_INCOMPLETE`
- `FULL_SEMANTIC_REVIEW_INCOMPLETE`
- `UNIQUE_PRODUCT_MINIMUM_NOT_PROVEN`

Không được tái tạo file 729 candidate lịch sử rồi gắn lại identity cũ. Nếu backup
không thể phục hồi, full queue 2.896 ảnh của lineage R2 phải được review như một
audit mới.

## Artifacts

- `artifacts/new_work/r2_provenance_semantic_audit_v1/legacy_inventory.json`
- `artifacts/new_work/r2_provenance_semantic_audit_v1/cross_label_visual_pairs.json`
- `artifacts/new_work/r2_provenance_semantic_audit_v1/semantic_review_queue.json`
- `artifacts/new_work/r2_provenance_semantic_audit_v1/audit_report.json`
- `artifacts/new_work/r2_provenance_semantic_audit_v1/artifact_manifest.json`
- `docs/R2_VERIFICATION.json`

## R2 Exit Gate

R2 chỉ pass khi R0 và R1 pass, historical authority được xử lý đúng lineage,
mọi ảnh gốc có semantic decision đã resolve, mọi accepted asset có provenance và
mỗi lớp đạt ngưỡng unique product đã khai báo.
