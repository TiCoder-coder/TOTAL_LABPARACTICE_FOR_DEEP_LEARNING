# Tổng Hợp Refactor Practice 2.2 Từ R0 Đến R12

## 1. Thông tin tài liệu

| Thuộc tính | Giá trị |
|---|---|
| Project | Practice 2.2 - Cosmetic Product Image Classification |
| Ngày tổng hợp | 06/08/2026 |
| Snapshot kiểm tra cuối | 2026-08-06 09:39:47 +0700 |
| Historical canonical lineage | `canonical_26dc4625_52aaf974_s42_v1` |
| New candidate split lineage | `v3_product_visual_group_s42_v1` |
| Phạm vi | Toàn bộ thay đổi và bằng chứng từ R0 đến R12 |
| Canonical notebook | `notebooks/04_canonical_report.ipynb` |
| New report notebook dự kiến | `notebooks/05_product_visualsafe_report.ipynb` |
| Final Test mới | Chưa được thực hiện |

Tài liệu này tổng hợp trạng thái thực tế của refactor, bao gồm mục tiêu, file đã
thêm, logic đã thay đổi, artifact đã sinh, gate đã kiểm tra, blocker còn tồn tại
và flow bắt buộc để hoàn thiện project.

Điểm quan trọng nhất khi đọc tài liệu này là phân biệt hai khái niệm:

- `protocol implementation passed`: cơ chế, invariant và fail-closed guard đã
  được kiểm tra bằng dữ liệu audit hoặc synthetic fixture;
- `phase gate passed`: toàn bộ dữ liệu thật, review thủ công, predecessor và
  model evidence đã đầy đủ để phase được phép chuyển tiếp.

Hiện chưa có phase R0-R10 nào vượt toàn bộ gate. R7-R10 đã vượt các kiểm tra
correctness/protocol nội bộ, nhưng vẫn bị chặn bởi các predecessor gate. R11 chưa
được thực thi. R12 mới hoàn thành phần policy, registry và implementation source;
notebook/HTML cuối chưa được sinh thành công.

## 2. Kết luận tổng quan

### 2.1 Vấn đề ban đầu

Historical canonical result đang ghi nhận:

| Metric | Giá trị historical |
|---|---:|
| Validation Accuracy | 78.31% |
| Validation Macro F1 | 0.7821 |
| Final Test Accuracy | 76.36% |
| Final Test Macro F1 | 0.7617 |
| Final Test evaluation count | 1 |

Kết quả audit không ủng hộ giả thuyết rằng Accuracy thấp vì một split bị thiếu
label. Cả mười class đều có mặt trong Train, Validation và Test của historical
canonical split. Phân bố class chỉ mất cân bằng nhẹ và dự đoán không collapse về
một vài class.

Các nguyên nhân gốc được xác định theo thứ tự ưu tiên:

1. Semantic contamination và annotation contract chưa đủ chặt.
2. Dataset legacy không lưu `product_id`, listing URL và image URL theo từng ảnh.
3. Product-level independence chưa được chứng minh.
4. Visual near-duplicate và template quảng cáo có thể tạo shortcut.
5. Số product độc lập hiệu dụng nhỏ so với số trainable parameter.
6. Transform cũ có thể crop mất nhãn sản phẩm, đảo chữ và xóa vùng phân biệt.
7. Weighted epoch-loss aggregation trong canonical path sai về mẫu số.
8. Model được fine-tune `layer4` ngay khi classifier head còn khởi tạo ngẫu nhiên.
9. Model selection dùng một seed và một split nên uncertainty cao.

E2 historical đạt Train Accuracy 98.31% nhưng Validation Accuracy 78.31%, tạo
generalization gap 20 điểm phần trăm. Đây là bằng chứng model có đủ capacity để
memorize Train; tăng model size đơn thuần không phải hướng sửa có căn cứ.

### 2.2 Thay đổi kiến trúc chính

Refactor chuyển project từ một pipeline thiên về notebook và metric historical
sang một pipeline có policy, lineage, artifact registry và gate rõ ràng:

```text
Authority audit
    -> Label contract
    -> Provenance and semantic audit
    -> Product/visual grouping
    -> Stratified component split
    -> Train-only data readiness
    -> Domain-safe transform ablation
    -> Correct training mathematics
    -> Staged transfer protocol
    -> Controlled architecture comparison
    -> Repeated-seed robust selection
    -> One-time Test evaluation
    -> Artifact-only report notebook
```

Mọi phase mới đều tuân thủ các nguyên tắc:

- data correctness trước model tuning;
- không ghi đè historical canonical lineage;
- source image là immutable;
- generated derivative không được tính là product mới;
- Validation chỉ dùng sau khi data/transform authority hợp lệ;
- Test bị khóa đến khi finalist và checkpoint đã freeze;
- synthetic fixture không được trình bày như model-performance evidence;
- mọi output quan trọng có byte size và SHA-256 trong artifact manifest;
- predecessor chưa pass thì successor phải fail closed.

### 2.3 Trạng thái R0-R12 tại snapshot

| Phase | Vai trò | Implementation | Gate tổng thể | Dữ liệu/model thật |
|---|---|---|---|---|
| R0 | Authority inventory | Đã triển khai và chạy | `BLOCKED` | Không train, không Test |
| R1 | Label/image contract | Đã triển khai, pilot đã sinh | `PENDING_REVIEW` | 0/2 reviewer |
| R2 | Provenance/semantic audit | Đã chạy trên 3.202 file | `BLOCKED` | Audit read-only |
| R3 | Product/visual groups | Đã chạy trên 2.896 ảnh gốc | `BLOCKED` | Embedding/group audit |
| R4 | Stratified group split | Candidate split đã sinh | `BLOCKED` | Không train, Test sealed |
| R5 | Train-only readiness | Đã chạy trên 2.034 Train asset | `BLOCKED` | Chỉ mở Train content |
| R6 | Transform ablation | T0-T4 đã audit | `BLOCKED` | Chỉ mở Train content |
| R7 | Training correctness | Correctness checks pass | `BLOCKED` | Synthetic tensor only |
| R8 | Staged transfer | Protocol checks pass | `BLOCKED` | Synthetic fixture only |
| R9 | Architecture ceiling | Protocol/profile checks pass | `BLOCKED` | Synthetic input only |
| R10 | Robust selection | Selection/freeze logic pass | `BLOCKED` | Synthetic fixture only |
| R11 | One-time Test | Chưa triển khai/chưa chạy | `NOT_EXECUTED` | Test count mới bằng 0 |
| R12 | Artifact-only report | Source/registry đã triển khai một phần | `INCOMPLETE/BLOCKED` | Không train, không Test |

## 3. Kiến trúc quản trị sau refactor

### 3.1 Ba vùng authority

R0 phân loại tài nguyên thành ba vùng:

| Classification | Ý nghĩa |
|---|---|
| `frozen_historical` | Canonical dataset, split, checkpoint, report và notebook cũ |
| `v2_experimental` | Manifest/checkpoint từ các thử nghiệm visual-safe trước đây |
| `new_work` | Toàn bộ R1-R12 artifact mới, không ghi đè historical evidence |

Historical canonical result 76.36% được giữ làm bằng chứng quá khứ. Nó không được
đổi tên hoặc trình bày như kết quả của candidate V3 vì data lineage và split khác
nhau.

### 3.2 Gate fail-closed

Mỗi report có `status`, `gate_passed`, `blocked_reasons` và các guard liên quan.
Một phase có thể hoàn thành code và protocol test nhưng vẫn `blocked`. Đây là hành
vi có chủ đích, không phải lỗi của pipeline.

Ví dụ:

- R4 có split 70/15/15 rất sát target nhưng không authorize training vì R3 chưa
  hoàn thành manual review;
- R7 sửa đúng loss math nhưng không train real data vì R6 chưa chọn transform;
- R10 kiểm tra winner/freeze logic nhưng không chọn finalist từ synthetic data;
- R12 có thể dựng report trạng thái blocked nhưng không được bịa R11 Test metric.

### 3.3 Test firewall

Từ R0 đến R10, các verification report đều ghi nhận:

- không tạo Test DataLoader;
- không evaluate Test;
- không đọc Test image content ở R5-R10;
- không dùng Test field trong model selection;
- không authorize successor tự động.

Candidate Test partition của R4 được seal bằng fingerprint. Guard chỉ cho phép
access khi lineage đã authorize, token hợp lệ, split fingerprint khớp và Test
chưa từng được access. Current candidate guard có:

| Trường | Giá trị |
|---|---|
| State | `sealed_candidate_blocked` |
| Test access allowed | `false` |
| Test access count | `0` |
| Test evaluation count | `0` |
| Repeat evaluation allowed | `false` |

### 3.4 Artifact và hash lineage

Mỗi phase ghi output dưới `artifacts/new_work/<lineage>/`. Artifact manifest lưu
relative path, byte size và SHA-256. Cách này bảo đảm report có thể phát hiện file
thiếu, file bị thay đổi và evidence trộn lineage.

Các path source không được suy ra từ current working directory tùy ý. Module dùng
project-root discovery và path policy. Candidate manifest, component membership,
split assignment và Test partition đều có fingerprint độc lập.

### 3.5 Tách protocol evidence khỏi model evidence

R7-R10 dùng synthetic tensor hoặc synthetic metric fixture để chứng minh logic.
Các artifact này có trường như:

- `synthetic_tensor_verification_only`;
- `synthetic_protocol_verification_only`;
- `fixture_is_performance_evidence = false`;
- `fixture_is_model_quality_evidence = false`;
- `fixture_is_model_selection_evidence = false`.

Do đó không được lấy số trong synthetic uncertainty report của R10 làm Validation
Accuracy thực tế.

## 4. R0 - Restore and Freeze Authority

### 4.1 Mục tiêu

R0 tạo inventory read-only cho toàn bộ canonical và V2 authority trước khi bất kỳ
refactor nào được phép dùng chúng. Phase này không tự tái tạo file mất, không
train và không mở Final Test.

### 4.2 File đã thêm

- Policy: [`configs/r0_authority_policy.json`](../configs/r0_authority_policy.json)
- Module: [`src/practice_2_2/r0_authority.py`](../src/practice_2_2/r0_authority.py)
- Entrypoint: [`scripts/verify_r0_authority.py`](../scripts/verify_r0_authority.py)
- Verification: [`docs/R0_AUTHORITY_INVENTORY.json`](../docs/R0_AUTHORITY_INVENTORY.json)
- Test: [`tests/test_r0_authority.py`](../tests/test_r0_authority.py)

### 4.3 Logic đã triển khai

`r0_authority.py` thực hiện:

1. Load policy schema và validate classification.
2. Resolve project-relative path an toàn.
3. Tính SHA-256 cho file.
4. Tính directory digest từ relative path, file size và file hash.
5. Kiểm tra pandas manifest theo logical hash khi được khai báo.
6. So sánh actual size/hash/count với expected authority.
7. Sinh inventory nhưng không sửa resource.
8. `assert_r0_ready()` fail nếu bất kỳ required authority nào missing/mismatch.

### 4.4 Kết quả

Dataset canonical được tìm thấy và verify:

| Thuộc tính | Giá trị |
|---|---:|
| File count | 3.202 |
| Total bytes | 48.502.262 |
| Directory SHA-256 | `fd1cd4557352b28054b976feebd69932c34af9edda44eb1251e8f46b7e97e8a5` |

Canonical registry và canonical HTML cũng verify tại thời điểm R0. Tuy nhiên R0
blocked với 14 failed authorities. Các nhóm quan trọng bị thiếu gồm canonical
split, quarantine CSV, canonical checkpoint/output, E3/E4 checkpoint/output,
V2 manifests, V2 initial state và V2 E1/E2 checkpoints.

Canonical notebook cũng mismatch policy. Policy R0 kỳ vọng:

```text
48793d9ef49529b978ad30047f3f514f81bff2e0741e96c13a27e5b147503a2a
```

Inventory R0 quan sát notebook hash:

```text
a4eb294b3c4e8ec9619255f4acdd529094c0abd47eed9a263cf8bc96a7a8ff0d
```

R0 vì vậy không authorize implementation, canonical notebook mutation hoặc Final
Test re-evaluation.

## 5. R1 - Label and Image-Content Contract

### 5.1 Mục tiêu

R1 biến mười tên folder thành ontology có quy tắc operational. Ground truth lấy
từ sản phẩm nhìn thấy trong ảnh. Query crawl, folder và filename chỉ là bằng
chứng hỗ trợ, không được ghi đè nội dung ảnh.

### 5.2 File đã thêm

- Contract: [`configs/r1_label_contract.json`](../configs/r1_label_contract.json)
- Module: [`src/practice_2_2/r1_contract.py`](../src/practice_2_2/r1_contract.py)
- Pilot builder: [`scripts/build_r1_pilot.py`](../scripts/build_r1_pilot.py)
- Contract verifier: [`scripts/verify_r1_contract.py`](../scripts/verify_r1_contract.py)
- Human guide: [`docs/R1_LABEL_AND_IMAGE_CONTRACT.md`](../docs/R1_LABEL_AND_IMAGE_CONTRACT.md)
- Pilot: [`docs/R1_PILOT_SAMPLE.json`](../docs/R1_PILOT_SAMPLE.json)
- Review template: [`docs/R1_REVIEW_TEMPLATE.json`](../docs/R1_REVIEW_TEMPLATE.json)
- Verification: [`docs/R1_VERIFICATION.json`](../docs/R1_VERIFICATION.json)
- Test: [`tests/test_r1_contract.py`](../tests/test_r1_contract.py)

### 5.3 Decision contract

Ba quyết định hợp lệ:

| Decision | Điều kiện |
|---|---|
| `accept` | Dominant product rõ và khớp current label |
| `relabel` | Dominant product rõ, thuộc đúng một target class khác |
| `quarantine` | Ambiguous, multi-class, hybrid, ngoài ontology hoặc thiếu thông tin |

Contract định nghĩa 15 reason code. Nó bao phủ clear single product, same-class
variants, wrong target class, ambiguous class, multiple target classes, hybrid,
no dominant product, text-only, lifestyle-only, before-after, non-cosmetic,
outside ontology, low information, conflicting evidence và invalid image.

Source image không bị xóa hoặc ghi đè khi quarantine.

### 5.4 Quy tắc nội dung

R1 thêm quy tắc cụ thể cho:

- bundle cùng class và bundle nhiều class;
- lifestyle và before-after;
- banner, watermark và text-only image;
- hybrid product như shampoo-body wash hoặc moisturizer-sunscreen;
- offline-generated derivative;
- positive, negative và borderline definition cho cả mười class.

Ví dụ, `toner` phải là sản phẩm cân bằng da leave-on. Flower food, perfume,
micellar water và chai dung dịch không đủ bằng chứng không được accept vào toner.

### 5.5 Pilot review

Pilot `r1_semantic_pilot_s42_v1` được chọn bằng SHA-256 ranking với seed 42:

| Thuộc tính | Giá trị |
|---|---:|
| Tổng mẫu | 100 |
| Mỗi class | 10 |
| Generated image | 0 |
| Split blinded | `true` |
| Model output blinded | `true` |
| Pilot SHA-256 | `d7df3e3ae558cbb1e5e5c8093b991ef0a355d236ca4ff4e362ac65d9a392bfe4` |

Hai reviewer độc lập phải đạt exact agreement ít nhất 0.85, Cohen's kappa ít
nhất 0.80 và agreement từng class ít nhất 0.70.

Current review count là 0/2. R1 đang `pending_independent_review` và chưa pass.

## 6. R2 - Provenance and Semantic-Quality Audit

### 6.1 Mục tiêu

R2 thiết lập provenance contract cho crawl mới và audit read-only toàn bộ legacy
dataset. Phase này không tự động accept, relabel, quarantine hoặc delete ảnh.

### 6.2 File đã thêm

- Policy: [`configs/r2_provenance_policy.json`](../configs/r2_provenance_policy.json)
- Collector/provenance: [`src/practice_2_2/r2_provenance.py`](../src/practice_2_2/r2_provenance.py)
- Semantic audit: [`src/practice_2_2/r2_semantic_audit.py`](../src/practice_2_2/r2_semantic_audit.py)
- Crawl entrypoint: [`scripts/collect_tiki_r2.py`](../scripts/collect_tiki_r2.py)
- Audit entrypoint: [`scripts/run_r2_audit.py`](../scripts/run_r2_audit.py)
- Documentation: [`docs/R2_PROVENANCE_AND_SEMANTIC_AUDIT.md`](../docs/R2_PROVENANCE_AND_SEMANTIC_AUDIT.md)
- Verification: [`docs/R2_VERIFICATION.json`](../docs/R2_VERIFICATION.json)
- Test: [`tests/test_r2_provenance_semantic_audit.py`](../tests/test_r2_provenance_semantic_audit.py)
- Artifact root: [`artifacts/new_work/r2_provenance_semantic_audit_v1`](../artifacts/new_work/r2_provenance_semantic_audit_v1)

### 6.3 Provenance logic mới

Ảnh crawl mới phải lưu `asset_id`, provider, `product_id`, listing URL, image URL,
query, category ID/name, image index/role, UTC timestamp, raw SHA-256 và class.

Collector mới:

- bắt buộc category constraint thay vì chỉ search query;
- dùng required terms và negative terms theo class;
- ưu tiên primary thumbnail/primary gallery;
- lấy tối đa hai ảnh cho mỗi product;
- loại detail gallery;
- không overwrite raw bytes hoặc metadata identity đã tồn tại;
- filter pass chỉ đưa ảnh vào semantic review, không tự biến thành ground truth.

### 6.4 Legacy audit result

| Thuộc tính | Giá trị |
|---|---:|
| Tổng file | 3.202 |
| Ảnh gốc | 2.896 |
| Generated derivative bị loại | 306 |
| Invalid image | 0 |
| Provenance đầy đủ | 0 |
| Provenance coverage | 0% |
| Cross-label visual pair | 325 |
| Cross-label exact-pixel pair | 0 |

Quality flags chỉ dùng để ưu tiên review:

| Flag | Số ảnh |
|---|---:|
| `EXTREMELY_BRIGHT` | 1.025 |
| `EXTREMELY_DARK` | 14 |
| `LOW_COLOR_CONTENT` | 5 |
| `LOW_CONTRAST` | 4 |
| `POTENTIALLY_BLURRY` | 1 |

Nền trắng sáng phổ biến trong e-commerce nên `EXTREMELY_BRIGHT` không phải rule
tự động loại.

### 6.5 Semantic review queue

Toàn bộ 2.896 ảnh gốc được đưa vào queue:

| Priority | Số ảnh |
|---|---:|
| P1 | 251 |
| P2 | 964 |
| P3 | 1.681 |

Historical 729-candidate authority không được khôi phục từ backup. Pipeline không
regenerate rồi giả làm artifact cũ; nó tạo full audit lineage mới.

Unique product threshold được predeclare là ít nhất 100 product mỗi class. Vì
legacy dataset không có `product_id`, mọi unique-product count hiện là `unknown`.
Source stem hoặc image hash không được thay thế product identity.

R2 blocked bởi R0/R1, missing historical authority, provenance coverage 0%, full
semantic review chưa hoàn thành và unique-product minimum chưa được chứng minh.

## 7. R3 - Product and Visual Grouping

### 7.1 Mục tiêu

R3 xây deterministic connected component cho 2.896 ảnh gốc. Generated derivative
không được tham gia. R3 chỉ group, không split, train hoặc Test evaluation.

### 7.2 File đã thêm

- Policy: [`configs/r3_grouping_policy.json`](../configs/r3_grouping_policy.json)
- Module: [`src/practice_2_2/r3_grouping.py`](../src/practice_2_2/r3_grouping.py)
- Entrypoint: [`scripts/run_r3_grouping.py`](../scripts/run_r3_grouping.py)
- Documentation: [`docs/R3_PRODUCT_VISUAL_GROUPING.md`](../docs/R3_PRODUCT_VISUAL_GROUPING.md)
- Verification: [`docs/R3_VERIFICATION.json`](../docs/R3_VERIFICATION.json)
- Test: [`tests/test_r3_product_visual_grouping.py`](../tests/test_r3_product_visual_grouping.py)
- Artifact root: [`artifacts/new_work/r3_product_visual_groups_v1`](../artifacts/new_work/r3_product_visual_groups_v1)

### 7.3 Identity evidence

R3 kiểm tra bốn identity key:

1. `product_id` khi có provenance.
2. Raw-file SHA-256.
3. Decoded-pixel SHA-256.
4. R2 source group.

Same-label identity edge có thể union. Cross-label identity edge luôn đi vào
manual review, không tự union hoặc quarantine.

### 7.4 Visual evidence

Mỗi ảnh có bốn tín hiệu:

| Signal | Vai trò |
|---|---|
| pHash | Low-frequency perceptual structure |
| dHash | Relative grayscale transitions |
| SSIM | Local luminance/contrast/structure ở 96x96 |
| ResNet18 embedding | 512-dimensional ImageNet visual semantics |

Embedding dùng `ResNet18_Weights.IMAGENET1K_V1`, checkpoint SHA-256:

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

Embedding được L2 normalize, chạy CPU, sorted asset-ID order, không shuffle.

Candidate pair được xét sâu khi thuộc top-8 embedding neighbor với cosine ít nhất
0.94, hoặc pHash distance tối đa 8, hoặc dHash distance tối đa 5.

Same-label pair chỉ auto-confirm khi đồng thời thỏa:

- pHash distance tối đa 4;
- dHash distance tối đa 4;
- SSIM ít nhất 0.95;
- embedding cosine ít nhất 0.98.

### 7.5 Deterministic grouping result

Union-find dùng deterministic component ID từ SHA-256 của sorted member asset ID.
Component ID có prefix `r3g_` và không phụ thuộc traversal order.

| Thuộc tính | Giá trị |
|---|---:|
| Eligible original assets | 2.896 |
| Components | 2.462 |
| Singleton components | 2.042 |
| Largest component | 4 |
| Accepted identity/visual edges | 448 |
| Assets assigned exactly once | 2.896/2.896 |
| Cross-label components | 0 |
| Product provenance coverage | 0% |

Review queue còn 2.325 edge:

| Edge type | Số lượng |
|---|---:|
| Cross-label pending | 1.023 |
| Same-label uncertain | 1.302 |

R3 blocked vì R2 chưa pass, product provenance thiếu và hai nhóm edge chưa được
review. Component hiện là provisional visual component, không được báo cáo như
unique product đã chứng minh.

## 8. R4 - Stratified Product/Visual Group Split

### 8.1 Mục tiêu

R4 tạo candidate split 70/15/15 theo whole component, không split từng image độc
lập. Nó cân bằng đồng thời image count và component count theo class.

### 8.2 File đã thêm

- Policy: [`configs/r4_split_policy.json`](../configs/r4_split_policy.json)
- Module: [`src/practice_2_2/r4_split.py`](../src/practice_2_2/r4_split.py)
- Entrypoint: [`scripts/run_r4_split.py`](../scripts/run_r4_split.py)
- Documentation: [`docs/R4_STRATIFIED_GROUP_SPLIT.md`](../docs/R4_STRATIFIED_GROUP_SPLIT.md)
- Verification: [`docs/R4_VERIFICATION.json`](../docs/R4_VERIFICATION.json)
- Test: [`tests/test_r4_stratified_group_split.py`](../tests/test_r4_stratified_group_split.py)
- Artifact root: [`artifacts/new_work/v3_product_visual_group_s42_v1`](../artifacts/new_work/v3_product_visual_group_s42_v1)

### 8.3 Assignment logic

Trong từng class, component được sort theo size giảm dần rồi SHA-256 rank từ seed
42. Mỗi component được gán vào split làm nhỏ nhất combined squared fill ratio cho
image và component. SHA-256 rank làm deterministic tie-breaker.

Pipeline recompute assignment sau khi đảo input component order. Nếu hai map
khác nhau, R4 abort.

### 8.4 Candidate distribution

| Split | Images | Image ratio | Components | Component ratio |
|---|---:|---:|---:|---:|
| Train | 2.034 | 70.2348% | 1.725 | 70.0650% |
| Validation | 431 | 14.8826% | 369 | 14.9878% |
| Test | 431 | 14.8826% | 368 | 14.9472% |

Cả mười class có mặt trong cả ba split. Maximum deviation:

| Check | Deviation |
|---|---:|
| Overall image ratio | 0.2348 percentage points |
| Per-class image ratio | 0.4403 percentage points |
| Per-class component ratio | 0.3061 percentage points |

Mọi giá trị đều thấp hơn threshold 1 percentage point.

### 8.5 Leakage proof và giới hạn

Candidate split có zero confirmed violation cho component ID, product ID khi có,
raw hash, decoded hash, source group và accepted visual edge.

Tuy nhiên còn 1.172 review-only pair đi qua split boundary:

| Pair type | Số lượng |
|---|---:|
| Same-label uncertain | 663 |
| Cross-label uncertain | 509 |

Đây chưa phải confirmed leakage, nhưng bắt buộc block lineage cho đến khi review
xong và R3/R4 được rebuild.

Mọi manifest row có `use_for_model = false` và
`eligibility = provisional_pending_predecessor_gates`. R4 không authorize model
training. Historical metric không được so trực tiếp với candidate V3.

## 9. R5 - Train-Only Data Readiness

### 9.1 Mục tiêu

R5 thay count-only EDA bằng data-readiness audit trên Train content. Validation và
Test chỉ xuất hiện dưới dạng manifest ID để kiểm tra disjointness; image content
và embedding của chúng không đi vào EDA, threshold hoặc figure.

### 9.2 File đã thêm

- Policy: [`configs/r5_data_readiness_policy.json`](../configs/r5_data_readiness_policy.json)
- Module: [`src/practice_2_2/r5_data_readiness.py`](../src/practice_2_2/r5_data_readiness.py)
- Entrypoint: [`scripts/run_r5_data_readiness.py`](../scripts/run_r5_data_readiness.py)
- Documentation: [`docs/R5_TRAIN_ONLY_DATA_READINESS.md`](../docs/R5_TRAIN_ONLY_DATA_READINESS.md)
- Verification: [`docs/R5_VERIFICATION.json`](../docs/R5_VERIFICATION.json)
- Test: [`tests/test_r5_train_only_data_readiness.py`](../tests/test_r5_train_only_data_readiness.py)
- Artifact root: [`artifacts/new_work/r5_train_only_data_readiness_v1`](../artifacts/new_work/r5_train_only_data_readiness_v1)

### 9.3 Content isolation

| Thuộc tính | Giá trị |
|---|---:|
| Candidate Train assets | 2.034 |
| Unique Train assets opened | 2.034 |
| Held-out manifest assets | 862 |
| Validation content reads | 0 |
| Test content reads | 0 |
| Invalid/held-out IDs in access log | 0 |

Mọi image-opening function reject row không thuộc Train. Content access manifest
khớp đúng 2.034 Train asset ID.

### 9.4 EDA logic mới

R5 bổ sung:

- image and component distribution theo class;
- component diversity;
- Train-only RGB mean/std;
- brightness, contrast, blur, edge và border-edge metrics;
- text-like region, banner-band và crop-risk heuristics;
- deterministic class contact sheets;
- deterministic cosine k-means trên Train embedding;
- nearest-neighbor class overlap;
- class/source/vendor/template concentration fields;
- full manual semantic review template.

Train pixel statistics:

| Metric | Giá trị |
|---|---|
| Pixel count | 102.057.984 |
| RGB mean | `[0.77639, 0.76001, 0.74998]` |
| RGB std | `[0.27712, 0.26809, 0.28430]` |
| Mean grayscale brightness | 0.76213 |
| Mean grayscale contrast | 0.19729 |
| Image size | 224x224 |

### 9.5 Review và embedding findings

Review queue bao phủ 2.034 Train image:

| Priority | Số ảnh |
|---|---:|
| P1 | 1.039 |
| P2 | 995 |
| Completed | 0 |

Automated signals gồm 701 extremely bright, 188 border-crop risk, 174 high
text-like area, 97 banner candidates và 21 strict high-similarity cross-label
neighbors. Chúng chỉ ưu tiên review, không tự quyết định nhãn.

K-means tạo 20 cluster và hội tụ sau 33 iteration. `body_wash` có maximum
per-class cluster concentration 38.97%, vượt threshold review 35%.

Trong nearest-neighbor audit:

- 440 image có top-1 neighbor khác label;
- top-1 cross-label ratio là 21.63%;
- có 3.878 unique cross-label neighbor pair;
- 21 pair vượt strict cosine review threshold 0.95.

R5 sinh 31 PNG: 10 class sheets, 20 cluster sheets và một cross-label pair sheet.

### 9.6 Gate

Train-only isolation và component-diversity check pass. R5 vẫn blocked vì R4 chưa
authorize manifest, provenance còn 0%, toàn bộ review pending, P1 unresolved,
source/vendor unknown và embedding concentration vượt threshold. `data_ready`
vẫn là `false`.

## 10. R6 - Domain-Safe Transform Ablation

### 10.1 Mục tiêu

R6 kiểm tra transform theo one-variable ablation để giữ product shape và chữ trên
bao bì. Không recipe nào được auto-select khi Validation evidence chưa hợp lệ.

### 10.2 File đã thêm

- Policy: [`configs/r6_transform_ablation_policy.json`](../configs/r6_transform_ablation_policy.json)
- Module: [`src/practice_2_2/r6_transform_ablation.py`](../src/practice_2_2/r6_transform_ablation.py)
- Entrypoint: [`scripts/run_r6_transform_ablation.py`](../scripts/run_r6_transform_ablation.py)
- Documentation: [`docs/R6_DOMAIN_SAFE_TRANSFORM_ABLATION.md`](../docs/R6_DOMAIN_SAFE_TRANSFORM_ABLATION.md)
- Verification: [`docs/R6_VERIFICATION.json`](../docs/R6_VERIFICATION.json)
- Test: [`tests/test_r6_transform_ablation.py`](../tests/test_r6_transform_ablation.py)
- Artifact root: [`artifacts/new_work/r6_transform_ablation_s42_v1`](../artifacts/new_work/r6_transform_ablation_s42_v1)

### 10.3 Controlled recipes

| Recipe | Parent | Thay đổi duy nhất |
|---|---|---|
| T0 | none | Existing baseline |
| T1 | T0 | Evaluation từ Resize 256 + CenterCrop 224 sang direct Resize 224 |
| T2 | T1 | Train crop scale floor từ 0.70 lên 0.90 |
| T3 | T2 | Horizontal flip probability từ 0.50 xuống 0.00 |
| T4 | T2 | Random Erasing probability từ 0.10 xuống 0.00 |

T3 và T4 là sibling của T2 để mỗi ablation chỉ đổi một biến. Tất cả recipe giữ
light ColorJitter và ImageNet normalization.

### 10.4 Full Train audit

2.034 Train asset nhân 5 recipe tạo 10.170 transformation checks:

| Recipe | Mean area retained | Minimum | Eval area | Flips | Erasing |
|---|---:|---:|---:|---:|---:|
| T0 | 80.12% | 70.02% | 76.56% | 1.005 | 224 |
| T1 | 80.12% | 70.02% | 100.00% | 1.005 | 224 |
| T2 | 94.29% | 90.01% | 100.00% | 1.005 | 224 |
| T3 | 94.29% | 90.01% | 100.00% | 0 | 224 |
| T4 | 94.29% | 90.01% | 100.00% | 1.005 | 0 |

R6 ghi nhận zero generated black-border artifact và zero non-finite normalized
tensor. Random seeds được derive từ seed 42, asset ID và operation name, nên
shared operation giữa parent/child nhận cùng random draw.

R6 sinh 50 contact sheet, mỗi recipe-class một sheet với ba cột: original, Train
transform và evaluation transform.

### 10.5 Gate

T3 có vẻ an toàn hơn về text reversal nhưng không được authorize bằng nhận xét
định tính. Cả 50 contact-sheet review vẫn pending, Validation chưa được dùng và
`selected_recipe_id = null`. R6 blocked và không execute real training.

## 11. R7 - Training Mathematics and Offline Checkpoint Correctness

### 11.1 Bug được sửa về mặt logic

Canonical weighted CrossEntropy path từng cộng:

```text
loss.item() * batch_size
```

Với weighted CrossEntropy mean reduction, mẫu số đúng là tổng class weight của
target trong toàn epoch, không phải sample count. Vì vậy Train/Validation/Test
loss historical, scheduler signal và early-stopping signal có thể bị sai.

R7 tạo contract `exact_global_cross_entropy_mean`:

```text
epoch_loss = sum(unreduced_cross_entropy_numerator)
             / sum(exact_batch_denominator)
```

Mẫu số là target count cho unweighted CE và tổng target-class weight cho weighted
CE. Accumulator chỉ chia một lần sau khi cộng toàn epoch, nên invariant với cách
chia batch.

### 11.2 File đã thêm

- Policy: [`configs/r7_training_correctness_policy.json`](../configs/r7_training_correctness_policy.json)
- Module: [`src/practice_2_2/r7_training_correctness.py`](../src/practice_2_2/r7_training_correctness.py)
- Entrypoint: [`scripts/run_r7_training_correctness.py`](../scripts/run_r7_training_correctness.py)
- Documentation: [`docs/R7_TRAINING_CORRECTNESS.md`](../docs/R7_TRAINING_CORRECTNESS.md)
- Verification: [`docs/R7_VERIFICATION.json`](../docs/R7_VERIFICATION.json)
- Test: [`tests/test_r7_training_correctness.py`](../tests/test_r7_training_correctness.py)
- Artifact root: [`artifacts/new_work/r7_training_correctness_v1`](../artifacts/new_work/r7_training_correctness_v1)

### 11.3 Training contract mới

- Baseline là unweighted CrossEntropy, label smoothing 0.0.
- Class weighting và label smoothing phải là experiment riêng.
- Scheduler và early stopping chỉ nhận metric có split `Validation` và contract
  `exact_global_cross_entropy_mean`.
- ResNet18 checkpoint reload được construct với `weights=None`.
- Full state dict được strict-load và verify state hash.
- Head/backbone parameter group không overlap và log learning rate riêng.
- Frozen BatchNorm được giữ ở evaluation mode sau `model.train()`.
- Offline reload phải cho output tensor byte-identical.

R7 kiểm tra unweighted, weighted, smoothed và weighted-smoothed loss qua nhiều
batch partition. Maximum numerical difference chỉ ở mức floating-point khoảng
`1e-16`.

### 11.4 Gate

Toàn bộ R7 correctness checks pass. Tuy nhiên R7 policy cấm real training và cấm
Validation/Test content vì R6 chưa chọn recipe và Train manifest chưa authorize.
R7 tổng thể vẫn `blocked`.

## 12. R8 - Staged Transfer Learning

### 12.1 Mục tiêu

R8 thay immediate partial fine-tuning bằng staged protocol có state machine rõ
ràng, đồng thời giữ M0 để so sánh causal.

### 12.2 File đã thêm

- Policy: [`configs/r8_staged_transfer_policy.json`](../configs/r8_staged_transfer_policy.json)
- Module: [`src/practice_2_2/r8_staged_transfer.py`](../src/practice_2_2/r8_staged_transfer.py)
- Entrypoint: [`scripts/run_r8_staged_transfer.py`](../scripts/run_r8_staged_transfer.py)
- Documentation: [`docs/R8_STAGED_TRANSFER_LEARNING.md`](../docs/R8_STAGED_TRANSFER_LEARNING.md)
- Verification: [`docs/R8_VERIFICATION.json`](../docs/R8_VERIFICATION.json)
- Test: [`tests/test_r8_staged_transfer.py`](../tests/test_r8_staged_transfer.py)
- Artifact root: [`artifacts/new_work/r8_staged_transfer_s42_123_2026_v1`](../artifacts/new_work/r8_staged_transfer_s42_123_2026_v1)

### 12.3 Controlled experiments

| ID | Parent | Protocol |
|---|---|---|
| M0 | none | Immediate layer4 fine-tuning, dropout 0.20 |
| M1 | M0 | Thêm 4 head-warmup epochs |
| M2 | M1 | Chỉ đổi dropout 0.20 thành 0.35 |
| M3 | M1 | Chỉ đổi label smoothing 0.00 thành 0.05 |

Mọi experiment phải dùng seeds 42, 123, 2026 và cùng pretrained backbone state
hash cùng deterministic head initialization.

### 12.4 Stage state machine

1. Construct model từ authorized offline pretrained state.
2. Chỉ mở classifier head trong warmup.
3. Chạy đúng bốn warmup epoch.
4. Chọn best warmup checkpoint bằng Validation Macro F1, tie-break bằng corrected
   Validation loss.
5. Strict reload đúng best warmup state.
6. Sau đó mới unfreeze `layer4` và `fc`.
7. Recreate optimizer/scheduler/early stopping cho fine-tune stage.

Hyperparameter predeclared:

| Item | Giá trị |
|---|---:|
| Head LR | `5e-4` |
| Backbone LR | `2e-5` |
| Weight decay | `2e-4` |
| Scheduler factor | 0.5 |
| Scheduler patience | 2 |
| Early-stopping patience | 4 |

### 12.5 Repeated-seed selection logic

R8 yêu cầu đủ 4 experiment x 3 seed, không duplicate, cùng initial-state hash và
chỉ chứa Train/Validation field.

Một staged candidate chỉ eligible nếu:

- mean Validation Macro F1 tăng ít nhất 0.005;
- mean Validation Accuracy không giảm quá 0.01;
- mean generalization gap giảm ít nhất 0.02 so với M0.

Test field làm validation fail ngay.

### 12.6 Gate

Protocol checks pass bằng offline synthetic fixture. Không có pretrained state
authority, selected transform, authorized Train manifest hoặc repeated-seed
Validation evidence thật. `authorized_candidate_experiment_id = null`; R8 vẫn
blocked và không train project data.

## 13. R9 - Feature Extractor Ceiling

### 13.1 Mục tiêu

R9 chỉ cho phép so sánh backbone sau khi R8 có candidate thật. Controlled change
hiện tại là ResNet18 sang EfficientNet-B0 ở cùng resolution 224 và cùng staged
protocol.

### 13.2 File đã thêm

- Policy: [`configs/r9_feature_extractor_ceiling_policy.json`](../configs/r9_feature_extractor_ceiling_policy.json)
- Module: [`src/practice_2_2/r9_feature_extractor_ceiling.py`](../src/practice_2_2/r9_feature_extractor_ceiling.py)
- Entrypoint: [`scripts/run_r9_feature_extractor_ceiling.py`](../scripts/run_r9_feature_extractor_ceiling.py)
- Documentation: [`docs/R9_FEATURE_EXTRACTOR_CEILING.md`](../docs/R9_FEATURE_EXTRACTOR_CEILING.md)
- Verification: [`docs/R9_VERIFICATION.json`](../docs/R9_VERIFICATION.json)
- Test: [`tests/test_r9_feature_extractor_ceiling.py`](../tests/test_r9_feature_extractor_ceiling.py)
- Artifact root: [`artifacts/new_work/r9_resnet18_vs_efficientnet_b0_s42_123_2026_v1`](../artifacts/new_work/r9_resnet18_vs_efficientnet_b0_s42_123_2026_v1)

### 13.3 Architecture registry

| ID | Architecture | Final block | Image size |
|---|---|---|---:|
| A0 | ResNet18 | `layer4` | 224 |
| A1 | EfficientNet-B0 | `features.7_and_8` | 224 |

Higher-resolution experiment, OCR và image-text feature đang disable để không
trộn nhiều controlled variables.

R9 adapter hỗ trợ head-only warmup, final-block unfreeze, frozen BatchNorm,
named LR groups, `weights=None` reload và strict output equality.

### 13.4 Engineering profile

Profile dùng synthetic input `1x3x224x224` trên CPU. Đây không phải Accuracy
evidence:

| Architecture | Parameters | Trainable | Parameter memory | Mean latency |
|---|---:|---:|---:|---:|
| ResNet18 | 11.181.642 | 8.398.858 | 44.726.568 bytes | 17.54 ms |
| EfficientNet-B0 | 4.020.358 | 1.142.202 | 16.081.432 bytes | 216.08 ms |

Latency chỉ phản ánh synthetic profiling ở environment của verification run.

### 13.5 Selection contract và gate

A1 chỉ advance khi mean Validation Macro F1 tăng ít nhất 0.005, Accuracy không
giảm quá 0.01 và gap giảm ít nhất 0.01. Train-only improvement bị reject.

Protocol/profile checks pass. Không có authorized R8 candidate, selected
transform, Train authority, pretrained comparator hash hoặc repeated-seed result
thật. `authorized_architecture_experiment_id = null`; R9 blocked.

## 14. R10 - Robust Validation Model Selection

### 14.1 Mục tiêu

R10 loại bỏ việc chọn model từ một seed hoặc một peak epoch. Phase này định nghĩa
selection statistic và freeze contract trước khi R11 được phép thấy Test.

### 14.2 File đã thêm

- Policy: [`configs/r10_robust_model_selection_policy.json`](../configs/r10_robust_model_selection_policy.json)
- Module: [`src/practice_2_2/r10_robust_model_selection.py`](../src/practice_2_2/r10_robust_model_selection.py)
- Entrypoint: [`scripts/run_r10_robust_model_selection.py`](../scripts/run_r10_robust_model_selection.py)
- Documentation: [`docs/R10_ROBUST_MODEL_SELECTION.md`](../docs/R10_ROBUST_MODEL_SELECTION.md)
- Verification: [`docs/R10_VERIFICATION.json`](../docs/R10_VERIFICATION.json)
- Test: [`tests/test_r10_robust_model_selection.py`](../tests/test_r10_robust_model_selection.py)
- Artifact root: [`artifacts/new_work/r10_robust_validation_selection_s42_123_2026_v1`](../artifacts/new_work/r10_robust_validation_selection_s42_123_2026_v1)

### 14.3 Metric contract

R10 yêu cầu seeds `[42, 123, 2026]` và ít nhất hai finalist.

Ranking order:

1. Mean Validation Macro F1.
2. Mean Validation Accuracy.
3. Lower mean Train/Validation generalization gap.

Accuracy, precision, recall, F1 và support được derive trực tiếp từ ordered 10x10
confusion matrix. Result row không được tự cung cấp scalar metric thay thế.

Mỗi finalist summary chứa:

- mean và sample standard deviation theo seed;
- per-seed Wilson 95% Accuracy interval;
- descriptive pooled Wilson interval;
- per-class recall mean/std/min/max;
- row-normalized confusion stability;
- mean generalization gap.

### 14.4 Consistent winner rule

Leader phải thắng ít nhất 2/3 seed và hơn runner-up mean Macro F1 ít nhất 0.003.
Một model có mean cao nhưng chỉ thắng một seed không được chọn.

R10 reject missing seed, duplicate seed, mixed config hash, mixed initialization
hash, invalid confusion matrix và mọi Test-derived field.

### 14.5 Checkpoint freeze

Sau khi có winner thật, checkpoint cuối được predeclare là checkpoint seed 42
được chọn bằng best Validation Macro F1. Freeze record phải xác minh:

- selection-record SHA-256;
- configuration SHA-256;
- model-state SHA-256;
- checkpoint-file SHA-256.

Tampered selection, config mismatch hoặc checkpoint mutation đều fail.

### 14.6 Gate

Selection và freeze logic pass bằng synthetic fixtures. `uncertainty_report.json`
là fixture, không phải measured Validation result.

Hiện:

| Trường | Giá trị |
|---|---|
| Repeated-seed finalist results | `false` |
| Selected finalist | `null` |
| Final checkpoint frozen | `false` |
| Test evidence used | `false` |
| Validation/Test content reads | `0/0` |
| Real training | `false` |

R10 blocked và không authorize R11.

## 15. R11 - One-Time Test Evaluation

### 15.1 Trạng thái

R11 chưa được triển khai và chưa được thực thi. Không có:

- `configs/r11_*.json`;
- `src/practice_2_2/r11_*.py`;
- `scripts/run_r11_*.py`;
- `docs/R11_VERIFICATION.json`;
- new-lineage Test metric;
- new-lineage confusion/calibration artifact.

New-lineage Test evaluation count vẫn bằng 0.

### 15.2 Điều kiện trước khi được thực thi

R11 chỉ được phép chạy sau khi:

1. R0-R10 gate pass theo đúng lineage.
2. Một finalist thật đã thắng repeated-seed Validation selection.
3. Final configuration và checkpoint đã freeze hash.
4. Test partition được chứng minh chưa dùng cho tuning.
5. One-time token được authorize.
6. Model state hash được kiểm tra trước và sau inference.

R11 phải report Accuracy, Macro F1, per-class metrics, uncertainty, confusion,
calibration và evaluation count bằng đúng một. Không được retrain hoặc reselect
sau khi xem Test.

## 16. R12 - New Artifact-Only Report

### 16.1 Phần đã triển khai

R12 đã thêm:

- Policy: [`configs/r12_artifact_report_policy.json`](../configs/r12_artifact_report_policy.json)
- Module: [`src/practice_2_2/r12_artifact_report.py`](../src/practice_2_2/r12_artifact_report.py)
- Builder: [`scripts/build_r12_artifact_report.py`](../scripts/build_r12_artifact_report.py)
- Documentation: [`docs/R12_ARTIFACT_ONLY_REPORT.md`](../docs/R12_ARTIFACT_ONLY_REPORT.md)
- Test declaration: [`tests/test_r12_artifact_report.py`](../tests/test_r12_artifact_report.py)
- Partial artifact root: [`artifacts/new_work/r12_artifact_report_v1`](../artifacts/new_work/r12_artifact_report_v1)

`requirements.txt` được bổ sung `torch`, `torchvision`,
`opencv-python-headless`, `nbformat`, `nbclient` và `nbconvert` so với HEAD hiện
tại của repository. Ba dependency cuối phục vụ notebook execution/export R12.

### 16.2 Report policy

R12 yêu cầu predecessor R11 đã complete và cấm:

- training;
- Test evaluation;
- Test loader construction;
- network access;
- persisted artifact write từ bên trong notebook;
- canonical notebook mutation;
- successor authorization.

Source registry bao phủ 11 verification report R0-R10 và 12 evidence artifact,
tổng cộng 23 source path. Registry verification hiện pass 23/23 path, size và
SHA-256 tại thời điểm nó được tạo.

### 16.3 Notebook design đã code

In-memory notebook builder tạo 17 cell, gồm 8 code cell. Nội dung dự kiến:

1. Lineage status và fail-closed contract.
2. R0-R10 status table.
3. R1/R2/R5 data-quality decisions.
4. R3 visual components và provenance limitation.
5. R4 leakage proof cùng unresolved review-only edges.
6. R6-R10 transform/training/selection protocol status.
7. Repeated-seed/uncertainty limitation.
8. R11 absence và zero Test evaluation.
9. SHA-256 verification cho mọi registered artifact.
10. Blocked conclusion không authorize phase khác.

AST safety audit trên notebook in-memory ghi nhận:

| Check | Kết quả |
|---|---|
| Imports | `hashlib`, `json`, `pathlib` |
| Forbidden imports | 0 |
| Forbidden training calls | 0 |
| Training operations detected | `false` |
| Test evaluation operations detected | `false` |
| Internal anchors | 8 |
| Internal links | 8 |
| Missing link targets | 0 |

### 16.4 Phần chưa hoàn thành

Lần chạy builder đầu tiên dừng khi Jupyter kernel cần bind local socket trong
sandbox:

```text
PermissionError: [Errno 1] Operation not permitted
```

Lượt xin quyền chạy kernel cục bộ sau đó bị abort. Vì vậy các file sau chưa tồn
tại tại snapshot này:

- `notebooks/05_product_visualsafe_report.ipynb`;
- `reports/html/practice_2_2_product_visualsafe_report.html`;
- `docs/R12_VERIFICATION.json`;
- R12 notebook execution/safety/link/hash reports cuối;
- R12 artifact manifest cuối.

Partial artifact hiện chỉ có:

- `policy_snapshot.json`;
- `source_artifact_registry.json`;
- `registry_verification.json`.

R12 không được ghi là completed hoặc passed.

### 16.5 Ảnh hưởng của canonical notebook drift lên R12

R12 policy đang freeze canonical notebook hash:

```text
a4eb294b3c4e8ec9619255f4acdd529094c0abd47eed9a263cf8bc96a7a8ff0d
```

Current notebook hash tại 09:30:29 ngày 06/08/2026 là:

```text
45ebe8683be6cd7561fd2cf6ef2e43c1f112283cfcd8ac9b82b7328331473920
```

Do đó ngay cả khi được cấp quyền chạy kernel, R12 builder hiện phải fail trước
execution vì canonical hash không còn khớp policy. Không được sửa expected hash
âm thầm. Cần quyết định authority rõ ràng ở R0 trước.

## 17. Flow của canonical notebook hiện tại

### 17.1 Vai trò

[`notebooks/04_canonical_report.ipynb`](../notebooks/04_canonical_report.ipynb)
là historical artifact-only presentation, không phải notebook để train lại model.
Metadata vẫn khai báo:

```text
mode = canonical_report_only
training_allowed = false
final_test_evaluation_allowed = false
```

Notebook có 40 cell, trong đó 19 code cell, theo 12 phase:

| Notebook phase | Nội dung |
|---|---|
| Phase 1 | Problem Definition |
| Phase 2 | Environment Setup và canonical artifact resolution |
| Phase 3 | Data provenance, cleaning và historical split summary |
| Phase 4 | Historical EDA |
| Phase 5 | Historical preprocessing |
| Phase 6 | Historical model building và E1 baseline |
| Phase 7 | Historical training dynamics |
| Phase 8 | E1-E4 controlled experiment comparison |
| Phase 9 | Validation-only model selection/checkpoint verification |
| Phase 10 | Load persisted one-time Final Test evidence |
| Phase 11 | Historical error analysis/visualization |
| Phase 12 | Save/load verification, limitation và conclusion |

### 17.2 Flow đúng khi artifact đầy đủ

```text
Resolve active registry
    -> Verify immutable canonical resources and guard
    -> Load persisted manifests and JSON/CSV artifacts
    -> Render provenance/split/EDA tables
    -> Load persisted histories and experiment comparison
    -> Load final selection and frozen checkpoint metadata
    -> Load persisted one-time Final Test result
    -> Render stored figures/error analysis
    -> Verify save/load and report limitations
```

Không cell nào được phép train hoặc lặp Final Test. Nếu artifact thiếu, notebook
phải fail thay vì regenerate.

### 17.3 Trạng thái notebook tại snapshot

Current file có 19/19 code cell với `execution_count = null` và tổng output count
bằng 0. Git diff cho thấy persisted outputs đã bị clear. File size hiện là 31.056
bytes và hash là `45ebe...`.

R7-R10 verification được tạo khi notebook hash là `a4eb...`; vì vậy claim
`canonical_notebook_mutated = false` của từng run có nghĩa notebook không đổi
trong chính execution window đó. Nó không chứng minh notebook chưa đổi sau này.

R0 cũng cho biết nhiều canonical resource được README mô tả không còn trong
active checkout. Vì vậy current notebook không nên Run All để tạo lại historical
evidence và tuyệt đối không được dùng Test path để bù file thiếu.

## 18. Flow tổng thể của new lineage

### 18.1 Data flow

```text
data/final/data_clean_balanced
    -> R2 legacy inventory and semantic queue
    -> R3 deterministic product/visual components
    -> R4 candidate component-level 70/15/15 manifest
    -> R5 Train-only data-readiness audit
    -> R6 Train-only transform audit
```

Không có image nào được copy, move, relabel hoặc delete tự động. Label decision
sau review phải nằm trong audit artifact, còn raw image giữ immutable.

### 18.2 Model-development flow

```text
Authorized Train manifest and selected R6 transform
    -> R7 corrected unweighted baseline contract
    -> R8 M0-M3 staged transfer experiments over 3 seeds
    -> R9 A0/A1 controlled architecture comparison over 3 seeds
    -> R10 robust Validation finalist selection
    -> Freeze configuration/model/checkpoint hashes
    -> R11 one-time Test
    -> R12 artifact-only report
```

Current execution mới dừng ở protocol verification. Chưa có authorized Train
run trong R7-R10.

### 18.3 Split/content access matrix

| Phase | Train content | Validation content | Test content | Model optimization |
|---|---:|---:|---:|---:|
| R0 | 0 | 0 | 0 | 0 |
| R1 | Pilot image review | Blinded | Blinded | 0 |
| R2 | Full legacy audit | Không dùng split | Không dùng split | 0 |
| R3 | 2.896 original images để grouping | Không dùng model split | Không dùng model split | 0 |
| R4 | Manifest assignment only | Manifest assignment only | Manifest assignment only | 0 |
| R5 | 2.034 images | 0 | 0 | 0 |
| R6 | 2.034 images | 0 | 0 | 0 |
| R7 | Synthetic tensors | 0 | 0 | 0 real steps |
| R8 | Synthetic fixture | 0 | 0 | 0 real steps |
| R9 | Synthetic input | 0 | 0 | 0 real steps |
| R10 | Synthetic metric fixture | 0 | 0 | 0 real steps |
| R11 | Chưa chạy | Chưa chạy | 0 | 0 |
| R12 | Persisted JSON only | Persisted status only | Không evaluate | 0 |

## 19. So sánh logic trước và sau refactor

| Khu vực | Trước | Sau refactor |
|---|---|---|
| Authority | Path/artifact tồn tại được ngầm tin | R0 inventory, classification, size/hash gate |
| Label | Folder/query gần như ground truth | R1 visual-content contract và human decisions |
| Provenance | Không có product ID/URL per image | R2 immutable metadata contract cho crawl mới |
| Grouping | Hash/perceptual grouping chưa chứng minh product | R3 product + raw/pixel hash + pHash/dHash/SSIM/embedding |
| Split | Historical split khó re-audit | R4 deterministic whole-component 70/15/15 candidate |
| EDA | Chủ yếu class count và hình mẫu | R5 Train-only purity, cluster, neighbor, template audit |
| Evaluation leakage | Không có access manifest chi tiết | Content access manifest và split guard |
| Transform | Nhiều generic augmentation cùng lúc | R6 T0-T4 one-variable domain-safe ablation |
| Epoch loss | Weighted batch mean nhân batch size | R7 exact numerator/denominator accumulator |
| Baseline loss | Weighted CE + smoothing mặc định | Unweighted CE baseline; weight/smoothing là experiment |
| Checkpoint reload | Có thể construct pretrained model | `weights=None`, strict full-state offline reload |
| Fine-tuning | Immediate layer4 | R8 head warmup, best restore, layer4 unfreeze |
| Model capacity | Chủ yếu ResNet18 experiments | R9 controlled ResNet18/EfficientNet-B0 ceiling test |
| Selection | Một seed, one peak | R10 3 seeds, uncertainty, consistency rule |
| Test | Historical one-time result đã xem | R11 guard cho đúng một new-lineage evaluation |
| Reporting | Historical canonical notebook | R12 new artifact-only report, old lineage giữ nguyên |

## 20. Bản đồ file theo phase

| Phase | Config | Core module | Entrypoint | Verification/Test |
|---|---|---|---|---|
| R0 | `r0_authority_policy.json` | `r0_authority.py` | `verify_r0_authority.py` | `R0_AUTHORITY_INVENTORY.json`, `test_r0_authority.py` |
| R1 | `r1_label_contract.json` | `r1_contract.py` | `build_r1_pilot.py`, `verify_r1_contract.py` | `R1_VERIFICATION.json`, `test_r1_contract.py` |
| R2 | `r2_provenance_policy.json` | `r2_provenance.py`, `r2_semantic_audit.py` | `collect_tiki_r2.py`, `run_r2_audit.py` | `R2_VERIFICATION.json`, `test_r2_provenance_semantic_audit.py` |
| R3 | `r3_grouping_policy.json` | `r3_grouping.py` | `run_r3_grouping.py` | `R3_VERIFICATION.json`, `test_r3_product_visual_grouping.py` |
| R4 | `r4_split_policy.json` | `r4_split.py` | `run_r4_split.py` | `R4_VERIFICATION.json`, `test_r4_stratified_group_split.py` |
| R5 | `r5_data_readiness_policy.json` | `r5_data_readiness.py` | `run_r5_data_readiness.py` | `R5_VERIFICATION.json`, `test_r5_train_only_data_readiness.py` |
| R6 | `r6_transform_ablation_policy.json` | `r6_transform_ablation.py` | `run_r6_transform_ablation.py` | `R6_VERIFICATION.json`, `test_r6_transform_ablation.py` |
| R7 | `r7_training_correctness_policy.json` | `r7_training_correctness.py` | `run_r7_training_correctness.py` | `R7_VERIFICATION.json`, `test_r7_training_correctness.py` |
| R8 | `r8_staged_transfer_policy.json` | `r8_staged_transfer.py` | `run_r8_staged_transfer.py` | `R8_VERIFICATION.json`, `test_r8_staged_transfer.py` |
| R9 | `r9_feature_extractor_ceiling_policy.json` | `r9_feature_extractor_ceiling.py` | `run_r9_feature_extractor_ceiling.py` | `R9_VERIFICATION.json`, `test_r9_feature_extractor_ceiling.py` |
| R10 | `r10_robust_model_selection_policy.json` | `r10_robust_model_selection.py` | `run_r10_robust_model_selection.py` | `R10_VERIFICATION.json`, `test_r10_robust_model_selection.py` |
| R11 | Chưa có | Chưa có | Chưa có | Chưa có |
| R12 | `r12_artifact_report_policy.json` | `r12_artifact_report.py` | `build_r12_artifact_report.py` | Chưa sinh report cuối, `test_r12_artifact_report.py` đã khai báo |

## 21. Artifact inventory

| Lineage directory | File count | Dung lượng xấp xỉ | Nội dung chính |
|---|---:|---:|---|
| `r2_provenance_semantic_audit_v1` | 5 | 4.42 MiB | Inventory, pair audit, semantic queue |
| `r3_product_visual_groups_v1` | 10 | 54.54 MiB | Embeddings, edges, groups, review queue |
| `v3_product_visual_group_s42_v1` | 9 | 3.52 MiB | Candidate split, fingerprint, leakage, Test guard |
| `r5_train_only_data_readiness_v1` | 41 | 16.18 MiB | EDA JSON và 31 PNG |
| `r6_transform_ablation_s42_v1` | 58 | 31.77 MiB | Transform audit và 50 PNG |
| `r7_training_correctness_v1` | 7 | 0.01 MiB | Loss/checkpoint/control verification |
| `r8_staged_transfer_s42_123_2026_v1` | 7 | 0.02 MiB | Protocol and synthetic selection verification |
| `r9_resnet18_vs_efficientnet_b0_s42_123_2026_v1` | 8 | 0.01 MiB | Architecture profiles/protocol verification |
| `r10_robust_validation_selection_s42_123_2026_v1` | 7 | 0.01 MiB | Robust selection/freeze fixture reports |
| `r12_artifact_report_v1` | 3 partial | 0.01 MiB | Policy, source registry, registry verification |

## 22. Verification snapshot ngày 06/08/2026

### 22.1 Source validation

Các module R0-R10, R12 module và R12 builder compile thành công bằng Python 3.10
venv.

R12 in-memory checks:

- policy/registry tests: 2 passed;
- source registry: 23/23 path verified;
- notebook AST safety: passed;
- internal link audit: 8/8 links resolved;
- 10 R12 tests được collect;
- 8 test phụ thuộc generated notebook/HTML/report chưa thể chạy hoàn chỉnh.

### 22.2 R0-R10 test suite

Current targeted suite collect 139 test cases sau parametrization:

```text
135 passed
4 failed
```

Bốn failure duy nhất:

- `test_r7_did_not_mutate_canonical_notebook`;
- `test_r8_did_not_mutate_canonical_notebook`;
- `test_r9_did_not_mutate_canonical_notebook`;
- `test_r10_did_not_mutate_canonical_notebook`.

Tất cả fail cùng một nguyên nhân: report freeze hash `a4eb...`, current notebook
hash `45ebe...`. Không có failure nào trong loss math, split logic, data isolation,
transform logic, staged protocol, architecture selection hoặc robust selection.

### 22.3 Dependency verification

Để tái chạy test trong đúng deep-learning venv, `pytest` và
`opencv-python-headless` đã được cài vào environment. `opencv-python-headless` đã
có trong requirements; `pytest` hiện là test-runner cài trong environment nhưng
chưa được thêm vào project requirements.

Jupyter execution cần local kernel socket. Sandbox hiện chặn thao tác đó nếu
không được cấp quyền bên ngoài sandbox.

## 23. Current blockers cần xử lý

### 23.1 Authority drift

Đầu tiên phải quyết định current output-cleared canonical notebook có phải thay
đổi được chủ project chấp thuận hay không.

Hai hướng hợp lệ:

1. Restore đúng verified frozen notebook từ backup có hash authority được duyệt.
2. Nếu output-cleared notebook là authority mới có chủ đích, tạo một authority
   migration record mới, giải thích thay đổi, re-freeze hash và regenerate các
   verification phụ thuộc.

Không được chỉ sửa JSON hash để làm test xanh mà không có authority decision.

### 23.2 Data and review gates

Thứ tự bắt buộc:

1. Restore hoặc xử lý chính thức các R0 authority còn thiếu.
2. Hoàn thành hai R1 independent reviews và adjudication.
3. Hoàn thành semantic review 2.896 ảnh và provenance cho accepted data.
4. Thu thập thêm product nếu class chưa đủ 100 unique products.
5. Review 2.325 R3 uncertain/cross-label edges.
6. Rebuild R3 component và R4 split sau decision.
7. Chứng minh zero confirmed cross-split leakage.
8. Hoàn thành 2.034 R5 Train semantic reviews.
9. Resolve cluster/source-template concentration.
10. Hoàn thành 50 R6 contact-sheet reviews.
11. Dùng Validation hợp lệ để chọn đúng một transform recipe.

### 23.3 Model gates

Sau khi data gate pass:

1. Chạy corrected unweighted R7 baseline trên authorized Train/Validation.
2. Chạy M0-M3 R8 ở seeds 42, 123, 2026 từ cùng initial state.
3. Chỉ authorize staged candidate khi metric và gap thresholds pass.
4. Chạy controlled A0/A1 R9 với cùng split/transform/seeds/protocol.
5. Chạy R10 finalist selection từ real Validation confusion matrices.
6. Freeze finalist config, model state và checkpoint hash.
7. Xin authorize R11 đúng một lần.
8. Sau R11 mới hoàn thiện R12 notebook/HTML.

## 24. Điều không được kết luận ở thời điểm hiện tại

Không được tuyên bố bất kỳ điều nào sau đây:

- V3 candidate split đã được authorize để train;
- 2.462 component tương đương 2.462 unique product;
- T3 hoặc T4 đã thắng transform selection;
- R7-R10 đã train model thật;
- EfficientNet-B0 chính xác hơn ResNet18;
- synthetic R10 uncertainty là measured Validation result;
- một finalist đã được chọn;
- final checkpoint mới đã freeze;
- R11 đã evaluate Test;
- R12 report notebook đã hoàn tất;
- historical 76.36% là kết quả của V3 lineage.

## 25. Trạng thái cuối của refactor tại thời điểm tổng hợp

Refactor đã hoàn thành một nền tảng kỹ thuật lớn: authority audit, label contract,
provenance schema, semantic queue, deterministic visual grouping, candidate
component split, Train-only EDA, transform ablation, corrected loss, staged
transfer state machine, controlled architecture adapter, repeated-seed selector,
checkpoint freeze contract và artifact-only report builder.

Tuy nhiên project đang ở trạng thái `implemented but intentionally blocked`.
Pipeline đang làm đúng nhiệm vụ khi từ chối train/chọn model/Test trong lúc data
authority và human review chưa hoàn chỉnh.

Mốc kỹ thuật hiện tại:

```text
R0-R6: audit artifacts đã sinh, data gates còn blocked
R7-R10: protocol checks pass, real model evidence chưa có
R11: chưa thực thi
R12: registry pass, notebook execution/export chưa hoàn thành
Canonical notebook: current hash drift cần authority decision
New-lineage Test evaluation count: 0
```

Đây là trạng thái nguồn sự thật cần dùng để tiếp tục project. Mọi bước tiếp theo
phải bắt đầu bằng xử lý canonical authority drift và predecessor data gates,
không bắt đầu bằng tuning model hoặc chạy Test.
