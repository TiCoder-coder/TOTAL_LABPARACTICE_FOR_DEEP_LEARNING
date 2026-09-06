# Notebook Cells Walkthrough Link Mapping Refactor Plan

## 1. Trạng thái plan

```text
Plan ID: NOTEBOOK-CELLS-WALKTHROUGH-LINK-MAPPING-REFACTOR-v1
Issue ID: NOTEBOOK-CELLS-WALKTHROUGH-LINK-MAPPING-v1
Status: WAITING_FOR_HUMAN_APPROVAL
Execution mode: STRICTLY_SEQUENTIAL
```

Plan này chỉ cho phép refactor tài liệu walkthrough. Chưa được sửa target trước khi Human duyệt.

## 2. Mục tiêu

- Đồng bộ toàn bộ cell mapping với notebook hiện tại.
- Đưa người đọc tới đúng walkthrough section và đúng notebook output cell.
- Loại bỏ cell number, cell ID, output count và source locator bị stale.
- Giữ nguyên notebook, output và scientific artifacts.
- Tạo một verification contract để mapping không tiếp tục lệch âm thầm.

## 3. Source of truth

```text
Notebook: notebook_course_work/CourseWork.ipynb
Walkthrough: docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md
Current-flow references: docs/current_flow/
```

Notebook JSON là authority cho:

- Cell index.
- Cell ID.
- Cell type.
- Markdown heading.
- Code source locator.
- Execution count.
- Output count.
- Output MIME types.

## 4. Kiến trúc tài liệu sau refactor

### 4.1. Navigation layer

Mục lục walkthrough dùng internal anchors để luôn điều hướng đúng tới phần giải thích trong chính file Markdown.

Mỗi entry phải có:

- Phase hoặc section name.
- Heading cell index và heading cell ID nếu có.
- Output code cell index.
- Output code cell ID.
- Internal walkthrough link.

### 4.2. Notebook locator layer

Mỗi section có một notebook locator thống nhất:

```text
Notebook path
Output cell index
Output cell ID
Source search token
Execution count
Output count
```

Nếu môi trường render hỗ trợ notebook cell fragment, link phải dùng exact output cell ID. Nếu Cursor không nhảy theo fragment, cùng section vẫn phải có file link và source search token chính xác. Không được tuyên bố một fragment là portable nếu chưa verify.

### 4.3. Explanation layer

Phần giải thích chỉ mô tả output thực sự tồn tại trong notebook snapshot. Không suy diễn metric, trạng thái hoặc artifact ngoài output và canonical current-flow evidence.

## 5. Trình tự thực thi

### Step 1. Preservation gate

1. Ghi lại SHA-256 của notebook và walkthrough.
2. Ghi cell count và kiểm tra cell ID duy nhất.
3. Xác minh notebook không thay đổi từ snapshot dùng để tạo mapping.
4. Nếu notebook thay đổi, tạo lại mapping trước khi sửa walkthrough.

### Step 2. Build canonical cell inventory

1. Đọc toàn bộ notebook JSON.
2. Lập inventory cho 151 cell theo index và exact ID.
3. Xác định phase heading và output-producing code cell.
4. Ghi output count và MIME type của từng code cell.
5. Dùng source line đầu tiên làm search token nếu không có stable named locator.

### Step 3. Correct global structure

1. Cập nhật tổng cell count thành 151.
2. Sửa nhóm Phase 48–59 theo vị trí thực tế.
3. Đặt supplementary MAPE tại cell 126 theo notebook hiện tại.
4. Sửa Phase 44 thành cell 118–119.
5. Sửa Phase 59 thành cell 149–150.
6. Xóa mọi tham chiếu tới cell 151 không tồn tại.

### Step 4. Refactor table of contents

1. Thay các file-level duplicate links bằng internal section links.
2. Mỗi entry hiển thị đúng heading range và output cell.
3. Không dùng cùng một notebook link làm đích duy nhất cho 130 entry.
4. Giữ cách phân nhóm Foundation, Modeling, Sweeps, Final Pipeline và Analysis nếu còn khớp current flow.

### Step 5. Refactor every section locator

1. Thêm exact output cell ID cho từng section.
2. Thêm exact source search token.
3. Liên kết notebook bằng chiến lược đã verify ở target renderer.
4. Đối với phase có nhiều output cell như EDA, map riêng từng subsection tới đúng code cell.
5. Không gắn link subsection vào markdown heading nếu mục tiêu được mô tả là output.

### Step 6. Correct content inconsistencies

1. Đổi sign-off status mô tả từ `OK` sang `PASS` khi canonical output dùng `PASS`.
2. Đồng bộ MAPE position và role với notebook hiện tại.
3. Đồng bộ Phase 59 với code cell thực tế.
4. Sửa related-document links sang `../current_flow/`.
5. Loại bỏ decorative icon trong phần được refactor.

### Step 7. Immediate verification

1. Parse lại notebook và walkthrough.
2. Kiểm tra mọi documented cell index tồn tại.
3. Kiểm tra mọi documented cell ID tồn tại và khớp đúng index.
4. Kiểm tra mọi output locator trỏ tới code cell có ít nhất một output.
5. Kiểm tra mọi phase heading khớp source notebook.
6. Kiểm tra mọi relative file link tồn tại.
7. Kiểm tra không còn `Cell 151`, `Cell 119-119` hoặc mapping MAPE 149–150.
8. Kiểm tra không còn 130 file-level duplicate links dùng thay cho output locator.

### Step 8. Preservation verification

1. Notebook SHA-256 phải không đổi.
2. Notebook cell count, cell IDs, execution counts và outputs phải không đổi.
3. Không source code hoặc artifact nào bị thay đổi.
4. `git diff --check` phải PASS cho walkthrough.
5. Chỉ walkthrough và execution report được phép có thay đổi mới.

## 6. File được phép sửa

```text
docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md
docs/plan/plan_to_refactor&fix/notebook_cells_walkthrough_link_mapping_refactor_result.md
```

## 7. File không được sửa

```text
notebook_course_work/CourseWork.ipynb
src/course_work/**
tests/**
artifacts/**
docs/save_log_in_processing/**
docs/rule_base/**
docs/current_flow/**
```

## 8. Stop conditions

- Notebook thay đổi trong lúc mapping được tạo.
- Có cell ID trùng hoặc thiếu.
- Không xác định được output owner của một section.
- Deep-link syntax không hoạt động trong target renderer.
- Walkthrough mô tả scientific result không tồn tại trong notebook.
- Refactor yêu cầu sửa notebook hoặc source code.

## 9. Acceptance criteria

- 151 cell được đối chiếu từ notebook authority.
- Mọi phase và EDA subsection map đúng cell hiện tại.
- Mọi output link dùng đúng output-producing code cell ID hoặc locator fallback đã verify.
- Mục lục không còn mở cùng một vị trí notebook cho mọi mục.
- Phase 44, MAPE, Phase 48–59 và related-document links được sửa đúng.
- Notebook và toàn bộ output giữ nguyên byte.
- Không có code, comment hoặc icon mới.

## 10. Approval gate

```text
ISSUE_CONFIRMED=true
PRE_PROCESS_PLAN_CREATED=true
IMPLEMENTATION_STARTED=false
WAITING_FOR_HUMAN_APPROVAL=true
```
