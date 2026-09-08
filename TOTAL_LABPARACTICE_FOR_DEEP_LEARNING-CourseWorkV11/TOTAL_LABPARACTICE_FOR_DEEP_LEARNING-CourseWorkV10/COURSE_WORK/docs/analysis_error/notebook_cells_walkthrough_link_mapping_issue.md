# Notebook Cells Walkthrough Link Mapping Issue

## 1. Trạng thái

```text
Issue ID: NOTEBOOK-CELLS-WALKTHROUGH-LINK-MAPPING-v1
Status: CONFIRMED
Target: docs/link&discussion_to_result/NOTEBOOK_CELLS_WALKTHROUGH.md
Notebook authority: notebook_course_work/CourseWork.ipynb
Implementation status: NOT_STARTED
```

## 2. Snapshot được kiểm tra

```text
Notebook SHA-256: cd2a7f004998173057a66c2829211920fef99444fcd39831f2aa731b7269bcaf
Notebook cell count: 151
Unique cell IDs: 151
Walkthrough SHA-256: bb0a02a6b2c0fd88c2d68060cc35aaa9f06694903fb060a4a1e0bee3e91e6d10
```

## 3. Vấn đề đã xác nhận

### 3.1. Link không định vị output cell

Walkthrough có 130 liên kết cùng trỏ tới:

```text
../../notebook_course_work/CourseWork.ipynb
```

Các liên kết chỉ mở notebook và không mang thông tin định vị code cell tạo output. Vì vậy link của nhiều phase có cùng đích dù nội dung mô tả các output khác nhau.

### 3.2. Cell mapping bị stale sau khi di chuyển MAPE

Notebook hiện đặt MAPE renderer ở cell index 126 với cell ID `ac4fe7dd`. Sau vị trí này:

- Phase 48 ở cell 127–128.
- Phase 49 ở cell 129–130.
- Phase 50 ở cell 131–132.
- Phase 51 ở cell 133–134.
- Phase 52 ở cell 135–136.
- Phase 53 ở cell 137–138.
- Phase 54 ở cell 139–140.
- Phase 55 ở cell 141–142.
- Phase 56 ở cell 143–144.
- Phase 57 ở cell 145–146.
- Phase 58 ở cell 147–148.
- Phase 59 ở cell 149–150.

Walkthrough vẫn mô tả Phase 48 bắt đầu ở cell 126, Phase 59 ở cell 148, MAPE ở cell 149–150 và một Phase 59 dashboard riêng ở cell 151. Mapping này không còn đúng với notebook hiện tại.

### 3.3. Lỗi cell range cụ thể

Phase 44 được ghi là `Cell 119-119`; notebook thực tế có heading ở cell 118 và output code ở cell 119.

Walkthrough ghi notebook có cell 151, trong khi index cuối hiện tại là 150.

### 3.4. Cell ID chưa được mô tả chính xác

Một số section chỉ ghi dạng `phase-N ID` hoặc chỉ ghi cell number. Notebook có cell ID cụ thể và duy nhất cho từng cell. Thiếu exact ID làm mapping dễ lệch sau khi chèn hoặc di chuyển cell.

### 3.5. Related-document links sai thư mục

Hai file current-flow nằm trong `docs/current_flow/`, nhưng walkthrough dùng link tương đối như thể chúng nằm cùng `docs/link&discussion_to_result/`.

### 3.6. Mô tả trạng thái chưa đồng nhất

Một số đoạn dùng trạng thái `OK` cho sign-off trong khi contract hiện hành sử dụng `PASS`. Một số mô tả MAPE và Phase 59 vẫn phản ánh thứ tự notebook cũ.

## 4. Root cause

Walkthrough được duy trì bằng cell number viết tay và link file-level. Khi notebook chèn, xóa hoặc di chuyển cell, toàn bộ mapping phía sau điểm thay đổi bị lệch. Không có bước đối chiếu tự động giữa walkthrough và notebook JSON trước khi tài liệu được khóa.

## 5. Hướng sửa

- Dùng notebook JSON hiện tại làm nguồn duy nhất cho cell index, cell ID, code locator và output count.
- Dùng internal walkthrough anchors cho mục lục để người đọc đến đúng phần giải thích.
- Mỗi phase phải liên kết đến code cell có output, không liên kết chung chung đến heading cell.
- Mỗi section phải ghi cả cell index và exact cell ID.
- Giữ fallback search token vì Cursor và VS Code không bảo đảm deep-link tới notebook cell từ external Markdown trên mọi phiên bản.
- Không sửa notebook chỉ để làm walkthrough link hoạt động.

## 6. Phạm vi ảnh hưởng

Lỗi hiện tại chỉ làm sai navigation và mô tả tài liệu. Nó không thay đổi notebook output, scientific artifacts, phase sign-off hoặc model results.

## 7. Điều cấm

- Không thay đổi `CourseWork.ipynb` trong refactor tài liệu này.
- Không thay cell ID notebook.
- Không xóa notebook output.
- Không tạo link giả tuyên bố nhảy trực tiếp khi target renderer không hỗ trợ.
- Không viết lại scientific interpretation ngoài bằng chứng đang có trong notebook.
