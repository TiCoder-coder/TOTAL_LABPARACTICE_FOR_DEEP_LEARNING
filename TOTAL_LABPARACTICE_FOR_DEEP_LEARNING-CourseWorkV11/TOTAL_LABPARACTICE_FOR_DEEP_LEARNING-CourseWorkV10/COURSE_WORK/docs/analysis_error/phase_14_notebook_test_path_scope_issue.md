# Phase 14 Notebook Test Path Scope Issue

## Trạng thái

RESOLVED_AFTER_SECOND_OCCURRENCE

## Lỗi quan sát được

Lệnh inspection được chạy với working directory là `COURSE_WORK` nhưng vẫn dùng path `COURSE_WORK/tests/integration/test_notebook_boundary.py`.

Kết quả:

`No such file or directory`

## Root cause

Path tương đối bị lặp project directory prefix.

## Ảnh hưởng

- Chưa đọc được phần cuối notebook boundary test.
- Không có source, notebook, registry hoặc artifact nào bị thay đổi bởi lệnh lỗi.

## Điều kiện đóng issue

- Dùng path `tests/integration/test_notebook_boundary.py` từ working directory hiện tại.
- Đọc được phần file còn lại.
- Notebook patch chỉ tiếp tục sau inspection thành công.

## Kết quả xác minh

- Path đã được sửa thành `tests/integration/test_notebook_boundary.py`.
- Phần cuối file được đọc thành công với exit code 0.

## Tái diễn lần 2

Một lệnh verify tiếp theo vẫn dùng `COURSE_WORK/notebook_course_work/CourseWork.ipynb` trong khi working directory đã là `COURSE_WORK`.

Vế jq dừng trước khi test được chạy. Không có file nào bị sửa bởi lệnh lỗi.

## Kiểm soát bổ sung

- Mọi target trong các lệnh còn lại của Bước 9 phải được viết tương đối trực tiếp từ `COURSE_WORK`.
- Notebook target phải là `notebook_course_work/CourseWork.ipynb`.
- Test target phải là `tests/...`.

## Kết quả xác minh cuối

- JQ inspection exit code 0 và xác định Phase 14 orchestration tại cell 58.
- Notebook boundary và phase summary suites đạt 28/28 test.
- Không có lần tái diễn thứ 3.
