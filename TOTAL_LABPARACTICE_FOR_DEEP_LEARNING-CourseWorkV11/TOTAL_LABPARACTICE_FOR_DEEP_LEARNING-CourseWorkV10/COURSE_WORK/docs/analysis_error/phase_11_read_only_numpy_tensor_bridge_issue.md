# Phase 11 Read-Only NumPy Tensor Bridge Issue

## Hiện tượng

Synthetic Dataset gate phát cảnh báo khi chuyển cửa sổ NumPy read-only sang Torch bằng `torch.from_numpy`.

## Nguyên nhân

Phase 11 khóa feature timeline read-only để ngăn mutation. Phase 10 materializer có thể trả một contiguous view vẫn kế thừa trạng thái read-only. `torch.from_numpy` chia sẻ memory với view này và cảnh báo rằng mutation qua Tensor có thể tạo undefined behavior.

## Phạm vi ảnh hưởng

Dataset shape, value, population và Test firewall chưa sai. Rủi ro nằm tại ownership boundary giữa NumPy và Torch.

## Yêu cầu sửa

Tensor đầu ra phải sở hữu memory float32 độc lập, không chia sẻ writable bridge với canonical feature timeline và không thay đổi bất kỳ upstream artifact nào.
