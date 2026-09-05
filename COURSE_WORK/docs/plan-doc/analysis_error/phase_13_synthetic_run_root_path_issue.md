# Phase 13 Synthetic Run Root Path Issue

## Hiện tượng

Synthetic registry audit dừng khi đăng ký CONFIG và STATUS artifacts vì temporary `run_root` không nằm dưới project root hoặc registry root.

## Nguyên nhân

Path normalizer chưa đưa explicit `run_root` vào tập canonical base paths dù constructor cho phép tách run root để cô lập test.

## Phạm vi ảnh hưởng

Production registry và `artifacts/experiments` chưa được materialize. Config fingerprint, lineage validation và source import vẫn hợp lệ.

## Yêu cầu sửa

Path normalizer và resolver phải hỗ trợ project root, registry root và explicit run root, đồng thời vẫn từ chối path ngoài ba phạm vi này.
