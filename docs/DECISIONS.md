# DECISIONS — Nhật ký quyết định

Một dòng mỗi quyết định: ngày · task · quyết định · lý do (kèm số liệu nếu có) · ai quyết.
Mới nhất ở trên.

| Ngày | Task | Quyết định | Lý do | Ai |
|---|---|---|---|---|
| 2026-10-12 | Khởi tạo | Test pipeline tích hợp đo **Lift chuẩn hoá** thay vì Lift thô | Với tỷ lệ lỗi nền 62%, trần Lift = 1,61; ngưỡng Lift > 1,5 là đòi gần hoàn hảo | Opus |
| 2026-10-12 | Khởi tạo | Dữ liệu tổng hợp mặc định `class_strength=3.5`, `driver_strength=6.0` | Tỷ lệ lỗi nền ~19% gần thực tế hơn; embedding vẫn bị chi phối bởi tài xế | Opus |
| 2026-10-12 | Khởi tạo | Khử danh tính phải quét hạng r | Trên dữ liệu tổng hợp 14 tài xế, r = 8 chỉ hạ độ chính xác đoán ID xuống ~0,83; r = 13 xoá gần hết | Opus |
| 2026-10-12 | Khởi tạo | Giữ tín hiệu thời gian S4 dù vô ích trên dữ liệu tổng hợp | Cửa sổ tổng hợp độc lập theo thời gian; bộ hiệu chuẩn tự gán trọng số nhỏ nhất (đã có test). Giá trị của S4 chỉ đo được trên video thật (H3) | Opus |

## Đang bị chặn

(trống)
