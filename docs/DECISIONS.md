# DECISIONS — Nhật ký quyết định

Một dòng mỗi quyết định: ngày · task · quyết định · lý do (kèm số liệu nếu có) · ai quyết.
Mới nhất ở trên.

| Ngày | Task | Quyết định | Lý do | Ai |
|---|---|---|---|---|
| 2026-10-12 | Demo | Giấy phép mã nguồn đổi sang MIT | Ngắn gọn, phổ biến cho repo nghiên cứu cá nhân; README và pyproject đã cập nhật | Opus |
| 2026-10-12 | Demo | Demo tổng hợp: HardMine không tách biệt khỏi baseline bất định ở tầng clip (macro-F1 test 0.905 so với 0.913/0.919) | Dữ liệu tổng hợp thiếu cấu trúc thời gian thật; giữ nguyên kết quả, không chọn lại seed. Câu trả lời thật ở G3 | Opus |
| 2026-10-12 | Demo | Tầng sự kiện tổng hợp: hardmine_event 100% trúng lỗi; bất định frame trung bình 48% (< ngẫu nhiên 57%) | 100% là theo cấu tạo của bộ sinh; phát hiện đáng giữ là bất định frame trung bình bỏ sót ca cắt vụn | Opus |
| 2026-10-12 | Khởi tạo | Test pipeline tích hợp đo **Lift chuẩn hoá** thay vì Lift thô | Với tỷ lệ lỗi nền 62%, trần Lift = 1,61; ngưỡng Lift > 1,5 là đòi gần hoàn hảo | Opus |
| 2026-10-12 | Khởi tạo | Dữ liệu tổng hợp mặc định `class_strength=3.5`, `driver_strength=6.0` | Tỷ lệ lỗi nền ~19% gần thực tế hơn; embedding vẫn bị chi phối bởi tài xế | Opus |
| 2026-10-12 | Khởi tạo | Khử danh tính phải quét hạng r | Trên dữ liệu tổng hợp 14 tài xế, r = 8 chỉ hạ độ chính xác đoán ID xuống ~0,83; r = 13 xoá gần hết | Opus |
| 2026-10-12 | Khởi tạo | Giữ tín hiệu thời gian S4 dù vô ích trên dữ liệu tổng hợp | Cửa sổ tổng hợp độc lập theo thời gian; bộ hiệu chuẩn tự gán trọng số nhỏ nhất (đã có test). Giá trị của S4 chỉ đo được trên video thật (H3) | Opus |

## Đang bị chặn

(trống)
