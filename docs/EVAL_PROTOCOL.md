# EVAL_PROTOCOL — Luật đánh giá

Mọi kết quả công bố phải tuân thủ file này. Một kết quả vi phạm bất kỳ mục nào là không hợp lệ.

## 1. Chia dữ liệu
- Theo **người lái**. Drive&Act: 3 split chính thức. DMD và dataset khác: chia theo người, ghi danh sách vào
  `configs/splits/<dataset>.yaml`, cố định một lần.
- Không dùng tài xế test cho bất kỳ việc fit nào: head, nhiệt độ, khử danh tính, bộ hiệu chuẩn, ngưỡng.
- Báo cáo thêm leave-one-subject-out cho model mắt khi số người ≤ 20.

## 2. Thứ tự sử dụng tập
1. Tinh chỉnh mọi siêu tham số trên **val**.
2. Đóng băng config (commit) trước khi chạy test.
3. Chạy test **một lần** cho mỗi config cuối. Không lặp lại để "thử".

## 3. Metric bắt buộc
| Tầng | Metric |
|---|---|
| Clip | macro-F1, recall lớp đuôi (30% lớp ít mẫu nhất), AUBC, ECE |
| Mining | AUROC phát hiện lỗi, HHR@K, Lift@K, **Lift chuẩn hoá** `(HHR@K − base)/(1 − base)` |
| Sự kiện | F1 theo lớp (IoU ≥ 0,5, cùng lớp), MAE thời lượng (s), sự kiện sai mỗi giờ; thô và có lấp khe |
| Thực tế (v1.0) | TPR theo kịch bản, FA/giờ trên lái tự nhiên, TPR ô tệ nhất trong ma trận điều kiện |
| Chi phí | GPU-giờ trích đặc trưng; thời gian mỗi vòng; % dữ liệu bị loại khi khử trùng |

K mặc định = ngân sách mỗi vòng B. Luôn báo cáo `base` (tỷ lệ lỗi nền) cạnh Lift.

## 4. Thống kê
- 3 split × 3 seed = 9 lần chạy cho mỗi (phương pháp, cấu hình).
- So sánh hai phương pháp: `paired_win` (paired t-test, p < 0,05) ở mỗi vòng; báo cáo tỷ lệ vòng thắng/thua/hoà.
- Khoảng tin cậy 95%: `bootstrap_ci_by_group` theo **tài xế**, 1.000 lần.
- Không báo cáo chỉ số tốt nhất qua seed; báo cáo trung bình ± CI.

## 5. Điều cấm
- Chia theo frame hoặc cửa sổ.
- Chọn seed, split, hay K sau khi xem kết quả test.
- Dùng dữ liệu buồn ngủ diễn (NTHU-DDD) để kết luận về phát hiện buồn ngủ thật.
- So sánh với số trích từ paper khác khi giao thức khác nhau, mà không ghi rõ.
- Báo cáo accuracy theo frame thay cho metric sự kiện ở tầng sự kiện.

## 6. Định dạng kết quả
`results/<run_id>/rounds.csv` cột tối thiểu:
`strategy, split, seed, round, n_labeled, macro_f1_val, tail_recall_val, ece_val, auroc_err, hhr_at_k, lift_at_k, lift_norm, base_err, w_<signal>..., seconds`.
`results/<run_id>/final_test.csv`: một dòng mỗi (strategy, split, seed).
