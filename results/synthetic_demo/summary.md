| Chiến lược | Macro-F1 val (vòng cuối) | Macro-F1 test | Tỷ lệ chọn trúng lỗi | Lift chuẩn hoá |
|---|---|---|---|---|
| random | 0.805 ± 0.010 | 0.866 ± 0.083 | 0.06 | 0.00 |
| entropy | 0.844 ± 0.073 | 0.913 ± 0.019 | 0.28 | 0.25 |
| margin_diverse | 0.835 ± 0.082 | 0.919 ± 0.026 | 0.14 | 0.39 |
| hardmine | 0.853 ± 0.089 | 0.905 ± 0.033 | 0.19 | 0.34 |

| Chiến lược (sự kiện) | Tỷ lệ lỗi trong top-20 | Tỷ lệ lỗi nền |
|---|---|---|
| random | 0.57 ± 0.10 | 0.54 |
| mean_frame_uncertainty | 0.48 ± 0.15 | 0.54 |
| hardmine_event | 1.00 ± 0.00 | 0.54 |

Hậu xử lý lấp khe 0,2 s: F1 sleep 0.65 → 0.94; F1 microsleep 0.48 → 0.88.
