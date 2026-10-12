# DATA — Cách lấy và đặt dữ liệu

**Repo không chứa và không phân phối dữ liệu.** Người dùng tự đăng ký và tải từng dataset theo điều khoản
của nó, rồi đặt vào thư mục `data/` (đã được `.gitignore`). Claude không tự đăng ký hay tải dataset có điều khoản.

Kiểm tra: `python scripts/check_env.py` in ra dataset nào đã có.

| Dataset | Dùng cho | Lấy ở đâu | Đặt vào |
|---|---|---|---|
| Drive&Act | Tầng clip (chính) | driveandact.com (đăng ký) | `data/driveact/` |
| DMD | Chuyển miền; gaze; nhãn mắt nếu còn | github.com/Vicomtech/DMD-Driver-Monitoring-Dataset (form tải) | `data/dmd/` |
| NTHU-DDD | Nhãn mắt theo frame (dự phòng) | liên hệ nhóm tác giả NTHU | `data/nthu_ddd/` |
| UTA-RLDD | Buồn ngủ thật (v1.0) | trang dataset UTA | `data/uta_rldd/` |
| DROZY | Buồn ngủ + sinh lý (v1.0) | trang dataset ULg | `data/drozy/` |
| Tự quay NIR | Kiểm tra thực tế (v1.0) | SPEC §8 | `data/own_nir/` |

## Cấu trúc mong đợi (điều chỉnh loader nếu bản tải khác)

```
data/driveact/
  <view>/vp<subject>/run<k>_*.mp4          # video theo góc
  activities_3s/<view>/midlevel.chunks_90.split_<s>.<train|val|test>.csv
data/dmd/
  gA/<subject>/<session>/<...>_rgb_face.mp4
  gA/<subject>/<session>/<...>_ann_distraction.json   # OpenLABEL/VCD
```

## Lưu ý đã biết (tính đến 10/2026)
- DMD sửa đổi 2026: chỉ còn RGB xe thật của các đối tượng 1, 5, 6, 7, 9, 10, 13, 14, 23, 28, 29, 33, 36, 37;
  IR, depth, mô phỏng đã bị gỡ. Kiểm tra lại phần buồn ngủ còn tải được không.
- NTHU-DDD: buồn ngủ là diễn; nhãn thời gian bị đánh giá là kém tin cậy → chỉ dùng cho cơ chế sự kiện.
- Ghi chính xác phiên bản/ngày tải vào `docs/DECISIONS.md`.
