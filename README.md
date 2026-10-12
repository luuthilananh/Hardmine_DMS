# HardMine-DMS

**Khai phá ca khó cho video giám sát tài xế (Driver Monitoring System).** Từ nhiều giờ video cabin chưa gán
nhãn, chọn ra tập nhỏ mà model đang sai — ở đúng tầng thời gian của trạng thái tài xế — để mỗi giờ gán nhãn
giảm được nhiều cảnh báo sai và ca bỏ sót nhất.

> Trạng thái: **đang phát triển (trước v0.1)**. Chưa có kết quả trên dữ liệu thật. Mọi bảng kết quả sẽ được sinh
> bằng script từ `configs/` và lưu trong `results/`.

## Ý tưởng

| Tầng | Đơn vị mining | Trạng thái | Tín hiệu đặc thù |
|---|---|---|---|
| Clip | cửa sổ 2 s | hành vi, điện thoại | bất ổn định thời gian, bất đồng đa góc, khử danh tính tài xế |
| Sự kiện | đoạn nhắm mắt | ngủ gật, ngủ, mất phản ứng (ngưỡng Euro NCAP 2026) | sát ngưỡng thời lượng, bị cắt vụn |
| Cửa sổ dài | 5–10 phút | buồn ngủ | mâu thuẫn xu hướng PERCLOS (v1.0) |

Các tín hiệu được kết hợp bằng **bộ hiệu chuẩn độ khó** (hồi quy logistic trọng số không âm học từ lỗi trên
validation), rồi chọn mẫu đa dạng với NMS theo thời gian và hạn ngạch lớp.

## Cài đặt và kiểm thử

```bash
pip install -e ".[dev]"
python tests/run_tests.py      # hoặc: pytest -q   — chạy trên CPU, không cần dữ liệu thật
```

## Dữ liệu
Repo không phân phối dữ liệu. Xem [docs/DATA.md](docs/DATA.md).

## Tài liệu
- [docs/MILESTONES.md](docs/MILESTONES.md) — lộ trình và tiêu chí nghiệm thu
- [docs/SPEC.md](docs/SPEC.md) — đặc tả kỹ thuật
- [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) — luật đánh giá
- [CLAUDE.md](CLAUDE.md) — hướng dẫn cho agent lập trình

## Giấy phép
Mã nguồn: Apache-2.0. Mỗi dataset và model nền có điều khoản riêng; người dùng tự chịu trách nhiệm tuân thủ.
