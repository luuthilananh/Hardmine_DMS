# DATA — Cách lấy và đặt dữ liệu

**Repo không chứa và không phân phối dữ liệu.** Người dùng tự đăng ký và tải từng dataset theo điều khoản
của nó, rồi đặt vào thư mục `data/` (đã được `.gitignore`). Claude không tự đăng ký hay tải dataset có điều khoản.

Kiểm tra: `python scripts/check_env.py` in ra dataset nào đã có.

| Dataset | Dùng cho | Lấy ở đâu | Đặt vào |
|---|---|---|---|
| Drive&Act | Tầng clip (chính) | driveandact.com — **không cần đăng ký** (kiểm tra 12/10/2026), "chỉ dùng cho nghiên cứu" | `data/driveact/` |
| DMD | Chuyển miền; gaze; nhãn mắt nếu còn | github.com/Vicomtech/DMD-Driver-Monitoring-Dataset (form tải) | `data/dmd/` |
| NTHU-DDD | Nhãn mắt theo frame (dự phòng) | liên hệ nhóm tác giả NTHU | `data/nthu_ddd/` |
| UTA-RLDD | Buồn ngủ thật (v1.0) | trang dataset UTA | `data/uta_rldd/` |
| DROZY | Buồn ngủ + sinh lý (v1.0) | trang dataset ULg | `data/drozy/` |
| Tự quay NIR | Kiểm tra thực tế (v1.0) | SPEC §8 | `data/own_nir/` |

## Drive&Act: tải gì cho MVP

Trang https://www.driveandact.com/ đã bỏ yêu cầu đăng nhập; các file tải trực tiếp. Bản quyền Fraunhofer IOSB, chỉ dùng
cho nghiên cứu, không phân phối lại (không đăng lại công khai, kể cả trên Kaggle).

| File | Nội dung | Dung lượng (theo bản mirror, chưa kiểm chứng) | MVP cần? |
|---|---|---|---|
| `a_column_driver.zip` | NIR, cột A phía tài xế — gần vị trí camera DMS thật nhất | ~2,5 GB | **Có** |
| `iccv_activities_3s.zip` | Nhãn hoạt động (3 cấp, gồm 34 lớp cấp giữa) và split | ~4,4 GB | **Có** |
| `inner_mirror.zip` | NIR, gương chiếu hậu | ~6,5 GB | P5 (đa góc) |
| `ceiling.zip`, `steering_wheel.zip` | NIR, trần xe / vô lăng | ~4,4 GB / ~18 GB | Không |
| `iccv_openpose_3d.zip` | Tư thế 3D | | Không |

```bash
mkdir -p data/driveact && cd data/driveact
wget https://www.driveandact.com/dataset/a_column_driver.zip
wget https://www.driveandact.com/dataset/iccv_activities_3s.zip
unzip -q a_column_driver.zip && unzip -q iccv_activities_3s.zip
```

Trên Windows có thể tải bằng trình duyệt rồi giải nén vào `data\driveact\`.

**Việc đầu tiên của P1.1:** liệt kê cấu trúc thư mục thật sau khi giải nén (`find data/driveact -maxdepth 3 | head -50`),
mở vài file nhãn, rồi viết loader theo cấu trúc thật. Cấu trúc dưới đây chỉ là phỏng đoán ban đầu.

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
