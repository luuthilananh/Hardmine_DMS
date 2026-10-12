# CLAUDE.md — Luật làm việc cho Claude trong repo `hardmine-dms`

Bạn (Claude) là kỹ sư nghiên cứu chính của dự án này. Đọc hết file này trước khi làm bất cứ việc gì,
rồi đọc `docs/MILESTONES.md` để biết đang ở giai đoạn nào.

## 0. Môi trường Python của dự án (bắt buộc)

Dự án dùng môi trường riêng `.venv/` trong thư mục repo, tạo bằng `scripts/bootstrap.py`.
**Mọi lệnh Python gọi thẳng Python của dự án**, không dùng Python hệ thống, không cần activate:

- Windows: `.venv/Scripts/python.exe` (Git Bash) · `.venv\Scripts\python.exe` (PowerShell)
- macOS/Linux: `.venv/bin/python`

Chưa có `.venv/` → làm theo `docs/AUTOPILOT.md` bước A–B trước. Cài thêm gói: `<python dự án> -m pip install …`
và ghi phụ thuộc lâu dài vào `pyproject.toml`. Khi người dùng nói "đọc docs/AUTOPILOT.md và làm theo",
file đó là kịch bản chính của phiên làm việc.

## 1. Dự án là gì (đọc trong 30 giây)

**HardMine-DMS**: engine khai phá ca khó (hard-case mining) cho dữ liệu video trong cabin, phục vụ
Driver Monitoring System. Painpoint duy nhất: *trong nhiều giờ video chưa gán nhãn, chọn ra tập nhỏ
mà model hiện tại đang/sẽ sai, với chi phí GPU và chi phí gán nhãn thấp nhất.*

Engine hoạt động ở ba tầng thời gian:

| Tầng | Đơn vị | Trạng thái |
|---|---|---|
| Clip | cửa sổ 2 s, bước 1 s | hành vi, dùng điện thoại |
| Frame → sự kiện | đoạn nhắm mắt độ dài thay đổi | chớp mắt, ngủ gật 1–2 s, ngủ ≥ 3 s, mất phản ứng ≥ 6 s |
| Cửa sổ dài | 5–10 phút | buồn ngủ (KSS ≥ 7), chỉ ở v1.0 |

Tài liệu nguồn (đọc khi cần, theo thứ tự ưu tiên khi mâu thuẫn):
1. `docs/MILESTONES.md` — việc cần làm, cổng kiểm tra, tiêu chí nghiệm thu.
2. `docs/EVAL_PROTOCOL.md` — luật đánh giá. **Không bao giờ vi phạm.**
3. `docs/SPEC.md` — đặc tả module, công thức, giao diện, định dạng dữ liệu.
4. `docs/DATA.md` — cách đặt dữ liệu.
5. `docs/DECISIONS.md` — nhật ký quyết định. Bạn ghi vào đây.

## 2. Trạng thái khởi đầu

Phần toán lõi đã được viết và kiểm chứng (22 test xanh; CI chạy trên Python 3.10–3.12):

- `src/hardmine/signals/core.py` — S1 margin, S2 BALD, S3 thầy–trò, S4 bất ổn định thời gian,
  S5 đa góc, S6 độ hiếm kNN, S7 độ hiếm từ khoá.
- `src/hardmine/features/deidentify.py` — khử danh tính tài xế bằng chiếu trực giao.
- `src/hardmine/events/core.py` — ghép sự kiện nhắm mắt, phân lớp theo ngưỡng Euro NCAP,
  tín hiệu sát ngưỡng, tín hiệu cắt vụn, metric sự kiện.
- `src/hardmine/select/core.py` — bộ hiệu chuẩn độ khó, Pareto, chọn batch (k-means + NMS + hạn ngạch).
- `src/hardmine/eval/metrics.py` — HHR@K, Lift@K, AUROC lỗi, AUBC, recall lớp đuôi, t-test, bootstrap theo nhóm.
- `src/hardmine/synthetic/generators.py` — dữ liệu tổng hợp có ca khó biết trước.

Ngoài ra đã có (kiểm thử một phần):

- `src/hardmine/al/simple_loop.py` — vòng mô phỏng rút gọn trên CPU (logistic head, 4 chiến lược). Dùng làm khung
  tham chiếu khi viết `al/loop.py` đầy đủ ở P2.2.
- `src/hardmine/features/extract.py` + `scripts/extract_features.py` — trích embedding HF (VideoMAE, V-JEPA 2,
  SigLIP 2), resume được. Phần đọc video đã chạy thử; phần GPU được kiểm tra bởi workflow `extract-smoke.yml`
  với model tí hon và trên Kaggle (`notebooks/kaggle_hardmine.ipynb`).
- `scripts/demo_synthetic.py` — demo đầu-cuối, sinh `results/synthetic_demo/` và `demo/data.js`.
- `demo/index.html` — trang demo tương tác (GitHub Pages). Công thức sự kiện trong JS được kiểm tra khớp
  tuyệt đối với Python; nếu sửa `events/core.py`, sửa cả JS và chạy lại kiểm tra đối chiếu.
- `.github/workflows/` — `ci.yml` (test + demo), `extract-smoke.yml`, `pages.yml`.

Các module toán lõi là **nền đã kiểm chứng**. Bạn được phép mở rộng; chỉ được sửa hành vi hiện có
khi có test chứng minh nó sai, và phải ghi lý do vào `docs/DECISIONS.md`.

Phần còn trống, bạn xây: `data/`, `dedup/`, `models/`, `al/`, `scripts/`, `configs/`, README, báo cáo.

## 3. Vòng làm việc bắt buộc cho mỗi task

```
1. ĐỌC     task trong MILESTONES.md + mục SPEC.md liên quan.
2. TEST    viết test trước (tests/test_<module>.py): ít nhất 1 test đơn vị + 1 test trên dữ liệu tổng hợp
           chứng minh hành vi mong muốn.
3. CODE    viết code tối thiểu để test xanh. Hàm thuần numpy khi có thể; GPU chỉ ở features/ và models/.
4. CHẠY    python tests/run_tests.py   (hoặc pytest -q nếu có). Toàn bộ suite phải xanh, không chỉ test mới.
5. SMOKE   chạy script liên quan với --synthetic để chắc pipeline chạy đầu-cuối.
6. GHI     thêm một dòng vào docs/DECISIONS.md nếu bạn đã chọn giữa nhiều phương án.
7. COMMIT  một commit cho một task, message dạng "P2.3: <việc đã làm>".
```

Không chuyển sang giai đoạn mới khi cổng của giai đoạn trước chưa qua (xem MILESTONES.md).

## 4. Khi có lỗi: giao thức sửa lỗi

1. **Tái hiện**: viết lại lỗi thành một test nhỏ nhất có thể, chạy để thấy nó đỏ.
2. **Giả thuyết**: viết 1–2 câu nguyên nhân nghi ngờ (trong comment hoặc trong lời nhắn của bạn).
3. **Kiểm tra giả thuyết** bằng in giá trị trung gian, không đoán mò sửa nhiều chỗ cùng lúc.
4. **Sửa**, chạy lại test vừa viết, rồi chạy cả suite.
5. Giữ test đó lại làm test hồi quy.

**Phân biệt lỗi code và lỗi kỳ vọng.** Nếu một test thống kê fail, trước khi sửa code hãy kiểm tra kỳ
vọng có khả thi không. Ví dụ đã gặp: Lift@K tối đa = 1 / tỷ lệ lỗi nền; khi lỗi nền 62% thì trần Lift chỉ
là 1,61, nên đòi Lift > 1,5 là đòi gần như hoàn hảo. Cách đúng là đo Lift chuẩn hoá theo trần
`(HHR@K - base) / (1 - base)`.

**Giới hạn 3 lần.** Sau 3 lần sửa không thành cho cùng một lỗi: dừng, ghi vào `docs/DECISIONS.md` mục
"Đang bị chặn" (lỗi, đã thử gì, giả thuyết còn lại), rồi báo người dùng. Không được làm test "xanh" bằng
cách nới ngưỡng hoặc xoá assert mà không ghi lý do có số liệu.

## 5. Luật cứng (vi phạm = kết quả vô giá trị)

1. **Chia dữ liệu theo người lái**, không bao giờ theo frame hay theo cửa sổ. Tài xế ở test không xuất hiện
   ở train, pool, hay khi fit bộ khử danh tính.
2. **Không chạm vào tập test khi tinh chỉnh.** Mọi siêu tham số chọn trên validation. Tập test chỉ chạy ở
   bước báo cáo cuối của mỗi giai đoạn.
3. **Không bịa số.** Mọi con số trong README, báo cáo, DECISIONS phải sinh ra từ một lệnh script + một file
   config + một seed, và file CSV kết quả nằm trong `results/`. Nếu chưa chạy được, viết "chưa có".
4. **Không commit dữ liệu** (video, frame, embedding của dataset thật). `.gitignore` đã chặn `data/`, `cache/`.
5. **Test phải chạy trên CPU, không cần dữ liệu thật**, trong dưới 2 phút. Test cần GPU hoặc dữ liệu thật
   đặt tên `test_*_slow.py` và tự bỏ qua khi thiếu tài nguyên.
6. **Mọi baseline dùng cùng backbone, cùng head, cùng ngân sách, cùng tập khởi đầu** với phương pháp đề xuất.
7. **Báo cáo trung thực kết quả âm tính.** Nếu phương pháp đề xuất không thắng, ghi rõ; không chọn lại seed.
8. **Không tự xin quyền hay điền form tải dataset.** Ngoại lệ duy nhất: Drive&Act (không cần đăng ký) được tải
   bằng `scripts/download_driveact.py` **sau khi người dùng đồng ý điều khoản** (AUTOPILOT bước C). Dataset khác
   cần đăng ký: dừng và nói chính xác cần file nào, đặt ở đâu.

## 6. Lệnh thường dùng

`PY` = Python của dự án (mục 0).

```bash
py -3.11 scripts/bootstrap.py           # tạo/cập nhật .venv và chạy test (Windows; macOS/Linux: python3 …)
PY tests/run_tests.py                   # chạy toàn bộ test (hoặc: PY -m pytest -q)
PY scripts/download_driveact.py         # tải Drive&Act (sau khi người dùng đồng ý)
PY scripts/demo_synthetic.py --quick    # demo đầu-cuối trên dữ liệu tổng hợp
PY scripts/run_al.py --config configs/mvp_clip.yaml --synthetic     # (tạo ở P2) smoke test đầu-cuối
PY scripts/run_al.py --config configs/mvp_clip.yaml                 # (tạo ở P2) chạy thật
PY scripts/evaluate.py --results results/<run_id>                   # (tạo ở P2) bảng và hình
```

## 7. Phong cách code

- Python ≥ 3.10, type hints, docstring ngắn nêu công thức nếu có.
- Hàm thuần, đầu vào/đầu ra numpy; trạng thái chỉ nằm trong class có `fit` / `transform` / `score`.
- Config bằng YAML (`configs/`), đọc vào dataclass; không hard-code đường dẫn.
- Mỗi lần chạy ghi `results/<run_id>/` gồm: `config.yaml` (bản sao), `meta.json` (git commit, seed, thời gian,
  phần cứng), `rounds.csv` (mỗi vòng một dòng), `selected.csv` (mẫu đã chọn), log.
- Seed hoá mọi thứ ngẫu nhiên (numpy Generator, torch, sklearn random_state).
- Comment và docstring bằng tiếng Việt hoặc tiếng Anh đều được, nhất quán trong một file.

## 8. Khi nào hỏi người dùng

Hỏi khi: cần dữ liệu hoặc quyền truy cập; một cổng (G0–G6) không qua và cần quyết định đổi hướng
(mục 7 kế hoạch nghiên cứu); cần GPU lớn hơn hiện có; phát hiện mâu thuẫn giữa các tài liệu.
Không hỏi cho những lựa chọn kỹ thuật nhỏ có mặc định hợp lý: chọn, ghi vào DECISIONS.md, làm tiếp.
