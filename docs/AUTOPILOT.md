# AUTOPILOT — Claude Code tự thiết lập và chạy dự án

**Người dùng chỉ cần mở Claude Code trong thư mục repo và gõ:**

```
Đọc docs/AUTOPILOT.md và làm theo.
```

---

## Dành cho Claude Code (đọc kỹ, làm theo thứ tự)

Bạn là kỹ sư nghiên cứu của dự án. Mục tiêu: thiết lập máy, lấy dữ liệu, rồi làm dự án theo `docs/MILESTONES.md`
với **ít thao tác nhất từ người dùng**. Luật trong `CLAUDE.md` luôn được ưu tiên.

### Nguyên tắc môi trường (bắt buộc)

- **Mọi lệnh Python đều chạy bằng Python riêng của dự án** trong `.venv/`:
  - Windows: `.venv/Scripts/python.exe` (Git Bash) hoặc `.venv\Scripts\python.exe` (PowerShell)
  - macOS/Linux: `.venv/bin/python`
- Không bao giờ chạy `pip install` bằng Python hệ thống. Cài thêm gói: `<python dự án> -m pip install <gói>`, và nếu
  gói là phụ thuộc lâu dài thì thêm vào `pyproject.toml`.
- Không cần "activate" môi trường: gọi thẳng đường dẫn Python ở trên (mỗi lệnh shell là một phiên mới).
- Ở dưới, `PY` là viết tắt của Python dự án theo hệ điều hành.

### Báo cáo cho người dùng

Sau mỗi bước, báo **một dòng**: đã xong gì, mất bao lâu, bước tiếp theo. Chỉ dừng lại hỏi khi gặp các mục
**[HỎI]** dưới đây, khi đến một cổng kiểm tra (G0–G6), hoặc khi bị chặn theo `CLAUDE.md` §4.

---

### Bước A. Đồng bộ và kiểm tra máy

1. `git pull origin main` để lấy bản mới nhất.
2. Xác định hệ điều hành và shell đang dùng.
3. Tìm Python >= 3.10, thử lần lượt: `py -3.11 --version`, `py -3 --version`, `python --version`, `python3 --version`.
   - Không có bản nào đạt: **[HỎI]** xin phép chạy `winget install -e --id Python.Python.3.11` (Windows) rồi mở lại
     terminal. Trên macOS/Linux, hướng dẫn người dùng cài Python 3.11 và dừng.
4. `git --version` (phải có, vì repo đã được clone).
5. Kiểm tra GPU: chạy `nvidia-smi`. Ghi lại tên GPU và VRAM nếu có. Không có lệnh này nghĩa là không có GPU NVIDIA.
6. Kiểm tra dung lượng trống của ổ chứa repo (cần ≥ 20 GB cho MVP).

### Bước B. Tạo môi trường riêng và chạy test

1. Chạy bằng Python vừa tìm được: `<python hệ thống> scripts/bootstrap.py`
   (ví dụ `py -3.11 scripts/bootstrap.py`). Script tạo `.venv/`, cài gói, chạy test.
2. Phải thấy `... passed, 0 failed`. Nếu lỗi: sửa theo giao thức `CLAUDE.md` §4; lỗi về mạng/proxy thì báo người dùng.
3. Chạy `PY scripts/demo_synthetic.py --quick --no-figures --web ""` để chắc pipeline chạy đầu-cuối trên máy này.

### Bước C. Tải dữ liệu Drive&Act

1. **[HỎI một lần]** "Drive&Act chỉ dùng cho nghiên cứu, không phân phối lại; sẽ tải khoảng 7 GB vào `data/driveact/`.
   Đồng ý không?" Người dùng đồng ý thì làm tiếp.
2. `PY scripts/download_driveact.py --dry-run` để xem dung lượng, rồi `PY scripts/download_driveact.py`.
   Lệnh chạy lâu (tuỳ mạng): chạy nền nếu công cụ cho phép, kiểm tra tiến độ định kỳ. Bị ngắt thì chạy lại đúng lệnh.
3. Ghi vào `docs/DECISIONS.md`: ngày tải, các file, dung lượng thực tế.
4. **Không bao giờ** `git add` thư mục `data/` (đã có trong `.gitignore` và bị chặn trong `.claude/settings.json`).

### Bước D. Làm dự án theo MILESTONES

Thực hiện đúng "Lời nhắn khởi động" trong `docs/KICKOFF_PROMPT.md`, bắt đầu từ task đầu tiên chưa đánh dấu trong
`docs/MILESTONES.md`. Luật bổ sung khi chạy tự động:

- Mỗi task xong: chạy toàn bộ test, commit theo mẫu `P<x>.<y>: ...`.
- Mỗi giai đoạn xong (hoặc mỗi ~5 task): `git push origin main`, rồi kiểm tra CI trên GitHub
  (`https://github.com/luuthilananh/Hardmine_DMS/actions`). CI đỏ thì sửa trước khi làm tiếp.
- Đến cổng: dừng, trình bày số liệu theo tiêu chí của cổng, chờ người dùng trả lời.

### Bước E. Trích embedding (P1.5) — chọn nơi chạy GPU

Sau khi có `cache/windows/driveact.csv` (P1.1–P1.4):

- **Máy có GPU NVIDIA ≥ 6 GB VRAM** (bước A.5): cài PyTorch bản CUDA vào `.venv` theo lệnh chính thức ở
  https://pytorch.org/get-started/locally/ (chọn đúng phiên bản CUDA mà `nvidia-smi` báo), rồi
  `PY -m pip install -e ".[gpu]"`. Đo thông lượng trước với `--limit 200`, báo người dùng thời gian dự kiến, rồi chạy:
  `PY scripts/extract_features.py --windows cache/windows/driveact.csv --root data/driveact --out cache/features/driveact/videomae_base --preset videomae_base`
- **Không có GPU phù hợp:** push `cache/windows/driveact.csv` lên GitHub, rồi **[HỎI]** người dùng chạy Kaggle theo
  `docs/GETTING_STARTED.md` bước 5. Trong lúc chờ, làm tiếp các phần không cần embedding thật (P2 trên dữ liệu
  tổng hợp, viết baseline và test). Khi người dùng báo đã giải nén `features.zip` vào `cache/features/`, kiểm tra bằng
  `hardmine.features.extract.load()` rồi làm tiếp P1.6.
- Không chạy trích đặc trưng trên CPU cho toàn bộ dataset trừ khi người dùng đồng ý (rất chậm).

### Bước F. Khi bị chặn

Ghi vào `docs/DECISIONS.md` mục "Đang bị chặn" và nói với người dùng **chính xác một việc** họ cần làm
(ví dụ: "điền form tải DMD ở link X rồi đặt file vào `data/dmd/`").

---

## Những gì người dùng phải tự làm (Claude không làm thay được)

| Việc | Khi nào |
|---|---|
| Bấm đồng ý khi Claude Code xin chạy lệnh lần đầu | Bước A–C |
| Đồng ý điều khoản Drive&Act | Bước C |
| Chạy notebook trên Kaggle (nếu máy không có GPU) | Bước E |
| Trả lời ở các cổng G0–G6, quan trọng nhất là G3 | Bước D |
| Điền form tải các dataset cần đăng ký (DMD, NTHU-DDD…) | Khi Claude báo |
