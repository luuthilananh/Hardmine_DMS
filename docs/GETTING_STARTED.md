# Bắt đầu từ đâu — hướng dẫn từng bước

Tài liệu này dành cho chủ dự án: làm gì, ở đâu, theo thứ tự nào để Claude Sonnet hoàn thiện dự án.
Viết cho máy Windows; macOS/Linux làm tương tự.

## Bức tranh tổng thể: ba nơi, ba việc

| Nơi | Làm gì | Vì sao |
|---|---|---|
| **Máy của bạn** | Giữ dữ liệu Drive&Act, chạy Claude Code (Sonnet) để viết code, chạy mọi thí nghiệm mining | Dữ liệu ~7 GB nằm sẵn trên máy; sau khi có embedding, mọi vòng mining chạy trên CPU trong vài phút |
| **Kaggle** (GPU miễn phí) | Trích embedding **một lần duy nhất** | Đây là bước duy nhất cần GPU |
| **GitHub** | Lưu code, chạy CI tự động, đăng trang demo | Mỗi lần Sonnet push, CI kiểm tra lại toàn bộ test |

Dòng chảy công việc:

```
Máy bạn: P0 → P1.1–P1.4 (loader, split, lọc)  ──push──▶  GitHub
Kaggle:  P1.5 trích embedding bằng GPU  ──tải về──▶  máy bạn: cache/features/
Máy bạn: P1.6 → P2 → P3 (cổng G3) → P4 (v0.1)
```

---

## Bước 1. Cài công cụ trên máy (một lần, ~20 phút)

1. **Git for Windows:** https://git-scm.com/downloads/win — cài với lựa chọn mặc định.
2. **Python 3.11:** https://www.python.org/downloads/ — khi cài, **tick ô "Add python.exe to PATH"**.
3. **Claude Code** — chọn một trong hai:
   - **Ứng dụng Claude Desktop** (dễ nhất): tải ở https://claude.ai/download, đăng nhập, bấm thẻ **Code**.
     Cần gói Claude trả phí.
   - **Dòng lệnh**: mở PowerShell, chạy `irm https://claude.ai/install.ps1 | iex`, mở cửa sổ PowerShell mới,
     chạy `claude --version` để kiểm tra.
   Hướng dẫn chính thức: https://code.claude.com/docs/en/overview

Kiểm tra trong PowerShell:
```powershell
git --version
python --version
```

## Bước 2. Lấy repo về máy và kiểm tra (~5 phút)

```powershell
cd $HOME\Documents
git clone https://github.com/luuthilananh/Hardmine_DMS.git
cd Hardmine_DMS
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,video]"
python tests/run_tests.py
```

Dòng cuối phải là `... passed, 0 failed`. Nếu PowerShell báo lỗi khi chạy `Activate.ps1`, chạy trước
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi thử lại.

## Bước 3. Tải dữ liệu Drive&Act (~7 GB)

Drive&Act **không cần đăng ký** (kiểm tra 12/10/2026). Chỉ dùng cho nghiên cứu, không đăng lại công khai.

1. Tạo thư mục `data\driveact\` bên trong thư mục repo.
2. Tải hai file bằng trình duyệt vào thư mục đó:
   - https://www.driveandact.com/dataset/a_column_driver.zip (camera NIR cột A phía tài xế, gần vị trí DMS thật nhất)
   - https://www.driveandact.com/dataset/iccv_activities_3s.zip (nhãn hoạt động và split)
3. Giải nén cả hai ngay tại `data\driveact\`.

Thư mục `data\` đã được Git bỏ qua, nên dữ liệu không bao giờ bị đẩy lên GitHub.

## Bước 4. Giao việc cho Sonnet

1. Mở Claude Code **trong thư mục repo**:
   - Desktop: thẻ **Code** → chọn thư mục `Documents\Hardmine_DMS`.
   - Dòng lệnh: trong PowerShell đang ở thư mục repo, gõ `claude`.
2. Chọn model **Sonnet** (dòng lệnh: gõ `/model` rồi chọn Sonnet; Desktop: dùng ô chọn model).
3. Mở file `docs/KICKOFF_PROMPT.md`, copy khối "Lời nhắn khởi động" và dán vào Claude Code.

Claude Code tự đọc `CLAUDE.md` khi mở repo, nên Sonnet biết luật làm việc của dự án.

**Việc của bạn trong lúc Sonnet làm:**
- Khi Sonnet xin phép chạy lệnh, đọc lướt rồi đồng ý. Từ chối nếu lệnh xoá thư mục `data\` hoặc đẩy dữ liệu lên mạng.
- Sau mỗi task Sonnet tóm tắt 3 dòng. Thỉnh thoảng nhắn "push lên GitHub" để lưu tiến độ và để CI kiểm tra.
- Khi Sonnet dừng ở một **cổng** (G0, G1, …), đọc kết quả rồi trả lời "đi tiếp" hoặc hỏi lại.
- Khi Sonnet báo **"đang bị chặn"**, làm đúng việc nó yêu cầu (thường là thiếu file hoặc cần GPU).

## Bước 5. Trích embedding bằng GPU trên Kaggle (khi Sonnet tới P1.5)

Lúc này Sonnet đã viết loader và sinh ra bảng cửa sổ `cache/windows/driveact.csv` (đường dẫn video tương đối).

1. Nhắn Sonnet: "push lên GitHub, kể cả `cache/windows/driveact.csv`". Bảng này nhỏ, chỉ có thời điểm và tên file, không có dữ liệu ảnh.
2. Mở notebook: https://kaggle.com/kernels/welcome?src=https://github.com/luuthilananh/Hardmine_DMS/blob/main/notebooks/kaggle_hardmine.ipynb
3. Thanh bên phải: **Accelerator → GPU T4 x2**, **Internet → On** (lần đầu Kaggle yêu cầu xác minh số điện thoại).
4. Chạy các ô kiểm tra GPU, cài repo, chạy test. Sau đó thêm một ô mới:

```python
!mkdir -p /kaggle/working/data/driveact
%cd /kaggle/working/data/driveact
!wget -q https://www.driveandact.com/dataset/a_column_driver.zip && unzip -q a_column_driver.zip && rm a_column_driver.zip
%cd /kaggle/working/Hardmine_DMS
# đo thông lượng trước
!python scripts/extract_features.py --windows cache/windows/driveact.csv --root /kaggle/working/data/driveact \
    --out /kaggle/working/features/driveact/videomae_base --preset videomae_base --limit 200
# rồi chạy toàn bộ (bị ngắt thì chạy lại đúng lệnh này để tiếp tục)
!python scripts/extract_features.py --windows cache/windows/driveact.csv --root /kaggle/working/data/driveact \
    --out /kaggle/working/features/driveact/videomae_base --preset videomae_base
!cd /kaggle/working && zip -qr features.zip features
```

5. Bấm **Save Version** (chọn "Save & Run All" nếu phiên có thể hết giờ), tải `features.zip` từ tab **Output**.
6. Giải nén vào `cache\features\` trong thư mục repo trên máy, rồi nhắn Sonnet: "embedding đã ở cache/features/driveact/videomae_base, làm tiếp P1.6".

Phiên GPU miễn phí của Kaggle có giới hạn giờ mỗi tuần. Bước đo thông lượng cho biết toàn bộ sẽ mất bao lâu.

## Bước 6. Các cổng quan trọng

| Cổng | Bạn cần làm |
|---|---|
| G0 | Xác nhận test xanh và dữ liệu đã có |
| G1 | Xem bảng khử danh tính: độ chính xác đoán ID tài xế có giảm khi tăng r không |
| G2 | Xem đường cong baseline có hợp lý không |
| **G3** | Quyết định quan trọng nhất. Dán "Lời nhắn khi đến cổng G3" trong `KICKOFF_PROMPT.md`. Nếu H1 không đạt, đi thẳng sang tầng sự kiện (P4) |
| G4 | Phát hành v0.1: nhắn Sonnet tạo tag `v0.1.0` và cập nhật README bằng kết quả thật |

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `python` không được nhận | Cài lại Python, nhớ tick "Add to PATH"; mở cửa sổ PowerShell mới |
| `claude` không được nhận | Mở cửa sổ PowerShell mới; xem https://code.claude.com/docs/en/troubleshoot-install |
| Sonnet đòi tải dataset khác (DMD, NTHU-DDD) | Làm theo `docs/DATA.md`; dataset nào cần điền form thì bạn tự điền |
| Kaggle hết giờ GPU giữa chừng | Chạy lại đúng lệnh trích đặc trưng ở phiên sau; phần đã xong được giữ nếu bạn đã Save Version |
| CI trên GitHub đỏ sau khi Sonnet push | Copy thông báo lỗi trong tab Actions, dán cho Sonnet: "CI đỏ, sửa giúp" |
