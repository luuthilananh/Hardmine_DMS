# Bắt đầu từ đâu

Bạn chỉ làm 3 việc. Phần còn lại Claude Code tự làm theo kịch bản `docs/AUTOPILOT.md`.

## Việc 1. Cài hai công cụ (một lần)

1. **Git for Windows:** https://git-scm.com/downloads/win (cài mặc định). Có Git, Claude Code dùng được Git Bash.
2. **Claude Code**, chọn một trong hai:
   - **Ứng dụng Claude Desktop:** https://claude.ai/download → đăng nhập → thẻ **Code**. (Cần gói Claude trả phí.)
   - **Dòng lệnh:** mở PowerShell, chạy `irm https://claude.ai/install.ps1 | iex`, mở cửa sổ mới, gõ `claude --version`.

   Hướng dẫn chính thức: https://code.claude.com/docs/en/overview

Python không cần cài trước: nếu máy chưa có Python 3.10 trở lên, Claude Code sẽ xin phép cài Python 3.11 cho bạn.

## Việc 2. Mở Claude Code trong thư mục repo

- **Desktop:** thẻ **Code** → chọn thư mục repo, ví dụ `C:\VINUNI_PROJECT\5_MY_PROJECT\Hardmine_DMS`.
- **Dòng lệnh:**
  ```powershell
  cd C:\VINUNI_PROJECT\5_MY_PROJECT\Hardmine_DMS
  claude
  ```

Chọn model **Sonnet** (dòng lệnh: gõ `/model`; Desktop: ô chọn model).

## Việc 3. Gõ đúng một câu

```
Đọc docs/AUTOPILOT.md và làm theo.
```

Claude Code sẽ tự làm:

| Bước | Claude làm gì | Bạn làm gì |
|---|---|---|
| A | Lấy code mới nhất, kiểm tra Python, Git, GPU, dung lượng đĩa | Bấm đồng ý khi được hỏi |
| B | Tạo môi trường Python **riêng của dự án** trong `.venv\`, cài thư viện, chạy test | — |
| C | Tải Drive&Act (~7 GB) vào `data\driveact\`, giải nén, kiểm tra | Đồng ý điều khoản "chỉ dùng cho nghiên cứu" |
| D | Làm dự án theo `docs/MILESTONES.md`: viết code, viết test, chạy, sửa lỗi, commit, push | Trả lời ở các cổng kiểm tra |
| E | Trích embedding bằng GPU: tự chạy nếu máy có GPU NVIDIA; nếu không, đưa bạn lệnh chạy trên Kaggle | Chạy Kaggle nếu được yêu cầu (bên dưới) |

### Về môi trường Python riêng

Mọi thư viện của dự án nằm trong thư mục `.venv\` bên trong repo. Python chung của máy và các dự án khác không bị ảnh hưởng.
Muốn xoá sạch, chỉ cần xoá thư mục `.venv\`; muốn tạo lại, nhắn Claude "chạy lại bootstrap".

Nếu tự chạy lệnh, gọi Python của dự án:
```powershell
.venv\Scripts\python.exe tests\run_tests.py
```

### Mở lại ở phiên sau

Mở Claude Code trong thư mục repo và gõ:
```
Tiếp tục dự án theo docs/AUTOPILOT.md, bắt đầu từ task chưa xong đầu tiên trong docs/MILESTONES.md.
```

## Khi Claude nhờ chạy Kaggle (máy không có GPU NVIDIA)

Chỉ cần một lần, để trích embedding.

1. Mở: https://kaggle.com/kernels/welcome?src=https://github.com/luuthilananh/Hardmine_DMS/blob/main/notebooks/kaggle_hardmine.ipynb
2. Thanh bên phải: **Accelerator → GPU T4 x2**, **Internet → On** (lần đầu Kaggle yêu cầu xác minh số điện thoại).
3. Chạy các ô kiểm tra GPU và cài repo. Thêm một ô mới, dán đoạn lệnh Claude đưa cho bạn (mẫu dưới), rồi chạy:

```python
!mkdir -p /kaggle/working/data/driveact
%cd /kaggle/working/data/driveact
!wget -q https://www.driveandact.com/dataset/a_column_driver.zip && unzip -q a_column_driver.zip && rm a_column_driver.zip
%cd /kaggle/working/Hardmine_DMS
!python scripts/extract_features.py --windows cache/windows/driveact.csv --root /kaggle/working/data/driveact \
    --out /kaggle/working/features/driveact/videomae_base --preset videomae_base
!cd /kaggle/working && zip -qr features.zip features
```

4. Bấm **Save Version**, tải `features.zip` ở tab **Output**, giải nén vào thư mục `cache\` trong repo.
5. Nhắn Claude: "đã có embedding trong cache/features, làm tiếp".

## Các cổng kiểm tra

| Cổng | Bạn cần làm |
|---|---|
| G0 | Xác nhận test xanh và dữ liệu đã có |
| G1 | Xem bảng khử danh tính: độ chính xác đoán ID tài xế có giảm khi tăng r không |
| G2 | Xem đường cong baseline có hợp lý không |
| **G3** | Quyết định quan trọng nhất: phương pháp có thắng baseline trên dữ liệu thật không. Không thắng thì đi thẳng sang tầng sự kiện (P4) |
| G4 | Đồng ý phát hành v0.1 |

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `claude` không được nhận | Mở cửa sổ PowerShell mới; xem https://code.claude.com/docs/en/troubleshoot-install |
| Claude hỏi quá nhiều lần cho cùng một lệnh | Chọn "Yes, and don't ask again" cho lệnh đó |
| Tải Drive&Act bị ngắt | Nhắn Claude "chạy lại lệnh tải", phần đã tải được giữ |
| CI trên GitHub đỏ | Nhắn Claude "CI đỏ, kiểm tra và sửa" |
