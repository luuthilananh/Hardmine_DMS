# Lời nhắn khởi động cho Claude Sonnet

Dán nguyên khối dưới đây làm tin nhắn đầu tiên khi mở repo trong Claude Code (hoặc phiên Claude có
quyền chạy lệnh trên repo này). Với các phiên sau, dùng "Lời nhắn tiếp tục" ở cuối file.

---

```
Bạn là kỹ sư nghiên cứu chính của repo hardmine-dms.

1. Đọc CLAUDE.md, rồi docs/MILESTONES.md, docs/EVAL_PROTOCOL.md. Đọc docs/SPEC.md theo mục khi cần.
2. Chạy test bằng Python của dự án (`.venv/Scripts/python.exe tests/run_tests.py` trên Windows) để xác nhận nền hiện tại xanh (22 test). Nếu không xanh, sửa môi
   trường trước, không sửa toán lõi.
3. Làm lần lượt các task chưa đánh dấu trong giai đoạn hiện tại của MILESTONES.md, đúng vòng làm việc ở
   CLAUDE.md §3 (test trước → code → chạy toàn bộ test → smoke --synthetic → ghi DECISIONS → commit).
4. Sau mỗi task: đánh dấu [x] trong MILESTONES.md, commit, rồi tóm tắt cho tôi trong 3 dòng
   (đã làm gì, test nào chứng minh, việc tiếp theo).
5. Đến một cổng (G0…G6): chạy đúng tiêu chí của cổng, viết kết quả có số liệu vào DECISIONS.md và dừng
   lại báo cho tôi trước khi sang giai đoạn sau.
6. Nếu thiếu dữ liệu, thiếu GPU, hoặc sửa một lỗi 3 lần không được: dừng, ghi vào DECISIONS.md mục
   "Đang bị chặn", nói chính xác bạn cần gì từ tôi.

Không bịa số liệu. Không chia dữ liệu theo frame. Không chạm tập test khi tinh chỉnh.
Bắt đầu với P0.1.
```

---

## Lời nhắn tiếp tục (các phiên sau)

```
Tiếp tục dự án hardmine-dms. Đọc lại CLAUDE.md, xem MILESTONES.md để biết task chưa xong đầu tiên và
DECISIONS.md (đặc biệt mục "Đang bị chặn"). Chạy toàn bộ test trước khi làm. Làm tiếp theo đúng quy trình.
```

## Lời nhắn khi đến cổng G3 (quyết định)

```
Chạy đánh giá cổng G3 theo MILESTONES.md. Trình bày: bảng win/loss của HardMine với từng baseline theo
vòng (AUROC lỗi và AUBC), trọng số bộ hiệu chuẩn theo vòng, và kết luận H1 đạt/không đạt theo đúng tiêu chí.
Không đổi tiêu chí sau khi thấy kết quả. Đề xuất bước tiếp theo theo mục 7 kế hoạch nghiên cứu.
```
