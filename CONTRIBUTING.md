# Đóng góp

1. Đọc `CLAUDE.md` (áp dụng cho cả người và agent) và `docs/EVAL_PROTOCOL.md`.
2. Mỗi thay đổi đi kèm test; `pytest -q` phải xanh trên CPU, không cần dữ liệu thật.
3. Mọi con số trong README/báo cáo phải sinh từ một script + config + seed, CSV nằm trong `results/`.
4. Không commit dữ liệu, frame hay embedding của dataset có điều khoản.
5. Ghi quyết định thiết kế vào `docs/DECISIONS.md`.
