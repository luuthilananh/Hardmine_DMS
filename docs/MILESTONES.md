# MILESTONES — Giai đoạn, task, cổng kiểm tra

Quy ước: mỗi task có **ID**, **file chính**, **nghiệm thu** (điều kiện kiểm chứng được bằng lệnh).
Đánh dấu `[x]` khi xong và commit. Không bắt đầu giai đoạn mới khi cổng trước chưa qua.

Trạng thái hiện tại: **P0 đang làm** (toán lõi, CI, demo tổng hợp, trích đặc trưng đã có; xem CLAUDE.md mục 2).

---

## P0 — Nền móng (tuần 1)

- [x] **P0.0 Môi trường.** `scripts/bootstrap.py` tạo `.venv/` riêng; `scripts/download_driveact.py`; `docs/AUTOPILOT.md`.
- [ ] **P0.1 Đóng gói.** `pyproject.toml` cài được bằng `pip install -e ".[dev]"`; `import hardmine` chạy.
  Nghiệm thu: CI xanh trên 3 phiên bản Python (đã có `ci.yml`; chỉ cần xác nhận lần chạy đầu trên GitHub).
- [ ] **P0.2 Config.** `src/hardmine/config.py`: dataclass đọc YAML (xem SPEC §7). Test đọc `configs/mvp_clip.yaml`.
- [ ] **P0.3 Ghi kết quả.** `src/hardmine/runlog.py`: tạo `results/<run_id>/`, ghi config, meta (git commit, seed,
  thời gian, GPU), CSV theo vòng. Test trên thư mục tạm.
- [ ] **P0.4 Kiểm tra môi trường.** `scripts/check_env.py`: in Python, torch, CUDA, VRAM, dung lượng đĩa,
  và kiểm tra các đường dẫn dữ liệu trong `docs/DATA.md` có tồn tại không. Không fail khi thiếu GPU.
- [ ] **P0.5 Kiểm tra dữ liệu.** Chạy `check_env.py`; ghi vào DECISIONS.md: dataset nào đã có, dataset nào thiếu,
  có nhãn mắt theo frame dùng được không (DMD hay NTHU-DDD).

**Cổng G0:** mọi test xanh; `check_env.py` chạy; có ít nhất Drive&Act (hoặc phương án thay thế theo
kế hoạch mục 7) trên đĩa. Nếu thiếu dữ liệu: dừng, báo người dùng chính xác cần gì.

---

## P1 — Dữ liệu và đặc trưng (tuần 2–3)

- [ ] **P1.1 Loader Drive&Act.** `data/driveact.py`: đọc annotation cấp giữa (34 lớp), sinh bảng cửa sổ
  (SPEC §2) cho một góc NIR. Gán nhãn đa số; cửa sổ có lớp đa số < 60% đánh dấu `is_transition`.
  Nghiệm thu: bảng không có cửa sổ vượt ranh giới phiên; thống kê số cửa sổ mỗi lớp lưu vào
  `results/data_stats/driveact_classes.csv`.
- [ ] **P1.2 Split theo tài xế.** `data/splits.py`: dùng đúng 3 split chính thức của Drive&Act; test khẳng định
  giao tài xế giữa train/val/test là rỗng.
- [ ] **P1.3 Lọc thô.** `dedup/filters.py`: năng lượng chuyển động, độ sáng, độ nét (phương sai Laplacian),
  công thức SPEC §3. Test trên frame tổng hợp.
- [ ] **P1.4 pHash.** `dedup/phash.py`: pHash 64-bit tự cài (DCT 32×32 → 8×8), khoảng cách Hamming, nhóm
  trùng trong cùng phiên. Test: ảnh dịch 1 pixel có Hamming ≤ 6; ảnh khác hẳn > 20.
- [ ] **P1.5 Trích đặc trưng.** Chạy trên Kaggle GPU theo `docs/GETTING_STARTED.md` bước 5 (người dùng thực hiện); Claude chuẩn bị `cache/windows/driveact.csv` và lệnh. Script đã hỗ trợ `--windows --root`. `features/extract.py` + `scripts/extract_features.py`: backbone cấu hình được,
  ghi `cache/features/<dataset>/<backbone>.npy` (float16) + `index.csv` (đã cài trong `features/extract.py`; cần nối với bảng cửa sổ của Drive&Act). Chạy một lần, có thể tiếp tục
  khi bị ngắt (resume theo chỉ số đã ghi). Đo thông lượng (cửa sổ/giây) trên 1.000 cửa sổ, ghi DECISIONS.md.
- [ ] **P1.6 Chọn backbone.** So sánh 2–3 backbone bằng linear probe trên val (SPEC §4). Chọn một, ghi lý do.
- [ ] **P1.7 SemDeDup.** `dedup/semdedup.py`: k-means trên embedding, loại cặp cosine ≥ 1 − ε trong cụm.
  Báo cáo % loại và recall theo lớp trước/sau (dùng nhãn thật chỉ để đánh giá).
- [ ] **P1.8 Khử danh tính trên dữ liệu thật.** Quét r ∈ {4, 8, 16, 32, n_train_drivers − 1}, cả
  `class_balanced` True/False. Fit **chỉ trên tài xế train**. Lưu bảng `diagnostics` cho từng r.

**Cổng G1:** linear probe macro-F1 trên val ổn định qua 3 seed (độ lệch chuẩn < 2 điểm); có ít nhất một r
làm `driver_acc_after` giảm ≥ 30% tương đối mà `behavior_acc_after` giảm ≤ 2 điểm. Nếu không có r nào
thoả: ghi kết quả, vẫn đi tiếp nhưng H2 coi như bị bác bỏ.

---

## P2 — Vòng mô phỏng và baseline (tuần 4–5)

- [ ] **P2.1 Ensemble head.** `models/heads.py`: M = 5 MLP 2 lớp (PyTorch) hoặc logistic regression
  (sklearn, khi chạy CPU), bootstrap + seed khác nhau, dropout 0,5. Temperature scaling trên val
  (`models/calibration.py`), báo cáo ECE trước/sau.
- [ ] **P2.2 Vòng AL mô phỏng.** `al/loop.py` theo SPEC §6: giấu nhãn, mở khi được chọn, huấn luyện lại,
  đánh giá trên **val** mỗi vòng, test chỉ ở cuối. Hỗ trợ `--synthetic`.
- [ ] **P2.3 Baseline.** `al/strategies.py`: Random, Entropy, Margin, BALD, Core-set, TypiClust, ProbCover,
  DropQuery (SPEC §6.3). Mỗi chiến lược có test trên dữ liệu tổng hợp (ít nhất: trả đúng số mẫu, không trùng,
  không chọn mẫu đã gán nhãn).
- [ ] **P2.4 Chạy baseline.** 3 split × 3 seed, ngân sách theo `configs/mvp_clip.yaml`. Kết quả vào
  `results/`, bảng tổng hợp bằng `scripts/evaluate.py`.

**Cổng G2:** Random tăng đơn điệu (trung bình) theo số nhãn; Entropy/DropQuery không thua Random ở vòng cuối;
độ lệch chuẩn macro-F1 giữa các seed ở vòng cuối < 3 điểm. Nếu Random không tăng: có bug trong vòng lặp.

---

## P3 — Tín hiệu và bộ hiệu chuẩn (tuần 6)

- [ ] **P3.1 Chiến lược HardMine.** `al/strategies.py::HardMine`: tính S1, S2, S4, S6 trên pool; fit
  `HardnessCalibrator` trên val với nhãn lỗi của model hiện tại; `select_batch` với NMS 10 s và hạn ngạch 25%.
  Embedding dùng cho k-means và S6 là bản đã khử danh tính.
- [ ] **P3.2 Phân tích tín hiệu.** Mỗi vòng ghi: AUROC lỗi và Lift@K của từng tín hiệu và của h(x), trọng số
  bộ hiệu chuẩn. Ghi vào `rounds.csv`.
- [ ] **P3.3 So sánh.** HardMine vs baseline, 3 split × 3 seed; win-matrix bằng `paired_win`.
- [ ] **P3.4 Ablation tối thiểu.** Bỏ khử danh tính; trọng số đều thay bộ hiệu chuẩn; bỏ NMS.

**Cổng G3 (quyết định):** H1 đạt khi HardMine thắng có ý nghĩa baseline tốt nhất ở ≥ 50% số vòng về
AUROC phát hiện lỗi **hoặc** AUBC. Không đạt: dừng mở rộng tầng clip, ghi kết quả âm tính, báo người dùng,
đi thẳng sang P4 (tầng sự kiện).

---

## P4 — Tầng sự kiện (tuần 7–8)

- [ ] **P4.1 Loader nhãn mắt.** `data/eyes.py`: đọc nhãn mắt mở/nhắm theo frame (DMD hoặc NTHU-DDD, theo
  P0.5). Xuất chuỗi nhị phân theo phiên + fps. Ánh xạ nhãn về định nghĩa SPEC §5.
- [ ] **P4.2 Model mắt.** `models/eye.py`: crop vùng mắt từ landmark (MediaPipe Face Landmarker hoặc tương
  đương) + CNN nhỏ (ví dụ MobileNetV3-Small) cho q_t; thêm EAR làm bộ ước lượng thứ hai (cho S9).
  Chia theo người. Báo cáo AUROC frame.
- [ ] **P4.3 Mining sự kiện.** `al/event_loop.py`: đơn vị mining = sự kiện dự đoán + 2 s ngữ cảnh. Tín hiệu:
  bất định frame trung bình trong sự kiện, `threshold_proximity`, `fragmentation`, (S9 nếu có). Bộ hiệu
  chuẩn với nhãn lỗi = sự kiện dự đoán không khớp sự kiện thật cùng lớp (IoU ≥ 0,5).
- [ ] **P4.4 Đánh giá.** `event_metrics` theo lớp, MAE thời lượng, FA/giờ; so với chọn theo bất định frame
  trung bình và Random.
- [ ] **P4.5 Phát hành v0.1.** README có hình pipeline, bảng kết quả kèm CI 95%, giới hạn; tag `v0.1.0`.

**Cổng G4:** có số liệu H4 (đạt hay không đều được báo cáo); README tái lập được bằng các lệnh ghi trong đó
trên một máy sạch (kiểm tra bằng cách clone vào thư mục mới và chạy).

---

## P5 — Mở rộng (tuần 9–12, v1.0)

- [ ] P5.1 S3 thầy–trò: student X3D-S hoặc MobileNetV3+TSM huấn luyện đầu-cuối.
- [ ] P5.2 S5 đa góc: thêm 1–2 góc NIR của Drive&Act.
- [ ] P5.3 S7 VLM: Qwen2.5-VL-7B (lượng tử 4-bit) chỉ trên tập ứng viên, prompt cố định có từ vựng đóng (SPEC §4.7); cache.
- [ ] P5.4 Đầu nhìn (gaze zone + head pose) trên DMD; S10 bất nhất giữa nguyên thuỷ.
- [ ] P5.5 Gắn thuộc tính điều kiện (ánh sáng, kính, khẩu trang…) bằng SigLIP 2 zero-shot; `coverage_boost`.
- [ ] P5.6 Chuyển miền: học trên Drive&Act, mining và đánh giá trên DMD.
- [ ] P5.7 Ablation đầy đủ (bảng 7.3 tài liệu thiết kế).

**Cổng G5:** H2, H3, H5, H6 có số liệu kèm khoảng tin cậy.

## P6 — Kiểm tra thực tế và công bố (tuần 13–18)

- [ ] P6.1 Giao thức tự quay NIR (SPEC §8) — cần người dùng thực hiện.
- [ ] P6.2 Đánh giá kiểu Euro NCAP: TPR theo kịch bản, FA/giờ trên lái tự nhiên, ô tệ nhất.
- [ ] P6.3 Đo độ trễ student và model mắt trên phần cứng nhúng mục tiêu (hoặc CPU ARM giả lập).
- [ ] P6.4 Tầng cửa sổ dài: buồn ngủ trên UTA-RLDD / DROZY.
- [ ] P6.5 Báo cáo kỹ thuật `reports/`, model card, tag `v1.0.0`.

**Cổng G6:** phát hành v1.0.
