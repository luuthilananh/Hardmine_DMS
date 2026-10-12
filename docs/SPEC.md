# SPEC — Đặc tả kỹ thuật

Tài liệu thiết kế đầy đủ (lý do chọn, tổng quan nghiên cứu) ở tài liệu "DMS Hard-Case Mining" do người
dùng cung cấp. File này là hợp đồng kỹ thuật: hình dạng dữ liệu, giao diện hàm, công thức.

## 1. Ký hiệu

- `N` số cửa sổ/mẫu, `C` số lớp, `M` số thành viên ensemble, `V` số góc camera, `d` chiều embedding.
- Mọi điểm tín hiệu: **càng cao càng khó**.
- `z` embedding gốc; `z_tilde` embedding đã khử danh tính và chuẩn hoá L2.

## 2. Bảng cửa sổ (data contract)

`cache/windows/<dataset>.csv`, một dòng một cửa sổ. File này **được commit** (nhỏ, không chứa ảnh) để Kaggle dùng chung; `video_path` là đường dẫn tương đối so với thư mục gốc dataset:

| cột | kiểu | ý nghĩa |
|---|---|---|
| `window_id` | int64 | khoá duy nhất |
| `dataset` | str | `driveact`, `dmd`, … |
| `session_id` | str | phiên quay |
| `driver_id` | str | tài xế (dùng để chia split, khử danh tính) |
| `view` | str | góc camera, ví dụ `a_column_driver` |
| `video_path` | str | đường dẫn video, tương đối so với `data/<dataset>/` |
| `t_start`, `t_end` | float | giây trong phiên |
| `frame_start`, `frame_end` | int | chỉ số frame |
| `label` | int | lớp đa số (−1 nếu không có nhãn) |
| `label_frac` | float | tỷ lệ thời gian của lớp đa số |
| `is_transition` | bool | `label_frac < 0.6` |
| `split` | str | `train` / `val` / `test` theo split chính thức |
| `keep` | bool | còn lại sau lọc thô và khử trùng |

Embedding: `cache/features/<dataset>/<backbone>/emb_<shard>.npy` (float16) + `index.csv` (window_id, shard, row),
đọc bằng `hardmine.features.extract.load()`.

Tầng sự kiện: `cache/eyes/<dataset>/<session_id>.npz` với `q_true` (uint8 theo frame), `fps`, và sau khi chạy
model `q_pred` (float16).

## 3. Lọc thô và khử trùng

- Năng lượng chuyển động: `E_mot = mean_t mean_pixels |I_{t+1} - I_t|` trên ảnh xám thu nhỏ 112 px.
  Cửa sổ có `E_mot` dưới phân vị 5% **và** thuộc đoạn tĩnh dài ≥ 10 s: giữ 1 đại diện mỗi 10 s.
- Độ sáng trung bình ngoài [10, 245] (ảnh 8-bit) hoặc phương sai Laplacian dưới phân vị 1%: loại.
- pHash: ảnh xám 32×32 → DCT 2D → khối 8×8 tần số thấp (bỏ DC) → so với trung vị → 64 bit.
  Trùng khi Hamming ≤ `tau_h` (mặc định 6, tinh chỉnh bằng cách xem 50 cặp ngẫu nhiên ở các mức 4/6/8/10).
  Chỉ so trong cùng phiên. Giữ cửa sổ có độ nét cao nhất mỗi nhóm.
- SemDeDup: k-means (k ≈ N/100) trên `z_tilde`; trong mỗi cụm loại cặp cosine ≥ 1 − ε (ε mặc định 0,02).
- **Khử trùng chỉ áp dụng cho pool chưa gán nhãn.** Không khử trùng tập val/test.

## 4. Đặc trưng, model, tín hiệu

### 4.1 Backbone (chọn ở P1.6)
Ứng viên: V-JEPA 2 ViT-L, VideoMAE V2 ViT-B/L (video, 16 frame), SigLIP 2 (ảnh, trung bình 4 frame).
Ảnh NIR 1 kênh → lặp thành 3 kênh. Tiêu chí chọn: macro-F1 linear probe trên val, rồi thông lượng.
GPU < 16 GB → ưu tiên ViT-B.

### 4.2 Khử danh tính — `features/deidentify.py` (đã có)
`DriverProjector(rank).fit(z_train, driver_train[, labels, class_balanced])`. Fit chỉ trên tài xế train.
Kết quả kiểm chứng trên dữ liệu tổng hợp: r < số chiều thật của biến thiên tài xế chỉ xoá một phần → **phải
quét r**. Biến thể `class_balanced=True` giảm rủi ro xoá nhầm thông tin hành vi khi tài xế có phân bố
hành vi khác nhau.

### 4.3 Ensemble head
M = 5, MLP 2 lớp (d → 512 → C), dropout 0,5, AdamW lr 1e-3, weight decay 1e-2, tối đa 100 epoch, early
stopping trên val loss. Bootstrap tập đã gán nhãn cho mỗi thành viên. Head huấn luyện trên `z` (không phải
`z_tilde`) — khử danh tính chỉ dùng cho khoảng cách/phân cụm, trừ khi ablation chứng minh khác.

### 4.4 Hiệu chuẩn nhiệt độ
Một T chung cho ensemble, tối ưu NLL trên val bằng tìm kiếm 1 chiều (T ∈ [0.5, 5]). Báo cáo ECE 15 bin.

### 4.5 Tín hiệu tầng clip — `signals/core.py` (đã có)
| Tên | Hàm | Ghi chú |
|---|---|---|
| S1 | `margin_score(p_mean)` | |
| S2 | `bald_score(p_ens)` | |
| S3 | `teacher_student_score(p_teacher, p_student)` | P5 |
| S4 | `temporal_instability(p_mean, session_ids, t_index, k=2)` | cần các cửa sổ liền kề của cùng phiên trong pool |
| S5 | `cross_view_disagreement(p_views)` | P5 |
| S6 | `knn_rarity(z_tilde_pool, z_tilde_labeled, k=10)` | pool > 50k: dùng FAISS IndexFlatIP |
| S7 | `keyword_rarity(list_of_keyword_sets)` | P5, chỉ trên ứng viên |

### 4.6 Bộ hiệu chuẩn và chọn mẫu — `select/core.py` (đã có)
`HardnessCalibrator(lam=1e-3).fit(signals_val, errors_val).score(signals_pool)`; tên tín hiệu ở val và pool
phải trùng khớp. `select_batch(h, z_tilde, B, alpha=10, extra_candidates=pareto_idx, session_ids, t_sec,
min_gap_s=10, pred_class, max_class_frac=0.25)`. Kênh Pareto: `pareto_front` trên (S2, S6[, S7]) của top
`5·alpha·B` theo h.

### 4.7 VLM (P5)
Prompt cố định, đầu ra JSON với từ vựng đóng:
`{"hands": [...], "held_object": [...], "gaze": [...], "eyes": [...], "lighting": [...], "accessories": [...], "occlusion": [...]}`.
Từ khoá ngoài từ vựng bị bỏ. Sinh 2 lần với nhiệt độ khác nhau; chỉ giữ từ khoá xuất hiện cả hai lần.
Cache theo `window_id`.

## 5. Tầng sự kiện — `events/core.py` (đã có)

- `extract_events(q, fps, thr=0.5, fill_gap_s=0.0)`; lớp theo `classify_duration`:
  0 chớp mắt (< 1 s), 1 ngủ gật [1, 2), 2 vùng đệm [2, 3), 3 ngủ [3, 6), 4 mất phản ứng (≥ 6 s).
- `threshold_proximity(q, events)`, `fragmentation(q, events, max_gap_s=0.2)`.
- `event_metrics(pred, gt, hours, iou_thr=0.5)`.
- **Hai chế độ hậu xử lý phải báo cáo riêng:** `fill_gap_s=0` (thô) và `fill_gap_s=0.2` (sản phẩm). Tín hiệu
  `fragmentation` chỉ nhìn thấy khe ở chế độ thô; vì vậy mining dùng chế độ thô, đánh giá báo cáo cả hai.
- Ánh xạ nhãn: nhãn "nhắm" và "đang nhắm/đang mở" (nếu dataset có) → nhắm khi mi che > 50% đồng tử; ghi
  ánh xạ cụ thể vào DECISIONS.md.

## 6. Vòng mô phỏng gán nhãn

```
input: pool U (train split, keep=True), val, test, strategy, B, R vòng, seed
L0 = strategy.initial(U, B0)          # TypiClust/ProbCover/centroid; Random cho baseline Random
for r in 1..R:
    fit ensemble trên L; hiệu chuẩn T trên val
    log metrics trên val (macro-F1, recall đuôi, ECE)
    A = strategy.select(U \ L, B, context)    # context: model, val, embeddings, metadata
    assert A ∩ L = ∅ và |A| = B
    L = L ∪ A                                  # "mở nhãn"
cuối cùng: fit lại, đánh giá test một lần; ghi rounds.csv
```

### 6.3 Baseline
- Entropy / Margin / BALD: top-B (bản gốc) **và** bản có đa dạng (top-10B → k-means B → gần tâm), như
  Gupte et al. 2024.
- Core-set: k-center greedy trên `z`.
- TypiClust: k-means B cụm (giới hạn 500), chọn điểm có typicality cao nhất (nghịch đảo khoảng cách trung
  bình tới 20 láng giềng) trong cụm chưa có nhãn lớn nhất.
- ProbCover: đồ thị bán kính δ (δ theo độ thuần cụm 0,95), greedy max coverage.
- DropQuery: dropout ρ = 0,75 lên đặc trưng, M = 3 lần; ứng viên = mẫu có > M/2 dự đoán khác bản gốc;
  k-means B, lấy gần tâm. Khởi tạo bằng tâm cụm.
- Tham khảo cài đặt: github.com/sanketx/AL-foundation-models (đối chiếu, không bắt buộc dùng).

## 7. Config

```yaml
run_name: mvp_clip_driveact
seed: [0, 1, 2]
dataset: {name: driveact, root: data/driveact, view: a_column_driver, split: [0, 1, 2]}
windows: {length_s: 2.0, stride_s: 1.0, fps_sample: 8, min_label_frac: 0.6}
backbone: {name: vjepa2_vitl, batch_size: 16, fp16: true}
deidentify: {rank: 8, class_balanced: false}
al: {init_frac: 0.01, budget_frac: 0.005, rounds: 10}
strategies: [random, entropy, bald, typiclust, dropquery, hardmine]
hardmine: {signals: [margin, bald, temporal, rarity], alpha: 10, min_gap_s: 10, max_class_frac: 0.25, lam: 0.001}
```

## 8. Giao thức tự quay NIR (P6, người dùng thực hiện)

Camera NIR 940 nm, đặt ở vị trí DMS (cột lái hoặc gương chiếu hậu), 30 fps. 10–20 người, cân bằng giới
và có người đeo kính. Mỗi người, khi xe đứng yên hoặc trên đường an toàn có người giám sát:
nhìn gương trái/phải/giữa, cluster, IVI, đùi (mỗi lần 3–4 s, trước đó nhìn đường 4 s); dùng điện thoại
ở đùi và trên vô lăng; nhắm mắt 1–2 s (10 lần), ≥ 3 s (5 lần); đeo kính râm 1 phút; khẩu trang 1 phút.
Ghi nhãn thời gian bằng nút bấm hoặc gán sau. **Không thực hiện nhắm mắt khi xe đang chạy.**
