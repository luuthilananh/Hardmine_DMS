"""Dữ liệu tổng hợp có ca khó biết trước.

Mục đích: (1) chạy toàn bộ pipeline không cần dữ liệu thật, (2) chứng minh mỗi tín hiệu bắt
đúng loại lỗi nó nhắm tới. Không dùng để báo cáo hiệu năng.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardmine.events.core import Event, extract_events


@dataclass
class SyntheticClips:
    z: np.ndarray            # (N, d) embedding
    y: np.ndarray            # (N,) nhãn hành vi
    driver: np.ndarray       # (N,) ID tài xế (= phiên)
    t_sec: np.ndarray        # (N,) thời điểm cửa sổ trong phiên
    is_rare: np.ndarray      # (N,) mẫu thuộc lớp đuôi
    is_boundary: np.ndarray  # (N,) mẫu nằm giữa hai lớp dễ nhầm


def make_clip_embeddings(n_drivers: int = 12, windows_per_driver: int = 300, n_classes: int = 8,
                         dim: int = 64, driver_strength: float = 6.0, class_strength: float = 3.5,
                         rare_classes: int = 2, rare_frac: float = 0.02, seed: int = 0) -> SyntheticClips:
    """Embedding = hướng lớp + hướng tài xế (mạnh hơn) + nhiễu.

    - driver_strength > class_strength mô phỏng phát hiện "embedding gom theo ngoại hình".
    - Lớp đuôi: `rare_classes` lớp cuối chỉ chiếm rare_frac mỗi lớp.
    - Ca biên: 10% mẫu của lớp 0 được trộn 50/50 với hướng lớp 1.
    - Hành vi kéo dài theo đoạn (segment 5–20 cửa sổ) để tín hiệu thời gian có ý nghĩa.
    """
    rng = np.random.default_rng(seed)
    class_dirs = rng.normal(size=(n_classes, dim))
    class_dirs /= np.linalg.norm(class_dirs, axis=1, keepdims=True)
    driver_dirs = rng.normal(size=(n_drivers, dim))
    driver_dirs /= np.linalg.norm(driver_dirs, axis=1, keepdims=True)

    common = n_classes - rare_classes
    probs = np.r_[np.full(common, (1 - rare_frac * rare_classes) / common), np.full(rare_classes, rare_frac)]

    Z, Y, D, T, R, B = [], [], [], [], [], []
    for d in range(n_drivers):
        t = 0
        while t < windows_per_driver:
            c = rng.choice(n_classes, p=probs)
            seg = int(rng.integers(5, 21))
            for _ in range(min(seg, windows_per_driver - t)):
                v = class_strength * class_dirs[c] + driver_strength * driver_dirs[d] + rng.normal(scale=0.6, size=dim)
                boundary = c == 0 and rng.random() < 0.10
                if boundary:
                    v = v - 0.5 * class_strength * class_dirs[0] + 0.5 * class_strength * class_dirs[1]
                Z.append(v); Y.append(c); D.append(d); T.append(float(t)); R.append(c >= common); B.append(boundary)
                t += 1
    return SyntheticClips(np.asarray(Z), np.asarray(Y), np.asarray(D), np.asarray(T), np.asarray(R), np.asarray(B))


@dataclass
class SyntheticEyeSeq:
    q_true: np.ndarray        # (T,) 0/1 nhắm mắt thật
    q_pred: np.ndarray        # (T,) xác suất nhắm do "model" giả lập
    fps: float
    gt_events: list[Event]
    hard_kind: dict           # chỉ số sự kiện thật -> "near_threshold" | "fragmented" | "easy"


def make_eye_sequence(minutes: float = 20.0, fps: float = 30.0, seed: int = 0) -> SyntheticEyeSeq:
    """Chuỗi độ mở mắt với sự kiện biết trước và lỗi model có chủ đích.

    - Chớp mắt thường 0.1–0.4 s (dễ).
    - Ngủ gật 1.2–1.8 s, ngủ 3.5–5 s (dễ).
    - Ca sát ngưỡng: thời lượng 0.95–1.05 s hoặc 2.9–3.1 s, mép bị làm mờ (q_pred lưỡng lự).
    - Ca cắt vụn: sự kiện ngủ 3.2–4 s nhưng q_pred bị "mở" 3 frame ở giữa.
    """
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 * fps)
    q_true = np.zeros(T)
    q_pred = np.clip(rng.normal(0.05, 0.03, T), 0, 1)
    events: list[tuple[int, int, str]] = []
    t = int(2 * fps)
    while t < T - int(10 * fps):
        r = rng.random()
        if r < 0.80:
            dur, kind = rng.uniform(0.1, 0.4), "easy"
        elif r < 0.86:
            dur, kind = rng.uniform(1.2, 1.8), "easy"
        elif r < 0.89:
            dur, kind = rng.uniform(3.5, 5.0), "easy"
        elif r < 0.95:
            dur, kind = (rng.uniform(0.95, 1.05) if rng.random() < 0.5 else rng.uniform(2.9, 3.1)), "near_threshold"
        else:
            dur, kind = rng.uniform(3.2, 4.0), "fragmented"
        n = max(1, int(round(dur * fps)))
        s, e = t, t + n
        q_true[s:e] = 1
        q_pred[s:e] = np.clip(rng.normal(0.95, 0.03, n), 0, 1)
        if kind == "near_threshold":
            for k in range(3):  # mép lưỡng lự
                if s + k < e:
                    q_pred[s + k] = 0.5 + 0.1 * (k - 1)
                if e - 1 - k >= s:
                    q_pred[e - 1 - k] = 0.5 + 0.1 * (k - 1)
        if kind == "fragmented":
            mid = (s + e) // 2
            q_pred[mid - 1: mid + 2] = rng.uniform(0.3, 0.45, 3)
        events.append((s, e, kind))
        t = e + int(rng.uniform(2, 6) * fps)

    gt = extract_events(q_true, fps)
    kind_by_start = {s: k for s, _, k in events}
    hard = {i: kind_by_start.get(ev.start, "easy") for i, ev in enumerate(gt)}
    return SyntheticEyeSeq(q_true, q_pred, fps, gt, hard)
