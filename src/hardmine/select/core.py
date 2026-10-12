"""Bộ hiệu chuẩn độ khó (bước 5) và chọn mẫu (bước 6)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """R_j(x) trong [0, 1]; giá trị bằng nhau nhận cùng hạng trung bình."""
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x)
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def rank_matrix(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = sorted(signals)
    return np.column_stack([percentile_rank(signals[n]) for n in names]), names


@dataclass
class HardnessCalibrator:
    """Hồi quy logistic, trọng số không âm, phạt L1, học bằng projected gradient descent.

        min_{w>=0, b} mean BCE(e, sigmoid(R w + b)) + lam * ||w||_1

    e = 1 nếu model hiện tại dự đoán sai mẫu validation.
    Rank được tính *trong* từng tập (val và pool) riêng, nên điểm so sánh được giữa hai tập.
    """
    lam: float = 1e-3
    lr: float = 0.5
    n_iter: int = 3000
    names: list[str] = field(default_factory=list)
    w: np.ndarray | None = None
    b: float = 0.0

    def fit(self, signals_val: dict[str, np.ndarray], errors_val: np.ndarray) -> "HardnessCalibrator":
        R, self.names = rank_matrix(signals_val)
        e = np.asarray(errors_val, dtype=float)
        w = np.zeros(R.shape[1])
        p0 = np.clip(e.mean(), 1e-4, 1 - 1e-4)
        b = float(np.log(p0 / (1 - p0)))
        for _ in range(self.n_iter):
            p = 1 / (1 + np.exp(-(R @ w + b)))
            g = p - e
            w = np.maximum(w - self.lr * (R.T @ g / len(e) + self.lam), 0.0)
            b -= self.lr * g.mean()
        self.w, self.b = w, b
        return self

    def score(self, signals_pool: dict[str, np.ndarray]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Gọi fit() trước.")
        if sorted(signals_pool) != self.names:
            raise ValueError(f"Tín hiệu pool {sorted(signals_pool)} khác lúc fit {self.names}")
        R, _ = rank_matrix(signals_pool)
        return 1 / (1 + np.exp(-(R @ self.w + self.b)))

    def weights(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.w)))


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Chỉ số các điểm không bị trội (mọi tiêu chí: càng lớn càng tốt). O(N^2) — đủ cho tập ứng viên."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if dominated.any():
            keep[i] = False
    return np.flatnonzero(keep)


def select_batch(h: np.ndarray, z: np.ndarray, budget: int, *, alpha: int = 10,
                 extra_candidates: np.ndarray | None = None,
                 session_ids: np.ndarray | None = None, t_sec: np.ndarray | None = None,
                 min_gap_s: float = 10.0,
                 pred_class: np.ndarray | None = None, max_class_frac: float = 0.25,
                 seed: int = 0) -> np.ndarray:
    """Bước 6: top alpha*B theo h (hợp với extra_candidates, ví dụ mặt Pareto) -> k-means B cụm trên z
    -> duyệt mẫu theo khoảng cách tới tâm, áp NMS thời gian và hạn ngạch lớp.

    Trả về chỉ số (trong pool) của <= budget mẫu.
    """
    from sklearn.cluster import KMeans

    n = len(h)
    budget = min(budget, n)
    cand = np.argsort(-h)[: min(n, alpha * budget)]
    if extra_candidates is not None:
        cand = np.unique(np.r_[cand, extra_candidates])
    if len(cand) <= budget:
        order = cand[np.argsort(-h[cand])]
        return _filter(order, budget, session_ids, t_sec, min_gap_s, pred_class, max_class_frac)

    km = KMeans(n_clusters=budget, n_init=3, random_state=seed).fit(z[cand])
    dist = np.linalg.norm(z[cand] - km.cluster_centers_[km.labels_], axis=1)
    # Vòng 1: đại diện gần tâm nhất mỗi cụm (ưu tiên cụm có h trung bình cao).
    # Vòng 2+: các mẫu còn lại theo h giảm dần, để bù chỗ bị NMS/hạn ngạch loại.
    reps, rest = [], []
    cluster_h = {c: h[cand[km.labels_ == c]].mean() for c in range(budget)}
    for c in sorted(cluster_h, key=lambda c: -cluster_h[c]):
        members = np.flatnonzero(km.labels_ == c)
        members = members[np.argsort(dist[members])]
        reps.append(cand[members[0]])
        rest.extend(cand[members[1:]])
    rest = np.array(rest, dtype=int)
    rest = rest[np.argsort(-h[rest])] if len(rest) else rest
    order = np.r_[np.array(reps, dtype=int), rest]
    return _filter(order, budget, session_ids, t_sec, min_gap_s, pred_class, max_class_frac)


def _filter(order, budget, session_ids, t_sec, min_gap_s, pred_class, max_class_frac):
    chosen: list[int] = []
    per_class: dict[int, int] = {}
    cap = max(1, int(np.floor(max_class_frac * budget))) if pred_class is not None else None
    for i in order:
        if len(chosen) >= budget:
            break
        if session_ids is not None and t_sec is not None:
            clash = any(session_ids[j] == session_ids[i] and abs(t_sec[j] - t_sec[i]) < min_gap_s for j in chosen)
            if clash:
                continue
        if cap is not None:
            c = int(pred_class[i])
            if per_class.get(c, 0) >= cap:
                continue
            per_class[c] = per_class.get(c, 0) + 1
        chosen.append(int(i))
    return np.array(chosen, dtype=int)


def coverage_boost(h: np.ndarray, cell_ids: np.ndarray, labeled_cell_counts: dict, beta: float = 1.0) -> np.ndarray:
    """h'(x) = h(x) * (1 + beta / (1 + n_c(x))) — ưu tiên ô điều kiện còn ít nhãn (mục 11.2)."""
    n_c = np.array([labeled_cell_counts.get(c, 0) for c in cell_ids], dtype=float)
    return h * (1 + beta / (1 + n_c))
