"""Khử danh tính tài xế khỏi embedding (đóng góp riêng của dự án).

Ý tưởng: mỗi phiên quay ~ một tài xế. Tâm embedding theo tài xế mu_d chứa chủ yếu thông tin
ngoại hình. Lấy r hướng chính của {mu_d - mu_bar} bằng PCA (ma trận U, d x r), rồi chiếu loại bỏ:

    z_tilde = (I - U U^T) z / ||(I - U U^T) z||

Lưu ý quan trọng: tâm theo tài xế cũng chứa *phân bố hành vi* của tài xế đó. Nếu mỗi tài xế
làm các hành vi với tỷ lệ khác nhau, PCA sẽ ăn mất một phần thông tin hành vi. Vì vậy có tuỳ chọn
`class_balanced`: tính tâm tài xế trên trung bình các tâm (tài xế, lớp) khi đã có nhãn, để
hướng tìm được ít lẫn hành vi hơn. Luôn kiểm chứng bằng `diagnostics` (G1 trong MILESTONES).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriverProjector:
    rank: int = 8
    U: np.ndarray | None = None  # (d, r)

    def fit(self, z: np.ndarray, driver_ids: np.ndarray, labels: np.ndarray | None = None,
            class_balanced: bool = False) -> "DriverProjector":
        drivers = np.unique(driver_ids)
        if len(drivers) < 2:
            raise ValueError("Cần ít nhất 2 tài xế để ước lượng hướng danh tính.")
        centers = []
        for d in drivers:
            m = driver_ids == d
            if class_balanced and labels is not None:
                cls = np.unique(labels[m])
                centers.append(np.mean([z[m & (labels == c)].mean(0) for c in cls], axis=0))
            else:
                centers.append(z[m].mean(0))
        C = np.asarray(centers)
        C = C - C.mean(0, keepdims=True)
        r = min(self.rank, len(drivers) - 1, z.shape[1])
        # SVD: hàng của Vt là hướng chính
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        self.U = Vt[:r].T
        return self

    def transform(self, z: np.ndarray, normalize: bool = True) -> np.ndarray:
        if self.U is None:
            raise RuntimeError("Gọi fit() trước.")
        zp = z - (z @ self.U) @ self.U.T
        if normalize:
            zp = zp / (np.linalg.norm(zp, axis=1, keepdims=True) + 1e-12)
        return zp


def diagnostics(z: np.ndarray, z_tilde: np.ndarray, driver_ids: np.ndarray, labels: np.ndarray,
                seed: int = 0) -> dict:
    """Đo độ chính xác đoán ID tài xế và đoán hành vi (logistic regression, 5-fold, chia theo mẫu).

    Kỳ vọng sau khử danh tính: driver_acc giảm rõ, behavior_acc không giảm đáng kể.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def acc(x, y):
        clf = LogisticRegression(max_iter=2000)
        return float(cross_val_score(clf, x, y, cv=5).mean())

    return {
        "driver_acc_before": acc(z, driver_ids),
        "driver_acc_after": acc(z_tilde, driver_ids),
        "behavior_acc_before": acc(z, labels),
        "behavior_acc_after": acc(z_tilde, labels),
    }
