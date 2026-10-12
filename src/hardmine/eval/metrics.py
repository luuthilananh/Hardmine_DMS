"""Metric tầng clip và thống kê."""
from __future__ import annotations

import numpy as np


def hhr_at_k(scores: np.ndarray, is_error: np.ndarray, k: int) -> float:
    """Tỷ lệ ca khó (model sai) trong top-K theo điểm."""
    top = np.argsort(-scores)[:k]
    return float(np.mean(is_error[top]))


def lift_at_k(scores: np.ndarray, is_error: np.ndarray, k: int) -> float:
    base = float(np.mean(is_error))
    return hhr_at_k(scores, is_error, k) / base if base > 0 else float("nan")


def error_auroc(scores: np.ndarray, is_error: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if is_error.min() == is_error.max():
        return float("nan")
    return float(roc_auc_score(is_error, scores))


def aubc(n_labels: np.ndarray, metric: np.ndarray) -> float:
    """Diện tích dưới đường cong ngân sách, chuẩn hoá theo độ dài trục (hình thang)."""
    n = np.asarray(n_labels, float)
    m = np.asarray(metric, float)
    span = n[-1] - n[0]
    return float(np.sum((m[1:] + m[:-1]) * np.diff(n)) / 2 / span) if span > 0 else float(m[0])


def tail_recall(y_true: np.ndarray, y_pred: np.ndarray, class_counts: dict, frac: float = 0.3) -> float:
    """Recall trung bình của nhóm frac lớp ít mẫu nhất (đếm trên tập train đầy đủ)."""
    classes = sorted(class_counts, key=lambda c: class_counts[c])
    tail = classes[: max(1, int(round(frac * len(classes))))]
    recs = [np.mean(y_pred[y_true == c] == c) for c in tail if np.any(y_true == c)]
    return float(np.mean(recs)) if recs else float("nan")


def paired_win(a_runs: np.ndarray, b_runs: np.ndarray, alpha: float = 0.05) -> int:
    """+1 nếu A thắng B có ý nghĩa (paired t-test hai phía), -1 nếu thua, 0 nếu hoà."""
    from scipy.stats import ttest_rel
    d = np.asarray(a_runs) - np.asarray(b_runs)
    if np.allclose(d, d[0]):
        return int(np.sign(d[0])) if d[0] != 0 else 0
    t, p = ttest_rel(a_runs, b_runs)
    return 0 if p >= alpha else (1 if t > 0 else -1)


def bootstrap_ci_by_group(values: np.ndarray, groups: np.ndarray, stat=np.mean, n_boot: int = 1000,
                          seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy 95% bằng bootstrap theo nhóm (theo tài xế), không theo frame."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_g = {g: values[groups == g] for g in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        stats.append(stat(np.concatenate([by_g[g] for g in pick])))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
