"""Tín hiệu độ khó tầng clip (S1, S2, S3, S4, S5, S6, S7).

Mọi hàm nhận numpy array, không phụ thuộc framework, trả về điểm "càng cao càng khó".
Quy ước hình dạng:
    probs_ens : (M, N, C)  xác suất của M thành viên ensemble, N mẫu, C lớp
    probs     : (N, C)     xác suất (đã hiệu chuẩn) của một model
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """H[p] = -sum_c p_c log p_c (nat)."""
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def margin_score(probs: np.ndarray) -> np.ndarray:
    """S1 = 1 - (p_(1) - p_(2)). Trong [0, 1]."""
    part = np.sort(probs, axis=-1)
    return 1.0 - (part[..., -1] - part[..., -2])


def bald_score(probs_ens: np.ndarray) -> np.ndarray:
    """S2 = H[mean_m p_m] - mean_m H[p_m]  (mutual information, >= 0)."""
    mean_p = probs_ens.mean(axis=0)
    return np.maximum(entropy(mean_p) - entropy(probs_ens).mean(axis=0), 0.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence giữa hai phân phối theo trục cuối (nat, trong [0, ln2])."""
    m = 0.5 * (p + q)
    return entropy(m) - 0.5 * (entropy(p) + entropy(q))


def teacher_student_score(p_teacher: np.ndarray, p_student: np.ndarray) -> np.ndarray:
    """S3 = JS(p_teacher || p_student)."""
    return js_divergence(p_teacher, p_student)


def temporal_instability(probs: np.ndarray, session_ids: np.ndarray, t_index: np.ndarray, k: int = 2) -> np.ndarray:
    """S4: bất ổn định thời gian.

    S4(x_t) = 1/(2k) sum_{j=-k..k, j!=0} JS(p_t || p_{t+j}) * 1[yhat_{t-k} == yhat_{t+k}]

    Cửa sổ được sắp theo (session_ids, t_index). Lân cận thiếu (đầu/cuối phiên) được bỏ qua
    và chuẩn hoá theo số lân cận thực có. Hệ số chỉ báo loại các ranh giới hành vi thật
    (hai đầu xa khác lớp dự đoán) -> điểm 0.
    """
    n = probs.shape[0]
    out = np.zeros(n)
    yhat = probs.argmax(-1)
    order = np.lexsort((t_index, session_ids))
    sess_sorted = session_ids[order]
    starts = np.flatnonzero(np.r_[True, sess_sorted[1:] != sess_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        L = len(idx)
        for pos in range(L):
            lo, hi = pos - k, pos + k
            if lo < 0 or hi >= L:
                # không đủ ngữ cảnh hai đầu -> không kết luận được về ranh giới
                continue
            if yhat[idx[lo]] != yhat[idx[hi]]:
                continue
            neigh = [idx[pos + j] for j in range(-k, k + 1) if j != 0]
            out[idx[pos]] = js_divergence(probs[idx[pos]][None, :], probs[neigh]).mean()
    return out


def cross_view_disagreement(probs_views: np.ndarray) -> np.ndarray:
    """S5 = H[mean_v p_v] - mean_v H[p_v]; probs_views có hình (V, N, C)."""
    return bald_score(probs_views)


def knn_rarity(z_query: np.ndarray, z_ref: np.ndarray, k: int = 10, exclude_self: bool = False) -> np.ndarray:
    """S6: trung bình (1 - cosine) tới k láng giềng gần nhất trong z_ref.

    Dùng cho pool vừa phải; với pool lớn thay bằng FAISS (xem SPEC.md).
    """
    zq = z_query / (np.linalg.norm(z_query, axis=1, keepdims=True) + EPS)
    zr = z_ref / (np.linalg.norm(z_ref, axis=1, keepdims=True) + EPS)
    sims = zq @ zr.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    kk = min(k, zr.shape[0] - int(exclude_self))
    top = -np.sort(-sims, axis=1)[:, :kk]
    return (1.0 - top).mean(axis=1)


def keyword_rarity(keywords_per_item: list[set[str]]) -> np.ndarray:
    """S7 (VLMine): max_w log(N / (1 + df(w))). Mục không có từ khoá -> 0."""
    n = len(keywords_per_item)
    df: dict[str, int] = {}
    for kws in keywords_per_item:
        for w in kws:
            df[w] = df.get(w, 0) + 1
    out = np.zeros(n)
    for i, kws in enumerate(keywords_per_item):
        if kws:
            out[i] = max(np.log(n / (1 + df[w])) for w in kws)
    return out
