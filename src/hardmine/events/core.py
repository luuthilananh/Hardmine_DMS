"""Tầng sự kiện: từ xác suất nhắm mắt theo frame q_t sang sự kiện nhắm mắt và tín hiệu độ khó.

Ngưỡng theo Euro NCAP Driver Engagement 2026:
    microsleep: nhắm 1–2 s; sleep: nhắm liên tục >= 3 s; unresponsive: nhắm >= 6 s.
Khoảng 2–3 s không được protocol định nghĩa -> lớp "buffer" để báo cáo riêng.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLDS_S = (1.0, 2.0, 3.0, 6.0)
CLASS_NAMES = ("blink", "microsleep", "buffer", "sleep", "unresponsive")


def classify_duration(d: float) -> int:
    """0 blink (<1s), 1 microsleep [1,2), 2 buffer [2,3), 3 sleep [3,6), 4 unresponsive (>=6)."""
    return int(np.searchsorted(THRESHOLDS_S, d, side="right"))


@dataclass
class Event:
    start: int  # chỉ số frame đầu (bao gồm)
    end: int    # chỉ số frame cuối (không bao gồm)
    fps: float

    @property
    def duration(self) -> float:
        return (self.end - self.start) / self.fps

    @property
    def cls(self) -> int:
        return classify_duration(self.duration)


def extract_events(q: np.ndarray, fps: float, thr: float = 0.5, fill_gap_s: float = 0.0,
                   min_dur_s: float = 0.0) -> list[Event]:
    """Ghép các frame liên tiếp q_t >= thr thành sự kiện.

    fill_gap_s: lấp khe "mở" ngắn hơn hoặc bằng giá trị này (hậu xử lý chống cắt vụn).
    Mặc định 0 để tín hiệu cắt vụn còn nhìn thấy khe; pipeline sản phẩm có thể đặt > 0.
    """
    closed = np.asarray(q) >= thr
    if not closed.any():
        return []
    diff = np.diff(np.r_[0, closed.astype(int), 0])
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    runs = list(zip(starts, ends))
    if fill_gap_s > 0 and runs:
        max_gap = int(round(fill_gap_s * fps))
        merged = [list(runs[0])]
        for s, e in runs[1:]:
            if s - merged[-1][1] <= max_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        runs = [tuple(r) for r in merged]
    min_len = int(round(min_dur_s * fps))
    return [Event(int(s), int(e), fps) for s, e in runs if e - s >= max(min_len, 1)]


def _boundary_uncertain_count(q: np.ndarray, center: int, half: int = 2, lo: float = 0.2, hi: float = 0.8) -> int:
    a, b = max(center - half, 0), min(center + half + 1, len(q))
    w = q[a:b]
    return int(((w >= lo) & (w <= hi)).sum())


def threshold_proximity(q: np.ndarray, events: list[Event]) -> np.ndarray:
    """S_thr = exp(-min_theta |d - theta| / sigma_d), sigma_d = dt * (u_s + u_e + 1).

    u_s, u_e: số frame lưỡng lự (q trong [0.2, 0.8]) trong 5 frame quanh mép đầu và mép cuối.
    Điểm cao = thời lượng nằm sát một ngưỡng phân lớp VÀ mép sự kiện không chắc chắn.
    """
    out = np.zeros(len(events))
    for i, ev in enumerate(events):
        dt = 1.0 / ev.fps
        u_s = _boundary_uncertain_count(q, ev.start)
        u_e = _boundary_uncertain_count(q, ev.end - 1)
        sigma = dt * (u_s + u_e + 1)
        dist = min(abs(ev.duration - th) for th in THRESHOLDS_S)
        out[i] = np.exp(-dist / sigma)
    return out


def fragmentation(q: np.ndarray, events: list[Event], max_gap_s: float = 0.2) -> np.ndarray:
    """S_frag cho mỗi sự kiện: = q_bar_gap nếu gộp với sự kiện liền kề (qua khe <= max_gap_s)
    làm đổi lớp so với lớp lớn nhất của hai mảnh; ngược lại 0. Lấy max qua hai phía.

    Bắt lỗi nguy hiểm nhất: một lần "sleep" bị đếm thành hai lần "microsleep".
    """
    out = np.zeros(len(events))
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        gap_frames = b.start - a.end
        if gap_frames <= 0 or gap_frames / a.fps > max_gap_s:
            continue
        merged = classify_duration((b.end - a.start) / a.fps)
        if merged != max(a.cls, b.cls):
            score = float(np.mean(q[a.end:b.start]))
            out[i] = max(out[i], score)
            out[i + 1] = max(out[i + 1], score)
    return out


def event_matching(pred: list[Event], gt: list[Event], iou_thr: float = 0.5) -> list[tuple[int, int]]:
    """Ghép tham lam theo IoU thời gian giảm dần; trả về cặp (chỉ số pred, chỉ số gt)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            inter = max(0, min(p.end, g.end) - max(p.start, g.start))
            union = max(p.end, g.end) - min(p.start, g.start)
            if union > 0 and inter / union >= iou_thr:
                pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def event_metrics(pred: list[Event], gt: list[Event], hours: float, iou_thr: float = 0.5,
                  classes: tuple[int, ...] = (1, 3, 4)) -> dict:
    """F1 sự kiện theo lớp (khớp khi IoU >= thr VÀ cùng lớp), MAE thời lượng trên các cặp khớp
    theo IoU, và số sự kiện báo sai mỗi giờ cho các lớp nguy hiểm (microsleep, sleep, unresponsive).
    """
    matches = event_matching(pred, gt, iou_thr)
    res: dict = {}
    for c in classes:
        tp = sum(1 for i, j in matches if pred[i].cls == c and gt[j].cls == c)
        n_pred = sum(1 for e in pred if e.cls == c)
        n_gt = sum(1 for e in gt if e.cls == c)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gt if n_gt else 0.0
        res[f"f1_{CLASS_NAMES[c]}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        res[f"fa_per_hour_{CLASS_NAMES[c]}"] = (n_pred - tp) / hours if hours > 0 else float("nan")
    res["duration_mae_s"] = float(np.mean([abs(pred[i].duration - gt[j].duration) for i, j in matches])) if matches else float("nan")
    return res
