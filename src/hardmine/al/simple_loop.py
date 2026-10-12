"""Vòng mô phỏng gán nhãn rút gọn, chạy trên CPU với head logistic regression.

Dùng cho: demo trên dữ liệu tổng hợp, CI, và làm khung tham chiếu cho `al/loop.py` đầy đủ (P2.2).
Đúng giao thức EVAL_PROTOCOL: chia theo tài xế, chọn mẫu chỉ từ pool, đánh giá theo vòng trên val,
test chỉ ở cuối.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from hardmine.eval.metrics import error_auroc, hhr_at_k
from hardmine.features.deidentify import DriverProjector
from hardmine.select.core import HardnessCalibrator, select_batch
from hardmine.signals.core import bald_score, entropy, knn_rarity, margin_score, temporal_instability

STRATEGIES = ("random", "entropy", "margin_diverse", "hardmine")


@dataclass
class LoopData:
    z: np.ndarray
    y: np.ndarray
    driver: np.ndarray
    t_sec: np.ndarray
    pool: np.ndarray   # chỉ số train
    val: np.ndarray
    test: np.ndarray


def ensemble_probs(z_tr, y_tr, z, n_classes, M=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(M):
        idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(z_tr[idx], y_tr[idx])
        p = np.full((len(z), n_classes), 1e-6)
        p[:, clf.classes_] = clf.predict_proba(z)
        out.append(p / p.sum(1, keepdims=True))
    return np.asarray(out)


def _signals(pe, z_t, z_t_lab, driver, t_sec):
    pm = pe.mean(0)
    return {
        "margin": margin_score(pm),
        "bald": bald_score(pe),
        "temporal": temporal_instability(pm, driver, t_sec),
        "rarity": knn_rarity(z_t, z_t_lab, k=5),
    }


def run_loop(data: LoopData, strategy: str, init_n: int, budget: int, rounds: int, seed: int = 0,
             deid_rank: int | None = None) -> dict:
    """Trả về {'rounds': list[dict], 'final_test_macro_f1': float, 'selected_round1': np.ndarray}."""
    assert strategy in STRATEGIES, strategy
    rng = np.random.default_rng(seed)
    C = int(data.y.max()) + 1
    train_drivers = np.unique(data.driver[data.pool])
    rank = deid_rank if deid_rank is not None else len(train_drivers) - 1
    proj = DriverProjector(rank=rank).fit(data.z[data.pool], data.driver[data.pool])
    zt = proj.transform(data.z)

    # khởi đầu: k-means trên pool, lấy điểm gần tâm (centroid init) — giống nhau cho mọi chiến lược
    km = KMeans(n_clusters=init_n, n_init=3, random_state=seed).fit(zt[data.pool])
    d = np.linalg.norm(zt[data.pool] - km.cluster_centers_[km.labels_], axis=1)
    labeled = []
    for c in range(init_n):
        m = np.flatnonzero(km.labels_ == c)
        labeled.append(data.pool[m[np.argmin(d[m])]])
    labeled = np.array(sorted(set(labeled)))

    log, sel_r1 = [], None
    for r in range(rounds + 1):
        pe_val = ensemble_probs(data.z[labeled], data.y[labeled], data.z[data.val], C, seed=seed + r)
        pm_val = pe_val.mean(0)
        f1_val = f1_score(data.y[data.val], pm_val.argmax(1), average="macro")
        unl = np.setdiff1d(data.pool, labeled)
        pe_u = ensemble_probs(data.z[labeled], data.y[labeled], data.z[unl], C, seed=seed + r)
        pm_u = pe_u.mean(0)
        err_u = pm_u.argmax(1) != data.y[unl]  # chỉ dùng để đo, không dùng để chọn
        row = {"strategy": strategy, "seed": seed, "round": r, "n_labeled": len(labeled),
               "macro_f1_val": f1_val, "base_err_pool": float(err_u.mean())}
        if r == rounds:
            log.append(row)
            break

        if strategy == "random":
            score = rng.random(len(unl))
            pick = np.argsort(-score)[:budget]
        elif strategy == "entropy":
            score = entropy(pm_u)
            pick = np.argsort(-score)[:budget]
        elif strategy == "margin_diverse":
            score = margin_score(pm_u)
            cand = np.argsort(-score)[: 10 * budget]
            km = KMeans(n_clusters=budget, n_init=3, random_state=seed).fit(zt[unl][cand])
            dd = np.linalg.norm(zt[unl][cand] - km.cluster_centers_[km.labels_], axis=1)
            pick = np.array([cand[np.flatnonzero(km.labels_ == c)][np.argmin(dd[km.labels_ == c])]
                             for c in range(budget)])
        else:  # hardmine
            s_val = _signals(pe_val, zt[data.val], zt[labeled], data.driver[data.val], data.t_sec[data.val])
            cal = HardnessCalibrator().fit(s_val, pm_val.argmax(1) != data.y[data.val])
            s_u = _signals(pe_u, zt[unl], zt[labeled], data.driver[unl], data.t_sec[unl])
            score = cal.score(s_u)
            pick = select_batch(score, zt[unl], budget, session_ids=data.driver[unl], t_sec=data.t_sec[unl],
                                min_gap_s=3.0, pred_class=pm_u.argmax(1), max_class_frac=0.25, seed=seed)
            for k, v in cal.weights().items():
                row[f"w_{k}"] = v

        k = budget
        row["auroc_err"] = error_auroc(score, err_u)
        base = err_u.mean()
        row["lift_norm"] = (hhr_at_k(score, err_u, k) - base) / (1 - base) if base < 1 else float("nan")
        row["hit_rate_selected"] = float(err_u[pick].mean())
        log.append(row)
        new = unl[pick]
        if r == 0:
            sel_r1 = new.copy()
        labeled = np.union1d(labeled, new)

    pe_test = ensemble_probs(data.z[labeled], data.y[labeled], data.z[data.test], C, seed=seed + 999)
    f1_test = f1_score(data.y[data.test], pe_test.mean(0).argmax(1), average="macro")
    return {"rounds": log, "final_test_macro_f1": float(f1_test), "selected_round1": sel_r1}
