"""Kiểm thử tích hợp trên dữ liệu tổng hợp: khử danh tính, bộ hiệu chuẩn, chọn mẫu, Lift@K."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from hardmine.eval.metrics import aubc, error_auroc, hhr_at_k, lift_at_k
from hardmine.features.deidentify import DriverProjector, diagnostics
from hardmine.select.core import HardnessCalibrator, coverage_boost, percentile_rank, select_batch
from hardmine.signals.core import bald_score, knn_rarity, margin_score, temporal_instability
from hardmine.synthetic.generators import make_clip_embeddings


def _split_by_driver(data, test_drivers):
    te = np.isin(data.driver, test_drivers)
    return ~te, te


def test_deidentify_reduces_driver_info_keeps_behavior():
    data = make_clip_embeddings(seed=0)
    n_drv = len(np.unique(data.driver))
    # Trên dữ liệu tổng hợp, biến thiên tài xế có hạng n_drv - 1; r nhỏ hơn chỉ xoá một phần.
    # Trên dữ liệu thật phải quét r (xem SPEC.md, mục khử danh tính).
    proj = DriverProjector(rank=n_drv - 1).fit(data.z, data.driver)
    zt = proj.transform(data.z)
    d = diagnostics(data.z, zt, data.driver, data.y)
    assert d["driver_acc_after"] < 0.5 * d["driver_acc_before"]
    assert d["behavior_acc_after"] >= d["behavior_acc_before"] - 0.05


def _ensemble_probs(z_tr, y_tr, z, M=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m in range(M):
        idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(z_tr[idx], y_tr[idx])
        p = np.zeros((len(z), y_tr.max() + 1))
        p[:, clf.classes_] = clf.predict_proba(z)
        out.append(p)
    return np.asarray(out)


def test_multisignal_calibrated_finds_errors_better_than_random_and_single_signal():
    data = make_clip_embeddings(seed=1, n_drivers=14)
    zt = DriverProjector(rank=13).fit(data.z, data.driver).transform(data.z)
    rng = np.random.default_rng(0)
    # tập đã gán nhãn nhỏ (3%), tập val (tài xế 10-11), pool (tài xế còn lại)
    val = np.isin(data.driver, [10, 11])
    pool = ~val
    lab = pool & (rng.random(len(data.y)) < 0.03)
    unl = pool & ~lab

    def signals(mask):
        pe = _ensemble_probs(zt[lab], data.y[lab], zt[mask])
        pm = pe.mean(0)
        return pm, {
            "margin": margin_score(pm),
            "bald": bald_score(pe),
            "temporal": temporal_instability(pm, data.driver[mask], data.t_sec[mask]),
            "rarity": knn_rarity(zt[mask], zt[lab], k=5),
        }

    pm_val, s_val = signals(val)
    err_val = pm_val.argmax(1) != data.y[val]
    cal = HardnessCalibrator().fit(s_val, err_val)
    pm_pool, s_pool = signals(unl)
    err_pool = pm_pool.argmax(1) != data.y[unl]
    h = cal.score(s_pool)

    K = 200
    base = err_pool.mean()
    # Lift chuẩn hoá theo trần: 0 = như ngẫu nhiên, 1 = top-K toàn là lỗi.
    norm_lift = (hhr_at_k(h, err_pool, K) - base) / (1 - base)
    assert norm_lift > 0.35, norm_lift
    assert lift_at_k(h, err_pool, K) > 1.5
    # không thua tín hiệu đơn tốt nhất quá nhiều (thường thắng)
    best_single = max(error_auroc(v, err_pool) for v in s_pool.values())
    assert error_auroc(h, err_pool) >= best_single - 0.02
    w = cal.weights()
    assert all(v >= 0 for v in w.values())
    # Tín hiệu thời gian không mang thông tin trên dữ liệu tổng hợp (cửa sổ độc lập theo thời gian),
    # nên bộ hiệu chuẩn phải gán nó trọng số nhỏ nhất.
    assert w["temporal"] == min(w.values())


def test_deidentify_partial_rank_is_partial():
    data = make_clip_embeddings(seed=0)
    zt = DriverProjector(rank=3).fit(data.z, data.driver).transform(data.z)
    d = diagnostics(data.z, zt, data.driver, data.y)
    assert d["driver_acc_after"] > 0.5  # r quá nhỏ: vẫn còn nhiều thông tin danh tính


def test_select_batch_respects_budget_nms_and_quota():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 8))
    h = rng.random(n)
    sess = rng.integers(0, 5, n)
    t = rng.uniform(0, 600, n)
    pred = rng.integers(0, 4, n)
    sel = select_batch(h, z, 40, session_ids=sess, t_sec=t, min_gap_s=10, pred_class=pred, max_class_frac=0.25)
    assert len(sel) == 40 and len(set(sel)) == 40
    for i in sel:
        for j in sel:
            if i != j and sess[i] == sess[j]:
                assert abs(t[i] - t[j]) >= 10
    counts = np.bincount(pred[sel], minlength=4)
    assert counts.max() <= 10
    assert h[sel].mean() > h.mean()  # ưu tiên mẫu khó


def test_coverage_boost_and_ranks():
    h = np.array([0.5, 0.5])
    out = coverage_boost(h, np.array(["night", "day"]), {"day": 100, "night": 0}, beta=1.0)
    assert out[0] > out[1]
    r = percentile_rank(np.array([3.0, 1.0, 2.0]))
    assert np.allclose(r, [1.0, 0.0, 0.5])


def test_aubc_constant():
    assert abs(aubc(np.array([10, 20, 40]), np.array([0.7, 0.7, 0.7])) - 0.7) < 1e-12
