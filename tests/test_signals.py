import numpy as np

from hardmine.signals.core import (bald_score, cross_view_disagreement, js_divergence, keyword_rarity,
                                   knn_rarity, margin_score, temporal_instability)


def test_margin_range_and_order():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.6, 0.3, 0.1]])
    s = margin_score(p)
    assert np.allclose(s, [0.0, 1.0, 0.7])


def test_bald_zero_when_members_agree_and_positive_when_not():
    agree = np.tile(np.array([[[0.7, 0.3]]]), (5, 1, 1))
    assert np.allclose(bald_score(agree), 0.0, atol=1e-9)
    disagree = np.array([[[0.99, 0.01]], [[0.01, 0.99]]])
    assert bald_score(disagree)[0] > 0.6  # gần ln2 = 0.693


def test_js_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=50)
    q = rng.dirichlet(np.ones(5), size=50)
    a, b = js_divergence(p, q), js_divergence(q, p)
    assert np.allclose(a, b)
    assert (a >= -1e-12).all() and (a <= np.log(2) + 1e-9).all()


def test_temporal_instability_flags_flicker_not_constant():
    T = 20
    sess = np.zeros(T, int)
    t = np.arange(T)
    stable = np.tile([0.9, 0.1], (T, 1))
    assert np.allclose(temporal_instability(stable, sess, t), 0.0)
    flick = stable.copy()
    flick[10] = [0.2, 0.8]  # một cửa sổ nhảy lớp giữa đoạn ổn định
    s = temporal_instability(flick, sess, t)
    assert s.argmax() == 10 and s[10] > 0.2


def test_temporal_instability_ignores_true_boundary():
    T = 20
    p = np.where(np.arange(T)[:, None] < 10, [0.9, 0.1], [0.1, 0.9])
    s = temporal_instability(p.astype(float), np.zeros(T, int), np.arange(T))
    assert np.allclose(s[8:12], 0.0)  # hai đầu xa khác lớp -> ranh giới thật -> không phạt


def test_cross_view_matches_bald_semantics():
    views = np.array([[[0.9, 0.1]], [[0.1, 0.9]], [[0.9, 0.1]]])
    assert cross_view_disagreement(views)[0] > 0.3


def test_knn_rarity_outlier_higher():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 16)) + 5
    query = np.vstack([ref[:5] + 0.01, -ref[:1]])
    s = knn_rarity(query, ref, k=5)
    assert s[-1] > s[:-1].max()


def test_keyword_rarity():
    items = [{"phone"}, {"phone"}, {"phone", "sunglasses"}, set()]
    s = keyword_rarity(items)
    assert s[2] > s[0] and s[3] == 0
