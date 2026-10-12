import numpy as np

from hardmine.events.core import (Event, classify_duration, event_metrics, extract_events, fragmentation,
                                  threshold_proximity)
from hardmine.synthetic.generators import make_eye_sequence


def test_classify_duration_thresholds():
    assert [classify_duration(d) for d in (0.3, 1.0, 1.99, 2.5, 3.0, 5.9, 6.0)] == [0, 1, 1, 2, 3, 3, 4]


def test_extract_events_and_gap_fill():
    fps = 10
    q = np.zeros(100)
    q[10:20] = 1; q[21:40] = 1  # khe 1 frame
    ev = extract_events(q, fps)
    assert [(e.start, e.end) for e in ev] == [(10, 20), (21, 40)]
    ev2 = extract_events(q, fps, fill_gap_s=0.1)
    assert [(e.start, e.end) for e in ev2] == [(10, 40)]
    assert ev2[0].duration == 3.0 and ev2[0].cls == 3  # gộp lại thành một lần "sleep"
    assert ev[0].cls == 1 and ev[1].cls == 1             # không gộp: hai lần "microsleep"


def test_merged_duration_class():
    ev = Event(10, 40, 10)
    assert ev.duration == 3.0 and ev.cls == 3


def test_threshold_proximity_ranks_near_threshold_events_higher():
    seq = make_eye_sequence(minutes=30, seed=1)
    pred = extract_events(seq.q_pred, seq.fps)
    s = threshold_proximity(seq.q_pred, pred)
    gt_kind = {seq.gt_events[i].start: k for i, k in seq.hard_kind.items()}
    near = [s[i] for i, e in enumerate(pred) if gt_kind.get(e.start) == "near_threshold"
            or any(abs(e.start - g) <= 3 and k == "near_threshold" for g, k in gt_kind.items())]
    easy_long = [s[i] for i, e in enumerate(pred) if e.duration > 1.1 and gt_kind.get(e.start) == "easy"]
    assert len(near) > 5 and len(easy_long) > 5
    assert np.median(near) > 3 * np.median(easy_long)


def test_fragmentation_finds_split_sleep():
    seq = make_eye_sequence(minutes=30, seed=2)
    pred = extract_events(seq.q_pred, seq.fps)
    s = fragmentation(seq.q_pred, pred)
    flagged = {pred[i].start for i in np.flatnonzero(s > 0)}
    frag_gt = [seq.gt_events[i] for i, k in seq.hard_kind.items() if k == "fragmented"]
    assert len(frag_gt) > 3
    hit = sum(any(g.start <= f < g.end for f in flagged) for g in frag_gt)
    assert hit / len(frag_gt) >= 0.9
    # không báo nhầm trên sự kiện dễ
    easy_gt = [seq.gt_events[i] for i, k in seq.hard_kind.items() if k == "easy"]
    false_hits = sum(any(g.start <= f < g.end for f in flagged) for g in easy_gt)
    assert false_hits / len(easy_gt) < 0.02


def test_event_metrics_perfect_and_fragmented():
    seq = make_eye_sequence(minutes=20, seed=3)
    hours = len(seq.q_true) / seq.fps / 3600
    perfect = event_metrics(seq.gt_events, seq.gt_events, hours)
    assert perfect["f1_sleep"] == 1.0 and perfect["duration_mae_s"] == 0.0
    raw = event_metrics(extract_events(seq.q_pred, seq.fps), seq.gt_events, hours)
    fixed = event_metrics(extract_events(seq.q_pred, seq.fps, fill_gap_s=0.2), seq.gt_events, hours)
    # lấp khe ngắn phải cải thiện F1 lớp sleep (vì có ca cắt vụn)
    assert fixed["f1_sleep"] > raw["f1_sleep"]
