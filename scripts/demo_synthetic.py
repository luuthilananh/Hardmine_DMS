"""Demo đầu-cuối trên dữ liệu tổng hợp (CPU, ~1–2 phút).

    python scripts/demo_synthetic.py [--out results/synthetic_demo] [--web demo/data.js] [--seeds 3] [--quick]

Tạo:
  <out>/rounds.csv          đường cong theo vòng cho 4 chiến lược × seed
  <out>/final_test.csv      macro-F1 trên tập test (tài xế chưa thấy) sau vòng cuối
  <out>/events.csv          so sánh mining sự kiện nhắm mắt
  <out>/summary.md          bảng tóm tắt (dán vào README)
  <out>/figures/*.png       hình
  <web>                     dữ liệu cho trang demo tương tác

Kết quả chỉ chứng minh cơ chế trên dữ liệu tổng hợp, KHÔNG phải hiệu năng trên dữ liệu thật.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardmine.al.simple_loop import STRATEGIES, LoopData, run_loop  # noqa: E402
from hardmine.events.core import (event_matching, event_metrics, extract_events, fragmentation,  # noqa: E402
                                  threshold_proximity)
from hardmine.features.deidentify import DriverProjector  # noqa: E402
from hardmine.select.core import HardnessCalibrator  # noqa: E402
from hardmine.synthetic.generators import make_clip_embeddings, make_eye_sequence  # noqa: E402


def clip_tier(seeds: int, rounds: int, out: pathlib.Path):
    data = make_clip_embeddings(seed=7, n_drivers=14)
    idx = np.arange(len(data.y))
    loop = LoopData(data.z, data.y, data.driver, data.t_sec,
                    pool=idx[data.driver < 10], val=idx[np.isin(data.driver, [10, 11])],
                    test=idx[data.driver >= 12])
    rows, finals, sel = [], [], {}
    for s in range(seeds):
        for st in STRATEGIES:
            res = run_loop(loop, st, init_n=40, budget=30, rounds=rounds, seed=s)
            rows += res["rounds"]
            finals.append({"strategy": st, "seed": s, "test_macro_f1": res["final_test_macro_f1"]})
            if s == 0:
                sel[st] = res["selected_round1"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "rounds.csv", index=False)
    pd.DataFrame(finals).to_csv(out / "final_test.csv", index=False)
    return data, loop, df, pd.DataFrame(finals), sel


def _event_errors(pred, gt):
    """Lỗi cấp sự kiện: sự kiện dự đoán không khớp (IoU>=0.5) sự kiện thật cùng lớp."""
    m = dict(event_matching(pred, gt))
    return np.array([not (i in m and pred[i].cls == gt[m[i]].cls) for i in range(len(pred))])


def _event_signals(q, pred):
    unc = np.array([np.mean(1 - np.abs(2 * q[e.start:e.end] - 1)) for e in pred])
    return {"frame_uncertainty": unc, "threshold_proximity": threshold_proximity(q, pred),
            "fragmentation": fragmentation(q, pred)}


def event_tier(out: pathlib.Path, budget: int = 20, n_eval: int = 5):
    # chỉ xét sự kiện >= 0.5 s (chớp mắt ngắn dễ, chiếm đa số)
    def prep(seed):
        seq = make_eye_sequence(minutes=30, seed=seed)
        pred = [e for e in extract_events(seq.q_pred, seq.fps) if e.duration >= 0.5]
        return seq, pred, _event_errors(pred, seq.gt_events), _event_signals(seq.q_pred, pred)

    _, _, err_v, s_v = prep(100)
    cal = HardnessCalibrator().fit(s_v, err_v)
    rows = []
    rng = np.random.default_rng(0)
    for sd in range(n_eval):
        seq, pred, err, s = prep(200 + sd)
        scores = {"random": rng.random(len(pred)), "mean_frame_uncertainty": s["frame_uncertainty"],
                  "hardmine_event": cal.score(s)}
        for name, sc in scores.items():
            top = np.argsort(-sc)[:budget]
            rows.append({"strategy": name, "seed": sd, "n_events": len(pred), "base_err": err.mean(),
                         "errors_found_in_top": int(err[top].sum()), "total_errors": int(err.sum()),
                         "hit_rate": float(err[top].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "events.csv", index=False)

    seq = make_eye_sequence(minutes=20, seed=42)
    hours = len(seq.q_true) / seq.fps / 3600
    m_raw = event_metrics(extract_events(seq.q_pred, seq.fps), seq.gt_events, hours)
    m_fill = event_metrics(extract_events(seq.q_pred, seq.fps, fill_gap_s=0.2), seq.gt_events, hours)
    return df, cal, seq, m_raw, m_fill


def figures(df, ev, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    g = df.groupby(["strategy", "n_labeled"])["macro_f1_val"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for st in STRATEGIES:
        d = g[g.strategy == st]
        ax.plot(d.n_labeled, d["mean"], marker="o", ms=3, label=st)
        ax.fill_between(d.n_labeled, d["mean"] - d["std"], d["mean"] + d["std"], alpha=0.15)
    ax.set_xlabel("Số nhãn đã dùng"); ax.set_ylabel("Macro-F1 (val, tài xế chưa thấy)")
    ax.set_title("Dữ liệu tổng hợp: hiệu quả nhãn theo chiến lược"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "label_efficiency.png", dpi=150); plt.close(fig)

    e = ev.groupby("strategy")["hit_rate"].agg(["mean", "std"]).reindex(["random", "mean_frame_uncertainty", "hardmine_event"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(e.index, e["mean"], xerr=e["std"], color=["#9aa3ad", "#6b8fb5", "#c0392b"])
    ax.set_xlabel("Tỷ lệ sự kiện bị phân lớp sai trong top-20 được chọn")
    ax.set_title("Dữ liệu tổng hợp: mining sự kiện nhắm mắt"); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "event_mining.png", dpi=150); plt.close(fig)


def summary(df, finals, ev, m_raw, m_fill, out: pathlib.Path):
    last = df[df["round"] == df["round"].max()]
    r = df[df["round"] < df["round"].max()]
    lines = ["| Chiến lược | Macro-F1 val (vòng cuối) | Macro-F1 test | Tỷ lệ chọn trúng lỗi | Lift chuẩn hoá |",
             "|---|---|---|---|---|"]
    for st in STRATEGIES:
        a = last[last.strategy == st]["macro_f1_val"]
        t = finals[finals.strategy == st]["test_macro_f1"]
        h = r[r.strategy == st]["hit_rate_selected"]
        ln = r[r.strategy == st]["lift_norm"]
        lines.append(f"| {st} | {a.mean():.3f} ± {a.std():.3f} | {t.mean():.3f} ± {t.std():.3f} | "
                     f"{h.mean():.2f} | {ln.mean():.2f} |")
    lines += ["", "| Chiến lược (sự kiện) | Tỷ lệ lỗi trong top-20 | Tỷ lệ lỗi nền |", "|---|---|---|"]
    for st in ["random", "mean_frame_uncertainty", "hardmine_event"]:
        d = ev[ev.strategy == st]
        lines.append(f"| {st} | {d.hit_rate.mean():.2f} ± {d.hit_rate.std():.2f} | {d.base_err.mean():.2f} |")
    lines += ["", f"Hậu xử lý lấp khe 0,2 s: F1 sleep {m_raw['f1_sleep']:.2f} → {m_fill['f1_sleep']:.2f}; "
              f"F1 microsleep {m_raw['f1_microsleep']:.2f} → {m_fill['f1_microsleep']:.2f}."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def web_data(data, loop, df, sel, cal_ev, seq, ev, web: pathlib.Path):
    from sklearn.decomposition import PCA
    pool = loop.pool
    zt = DriverProjector(rank=len(np.unique(data.driver[pool])) - 1).fit(data.z[pool], data.driver[pool]).transform(data.z)
    def proj(x):
        p = PCA(2, random_state=0).fit_transform(x[pool])
        p = (p - p.min(0)) / (p.max(0) - p.min(0))
        return np.round(p, 4).tolist()
    pos = {int(v): i for i, v in enumerate(pool)}
    g = df.groupby(["strategy", "n_labeled"])["macro_f1_val"].mean().reset_index()
    curves = {st: g[g.strategy == st][["n_labeled", "macro_f1_val"]].round(4).values.tolist() for st in STRATEGIES}
    evs = ev.groupby("strategy")["hit_rate"].mean().round(3).to_dict()
    payload = {
        "clip": {"raw": proj(data.z), "deid": proj(zt), "driver": data.driver[pool].tolist(),
                 "label": data.y[pool].tolist(), "rare": data.is_rare[pool].astype(int).tolist(),
                 "selected": {k: [pos[int(i)] for i in v] for k, v in sel.items()},
                 "curves": curves},
        "events": {"fps": seq.fps, "q": np.round(seq.q_pred, 3).tolist(),
                   "gt": [[e.start, e.end] for e in seq.gt_events],
                   "weights": cal_ev.weights(), "bias": cal_ev.b, "hit_rate": evs},
    }
    web.parent.mkdir(parents=True, exist_ok=True)
    web.write_text("window.DEMO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synthetic_demo")
    ap.add_argument("--web", default="demo/data.js")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 vòng — dùng cho CI nhanh")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.rounds = 1, 3
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data, loop, df, finals, sel = clip_tier(a.seeds, a.rounds, out)
    ev, cal_ev, seq, m_raw, m_fill = event_tier(out)
    if not a.no_figures:
        figures(df, ev, out)
    print(summary(df, finals, ev, m_raw, m_fill, out))
    if a.web:
        web_data(data, loop, df, sel, cal_ev, seq, ev, ROOT / a.web)
    print(f"\nXong trong {time.time() - t0:.1f} s. Kết quả: {out}")


if __name__ == "__main__":
    main()
