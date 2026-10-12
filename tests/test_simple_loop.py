import numpy as np

from hardmine.al.simple_loop import STRATEGIES, LoopData, run_loop
from hardmine.synthetic.generators import make_clip_embeddings


def test_loop_protocol_invariants():
    data = make_clip_embeddings(seed=3, n_drivers=8, windows_per_driver=120)
    idx = np.arange(len(data.y))
    loop = LoopData(data.z, data.y, data.driver, data.t_sec, pool=idx[data.driver < 5],
                    val=idx[data.driver == 5], test=idx[data.driver >= 6])
    for st in STRATEGIES:
        res = run_loop(loop, st, init_n=20, budget=10, rounds=2, seed=0, deid_rank=4)
        n = [r["n_labeled"] for r in res["rounds"]]
        assert n == [20, 30, 40], (st, n)                     # đúng ngân sách, không trùng
        sel = res["selected_round1"]
        assert np.isin(sel, loop.pool).all()                  # chỉ chọn từ pool (tài xế train)
        assert 0.0 <= res["final_test_macro_f1"] <= 1.0
