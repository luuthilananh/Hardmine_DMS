"""Kiểm thử trích đặc trưng với model tí hon (cần torch + transformers + mạng).

Chỉ chạy khi HARDMINE_SLOW=1; CI chạy nó trong workflow riêng `extract-smoke.yml`.
"""
import os
import tempfile

import numpy as np

SLOW = os.environ.get("HARDMINE_SLOW") == "1"


def _make_video(path, seconds=4, fps=10, size=(96, 96)):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for i in range(seconds * fps):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        f[20:60, (i * 3) % 60: (i * 3) % 60 + 20] = 200  # vật thể di chuyển
        f = np.clip(f + rng.integers(0, 10, f.shape), 0, 255).astype(np.uint8)
        w.write(f)
    w.release()


def test_extract_tiny_model_end_to_end():
    if not SLOW:
        return
    from hardmine.features.extract import extract, load, windows_from_folder
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/vids/driverA")
        _make_video(f"{d}/vids/driverA/s1.mp4")
        win = windows_from_folder(f"{d}/vids", 2.0, 1.0)
        assert len(win) == 3 and set(win.driver_id) == {"driverA"}
        extract(win, lambda r: r.video_path, f"{d}/out", preset="tiny_test", batch_size=2, device="cpu")
        ids, emb = load(f"{d}/out")
        assert len(ids) == 3 and emb.ndim == 2 and np.isfinite(emb).all()
        # resume: chạy lại không thêm dòng
        extract(win, lambda r: r.video_path, f"{d}/out", preset="tiny_test", batch_size=2, device="cpu")
        ids2, _ = load(f"{d}/out")
        assert len(ids2) == 3
