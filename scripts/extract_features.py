"""Trích embedding cho các cửa sổ video (dùng trên Kaggle/Colab hoặc máy có GPU).

Hai chế độ:

1. Thư mục video bất kỳ (video tự quay): cửa sổ được tạo tự động, tên thư mục cha = ID tài xế.
    python scripts/extract_features.py --videos /kaggle/input/my-cabin-videos --out cache/features/own

2. Bảng cửa sổ có sẵn (Drive&Act, DMD — do loader ở P1.1 sinh ra):
    python scripts/extract_features.py --windows cache/windows/driveact.csv --root data/driveact \
        --out cache/features/driveact/videomae_base
   Bảng cần cột: window_id, video_path (tương đối so với --root), t_start, t_end.
   Dùng đường dẫn tương đối để cùng một bảng chạy được cả trên máy bạn lẫn trên Kaggle.

Embedding ghi vào <out>/emb_*.npy + index.csv (resume được: chạy lại đúng lệnh là tiếp tục).
"""
import argparse
import pathlib
import sys

# Cửa sổ dòng lệnh Windows thường dùng cp1252/cp437, không in được tiếng Việt: ép UTF-8.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardmine.features.extract import PRESETS, extract, windows_from_folder  # noqa: E402

REQUIRED = {"window_id", "video_path", "t_start", "t_end"}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", help="thư mục video (chế độ 1)")
    src.add_argument("--windows", help="CSV bảng cửa sổ (chế độ 2)")
    ap.add_argument("--root", default=".", help="thư mục gốc cho video_path trong --windows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="videomae_base", choices=sorted(PRESETS))
    ap.add_argument("--length", type=float, default=2.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N cửa sổ đầu (đo thông lượng)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.videos:
        win = windows_from_folder(a.videos, a.length, a.stride)
        if win.empty:
            sys.exit(f"Không tìm thấy video .mp4 trong {a.videos}")
        path_of = lambda r: r.video_path  # noqa: E731
    else:
        win = pd.read_csv(a.windows)
        missing = REQUIRED - set(win.columns)
        if missing:
            sys.exit(f"{a.windows} thiếu cột: {sorted(missing)}")
        root = pathlib.Path(a.root)
        path_of = lambda r: str(root / r.video_path)  # noqa: E731
        first = root / win.video_path.iloc[0]
        if not first.exists():
            sys.exit(f"Không thấy video {first}. Kiểm tra --root.")
    if a.limit:
        win = win.head(a.limit)
    win.to_csv(out / "windows.csv", index=False)
    n_drv = win.driver_id.nunique() if "driver_id" in win else "?"
    print(f"{len(win)} cửa sổ, {n_drv} tài xế")
    t0 = time.time()
    extract(win, path_of, str(out), preset=a.preset, batch_size=a.batch_size)
    dt = time.time() - t0
    print(f"Xong: {len(win)} cửa sổ trong {dt:.1f} s ({len(win) / max(dt, 1e-9):.1f} cửa sổ/giây)")


if __name__ == "__main__":
    main()
