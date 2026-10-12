"""Trích embedding cho một thư mục video (dùng trên Kaggle/Colab hoặc máy có GPU).

    python scripts/extract_features.py --videos /kaggle/input/my-cabin-videos --out cache/features/own \
        --preset videomae_base --batch-size 8

Bảng cửa sổ được ghi vào <out>/windows.csv; embedding vào <out>/emb_*.npy + index.csv (resume được).
"""
import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardmine.features.extract import PRESETS, extract, windows_from_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="videomae_base", choices=sorted(PRESETS))
    ap.add_argument("--length", type=float, default=2.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N cửa sổ đầu (đo thông lượng)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    win = windows_from_folder(a.videos, a.length, a.stride)
    if win.empty:
        sys.exit(f"Không tìm thấy video .mp4 trong {a.videos}")
    if a.limit:
        win = win.head(a.limit)
    win.to_csv(out / "windows.csv", index=False)
    print(f"{len(win)} cửa sổ từ {win.session_id.nunique()} video, {win.driver_id.nunique()} tài xế")
    t0 = time.time()
    extract(win, lambda r: r.video_path, str(out), preset=a.preset, batch_size=a.batch_size)
    dt = time.time() - t0
    print(f"Xong: {len(win)} cửa sổ trong {dt:.1f} s ({len(win) / max(dt, 1e-9):.1f} cửa sổ/giây)")


if __name__ == "__main__":
    main()
