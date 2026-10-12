"""Tải và giải nén Drive&Act vào data/driveact/ (tiếp tục được nếu bị ngắt).

    .venv/Scripts/python.exe scripts/download_driveact.py            # Windows
    .venv/bin/python scripts/download_driveact.py                    # macOS / Linux

Mặc định chỉ tải 2 file cần cho MVP (~7 GB): camera NIR cột A phía tài xế + nhãn hoạt động.
  --extra inner_mirror     thêm góc gương chiếu hậu (cho P5, đa góc)
  --delete-zip             xoá file zip sau khi giải nén để tiết kiệm đĩa
  --dry-run                chỉ in ra sẽ tải gì

Drive&Act: bản quyền Fraunhofer IOSB, chỉ dùng cho nghiên cứu, không phân phối lại.
Trang gốc: https://www.driveandact.com/ (không cần đăng ký, kiểm tra 12/10/2026).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.driveandact.com/dataset/"
MVP = ["a_column_driver.zip", "iccv_activities_3s.zip"]
EXTRA = {"inner_mirror": "inner_mirror.zip", "ceiling": "ceiling.zip", "steering_wheel": "steering_wheel.zip",
         "openpose_3d": "iccv_openpose_3d.zip"}
CHUNK = 1 << 20


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Tải có tiếp tục: nếu có file .part thì xin Range; server không hỗ trợ Range thì tải lại từ đầu."""
    part = dest.with_suffix(dest.suffix + ".part")
    total = remote_size(url)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                done = have if mode == "ab" else 0
                t0, last = time.time(), -1
                with open(part, mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if total:
                            pct = int(100 * done / total)
                            if pct // 5 != last:
                                last = pct // 5
                                mb_s = (done - (have if mode == "ab" else 0)) / max(time.time() - t0, 1e-6) / 1e6
                                print(f"    {dest.name}: {pct:3d}%  ({done / 1e9:.2f}/{total / 1e9:.2f} GB, {mb_s:.1f} MB/s)",
                                      flush=True)
            if total and part.stat().st_size != total:
                raise IOError(f"tải thiếu: {part.stat().st_size} / {total} byte")
            part.replace(dest)
            return
        except Exception as e:  # mạng chập chờn: thử lại, giữ phần đã tải
            print(f"    lỗi lần {attempt}/{retries}: {e}", flush=True)
            time.sleep(min(30, 5 * attempt))
    raise SystemExit(f"Không tải được {url} sau {retries} lần. Chạy lại lệnh để tiếp tục.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "driveact"))
    ap.add_argument("--extra", nargs="*", default=[], choices=sorted(EXTRA))
    ap.add_argument("--delete-zip", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base-url", default=BASE, help=argparse.SUPPRESS)  # dùng cho test
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = MVP + [EXTRA[x] for x in a.extra]
    sizes = {f: remote_size(a.base_url + f) for f in files}
    need = sum(s or 0 for s in sizes.values())
    free = shutil.disk_usage(out).free
    print(f"Sẽ tải vào {out}:")
    for f in files:
        s = sizes[f]
        print(f"  - {f}  {s / 1e9:.2f} GB" if s else f"  - {f}  (không rõ dung lượng)")
    print(f"Đĩa trống: {free / 1e9:.1f} GB; cần khoảng {2.2 * need / 1e9:.1f} GB (zip + giải nén)")
    if a.dry_run:
        return 0
    if need and free < 2.2 * need:
        print("KHÔNG ĐỦ CHỖ TRỐNG. Giải phóng đĩa hoặc dùng --delete-zip, rồi chạy lại.")
        return 2

    for f in files:
        marker = out / f".extracted_{f}"
        if marker.exists():
            print(f"[bỏ qua] {f} đã giải nén trước đó")
            continue
        z = out / f
        if not (z.exists() and zipfile.is_zipfile(z)):
            print(f"[tải] {f}")
            download(a.base_url + f, z)
        print(f"[giải nén] {f}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        marker.write_text("ok\n")
        if a.delete_zip:
            z.unlink()

    print("\nCấu trúc sau khi giải nén (3 cấp đầu):")
    shown = 0
    for p in sorted(out.rglob("*")):
        depth = len(p.relative_to(out).parts)
        if depth <= 3 and not p.name.startswith(".") and shown < 60:
            print("  " * depth + p.name + ("/" if p.is_dir() else ""))
            shown += 1
    print("\nXONG. Bước tiếp theo: P1.1 viết loader theo đúng cấu trúc ở trên.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
