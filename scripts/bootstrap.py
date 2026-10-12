"""Tạo môi trường Python RIÊNG cho dự án (thư mục .venv/) và cài đặt mọi thứ cần thiết.

Chạy bằng bất kỳ Python >= 3.10 nào có trên máy, từ thư mục gốc của repo:

    Windows:        py -3.11 scripts/bootstrap.py      (hoặc: python scripts/bootstrap.py)
    macOS / Linux:  python3 scripts/bootstrap.py

Script này:
  1. Tạo .venv/ trong thư mục dự án (không đụng tới Python chung của máy).
  2. Cài gói hardmine (chế độ editable) cùng thư viện cần cho test và đọc video vào .venv/.
  3. Chạy toàn bộ test bằng Python của .venv/.
  4. In ra đường dẫn Python của dự án để mọi lệnh sau dùng đúng môi trường.

Chạy lại bao nhiêu lần cũng được: .venv/ đã có thì dùng lại.
Thêm --recreate để xoá và tạo lại .venv/ từ đầu.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
MIN = (3, 10)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="xoá và tạo lại .venv/")
    ap.add_argument("--extras", default="dev,video", help="nhóm phụ thuộc cần cài (mặc định: dev,video)")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    if sys.version_info < MIN:
        print(f"Cần Python >= {MIN[0]}.{MIN[1]}, đang dùng {sys.version.split()[0]}.\n"
              "Windows: winget install -e --id Python.Python.3.11   rồi chạy: py -3.11 scripts/bootstrap.py")
        return 1

    print(f"[1/4] Môi trường riêng của dự án: {VENV}")
    if a.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not venv_python().exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    py = venv_python()
    out = subprocess.run([str(py), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()).resolve() == VENV.resolve(), "Python của .venv không trỏ đúng thư mục dự án"

    print("[2/4] Cập nhật pip trong .venv")
    run([py, "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

    print(f"[3/4] Cài hardmine[{a.extras}] vào .venv")
    run([py, "-m", "pip", "install", "--quiet", "-e", f".[{a.extras}]"])

    if not a.skip_tests:
        print("[4/4] Chạy test bằng Python của dự án")
        run([py, "tests/run_tests.py"])
    else:
        print("[4/4] Bỏ qua test")

    rel = py.relative_to(ROOT).as_posix()
    print("\nXONG. Mọi lệnh Python của dự án hãy gọi qua:", rel)
    print(f"  ví dụ: {rel} scripts/demo_synthetic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
