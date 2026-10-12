"""Hồi quy: script phải chạy được trong cửa sổ dòng lệnh Windows (cp1252), nơi không in được tiếng Việt.

Lỗi thật đã gặp: bootstrap.py dừng với UnicodeEncodeError trên windows-latest ở dòng in đầu tiên.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["scripts/bootstrap.py", "scripts/download_driveact.py", "scripts/demo_synthetic.py",
           "scripts/extract_features.py"]


def test_scripts_print_vietnamese_under_cp1252():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("PYTHONUTF8", None)
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, s, "--help"], cwd=ROOT, env=env, capture_output=True)
        assert r.returncode == 0, (s, r.stderr.decode("utf-8", "replace")[-400:])
        assert "UnicodeEncodeError" not in r.stderr.decode("utf-8", "replace"), s


def test_downloader_dry_run_under_cp1252(tmp_path=None):
    import tempfile
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    with tempfile.TemporaryDirectory() as d:
        # base-url trỏ tới cổng đóng: không có mạng vẫn phải in được kế hoạch tải
        r = subprocess.run([sys.executable, "scripts/download_driveact.py", "--dry-run", "--out", d,
                            "--base-url", "http://127.0.0.1:9/"], cwd=ROOT, env=env, capture_output=True)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]
        assert "Sẽ tải" in r.stdout.decode("utf-8")
