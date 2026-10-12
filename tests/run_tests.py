"""Trình chạy test tối giản, dùng khi không cài được pytest. Ưu tiên `pytest -q` nếu có.

    python tests/run_tests.py
"""
import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

failed = passed = 0
for f in sorted((ROOT / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(dir(mod)):
        if name.startswith("test_") and callable(getattr(mod, name)):
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {f.stem}::{name}")
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}::{name}")
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
