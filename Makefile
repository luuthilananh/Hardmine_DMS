.PHONY: install test demo extract
install:
	pip install -e ".[dev]"
test:
	pytest -q || python tests/run_tests.py
demo:
	python scripts/demo_synthetic.py
extract:
	python scripts/extract_features.py --videos $(VIDEOS) --out cache/features/own --preset videomae_base
