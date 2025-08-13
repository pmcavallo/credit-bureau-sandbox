.PHONY: setup lint test train-local evaluate sagemaker-pipeline

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	python -m pip install ruff
	ruff check src

test:
	python -m pip install pytest
	pytest -q

train-local:
	python scripts/train_local.py --config config/base.yaml

evaluate:
	python scripts/evaluate.py --config config/base.yaml

sagemaker-pipeline:
	python pipelines/sagemaker/pipeline.py --config config/base.yaml
