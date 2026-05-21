.PHONY: help install install-dev install-train test test-fast lint format typecheck e2e smoke smoke-prod docker-build docker-run frontend-install frontend-build frontend-dev clean

PYTHON ?= python3.11
PIP := $(PYTHON) -m pip

help:
	@echo "MES — Motor Engagement Signal"
	@echo ""
	@echo "Targets:"
	@echo "  install         Install runtime deps"
	@echo "  install-dev     Install runtime + dev deps"
	@echo "  install-train   Install training-side deps (heavy)"
	@echo "  test            Run all tests"
	@echo "  test-fast       Run fast unit tests only"
	@echo "  lint            Ruff + mypy"
	@echo "  format          Auto-format with ruff"
	@echo "  e2e             Build container + run Playwright e2e"
	@echo "  smoke           Boot backend locally + one round-trip"
	@echo "  kaggle-validate  Run validation notebook on Kaggle"
	@echo "  hf-jobs-eegnet   Train EEGNet on HF Jobs GPU (fast queue, ~\$2)"
	@echo "  preflight        Run pipeline preflight checks locally"
	@echo "  docker-build    Build the HF Space image"
	@echo "  docker-run      Run image locally on :7860"
	@echo "  frontend-build  npm build the React app"
	@echo "  clean           Remove caches + build artifacts"

install:
	$(PIP) install -e ".[backend]"

install-dev:
	$(PIP) install -e ".[backend,dev]"

install-train:
	$(PIP) install -e ".[backend,train,dev]"

test:
	$(PYTHON) -m pytest --cov=mes_core --cov=backend --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest tests/unit tests/property -x

lint:
	$(PYTHON) -m ruff check mes_core backend tests
	$(PYTHON) -m mypy mes_core backend

format:
	$(PYTHON) -m ruff check --fix mes_core backend tests
	$(PYTHON) -m ruff format mes_core backend tests

frontend-install:
	cd frontend && npm ci

frontend-build:
	cd frontend && npm run build

frontend-dev:
	cd frontend && npm run dev

docker-build:
	docker build -t mes-space:local .

docker-run: docker-build
	docker run --rm -p 7860:7860 --env-file .env.local mes-space:local

e2e: docker-build
	cd tests/e2e && npx playwright test

smoke:
	$(PYTHON) -m backend.app.main --smoke

smoke-prod:
	curl -fsS https://huggingface.co/spaces/abachu2005/mes/healthz || echo "Space sleeping or down"

# ---------- Kaggle (cloud compute) ----------

KAGGLE_CLI := $(shell pwd)/.venv/bin/kaggle
KSUBMIT := $(PYTHON) scripts/kaggle_submit.py

kaggle-preprocess:
	@set -a; . ./.env.local; set +a; KAGGLE_CLI=$(KAGGLE_CLI) \
	  $(KSUBMIT) notebooks/kaggle/00_preprocess.py \
	  --kernel-id $$HF_USERNAME/mes-00-preprocess --no-gpu --internet --poll

kaggle-train-riemannian:
	@set -a; . ./.env.local; set +a; KAGGLE_CLI=$(KAGGLE_CLI) \
	  $(KSUBMIT) notebooks/kaggle/01_train_riemannian.py \
	  --kernel-id $$HF_USERNAME/mes-01-train-riemannian --no-gpu --internet --poll

kaggle-train-eegnet:
	@set -a; . ./.env.local; set +a; KAGGLE_CLI=$(KAGGLE_CLI) \
	  $(KSUBMIT) notebooks/kaggle/02_train_eegnet.py \
	  --kernel-id $$HF_USERNAME/mes-02-train-eegnet --gpu --internet --poll

kaggle-validate:
	@set -a; . ./.env.local; set +a; KAGGLE_CLI=$(KAGGLE_CLI) \
	  $(KSUBMIT) notebooks/kaggle/03_validate.py \
	  --kernel-id $$HF_USERNAME/mes-03-validate --no-gpu --internet --poll

hf-jobs-eegnet:
	@set -a; . ./.env.local; set +a; \
	  hf jobs uv run scripts/hf_jobs_train_eegnet.py \
	  --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
	  -e HF_DATASET_REPO=$$HF_DATASET_REPO \
	  -e HF_MODEL_REPO=$$HF_MODEL_REPO \
	  -e HF_USERNAME=$$HF_USERNAME

preflight:
	@set -a; . ./.env.local; set +a; \
	  $(PYTHON) scripts/preflight.py --stage all

preflight-smoke:
	@set -a; . ./.env.local; set +a; \
	  $(PYTHON) scripts/preflight.py --stage mes-train --smoke

# ---------- HF Space deploy ----------

hf-deploy:
	@set -a; . ./.env.local; set +a; $(PYTHON) scripts/hf_deploy.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd frontend && rm -rf node_modules dist .vite
