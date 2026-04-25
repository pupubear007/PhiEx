# PhiEx — developer entry points
#
# Usage:
#   make env       Create the conda environment (one-time, ~20 min)
#   make weights   Download ML weights into ./models (one-time, ~5 GB)
#   make run       Start the backend + serve the UI
#   make test      End-to-end APX sandbox sanity check
#   make clean     Remove caches (not the conda env, not the weights)
#
# Requires miniforge (mamba). If `mamba` isn't on PATH, install from
# https://github.com/conda-forge/miniforge  before running `make env`.

ENV_NAME     := PhiEx
MODELS_DIR   := ./models
DATA_DIR     := ./data
PYTHON       := mamba run -n $(ENV_NAME) python
UVICORN      := mamba run -n $(ENV_NAME) uvicorn

.PHONY: env weights run test clean help

help:
	@echo "PhiEx — make targets:"
	@echo "  env      create conda environment '$(ENV_NAME)'"
	@echo "  weights  download ML weights to $(MODELS_DIR)"
	@echo "  run      launch FastAPI backend at http://localhost:8000"
	@echo "  test     run the APX end-to-end example"
	@echo "  clean    remove caches (keeps env and weights)"

env:
	@echo "==> creating conda env '$(ENV_NAME)' (this takes ~20 min)"
	mamba env create -f environment.yml
	@echo "==> done. activate with:  mamba activate $(ENV_NAME)"

weights: $(MODELS_DIR)
	@echo "==> downloading ML weights into $(MODELS_DIR)"
	$(PYTHON) scripts/download_weights.py --out $(MODELS_DIR)

$(MODELS_DIR):
	mkdir -p $(MODELS_DIR)

run:
	@echo "==> starting PhiEx at http://localhost:8000"
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	PHIEX_MODELS_DIR=$(MODELS_DIR) \
	$(UVICORN) app.main:app --reload --host 127.0.0.1 --port 8000

test:
	@echo "==> running APX end-to-end sandbox"
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	PHIEX_MODELS_DIR=$(MODELS_DIR) \
	$(PYTHON) -m PhiEx.tests.apx_end_to_end

clean:
	@echo "==> removing __pycache__, .pytest_cache, trajectory scratch"
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache $(DATA_DIR)/scratch
