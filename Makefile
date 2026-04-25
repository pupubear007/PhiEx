# PhiEx — developer entry points
#
# Usage:
#   make env       Create the conda environment (one-time, ~20 min)
#   make weights   Download ML weights into ./models (one-time, ~5 GB)
#   make run       Start the backend + serve the UI
#   make test      End-to-end APX sandbox sanity check
#   make clean     Remove caches (not the conda env, not the weights)
#
# Works with either `mamba` (faster) or stock `conda`.  Auto-detects which
# is available; you can override with `make env CONDA_TOOL=conda`.
# Install miniforge from https://github.com/conda-forge/miniforge if you
# have neither.

ENV_NAME     := PhiEx
MODELS_DIR   := ./models
DATA_DIR     := ./data

# Pick mamba if present, else conda.  Override on the command line:
#   make env CONDA_TOOL=conda
CONDA_TOOL   ?= $(shell command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)
PYTHON       := $(CONDA_TOOL) run -n $(ENV_NAME) python
UVICORN      := $(CONDA_TOOL) run -n $(ENV_NAME) uvicorn

.PHONY: env weights run test clean help

help:
	@echo "PhiEx — make targets:"
	@echo "  env      create conda environment '$(ENV_NAME)'"
	@echo "  weights  download ML weights to $(MODELS_DIR)"
	@echo "  run      launch FastAPI backend at http://localhost:8000"
	@echo "  test     run the APX end-to-end example"
	@echo "  clean    remove caches (keeps env and weights)"

env:
	@echo "==> creating conda env '$(ENV_NAME)' with $(CONDA_TOOL) (this takes ~20 min)"
	$(CONDA_TOOL) env create -f environment.yml
	@echo "==> done. activate with:  $(CONDA_TOOL) activate $(ENV_NAME)"

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
