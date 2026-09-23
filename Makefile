# ARC-AGI-3 RL assignment. Kaggle targets adapted from arcprize/ARC-AGI-3-Kaggle-Starter.
PYTHON         ?= python3.12
VENV           := .venv
PY             := $(VENV)/bin/python
KAGGLE         := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle
FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR  := vendor/ARC-AGI-3-Agents
SPLIT          ?= dev
CONFIG         ?= configs/dqn_baseline.yaml
GAME           ?= ls20
STEPS          ?= 300


.PHONY: download-games probe dataset dataset-push
.PHONY: help setup test smoke smoke-toy games splits eval experiments ablations aggregate \
        play-local notebook submit status clean _check-kaggle

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-12s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

setup: ## venv + package + Kaggle CLI + ARC-AGI-3-Agents framework
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); fi
	$(PY) scripts/slim_framework.py

test: ## CPU unit tests (run before every experiment)
	$(PY) -m pytest -q

smoke-toy: ## end-to-end on the offline toy game (no internet)
	$(PY) scripts/smoke_test.py --toy --steps $(STEPS)

smoke: ## end-to-end on one real game
	$(PY) scripts/smoke_test.py --game $(GAME) --steps $(STEPS)

games: ## list public games
	$(PY) scripts/list_games.py

download-games: ## cache all public games into environment_files/ (needs internet, once)
	$(PY) scripts/download_games.py

probe: ## actions/s per config + projected study cost
	$(PY) scripts/time_probe.py

dataset: ## package code + games for Kaggle (dist/arcrl-code); first time: bash scripts/package_kaggle_dataset.sh create
	bash scripts/package_kaggle_dataset.sh

dataset-push: ## upload a new version of the arcrl-code Kaggle dataset
	bash scripts/package_kaggle_dataset.sh version

splits: ## freeze dev/held-out split (once, before experiments)
	$(PY) scripts/make_splits.py

eval: ## one config on SPLIT, all seeds
	$(PY) scripts/evaluate.py --config $(CONFIG) --split $(SPLIT)

experiments: ## random + all algorithm versions on SPLIT
	bash scripts/run_experiments.sh $(SPLIT)

ablations: ## ablation configs on SPLIT
	bash scripts/run_ablations.sh $(SPLIT)

aggregate: ## tables + curves into outputs/figures
	$(PY) scripts/aggregate.py

play-local: ## Kaggle-parity run of agent/my_agent.py through the framework
	$(PY) scripts/play_local.py --game $(GAME) --max-steps $(STEPS)

_check-kaggle:
	@test -s .kaggle/access_token || { echo "ERROR: put your Kaggle token in .kaggle/access_token"; exit 1; }

notebook: ## build notebooks/submission.ipynb
	$(PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## push notebook to Kaggle
	$(KAGGLE) kernels push -p notebooks/

status: _check-kaggle ## latest Kaggle run status
	@$(KAGGLE) kernels status $$($(PY) -c "import json;print(json.load(open('notebooks/kernel-metadata.json'))['id'])")

clean:
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb .pytest_cache
