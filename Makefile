.PHONY: install prepare eval clean lab help solid-up solid-down solid-populate

REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
PY := .venv/bin/python
JUP := .venv/bin/jupyter

help:
	@echo "FLTA 2026 evaluation companion artefact (FL-only)"
	@echo ""
	@echo "Targets:"
	@echo "  make install        — pip install the harness in editable mode"
	@echo "  make prepare        — fetch BloodMNIST, build partition, pods, battery"
	@echo "  make eval           — run all six notebooks end-to-end; write to results/"
	@echo "  make lab            — start Jupyter Lab"
	@echo "  make solid-up       — start CSS via docker-compose"
	@echo "  make solid-populate — push the pod federation into the running CSS"
	@echo "  make solid-down     — stop CSS and remove its volume"
	@echo "  make clean          — remove generated data, pods, battery, results"

install:
	$(PY) -m pip install -e . 2>/dev/null || python3 -m pip install -e .

prepare:
	$(PY) scripts/prepare.py

eval:
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/00_walkthrough.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/02_gradient_inversion.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq2-metadata/01_per_subject_fidelity.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq3-composability/01_rule_battery.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq5-comparison/01_scalar_vs_stepwise.ipynb

lab:
	$(JUP) lab

solid-up:
	cd solid_deploy && docker compose up -d

solid-populate:
	$(PY) solid_deploy/populate.py --css http://localhost:3000

solid-down:
	cd solid_deploy && docker compose down -v

clean:
	rm -rf data/*.parquet data/*.npz data/_manifest.json
	rm -rf pods/persona_* pods/_manifest.json
	rm -rf card/chains card/battery_manifest.json
	rm -rf results/sq1/*.json results/sq2/*.json results/sq3/*.json results/sq5/*.json
