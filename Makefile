.PHONY: install install-accountants install-sota prepare eval sota accountants multi-seed clean lab help solid-up solid-down solid-populate

REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
PY := .venv/bin/python
JUP := .venv/bin/jupyter

help:
	@echo "FLTA 2026 evaluation companion artefact (FL-only)"
	@echo ""
	@echo "Tier A (default — numpy MLP, CPU, ~5 min):"
	@echo "  make install             — pip install the harness in editable mode"
	@echo "  make prepare             — fetch BloodMNIST, build partition, pods, battery"
	@echo "  make eval                — run all Tier A notebooks end-to-end; write to results/"
	@echo "  make multi-seed          — multi-seed SQ-5 ordering stability (10 seeds × 3 configs)"
	@echo "  make lab                 — start Jupyter Lab"
	@echo ""
	@echo "Tier B (Position B — PyTorch CNN + parity shadows, ~10 min on MPS):"
	@echo "  make install-sota        — additionally install torch + opacus (~2 GB wheels)"
	@echo "  make sota                — run SQ-1d SOTA-faithful calibration notebook"
	@echo ""
	@echo "Accountant comparison:"
	@echo "  make install-accountants — additionally install the PRV accountant"
	@echo "  make accountants         — compare RDP envelope vs PRV (Gopi 2021)"
	@echo ""
	@echo "Solid runtime:"
	@echo "  make solid-up            — start CSS via docker-compose"
	@echo "  make solid-populate      — push the pod federation into the running CSS"
	@echo "  make solid-down          — stop CSS and remove its volume"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean               — remove generated data, pods, battery, results"

install:
	$(PY) -m pip install -e . 2>/dev/null || python3 -m pip install -e .

install-accountants:
	$(PY) -m pip install -e ".[accountants]" 2>/dev/null || python3 -m pip install -e ".[accountants]"

install-sota:
	$(PY) -m pip install -e ".[sota]" 2>/dev/null || python3 -m pip install -e ".[sota]"

prepare:
	$(PY) scripts/prepare.py

accountants:
	$(PY) scripts/compare_accountants.py

multi-seed:
	$(PY) scripts/multi_seed_sq5.py --seeds 10 --paper-scale

sota:
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/04_sota_calibration.ipynb

eval:
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/00_walkthrough.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/02_gradient_inversion.ipynb
	$(JUP) nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/03_canary_audit.ipynb
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
	rm -f results/accountant_comparison.json
