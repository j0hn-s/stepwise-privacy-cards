# Stepwise privacy cards for federated learning

Companion artefact to the short paper *Beyond scalar epsilon: stepwise privacy cards for federated learning*. An end-to-end, reproducible evaluation harness across two tiers — Tier A (numpy MLP, ~5 min on CPU) and Tier B (PyTorch CNN + shadow-target FL parity, ~10 min on Apple Silicon MPS). The harness runs on the **MedMNIST v2 BloodMNIST** benchmark and exercises three FL-native attack evaluations (per-record MIA with both LiRA and RMIA-online meta-classifiers, Geiping-style gradient inversion, and a held-out canary audit producing an empirical ε lower bound), a 200-pod Solid federation mock with optional deployment against a real Community Solid Server, a deterministic rule engine, and a 16-chain composability battery cross-checked against hand-authored expectations.

This repo is the empirical complement to the *Privacy Explorer / YAPS* framework (the parent privacy-engineering project: https://github.com/j0hn-s/privacy-explorer).

---

## What is in this repo

| Path | What it is |
|---|---|
| [`flta_eval/`](flta_eval/) | The harness: `datasets`, `fl` (Tier A numpy MLP), `fl_torch` (Tier B PyTorch CNN), `attacks` (LiRA + RMIA-online + shadow pool + CIs), `pods`, `rules`, `chains`, `audit` |
| [`notebooks/`](notebooks/) | Walkthrough + study-question notebooks (SQ-1a MIA, SQ-1b gradient inversion, SQ-1c canary audit, **SQ-1d SOTA-faithful baseline**, SQ-2/3/5) |
| [`card/example_chain.json`](card/example_chain.json) | The worked privacy card the evaluation calibrates |
| [`card/battery_expected.json`](card/battery_expected.json) | Hand-authored expected rule firings (independent of the rule engine) |
| [`scripts/`](scripts/) | `prepare.py` (one-shot generator: BloodMNIST + partition + pods + battery), `compare_accountants.py` (RDP envelope vs PRV side-by-side), `multi_seed_sq5.py` (multi-seed ordering-stability runner) |
| [`solid_deploy/`](solid_deploy/) | Optional: docker-compose Community Solid Server + populate script |
| [`docs/`](docs/) | [`METHODOLOGY.md`](docs/METHODOLOGY.md) · [`SCHEMAS.md`](docs/SCHEMAS.md) · [`THREAT_MODEL.md`](docs/THREAT_MODEL.md) · [`ENVIRONMENT.md`](docs/ENVIRONMENT.md) · [`PRIVACY_EXPLORER_MAP.md`](docs/PRIVACY_EXPLORER_MAP.md) — extended methodology, data structures, threat profiles, environment notes, and the per-module mapping into the parent Privacy Explorer repo |

The paper itself (manuscript + figures) is held outside this repository and released separately. The configured-card example in [`card/example_chain.json`](card/example_chain.json) is the canonical worked example the paper refers to.

---

## Running the evaluation locally

The evaluation runs offline after a single dataset fetch (BloodMNIST is ~35 MB; cached on first use).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                    # Tier A core
python scripts/prepare.py           # BloodMNIST + partition + pods + battery
make eval                           # ~5 min at tutorial scale; runs Tier A notebooks (walkthrough + SQ-1a/b/c + SQ-2/3/5)
make install-sota && make sota      # optional: Tier B SOTA-faithful baseline (~10 min on Apple Silicon MPS)
```

`make eval` runs every Tier A notebook end-to-end via `jupyter nbconvert --execute --inplace`. Records are written to `results/sq1/`, `results/sq2/`, `results/sq3/`, `results/sq5/` as JSON, each carrying the five-field audit trail (`harness_commit`, `dataset.sha256`, `config.config_hash_sha256`, `seed_namespace`, `timestamp_utc`). Tier B records additionally include `config.device` and `config.model_class`.

### Step-through in Jupyter Lab

To inspect each notebook rather than batch-run:

```bash
make lab            # opens Jupyter Lab; the notebooks/ tree is in the file browser
```

Recommended order:

1. [`notebooks/00_walkthrough.ipynb`](notebooks/00_walkthrough.ipynb) — orientation; prints the dataset manifest, the example card, the pod federation, the chain battery.
2. [`notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb`](notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb) — SQ-1a per-record MIA against the FL-released model; default meta-classifier **LiRA** (Carlini 2022), with **RMIA-online** (Zarifzadeh 2024) as an ablation.
3. [`notebooks/sq1-calibration/02_gradient_inversion.ipynb`](notebooks/sq1-calibration/02_gradient_inversion.ipynb) — SQ-1b gradient inversion vs σ; default Total-Variation regularisation on, per Geiping 2020 §3.1.
4. [`notebooks/sq1-calibration/03_canary_audit.ipynb`](notebooks/sq1-calibration/03_canary_audit.ipynb) — SQ-1c held-out canary audit; produces an empirical ε lower bound (Jagielski 2020 / Nasr 2023).
5. [`notebooks/sq1-calibration/04_sota_calibration.ipynb`](notebooks/sq1-calibration/04_sota_calibration.ipynb) — **SQ-1d Tier B** SOTA-faithful baseline; CNN target + shadow-target FL parity + LiRA and RMIA-online side-by-side. Requires `make install-sota`.
6. [`notebooks/sq2-metadata/01_per_subject_fidelity.ipynb`](notebooks/sq2-metadata/01_per_subject_fidelity.ipynb) — 200-pod federation × the rule engine.
7. [`notebooks/sq3-composability/01_rule_battery.ipynb`](notebooks/sq3-composability/01_rule_battery.ipynb) — composability rules vs hand-authored expectations.
8. [`notebooks/sq5-comparison/01_scalar_vs_stepwise.ipynb`](notebooks/sq5-comparison/01_scalar_vs_stepwise.ipynb) — the head-to-head: scalar (ε, AUROC) ordering vs empirical worst-record TPR ordering.

### Tutorial scale vs paper scale

Each notebook has a `PAPER_SCALE` flag near the top of its config cell. Default is tutorial scale (small target counts, fewer shadow models, completes in seconds to a few minutes). Set the flag to `True` for the calibrated configuration (60 × 128 for MIA; 20 × 8 σ × 500 iter for gradient inversion) — 1–10 minutes per notebook.

### Tier A vs Tier B

The harness exposes two evaluation paths. **Tier A** ([notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb](notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb)) uses a numpy MLP target with plain-SGD shadows — ~45 s at paper scale on CPU, designed for one-command reproducibility without installing PyTorch. **Tier B** ([notebooks/sq1-calibration/04_sota_calibration.ipynb](notebooks/sq1-calibration/04_sota_calibration.ipynb)) uses a PyTorch CNN target with a 64-shadow pool where each shadow is itself a full FedAvg + DP-SGD federated run (the Carlini shadow-target parity convention), with LiRA and RMIA-online side-by-side and bootstrap CIs and per-class stratification on every TPR estimate — ~3.5 min on Apple Silicon MPS. Install via `pip install -e ".[sota]"` (~2 GB of wheels for torch + opacus; opt-in).

Both tiers write to the same audit-trailed result records under `results/sq1/`; the card's `evidence_ref` field may point at either or both.

### Optional: deploy the pod federation to a real Community Solid Server

```bash
make solid-up         # docker compose up -d  (Community Solid Server on :3000)
make solid-populate   # provisions 200 accounts + pods via WebID-OIDC + WAC
make solid-down       # tear down + remove the volume
```

See [`solid_deploy/README.md`](solid_deploy/README.md). The on-disk evaluation (`make eval`) does not require this — the CSS deployment is the runtime check for the Solid claims in the paper.

---

## What's committed vs generated

The repo commits **the generator, not the generated artefacts**. Everything under `data/`, `pods/`, `card/chains/`, and `results/` is built from a master seed at `prepare` time. This keeps the repo small and makes the audit trail meaningful — re-running `make prepare` on a fresh clone produces bit-for-bit identical files at the same seed.

| Committed | Generated by `make prepare` / `make eval` |
|---|---|
| `flta_eval/` — harness package | `data/bloodmnist.npz` + `data/_manifest.json` |
| `card/example_chain.json` — hand-authored card | `pods/persona_*/` + `pods/_manifest.json` |
| `card/battery_expected.json` — hand-authored expectations | `card/chains/*.json` + `card/battery_manifest.json` |
| `scripts/prepare.py`, `notebooks/`, `solid_deploy/`, docs | `results/sq{1,2,3,5}/*.json` |
| `Makefile`, `pyproject.toml`, `requirements.txt`, `LICENSE`, `CITATION.cff`, `.gitignore` | `.venv/`, `__pycache__/`, `flta_eval.egg-info/`, `.ipynb_checkpoints/` |

See [`.gitignore`](.gitignore) for the exact list.

---

## Working with the companion Privacy Explorer repo

A few documents in this repo reference the parent [Privacy Explorer / YAPS](https://github.com/j0hn-s/privacy-explorer) project — the YAPS schema, the base rule engine, the workshop materials. **The harness is self-contained** (no Python imports from privacy-explorer) — but if you want the relative-path links in [`docs/PRIVACY_EXPLORER_MAP.md`](docs/PRIVACY_EXPLORER_MAP.md) to resolve locally, check out the two repositories as siblings:

```
~/Documents/dev/
├── privacy-explorer/         # parent project (YAPS, workshops, survey paper)
└── stepwise-privacy-cards/   # this repo
```

Otherwise, follow the GitHub links in the docs.

---

## Citation

If you use this work, please cite both [`CITATION.cff`](CITATION.cff) (this repo) and the parent Privacy Explorer repository.
