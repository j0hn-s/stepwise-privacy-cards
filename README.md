# Stepwise privacy cards for federated learning

Companion artefact to the short paper *Beyond scalar epsilon: stepwise privacy cards for federated learning* ([`paper/draft.md`](paper/draft.md); FLTA 2026). An end-to-end, reproducible evaluation harness: a small numpy federated-learning training loop on the **MedMNIST v2 BloodMNIST** benchmark, two FL-native attacks (Carlini's LiRA membership inference + Geiping-style gradient inversion), a 200-pod Solid federation mock with optional deployment against a real Community Solid Server, a deterministic rule engine, and a 16-chain composability battery.

This repo is the empirical complement to the *Privacy Explorer / YAPS* framework (the parent privacy-engineering project: https://github.com/j0hn-s/privacy-explorer — replace with your final URL when publishing). It is otherwise self-contained.

---

## What is in this repo

| Path | What it is |
|---|---|
| [`paper/draft.md`](paper/draft.md) | The short paper (markdown source); [`paper/draft.docx`](paper/draft.docx) is the Word-pasteable form |
| [`flta_eval/`](flta_eval/) | The harness: `datasets`, `fl`, `attacks`, `pods`, `rules`, `chains`, `audit` |
| [`notebooks/`](notebooks/) | Six runnable notebooks (walkthrough + four study questions) |
| [`card/example_chain.json`](card/example_chain.json) | The worked privacy card the evaluation calibrates |
| [`card/battery_expected.json`](card/battery_expected.json) | Hand-authored expected rule firings (independent of the rule engine) |
| [`scripts/prepare.py`](scripts/prepare.py) | One-shot generator: fetches BloodMNIST, builds the partition + pods + 16-chain battery |
| [`solid_deploy/`](solid_deploy/) | Optional: docker-compose Community Solid Server + populate script |
| [`METHODOLOGY.md`](METHODOLOGY.md) · [`SCHEMAS.md`](SCHEMAS.md) · [`THREAT_MODEL.md`](THREAT_MODEL.md) · [`ENVIRONMENT.md`](ENVIRONMENT.md) · [`TUTORIAL.md`](TUTORIAL.md) | Methodology, data structures, threat profiles, environment, video-tutorial script |
| [`PRIVACY_EXPLORER_MAP.md`](PRIVACY_EXPLORER_MAP.md) | How this repo links to the parent Privacy Explorer / YAPS work |

---

## Testing locally (no push needed)

The whole evaluation runs offline after one fetch (BloodMNIST is ~35 MB; cached on first use). Recommended path before pushing anywhere:

```bash
# from this directory
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                    # ~30 s
python scripts/prepare.py           # ~5 s after the initial BloodMNIST fetch
make eval                           # ~5 min at tutorial scale; runs all six notebooks
```

`make eval` runs every notebook end-to-end via `jupyter nbconvert --execute --inplace`. Outputs land under `results/sq1/`, `results/sq2/`, `results/sq3/`, `results/sq5/` as JSON records, each carrying the four-field audit trail (harness commit hash, dataset SHA-256, configuration hash, seed namespace).

### Per-notebook testing in Jupyter Lab

If you want to step through and inspect rather than batch-run:

```bash
make lab            # starts Jupyter Lab; open notebooks/ in the file browser
```

Order to try first time:

1. [`notebooks/00_walkthrough.ipynb`](notebooks/00_walkthrough.ipynb) — orientation; runs `prepare`, prints the dataset manifest, the example card, the pod federation, the chain battery.
2. [`notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb`](notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb) — per-record MIA against the FL-released model (default meta-classifier is **LiRA**, Carlini 2022 SOTA).
3. [`notebooks/sq1-calibration/02_gradient_inversion.ipynb`](notebooks/sq1-calibration/02_gradient_inversion.ipynb) — gradient inversion vs σ; default **Total-Variation regularisation** on, per Geiping 2020 §3.1.
4. [`notebooks/sq2-metadata/01_per_subject_fidelity.ipynb`](notebooks/sq2-metadata/01_per_subject_fidelity.ipynb) — 200-pod federation × the rule engine.
5. [`notebooks/sq3-composability/01_rule_battery.ipynb`](notebooks/sq3-composability/01_rule_battery.ipynb) — composability rules vs the hand-authored expectations.
6. [`notebooks/sq5-comparison/01_scalar_vs_stepwise.ipynb`](notebooks/sq5-comparison/01_scalar_vs_stepwise.ipynb) — the head-to-head: scalar (ε, AUROC) ordering vs empirical worst-record TPR ordering.

### Tutorial scale vs paper scale

Each notebook has a `PAPER_SCALE` (or `TUTORIAL_SCALE`) flag near the top of its config cell. **Default is tutorial scale**: small target counts, fewer shadow models, completes in seconds-to-minutes. **Paper scale** (set the flag to `True`) runs at the calibrated configuration (60 × 128 for MIA; 20 × 8 σ × 500 iter for gradient inversion) and takes 15–60 minutes per notebook. Use paper scale for any numbers you actually want to report.

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

A few documents in this repo reference the parent [Privacy Explorer / YAPS](https://github.com/j0hn-s/privacy-explorer) project — the YAPS schema, the base rule engine, the workshop materials. **The harness is self-contained** (no Python imports from privacy-explorer) — but if you want the relative-path links in [`PRIVACY_EXPLORER_MAP.md`](PRIVACY_EXPLORER_MAP.md) to resolve locally, check out the two repositories as siblings:

```
~/Documents/dev/
├── privacy-explorer/         # parent project (YAPS, workshops, survey paper)
└── stepwise-privacy-cards/   # this repo
```

Otherwise, follow the GitHub links in the docs.

---

## Citation

If you use this work, please cite both [`CITATION.cff`](CITATION.cff) (this repo) and the parent Privacy Explorer repository.
