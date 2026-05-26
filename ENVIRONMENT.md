# Environment — FLTA 2026 evaluation companion artefact

## 1. Dependencies

### Python (notebooks + harness)

| Component | Pinned version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| numpy | ≥ 1.26 | Numerics; FL training loop |
| scipy | ≥ 1.11 | L-BFGS-B optimisation for gradient inversion |
| scikit-learn | ≥ 1.4 | Univariate logistic regression for MIA meta-classifier |
| pandas | ≥ 2.2 | Dataset handling |
| pyarrow | ≥ 15 | Parquet I/O |
| jupyterlab | ≥ 4.0 | Notebook runtime |
| matplotlib | ≥ 3.8 | Diagnostic plots (optional) |
| rdflib | ≥ 7.0 | RDF/Turtle parsing (pod metadata) |
| pyld | ≥ 2.0 | JSON-LD processing |
| PyYAML | ≥ 6.0 | yaps rule loading |
| medmnist | latest | BloodMNIST loader (fetches from Zenodo on first use) |
| requests | latest | Used by `solid_deploy/populate.py` |

Install with `pip install -e .` (uses [`pyproject.toml`](pyproject.toml)).

### Optional: Solid runtime

| Component | Version | Purpose |
|---|---|---|
| Docker + docker-compose | any modern | Run Community Solid Server locally |
| Community Solid Server | 7.x | Solid Protocol reference implementation |

See [`solid_deploy/`](solid_deploy/).

## 2. Datasets

BloodMNIST is fetched from Zenodo by the `medmnist` package on first call to `scripts/prepare.py`. Cached at `~/.medmnist/bloodmnist.npz` (~35 MB). The evaluation copies it into `data/bloodmnist.npz` and records its SHA-256 in `data/_manifest.json` so reproducibility is checkpointable against a known file.

## 3. Reproducibility expectations

- **Bit-for-bit determinism** of metrics across consecutive runs at the same harness commit, dataset checksum, configuration hash, and seed namespace.
- **Audit-trail completeness.** Every result record carries the four-field audit trail.
- **One-command re-run.** `make eval` runs the full notebook battery; takes ~5 min at tutorial scale.

## 4. Expected runtimes (tutorial scale)

| Notebook | Wall-clock |
|---|---|
| `00_walkthrough.ipynb` | < 10 s |
| `sq1-calibration/01_mia_per_record_sweep.ipynb` | ~ 60 s (FL training ~10 s + MIA sweep ~50 s) |
| `sq1-calibration/02_gradient_inversion.ipynb` | ~ 60–120 s (5 targets × 4 σ × 80 iters) |
| `sq2-metadata/01_per_subject_fidelity.ipynb` | ~ 1 s (pure rule-engine loop) |
| `sq3-composability/01_rule_battery.ipynb` | < 1 s |
| `sq5-comparison/01_scalar_vs_stepwise.ipynb` | ~ 1–2 min (three FL trainings + three MIA sweeps) |

Paper-scale configurations push wall-clock significantly higher — see each notebook's `PAPER_SCALE` flag.

## 5. Hardware

Minimum: workstation with 8 GB RAM, 4 CPU cores. No GPU required; the numpy MLP is small enough.

## 6. Deferred environment items

- **PyTorch / Flower integration.** Production FL benchmarks (FedScale, Flower) would replace `flta_eval/fl.py`. The harness API is structured so the FL implementation is one module.
- **TEE attestation harness.** The card's TEE step declares attestation as an assurance anchor; this evaluation does not run an attestation verifier.
- **Real cryptographic Consent Receipts.** Mock signatures are used. Verifiable Credentials integration is named for future work.
- **Cross-IdP federated identity.** The Solid deployment uses one CSS instance; cross-IdP scenarios are out of scope.

## 7. Licensing

Permissive open-source licence (Apache 2.0 / MIT to be aligned with the parent repository). BloodMNIST is licensed under CC BY 4.0 [Yang et al. 2023]. No copyleft dependencies.
