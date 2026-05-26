# Environment — FLTA 2026 evaluation companion artefact

## 1. Dependencies

### Python (notebooks + harness — Tier A core)

| Component | Pinned version | Purpose |
|---|---|---|
| Python | 3.10–3.13 | Runtime |
| numpy | == 2.4.6 | Numerics; Tier A FL training loop. Pinned for bit-for-bit reproducibility — see §3 below. |
| scipy | == 1.17.1 | L-BFGS-B optimisation for gradient inversion; Gaussian CDF for LiRA |
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

Install with `pip install -e .` (uses [`pyproject.toml`](../pyproject.toml)).

### Optional: Position B SOTA-faithful baseline

| Component | Version | Purpose |
|---|---|---|
| torch | ≥ 2.4 | PyTorch backend for [`flta_eval/fl_torch.py`](../flta_eval/fl_torch.py) — `TinyCNN` (25k-param CNN) + FedAvg + DP-SGD. MPS-accelerated on Apple Silicon. |
| opacus | ≥ 1.5 | Reserved for full PRV-accountant integration into the FedAvg loop (currently exercised only by `scripts/compare_accountants.py`; the FedAvg DP-SGD in `fl_torch.py` uses our own per-update clip + Gaussian noise to match `fl.py`). |

Install via `pip install -e ".[sota]"`. ~2 GB of wheels — opt-in only. Used by [`notebooks/sq1-calibration/04_sota_calibration.ipynb`](../notebooks/sq1-calibration/04_sota_calibration.ipynb) and by any shadow-pool MIA run that wants CNN-target / parity-shadow rigour.

### Optional: Solid runtime

| Component | Version | Purpose |
|---|---|---|
| Docker + docker-compose | any modern | Run Community Solid Server locally |
| Community Solid Server | 7.x | Solid Protocol reference implementation |

See [`solid_deploy/`](../solid_deploy/).

### Optional: tighter DP accountant (PRV)

| Component | Version | Purpose |
|---|---|---|
| prv-accountant | ≥ 0.2.0 | Privacy Loss Distribution (PRV) accountant of Gopi *et al.* (NeurIPS 2021) for tighter (ε, δ) reporting. Pure-Python; no PyTorch dependency. Install via `pip install -e ".[accountants]"`. |

Used by [`scripts/compare_accountants.py`](../scripts/compare_accountants.py); not required for the default evaluation. See §3 of [`METHODOLOGY.md`](METHODOLOGY.md).

## 2. Datasets

BloodMNIST is fetched from Zenodo by the `medmnist` package on first call to `scripts/prepare.py`. Cached at `~/.medmnist/bloodmnist.npz` (~35 MB). The evaluation copies it into `data/bloodmnist.npz` and records its SHA-256 in `data/_manifest.json` so reproducibility is checkpointable against a known file.

## 3. Reproducibility expectations

- **Bit-for-bit determinism** of Tier A metrics across consecutive runs at the same harness commit, dataset checksum, configuration hash, seed namespace, **and pinned (numpy, scipy, BLAS) tuple**.
- **Tier B reproducibility envelope.** PyTorch on MPS is *not* fully deterministic for every op even at fixed seed; reproducibility across consecutive runs is to ~3 decimal places, comparable to the cross-BLAS envelope. The SQ-1d *verdict* (LiRA fires AMBER, RMIA does not) is reproducible across runs; absolute TPR values may drift in the last decimal.
- **Audit-trail completeness.** Every result record carries the audit fields (`harness_commit`, `dataset.sha256`, `config.config_hash_sha256`, `seed_namespace`, `timestamp_utc`). Tier B records additionally include `config.device` (cpu / mps) and `config.model_class`.
- **One-command re-run.** `make eval` runs the full Tier A notebook battery in ~5 min; `make sota` (after `make install-accountants` and `pip install -e ".[sota]"`) runs the Tier B baseline in ~10 min.

### Why numpy + BLAS are pinned

numpy's linear-algebra calls dispatch to a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux, MKL on Intel-tuned wheels). BLAS implementations differ in their internal accumulation order — e.g., the dot-product reduction tree — and the difference perturbs the low-order bits of summed floats even at fixed seed. Reproducibility claims that span BLAS backends are therefore *not* bit-for-bit; they are "stable to ~3 decimal places." Two consequences:

1. **`numpy==2.4.6` and `scipy==1.17.1` are pinned** in [`pyproject.toml`](../pyproject.toml). Upgrading either may change the trailing decimals.
2. **The BLAS backend** (visible via `numpy.show_config()`) is part of the reproducibility envelope. The SQ-5 *ordering* claim (paper §V) is invariant across BLAS backends because the disagreement is order-of-magnitude in worst-record TPR; absolute (ε, TPR) values are not bit-for-bit portable across backends.

Authoritative discussion: NEP 50 [a], the numpy reproducibility note [b], and the BLAS-level discussion in the SciPy roadmap [c].

[a] R. Gommers *et al.*, *NEP 50 — Promotion rules for NumPy scalars*, 2022.
[b] NumPy maintainers, *Reproducibility and floating-point determinism*, numpy.org/doc, accessed 2026.
[c] M. Brett *et al.*, *On bit-exact reproducibility across BLAS backends*, SciPy ML 2018 thread.

## 4. Expected runtimes

### Tier A (numpy MLP — default)

| Notebook | Wall-clock |
|---|---|
| `00_walkthrough.ipynb` | < 10 s |
| `sq1-calibration/01_mia_per_record_sweep.ipynb` (paper scale) | ~ 45 s |
| `sq1-calibration/02_gradient_inversion.ipynb` | ~ 60–120 s |
| `sq1-calibration/03_canary_audit.ipynb` (200 canaries) | ~ 2.5 min |
| `sq2-metadata/01_per_subject_fidelity.ipynb` | ~ 1 s |
| `sq3-composability/01_rule_battery.ipynb` | < 1 s |
| `sq5-comparison/01_scalar_vs_stepwise.ipynb` | ~ 1–2 min |
| `scripts/multi_seed_sq5.py --seeds 10 --paper-scale` | ~ 7 min |

### Tier B (PyTorch CNN, requires `pip install -e ".[sota]"`)

| Notebook / script | Wall-clock (Apple Silicon MPS) |
|---|---|
| `sq1-calibration/04_sota_calibration.ipynb` (64 shadow CNNs, 60 targets) | ~ 3.5 min |
| `scripts/compare_accountants.py` | < 5 s |

## 5. Hardware

**Tier A (default).** Workstation with 8 GB RAM, 4 CPU cores. No GPU required; the numpy MLP is small.

**Tier B (optional).** Same minimum spec plus PyTorch-compatible hardware. Apple Silicon Macs use MPS automatically (`mps` device); on Linux/Intel CPUs torch will fall back to CPU and Tier B run-times rise by roughly 5–10× (shadow-pool builds become 15–30 min). NVIDIA GPUs are not auto-detected by the current code path; CUDA support is a one-line `DEVICE` change in [`flta_eval/fl_torch.py`](../flta_eval/fl_torch.py).

## 6. Deferred environment items (future iterations)

- **Full Flower + Opacus dataloader pipeline.** The Tier B path uses minimum-viable PyTorch — our own FedAvg loop and per-update DP-SGD rather than Opacus's `PrivacyEngine` + Flower's client orchestration. Production-scale FL on Flower + Opacus + multi-architecture targets remains future work (paper §VI).
- **CIFAR-10/100 benchmarks.** BloodMNIST is the primary benchmark; CIFAR is named as a comparable benchmark the SOTA-MIA literature uses but is out of scope for this artefact.
- **CUDA GPU path.** `fl_torch.DEVICE` selects MPS or CPU; CUDA is a one-line change but not exercised in CI.
- **TEE attestation harness.** The card's TEE step declares attestation as an assurance anchor; this evaluation does not run an attestation verifier.
- **Real cryptographic Consent Receipts.** Mock signatures are used. Verifiable Credentials integration is named for future work.
- **Cross-IdP federated identity.** The Solid deployment uses one CSS instance; cross-IdP scenarios are out of scope.

## 7. Licensing

Permissive open-source licence (Apache 2.0, see [`../LICENSE`](../LICENSE)). BloodMNIST is licensed under CC BY 4.0 [Yang et al. 2023]. PyTorch is BSD-3; Opacus is Apache 2.0; `prv-accountant` is MIT. No copyleft dependencies.
