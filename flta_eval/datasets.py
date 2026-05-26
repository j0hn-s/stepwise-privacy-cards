"""Dataset loader for the FLTA evaluation.

Primary dataset: **BloodMNIST** from MedMNIST v2 [Yang et al., *Sci. Data*
2023] — a real, open, peer-reviewed medical-imaging benchmark used in
recent FL privacy work [Hatamizadeh et al. 2023; Boenisch et al. 2023].
Choice rationale:

- *Open and reproducible.* Fetched from Zenodo via the `medmnist` Python
  package on first use; checksummed against the package's manifest.
- *Real medical idiosyncrasies.* Class imbalance from rare cell types,
  intra-class variation across imaging conditions, ~17k 28×28 RGB
  images across 8 classes. Captures the kinds of distribution
  properties that scalar reporting hides.
- *FL-benchmark provenance.* Used as a cross-device / cross-silo FL
  benchmark in the literature this paper draws on.

The dataset is **partitioned across pods** via a Dirichlet(α) split per
the standard non-IID FL methodology [Hsu et al. 2019, "Measuring the
effects of non-identical data distribution for federated visual
classification"]; α=0.5 by default gives moderately non-IID partitions.

Each pod is a *data subject* contributing one slice; the FL evaluation
treats per-pod slices as the unit of federation participation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flta_eval.audit import derive_seed, file_sha256


# ─── Dataset spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    citation: str


SPECS: dict[str, DatasetSpec] = {
    "bloodmnist": DatasetSpec(
        name="bloodmnist",
        description="MedMNIST v2 BloodMNIST: 17 092 28×28 RGB blood-cell images, 8 classes; train/val/test as published",
        citation="Yang et al., Sci. Data 2023; Acevedo et al., Data in Brief 2020 (underlying dataset)",
    ),
}


PARTITION_ALPHA = 0.5  # Dirichlet concentration; lower = more non-IID
DEFAULT_N_PODS = 50    # tutorial scale; paper scale up to 200


# ─── BloodMNIST fetch + cache ────────────────────────────────────────────────


def _bloodmnist_path(cache_dir: Path) -> Path:
    return cache_dir / "bloodmnist_28.npz"


def _fetch_bloodmnist(cache_dir: Path) -> Path:
    """Fetch the BloodMNIST 28×28 NPZ file via the `medmnist` package.

    The medmnist package itself fetches from Zenodo on first call and
    caches under `~/.medmnist/`; we copy the file into our local cache so
    the manifest checksum is stable across machines.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _bloodmnist_path(cache_dir)
    if target.exists():
        return target

    try:
        import medmnist
    except ImportError as e:
        raise RuntimeError(
            "BloodMNIST requires the `medmnist` package. Install with:\n"
            "    pip install medmnist\n"
            "or use `make install`."
        ) from e

    info = medmnist.INFO["bloodmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    # Trigger the download into medmnist's cache by instantiating the dataset.
    DataClass(split="train", download=True, size=28)
    # medmnist's npz cache is under ~/.medmnist/ by default.
    src_default = Path.home() / ".medmnist" / "bloodmnist_28.npz"
    src_fallback = Path.home() / ".medmnist" / "bloodmnist.npz"
    src = src_default if src_default.exists() else src_fallback
    if not src.exists():
        raise RuntimeError(f"medmnist cache file not found at {src_default} or {src_fallback}")
    target.write_bytes(src.read_bytes())
    return target


def load_bloodmnist(cache_dir: Path) -> dict[str, np.ndarray]:
    """Return `{'X_train', 'y_train', 'X_test', 'y_test'}` as numpy arrays.

    Images are normalised to [0, 1] float32, flattened to (n, 28*28*3).
    Labels are int64 in [0, 8).
    """
    path = _fetch_bloodmnist(cache_dir)
    npz = np.load(path)
    # MedMNIST NPZ files use 'train_images', 'train_labels', etc.
    X_train = npz["train_images"].astype(np.float32) / 255.0
    X_test = npz["test_images"].astype(np.float32) / 255.0
    y_train = npz["train_labels"].astype(np.int64).reshape(-1)
    y_test = npz["test_labels"].astype(np.int64).reshape(-1)
    # Flatten 28×28×3 → 2352
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "n_classes": int(npz["train_labels"].max() + 1),
        "input_dim": X_train.shape[1],
    }


# ─── Dirichlet partitioning across pods ──────────────────────────────────────


def dirichlet_partition(
    y: np.ndarray,
    n_pods: int,
    alpha: float = PARTITION_ALPHA,
    master_seed: int = 20260525,
    namespace: str = "datasets.partition.bloodmnist.v1",
) -> list[np.ndarray]:
    """Partition indices of `y` across `n_pods` pods via a Dirichlet split.

    Per [Hsu et al. 2019]: for each class c, draw a Dirichlet(α)
    proportions vector over pods and assign that class's samples to pods
    accordingly. Lower α → more non-IID; α=0.5 is the standard moderate
    setting in the FL benchmarking literature.

    Returns: list of length `n_pods` with each entry an int array of
    indices into `y`.
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    n_classes = int(y.max() + 1)
    pod_indices: list[list[int]] = [[] for _ in range(n_pods)]
    for c in range(n_classes):
        class_indices = np.where(y == c)[0]
        rng.shuffle(class_indices)
        # Draw proportions per pod for this class
        proportions = rng.dirichlet(alpha=[alpha] * n_pods)
        # Convert to cumulative cut points
        n = len(class_indices)
        cuts = (np.cumsum(proportions) * n).astype(int)
        prev = 0
        for p in range(n_pods):
            chunk = class_indices[prev:cuts[p]]
            pod_indices[p].extend(chunk.tolist())
            prev = cuts[p]
    return [np.array(p, dtype=np.int64) for p in pod_indices]


# ─── Manifest + helpers ──────────────────────────────────────────────────────


def write_manifest(
    out_dir: Path,
    *,
    n_pods: int = DEFAULT_N_PODS,
    master_seed: int = 20260525,
) -> dict:
    """Build the partition manifest and persist it.

    Note: we *do not* materialise per-pod data files for the imaging
    dataset — they would be large (each MedMNIST image is ~7 KB; 200 pods
    of 80 images each = ~100 MB of duplicated data on disk). Instead, the
    manifest records the *indices* each pod owns, and the FL training
    loop reads from the central NPZ at runtime using those indices.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_bloodmnist(out_dir)
    partition = dirichlet_partition(
        ds["y_train"], n_pods=n_pods, master_seed=master_seed,
    )
    manifest = {
        "dataset": "bloodmnist",
        "citation": SPECS["bloodmnist"].citation,
        "n_train": int(len(ds["y_train"])),
        "n_test": int(len(ds["y_test"])),
        "n_classes": ds["n_classes"],
        "input_dim": ds["input_dim"],
        "n_pods": n_pods,
        "partition_alpha": PARTITION_ALPHA,
        "master_seed": master_seed,
        "npz_sha256": file_sha256(_bloodmnist_path(out_dir)),
        "partition_namespace": "datasets.partition.bloodmnist.v1",
        "pod_sizes": [int(len(idx)) for idx in partition],
        "pod_class_distribution": [
            {int(c): int((ds["y_train"][idx] == c).sum()) for c in range(ds["n_classes"])}
            for idx in partition
        ],
        "pod_indices": [idx.tolist() for idx in partition],
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def load_partition(out_dir: Path) -> dict:
    """Load the partition manifest."""
    return json.loads((out_dir / "_manifest.json").read_text())


def load_pod_data(out_dir: Path, pod_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the given pod, reading from the central NPZ."""
    ds = load_bloodmnist(out_dir)
    manifest = load_partition(out_dir)
    indices = np.array(manifest["pod_indices"][pod_index], dtype=np.int64)
    return ds["X_train"][indices], ds["y_train"][indices]
