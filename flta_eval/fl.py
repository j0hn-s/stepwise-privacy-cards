"""Minimal federated-learning training loop.

A small, pure-numpy FL training implementation suitable for FLTA-scale
evaluation: a 2-layer MLP, FedAvg aggregation [McMahan et al., AISTATS
2017], DP-SGD-style per-client gradient clipping + Gaussian noise
[Abadi et al., ACM CCS 2016], an optional secure-aggregation flag (only
the aggregate is observable; the individual updates are not), and a
hook for capturing per-round gradients for downstream gradient-inversion
analysis.

The implementation deliberately does *not* use PyTorch — the training is
small enough to run in numpy and keeps the dependency surface narrow
for tutorial reproducibility. Composition tracking follows Rényi DP
[Mironov, IEEE CSF 2017].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from flta_eval.audit import derive_seed


# ─── Small MLP (2 layers) ────────────────────────────────────────────────────


@dataclass
class MLP:
    """input_dim → hidden_dim → n_classes; ReLU + softmax cross-entropy."""

    input_dim: int
    hidden_dim: int
    n_classes: int
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(0)
        # He init for ReLU
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.input_dim),
                             (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim),
                             (self.hidden_dim, self.n_classes)).astype(np.float32)
        self.b2 = np.zeros(self.n_classes, dtype=np.float32)

    def init_from_seed(self, master_seed: int, namespace: str) -> None:
        rng = np.random.default_rng(derive_seed(master_seed, namespace))
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.input_dim),
                             (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim),
                             (self.hidden_dim, self.n_classes)).astype(np.float32)
        self.b2 = np.zeros(self.n_classes, dtype=np.float32)

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, p: dict[str, np.ndarray]) -> None:
        self.W1 = p["W1"].astype(np.float32, copy=True)
        self.b1 = p["b1"].astype(np.float32, copy=True)
        self.W2 = p["W2"].astype(np.float32, copy=True)
        self.b2 = p["b2"].astype(np.float32, copy=True)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Returns (logits, cache) where cache holds activations needed for backward."""
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        logits = A1 @ self.W2 + self.b2
        return logits, {"X": X, "Z1": Z1, "A1": A1, "logits": logits}

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        """Cross-entropy loss and parameter gradients."""
        logits, cache = self.forward(X)
        # Numerically stable softmax
        L = logits - logits.max(axis=1, keepdims=True)
        exp_L = np.exp(L)
        probs = exp_L / exp_L.sum(axis=1, keepdims=True)
        n = X.shape[0]
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
        # Backprop
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dW2 = cache["A1"].T @ dlogits
        db2 = dlogits.sum(axis=0)
        dA1 = dlogits @ self.W2.T
        dZ1 = dA1 * (cache["Z1"] > 0)
        dW1 = cache["X"].T @ dZ1
        db1 = dZ1.sum(axis=0)
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        L = logits - logits.max(axis=1, keepdims=True)
        exp_L = np.exp(L)
        return exp_L / exp_L.sum(axis=1, keepdims=True)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict_proba(X).argmax(axis=1)
        return float((preds == y).mean())


# ─── DP-SGD noise + clipping at the client ──────────────────────────────────


def _flatten(grads: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([g.ravel() for g in grads.values()])


def _unflatten(flat: np.ndarray, template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    i = 0
    for k, t in template.items():
        n = t.size
        out[k] = flat[i:i + n].reshape(t.shape).astype(np.float32)
        i += n
    return out


def clip_gradients(grads: dict[str, np.ndarray], clip_norm: float) -> dict[str, np.ndarray]:
    flat = _flatten(grads)
    norm = float(np.linalg.norm(flat))
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}


def add_dp_noise(
    grads: dict[str, np.ndarray],
    *,
    noise_multiplier: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Gaussian mechanism: N(0, (noise_multiplier × clip_norm)²)."""
    sigma = noise_multiplier * clip_norm
    return {k: g + rng.normal(0.0, sigma, size=g.shape).astype(np.float32) for k, g in grads.items()}


# ─── FedAvg with optional DP and secure-aggregation flag ────────────────────


@dataclass
class FLConfig:
    n_rounds: int = 5
    client_lr: float = 0.05
    client_steps: int = 1            # local SGD steps per round
    client_batch_size: int = 32
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0    # 0 = no DP; > 0 = DP-SGD-style noise
    secure_aggregation: bool = False
    sample_clients_per_round: int | None = None  # None = all clients

    @property
    def has_dp(self) -> bool:
        return self.noise_multiplier > 0.0


@dataclass
class RoundRecord:
    """Per-round artefact: what an honest-but-curious coordinator observes."""

    round_index: int
    sampled_clients: list[int]
    # Per-client gradient (post-clip, post-noise) if secure_aggregation=False,
    # else only the aggregate sum is observable
    per_client_grads: list[dict[str, np.ndarray]] | None
    aggregate_grad: dict[str, np.ndarray]
    test_accuracy: float


def federated_train(
    *,
    model: MLP,
    pod_data: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: FLConfig,
    master_seed: int = 20260525,
    namespace: str = "fl.train.v1",
) -> tuple[MLP, list[RoundRecord]]:
    """Run a FedAvg loop and return (final model, per-round records).

    The per-round record captures exactly what an honest-but-curious
    coordinator observes: under secure_aggregation=True only the aggregate
    sum is recorded; otherwise per-client (clipped, DP-noised) gradients
    are recorded for downstream gradient-inversion analysis.
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    n_pods = len(pod_data)
    records: list[RoundRecord] = []

    for r in range(config.n_rounds):
        # Sample participating clients
        if config.sample_clients_per_round is None or config.sample_clients_per_round >= n_pods:
            sampled = list(range(n_pods))
        else:
            sampled = sorted(rng.choice(n_pods, size=config.sample_clients_per_round, replace=False))

        per_client_grads: list[dict[str, np.ndarray]] = []
        for client_idx in sampled:
            Xc, yc = pod_data[client_idx]
            if len(Xc) == 0:
                continue
            # Sample a batch
            batch_size = min(config.client_batch_size, len(Xc))
            idx = rng.choice(len(Xc), size=batch_size, replace=False)
            Xb, yb = Xc[idx], yc[idx]
            # Compute gradient (one local step)
            _, grads = model.loss_and_grad(Xb, yb)
            # Per-sample clipping is approximated here as per-update clipping
            # (the standard "user-level" DP-SGD variant used in cross-device FL).
            grads = clip_gradients(grads, config.clip_norm)
            if config.has_dp:
                grads = add_dp_noise(
                    grads, noise_multiplier=config.noise_multiplier,
                    clip_norm=config.clip_norm, rng=rng,
                )
            per_client_grads.append(grads)

        # Aggregate (FedAvg = mean of client updates)
        if not per_client_grads:
            continue
        agg: dict[str, np.ndarray] = {
            k: np.mean([g[k] for g in per_client_grads], axis=0)
            for k in per_client_grads[0]
        }

        # Server step: apply aggregate as a gradient step
        for k in model.params:
            getattr(model, k)[...] -= config.client_lr * agg[k]

        records.append(RoundRecord(
            round_index=r,
            sampled_clients=sampled,
            per_client_grads=None if config.secure_aggregation else per_client_grads,
            aggregate_grad=agg,
            test_accuracy=model.accuracy(X_test, y_test),
        ))

    return model, records


# ─── RDP accountant for the FedAvg + Gaussian-noise composition ──────────────


def rdp_epsilon(
    *,
    noise_multiplier: float,
    n_rounds: int,
    sample_rate: float,
    delta: float = 1e-5,
    orders: Iterable[float] = (1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64),
) -> float:
    """Convert (σ, n_rounds, q, δ) to an ε estimate via the RDP accountant.

    Uses the standard RDP bound for the subsampled Gaussian mechanism
    [Mironov, IEEE CSF 2017; Wang et al., Mirror 2019]:

        ε(α) ≤ q² · α / σ²                  (small-q regime, α ≥ 2)

    composed n_rounds times, then converted to (ε, δ) via
    ε = ε(α) + log(1/δ) / (α - 1).

    The result is the smallest ε across the orders tried — i.e. the
    standard "minimum-over-α" accountant. Conservative for the demo;
    swap in `opacus.accountants.RDPAccountant` for production accounting.
    """
    if noise_multiplier <= 0.0:
        return float("inf")
    best = float("inf")
    for alpha in orders:
        eps_alpha = (alpha * sample_rate ** 2) / (noise_multiplier ** 2) * n_rounds
        eps = eps_alpha + np.log(1.0 / delta) / (alpha - 1.0)
        if eps < best:
            best = float(eps)
    return best
