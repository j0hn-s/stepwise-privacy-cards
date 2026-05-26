"""PyTorch CNN + DP-SGD FedAvg trainer.

Companion to `flta_eval.fl` (numpy MLP) for the Position B SOTA-faithful
calibration. Substantive upgrades vs the numpy path:

- **Convolutional architecture.** A small CNN (~25k params) processes
  28×28×3 BloodMNIST images with their spatial structure intact rather
  than flattening to a 2,352-dimensional feature vector. This is the
  architecture class the FL-MIA literature attacks against [Hatamizadeh
  CVPR 2023; Boenisch USENIX 2023], not a 2-layer MLP.

- **PyTorch backward + MPS acceleration.** Per-sample gradients are
  computed by autograd; on Apple Silicon the MPS backend is used when
  available, giving roughly an order of magnitude speedup over CPU
  numpy. The harness still avoids a full Flower + Opacus dataloader
  pipeline — what we use here is the minimum-viable PyTorch path
  consistent with the Position B reframe. Flower + Opacus + CIFAR-class
  benchmarks are explicitly named as out-of-scope in paper §VI.

The DP-SGD convention is per-update clipping + Gaussian noise [Abadi
CCS 2016], matching the numpy path so that audit-trail records and
accountant ε remain comparable across the two trainers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flta_eval.audit import derive_seed
from flta_eval.fl import FLConfig

# MPS on Apple Silicon, otherwise CPU. CUDA paths are not exercised here.
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class TinyCNN(nn.Module):
    """Small CNN: two 3×3 conv blocks + global FC head."""

    def __init__(self, n_classes: int = 8, input_dim: int = 2352):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.reshape(-1, 28, 28, 3).permute(0, 3, 1, 2)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @torch.no_grad()
    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(DEVICE)
        return F.softmax(self.forward(X), dim=1).cpu().numpy()

    def accuracy(self, X_np: np.ndarray, y_np: np.ndarray) -> float:
        return float((self.predict_proba(X_np).argmax(axis=1) == y_np).mean())

    def init_from_seed(self, master_seed: int, namespace: str) -> None:
        seed = derive_seed(master_seed, namespace) % (2**31 - 1)
        gen = torch.Generator()
        gen.manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    fan_in = p.shape[1] * (p.shape[2] * p.shape[3] if p.dim() == 4 else 1)
                    std = (2.0 / fan_in) ** 0.5
                    p.normal_(mean=0.0, std=std, generator=gen)
                else:
                    p.zero_()


def federated_train_torch(
    *,
    model: TinyCNN,
    pod_data: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: FLConfig,
    master_seed: int = 20260525,
    namespace: str = "fl_torch.train.v1",
):
    """PyTorch FedAvg + DP-SGD training loop. Returns (model, [])."""
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    torch_gen = torch.Generator()
    torch_gen.manual_seed(derive_seed(master_seed, namespace + ".torch") % (2**31 - 1))

    model = model.to(DEVICE)
    n_pods = len(pod_data)

    for r in range(config.n_rounds):
        sampled = (list(range(n_pods)) if config.sample_clients_per_round is None
                   or config.sample_clients_per_round >= n_pods
                   else sorted(rng.choice(n_pods, size=config.sample_clients_per_round,
                                          replace=False).tolist()))

        client_grads: list[dict[str, torch.Tensor]] = []
        for client_idx in sampled:
            Xc, yc = pod_data[client_idx]
            if len(Xc) == 0:
                continue
            bs = min(config.client_batch_size, len(Xc))
            idx = rng.choice(len(Xc), size=bs, replace=False)
            Xb = torch.from_numpy(np.asarray(Xc[idx], dtype=np.float32)).to(DEVICE)
            yb = torch.from_numpy(np.asarray(yc[idx], dtype=np.int64)).to(DEVICE)

            model.zero_grad(set_to_none=False)
            loss = F.cross_entropy(model(Xb), yb)
            loss.backward()

            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
            total_norm = sum((g ** 2).sum() for g in grads.values()).sqrt()
            scale = torch.clamp(config.clip_norm / (total_norm + 1e-12), max=1.0)
            for n in grads:
                grads[n] = grads[n] * scale
            if config.noise_multiplier > 0.0:
                sigma = config.noise_multiplier * config.clip_norm
                for n in grads:
                    noise = torch.randn(grads[n].shape, generator=torch_gen) * sigma
                    grads[n] = grads[n] + noise.to(DEVICE)
            client_grads.append(grads)

        if not client_grads:
            continue
        avg = {n: torch.stack([g[n] for g in client_grads]).mean(0)
               for n in client_grads[0]}
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data -= config.client_lr * avg[n]

    return model, []


def make_torch_shadow_train_fn(input_dim: int, n_classes: int):
    """Factory: returns a shadow-pool-compatible FL training closure."""
    def train_fn(*, pod_data, X_test, y_test, config, master_seed, namespace):
        m = TinyCNN(n_classes=n_classes, input_dim=input_dim)
        m.init_from_seed(master_seed, namespace + ".init")
        trained, _ = federated_train_torch(
            model=m, pod_data=pod_data,
            X_test=X_test, y_test=y_test, config=config,
            master_seed=master_seed, namespace=namespace + ".train",
        )
        return trained
    return train_fn
