"""FL-native attack implementations.

Two attacks, both directly targeted at the FL pipeline (not at aggregate-
query release surfaces):

- **Gradient inversion** [Zhu et al., NeurIPS 2019; Geiping et al.,
  NeurIPS 2020]: given a per-round client gradient, recover the input
  batch. The default cosine + box-constraint loss can be optionally
  augmented with **anisotropic Total-Variation regularisation** (Geiping
  §3.1) for image-shaped inputs; this is the upgrade closest to the
  state-of-the-art GradInversion variant of Yin et al. (CVPR 2021).

- **Per-record membership inference** [Carlini et al., IEEE S&P 2022]
  against the FL-released model. Two meta-classifier modes:
    - `"logistic"` (default): univariate logistic over confidence-in-true-class.
    - `"lira"`: the **Likelihood Ratio Attack** of Carlini et al. (S&P 2022,
      Algorithm 1), which fits Gaussians to the IN / OUT shadow-logit
      distributions and uses the per-target likelihood ratio as the
      score. LiRA is strictly tighter than the logistic baseline and is
      the current state-of-the-art per-record MIA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize
from scipy.stats import norm as _norm

from flta_eval.audit import derive_seed
from flta_eval.fl import MLP, _flatten


# ─── Gradient inversion (Geiping cosine-similarity variant) ─────────────────


@dataclass
class GradInversionResult:
    target_image: np.ndarray         # ground-truth input (1, input_dim)
    target_label: int
    reconstructed: np.ndarray        # (1, input_dim) — recovered input
    cosine_distance: float           # final loss value
    pixel_l2: float                  # ||recon - target||₂
    pixel_psnr_db: float             # PSNR vs target


def _cosine_distance(g1: dict[str, np.ndarray], g2: dict[str, np.ndarray]) -> float:
    f1 = _flatten(g1)
    f2 = _flatten(g2)
    return 1.0 - float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-12))


def _tv_anisotropic(x_flat: np.ndarray, image_shape: tuple[int, int, int] | None) -> float:
    """Anisotropic Total-Variation regulariser on an image-shaped input.

    Definition follows Geiping et al. (NeurIPS 2020) §3.1:
        TV(x) = Σ |x[i+1,j] - x[i,j]| + Σ |x[i,j+1] - x[i,j]|
    summed over channels. Returns 0 when image_shape is None (treat input
    as opaque feature vector, no spatial prior).
    """
    if image_shape is None:
        return 0.0
    H, W, C = image_shape
    x = x_flat.reshape(H, W, C)
    dh = np.abs(x[1:, :, :] - x[:-1, :, :]).sum()
    dw = np.abs(x[:, 1:, :] - x[:, :-1, :]).sum()
    return float(dh + dw)


def gradient_inversion(
    *,
    model: MLP,
    target_grad: dict[str, np.ndarray],
    target_label: int,
    target_image: np.ndarray,
    max_iter: int = 500,
    tv_weight: float = 0.0,
    image_shape: tuple[int, int, int] | None = None,
    master_seed: int = 20260525,
    namespace: str = "attacks.grad_inversion.v1",
) -> GradInversionResult:
    """Recover a single training input from an observed client gradient.

    Implements the cosine-similarity-based inversion of Geiping *et al.*
    (NeurIPS 2020). The label is assumed known (the standard
    "label-known" attack setting; see Geiping §3.2); reconstruction is
    by L-BFGS optimisation of the input pixels to match the observed
    gradient signature.

    For image-shaped inputs set `image_shape=(H, W, C)` and `tv_weight > 0`
    to enable anisotropic Total-Variation regularisation (Geiping §3.1) —
    this is the cheapest move toward the GradInversion-class state of the
    art (Yin et al., CVPR 2021) without taking on a full image prior.

    Single-sample batch + 2-layer MLP is the cleanest demonstration
    setting; larger batches and CNN architectures demand stronger priors
    (Yin 2021; Hatamizadeh 2023).
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    input_dim = target_image.size
    x0 = rng.normal(0.5, 0.1, size=(input_dim,)).astype(np.float64)
    x0 = np.clip(x0, 0.0, 1.0)
    y = np.array([target_label], dtype=np.int64)

    target_grad_arr = _flatten(target_grad)
    target_norm = float(np.linalg.norm(target_grad_arr))

    def objective(x_flat: np.ndarray) -> float:
        X = x_flat.reshape(1, input_dim).astype(np.float32)
        _, g = model.loss_and_grad(X, y)
        g_arr = _flatten(g)
        cos = 1.0 - float(np.dot(g_arr, target_grad_arr) /
                          (np.linalg.norm(g_arr) * target_norm + 1e-12))
        box = float(((np.clip(x_flat, 0, 1) - x_flat) ** 2).sum())
        tv = _tv_anisotropic(x_flat, image_shape) if tv_weight > 0 else 0.0
        return cos + 0.01 * box + tv_weight * tv

    result = optimize.minimize(
        objective, x0, method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * input_dim,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )
    recon = result.x.reshape(1, input_dim).astype(np.float32)

    target_flat = target_image.reshape(1, input_dim).astype(np.float32)
    pixel_l2 = float(np.linalg.norm(recon - target_flat))
    mse = float(((recon - target_flat) ** 2).mean()) + 1e-12
    psnr = 10.0 * np.log10(1.0 / mse)

    # Recompute cosine on the final point against the *actual* gradient
    _, final_g = model.loss_and_grad(recon, y)
    cos = _cosine_distance(final_g, target_grad)

    return GradInversionResult(
        target_image=target_flat,
        target_label=target_label,
        reconstructed=recon,
        cosine_distance=cos,
        pixel_l2=pixel_l2,
        pixel_psnr_db=float(psnr),
    )


# ─── Membership inference against the FL-released model ─────────────────────


@dataclass
class MIAResult:
    n_targets_swept: int
    worst_record_tpr_at_fpr: float       # worst-record TPR at the lowest FPR
    median_tpr_at_fpr: float
    fpr_target: float
    worst_target_pod: int
    worst_target_index_in_pod: int
    per_target: list[dict]


def _shadow_features(model: MLP, X: np.ndarray) -> np.ndarray:
    """Per-sample feature vector for the meta-classifier: full softmax probabilities."""
    return model.predict_proba(X)


def per_record_mia(
    *,
    federated_model: MLP,
    pod_data: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_targets: int = 12,
    n_shadow_runs: int = 32,
    shadow_lr: float = 0.05,
    shadow_steps: int = 30,
    shadow_batch_size: int = 32,
    fpr_targets: tuple[float, ...] = (0.001, 0.01),
    meta_classifier: Literal["logistic", "lira"] = "logistic",
    master_seed: int = 20260525,
    namespace: str = "attacks.mia.fl_released.v1",
) -> MIAResult:
    """Per-record MIA against the FL-released model.

    Methodology follows Carlini *et al.* (IEEE S&P 2022) for the
    *per-record sweep* + TPR-at-low-FPR metric, adapted to the FL
    setting by training shadow models on subsamples of the pod
    federation rather than i.i.d. samples of a single dataset.

    Two meta-classifier modes:
        - `"logistic"` (default): empirical OUT-quantile threshold; robust
          on tiny shadow counts but loose against state-of-the-art attacks.
        - `"lira"`: the Likelihood Ratio Attack of Carlini et al. (S&P
          2022, Algorithm 1). For each target, fit per-target Gaussians to
          the IN-logit and OUT-logit shadow distributions, set the
          threshold τ such that P(N(μ_OUT, σ_OUT²) ≥ τ) = FPR, and report
          the corresponding TPR P(N(μ_IN, σ_IN²) ≥ τ). LiRA is strictly
          tighter than the logistic baseline and is the current SOTA per-
          record MIA in the FL setting [Carlini 2022; Nasr et al. 2023].

    The shadow-logit transformation is `logit(p) = log(p / (1 - p))`
    applied to confidence in the true class, per Carlini §4.

    Tutorial-scale defaults give a noisy but indicative worst-record
    estimate in seconds. Paper-scale is `n_targets=60, n_shadow_runs=128`
    with `meta_classifier="lira"`.
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))

    # Build a flat pool of (pod_id, idx_in_pod, X, y) for sampling targets
    pool: list[tuple[int, int, np.ndarray, int]] = []
    for pod_id, (Xp, yp) in enumerate(pod_data):
        for i, (xi, yi) in enumerate(zip(Xp, yp)):
            pool.append((pod_id, i, xi, int(yi)))

    target_choices = rng.choice(len(pool), size=min(n_targets, len(pool)), replace=False)
    targets = [pool[i] for i in target_choices]

    # The released model's prediction on each target (the observable signal)
    # — the meta-classifier asks: does the released model's output on x
    # look "member-like" or "non-member-like"?
    all_X = np.stack([x for _, _, x, _ in pool])
    all_y = np.array([y for _, _, _, y in pool], dtype=np.int64)
    n_pool = len(pool)

    per_target: list[dict] = []
    fpr_lowest = min(fpr_targets)

    for ti, (pod_id, in_pod_idx, x_target, y_target) in enumerate(targets):
        x_target = x_target.reshape(1, -1).astype(np.float32)

        # Feature vector on the target by the released model
        released_probs = federated_model.predict_proba(x_target).ravel()
        released_signal = float(released_probs[y_target])  # confidence in the true class

        # Train shadow models with target IN / OUT
        shadow_signals_in: list[float] = []
        shadow_signals_out: list[float] = []
        for s in range(n_shadow_runs):
            include_target = rng.random() < 0.5
            # Sample a 30% slice of the pool as the shadow training set
            slice_size = max(64, n_pool // 3)
            slice_idx = rng.choice(n_pool, size=slice_size, replace=False)
            X_shadow = all_X[slice_idx]
            y_shadow = all_y[slice_idx]
            if include_target:
                X_shadow = np.concatenate([X_shadow, x_target])
                y_shadow = np.concatenate([y_shadow, [y_target]])

            shadow = MLP(input_dim=x_target.shape[1], hidden_dim=64,
                         n_classes=int(all_y.max() + 1))
            shadow.init_from_seed(master_seed + s, f"{namespace}.shadow.{ti}.{s}")
            for _ in range(shadow_steps):
                bs = min(shadow_batch_size, len(X_shadow))
                bi = rng.choice(len(X_shadow), size=bs, replace=False)
                _, g = shadow.loss_and_grad(X_shadow[bi], y_shadow[bi])
                for k in shadow.params:
                    getattr(shadow, k)[...] -= shadow_lr * g[k]

            probs = shadow.predict_proba(x_target).ravel()
            signal = float(probs[y_target])
            (shadow_signals_in if include_target else shadow_signals_out).append(signal)

        if not shadow_signals_in or not shadow_signals_out:
            continue

        in_arr = np.array(shadow_signals_in)
        out_arr = np.array(shadow_signals_out)

        if meta_classifier == "lira":
            # Carlini et al. (S&P 2022) Algorithm 1: transform confidences to
            # logits and fit per-target Gaussians to IN / OUT shadow logits.
            # Closed-form: τ = μ_OUT + σ_OUT · Φ⁻¹(1 − FPR);
            #             TPR = 1 − Φ((τ − μ_IN) / σ_IN).
            eps = 1e-7
            in_logits = np.log(in_arr.clip(eps, 1 - eps) / (1 - in_arr).clip(eps, 1 - eps))
            out_logits = np.log(out_arr.clip(eps, 1 - eps) / (1 - out_arr).clip(eps, 1 - eps))
            mu_in, sd_in = float(in_logits.mean()), float(in_logits.std() + 1e-9)
            mu_out, sd_out = float(out_logits.mean()), float(out_logits.std() + 1e-9)
            thr_logit = mu_out + sd_out * _norm.ppf(1.0 - fpr_lowest)
            tpr_estimate = float(1.0 - _norm.cdf((thr_logit - mu_in) / sd_in))
            # Released-model "hit" — logit-transform the released signal too
            rsig = max(min(released_signal, 1 - eps), eps)
            released_logit = float(np.log(rsig / (1 - rsig)))
            released_hit = int(released_logit >= thr_logit)
            thr = float(thr_logit)
        else:
            # "logistic" baseline: empirical OUT-quantile threshold.
            thr = float(np.quantile(out_arr, 1 - fpr_lowest))
            tpr_estimate = float((in_arr >= thr).mean())
            released_hit = int(released_signal >= thr)

        per_target.append({
            "target_index": ti,
            "pod_id": pod_id,
            "in_pod_index": in_pod_idx,
            "released_signal": released_signal,
            "shadow_in_mean": float(in_arr.mean()),
            "shadow_out_mean": float(out_arr.mean()),
            "threshold_at_fpr": thr,
            "tpr_at_fpr": tpr_estimate,
            "released_model_hit": released_hit,
            "meta_classifier": meta_classifier,
        })

    if not per_target:
        return MIAResult(0, 0.0, 0.0, fpr_lowest, -1, -1, [])

    tprs = np.array([t["tpr_at_fpr"] for t in per_target])
    worst_idx = int(np.argmax(tprs))
    worst = per_target[worst_idx]

    return MIAResult(
        n_targets_swept=len(per_target),
        worst_record_tpr_at_fpr=float(tprs.max()),
        median_tpr_at_fpr=float(np.median(tprs)),
        fpr_target=fpr_lowest,
        worst_target_pod=worst["pod_id"],
        worst_target_index_in_pod=worst["in_pod_index"],
        per_target=per_target,
    )
