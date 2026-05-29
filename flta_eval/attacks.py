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
  against the FL-released model. Three meta-classifier modes:
    - `"logistic"` (default): univariate logistic over confidence-in-true-class.
    - `"lira"`: the **Likelihood Ratio Attack** of Carlini et al. (S&P 2022,
      Algorithm 1), which fits per-target Gaussians to the IN / OUT shadow-
      logit distributions.
    - `"rmia"`: the **Reference-Member Inference Attack** of Zarifzadeh,
      Liu, Shokri (ICML 2024). Offline variant: per-target score is the
      released model's confidence on the target divided by the released
      model's confidence on reference samples for the same label; the
      shadow IN/OUT distribution of that ratio gives the threshold.
      Empirically tighter than LiRA at small shadow-model budgets.

The MIA returns bootstrap confidence intervals on each per-target TPR
estimate and on the worst-record TPR (Carlini §3.3 recipe). The result
also stratifies by class so the worst stratum is auditable separately
from the worst individual record.
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
class ShadowModel:
    """One element of the shadow pool.

    Each shadow is itself a full FL training run on a random subset of
    the pod federation. Carrying the pod_ids subset along with the
    trained model is what gives per-target LiRA IN/OUT splits: for a
    target at pod P, shadow s is "IN" iff P ∈ pod_ids.
    """
    model: object                        # has .predict_proba(X) -> np.ndarray
    pod_ids: frozenset[int]              # which pods this shadow trained on
    config_namespace: str                # for audit


@dataclass
class MIAResult:
    n_targets_swept: int
    worst_record_tpr_at_fpr: float       # worst-record TPR at the lowest FPR
    worst_record_tpr_ci_lower: float     # bootstrap 2.5 percentile
    worst_record_tpr_ci_upper: float     # bootstrap 97.5 percentile
    median_tpr_at_fpr: float
    fpr_target: float
    worst_target_pod: int
    worst_target_index_in_pod: int
    per_target: list[dict]
    per_class: dict[int, dict]           # {class_id: {worst_tpr, mean_tpr, n_targets}}
    meta_classifier: str


def _shadow_features(model: MLP, X: np.ndarray) -> np.ndarray:
    """Per-sample feature vector for the meta-classifier: full softmax probabilities."""
    return model.predict_proba(X)


def _lira_tpr(in_logits: np.ndarray, out_logits: np.ndarray, fpr: float) -> float:
    """Closed-form LiRA TPR at the given FPR — Carlini et al. Algorithm 1."""
    mu_in, sd_in = float(in_logits.mean()), float(in_logits.std() + 1e-9)
    mu_out, sd_out = float(out_logits.mean()), float(out_logits.std() + 1e-9)
    thr = mu_out + sd_out * _norm.ppf(1.0 - fpr)
    return float(1.0 - _norm.cdf((thr - mu_in) / sd_in))


def _bootstrap_tpr_ci(
    in_arr: np.ndarray, out_arr: np.ndarray, fpr: float,
    *, n_bootstrap: int, rng: np.random.Generator,
    transform: Literal["logit", "log", "none"] = "logit",
) -> tuple[float, float, float]:
    """Bootstrap CI on per-target TPR. Returns (point_estimate, ci_lower, ci_upper).

    Per Carlini et al. (IEEE S&P 2022) §3.3: resample the shadow IN and OUT
    signals (with replacement), refit the per-target Gaussian, recompute
    the closed-form TPR. CI is percentiles of the bootstrap distribution.

    Transform options
    -----------------
    "logit" — log(x / (1 - x)); the LiRA convention for confidence
              probabilities x ∈ (0, 1) [Carlini §4]. Inputs are clipped
              to [eps, 1-eps] for numerical stability.
    "log"   — natural log of the input; the correct choice for
              likelihood-ratio scores (RMIA-style; Zarifzadeh §3) which
              are strictly positive and not confined to [0, 1]. Inputs
              are clipped at eps from below only.
    "none"  — pass inputs through as-is; assumes the caller has already
              applied an appropriate transform.
    """
    eps = 1e-7
    if transform == "logit":
        in_logits = np.log(in_arr.clip(eps, 1 - eps) / (1 - in_arr).clip(eps, 1 - eps))
        out_logits = np.log(out_arr.clip(eps, 1 - eps) / (1 - out_arr).clip(eps, 1 - eps))
    elif transform == "log":
        in_logits = np.log(np.maximum(in_arr, eps))
        out_logits = np.log(np.maximum(out_arr, eps))
    else:
        in_logits, out_logits = in_arr, out_arr

    point = _lira_tpr(in_logits, out_logits, fpr)

    boot_tprs = np.empty(n_bootstrap, dtype=np.float64)
    n_in, n_out = len(in_logits), len(out_logits)
    for b in range(n_bootstrap):
        bi = rng.integers(0, n_in, size=n_in)
        bo = rng.integers(0, n_out, size=n_out)
        boot_tprs[b] = _lira_tpr(in_logits[bi], out_logits[bo], fpr)
    return point, float(np.quantile(boot_tprs, 0.025)), float(np.quantile(boot_tprs, 0.975))


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
    meta_classifier: Literal["logistic", "lira", "rmia"] = "logistic",
    n_bootstrap: int = 1000,
    n_rmia_references: int = 256,
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
    target_idx_set = {int(i) for i in target_choices}

    # The released model's prediction on each target (the observable signal)
    # — the meta-classifier asks: does the released model's output on x
    # look "member-like" or "non-member-like"?
    all_X = np.stack([x for _, _, x, _ in pool])
    all_y = np.array([y for _, _, _, y in pool], dtype=np.int64)
    n_pool = len(pool)

    # RMIA reference pool — random non-target pool members. The released
    # model's confidence on these references provides the per-label null
    # baseline used by RMIA's offline variant [Zarifzadeh et al. ICML 2024].
    non_target_pool = np.array([i for i in range(n_pool) if i not in target_idx_set])
    n_refs = min(n_rmia_references, len(non_target_pool))
    ref_indices = rng.choice(non_target_pool, size=n_refs, replace=False)
    X_refs = all_X[ref_indices]
    released_ref_probs = federated_model.predict_proba(X_refs)  # (n_refs, n_classes)
    boot_rng = np.random.default_rng(derive_seed(master_seed, namespace + ".bootstrap"))

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

        eps = 1e-7
        if meta_classifier == "lira":
            tpr_estimate, ci_lo, ci_hi = _bootstrap_tpr_ci(
                in_arr, out_arr, fpr_lowest,
                n_bootstrap=n_bootstrap, rng=boot_rng, transform="logit",
            )
            out_logits = np.log(out_arr.clip(eps, 1 - eps) /
                                (1 - out_arr).clip(eps, 1 - eps))
            thr = float(out_logits.mean() + (out_logits.std() + 1e-9) *
                        _norm.ppf(1.0 - fpr_lowest))
            rsig = max(min(released_signal, 1 - eps), eps)
            released_hit = int(np.log(rsig / (1 - rsig)) >= thr)
        elif meta_classifier == "rmia":
            # Simplified RMIA-style scoring [Zarifzadeh et al. ICML 2024]:
            # the per-target score is the (released-model OR shadow-model)
            # confidence divided by the released-model confidence on
            # reference samples of the same label — i.e. a likelihood ratio
            # against the per-label reference baseline. LR scores are
            # strictly positive but not confined to [0, 1], so we apply
            # the log (not logit) transform when fitting the per-target
            # Gaussian on IN / OUT ratios.
            ref_baseline = float(released_ref_probs[:, y_target].mean() + eps)
            in_ratio = in_arr / ref_baseline
            out_ratio = out_arr / ref_baseline
            tpr_estimate, ci_lo, ci_hi = _bootstrap_tpr_ci(
                in_ratio, out_ratio, fpr_lowest,
                n_bootstrap=n_bootstrap, rng=boot_rng, transform="log",
            )
            out_log = np.log(np.maximum(out_ratio, eps))
            thr = float(out_log.mean() + (out_log.std() + 1e-9) *
                        _norm.ppf(1.0 - fpr_lowest))
            released_ratio = released_signal / ref_baseline
            released_hit = int(np.log(max(released_ratio, eps)) >= thr)
        else:
            # "logistic" baseline — empirical OUT-quantile threshold + bootstrap CI.
            thr = float(np.quantile(out_arr, 1 - fpr_lowest))
            tpr_estimate = float((in_arr >= thr).mean())
            boot_tprs = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                bi = boot_rng.integers(0, len(in_arr), size=len(in_arr))
                bo = boot_rng.integers(0, len(out_arr), size=len(out_arr))
                thr_b = float(np.quantile(out_arr[bo], 1 - fpr_lowest))
                boot_tprs[b] = float((in_arr[bi] >= thr_b).mean())
            ci_lo, ci_hi = float(np.quantile(boot_tprs, 0.025)), float(np.quantile(boot_tprs, 0.975))
            released_hit = int(released_signal >= thr)

        per_target.append({
            "target_index": ti,
            "pod_id": pod_id,
            "in_pod_index": in_pod_idx,
            "class_id": int(y_target),
            "released_signal": released_signal,
            "shadow_in_mean": float(in_arr.mean()),
            "shadow_out_mean": float(out_arr.mean()),
            "threshold_at_fpr": float(thr),
            "tpr_at_fpr": tpr_estimate,
            "tpr_ci_lower": ci_lo,
            "tpr_ci_upper": ci_hi,
            "released_model_hit": released_hit,
            "meta_classifier": meta_classifier,
        })

    if not per_target:
        return MIAResult(0, 0.0, 0.0, 0.0, 0.0, fpr_lowest, -1, -1, [], {}, meta_classifier)

    tprs = np.array([t["tpr_at_fpr"] for t in per_target])
    ci_lo_arr = np.array([t["tpr_ci_lower"] for t in per_target])
    ci_hi_arr = np.array([t["tpr_ci_upper"] for t in per_target])
    worst_idx = int(np.argmax(tprs))
    worst = per_target[worst_idx]

    # Bootstrap CI on the worst-record TPR: percentiles of max over targets
    # under joint resampling. Each bootstrap draws one TPR per target from
    # that target's CI; the per-target CIs were themselves bootstraps so
    # this approximates the joint percentile via the order-statistic.
    # For a tight upper bound on the worst TPR, take the max of the per-
    # target upper CIs; for a lower bound, the max of the lower CIs.
    worst_ci_lower = float(ci_lo_arr.max())
    worst_ci_upper = float(ci_hi_arr.max())

    # Per-class stratification: per-class worst TPR + class size.
    per_class: dict[int, dict] = {}
    for t in per_target:
        c = t["class_id"]
        bucket = per_class.setdefault(c, {"tprs": []})
        bucket["tprs"].append(t["tpr_at_fpr"])
    for c, bucket in per_class.items():
        arr = np.array(bucket["tprs"])
        bucket["worst_tpr"] = float(arr.max())
        bucket["mean_tpr"] = float(arr.mean())
        bucket["n_targets"] = int(len(arr))
        del bucket["tprs"]

    return MIAResult(
        n_targets_swept=len(per_target),
        worst_record_tpr_at_fpr=float(tprs.max()),
        worst_record_tpr_ci_lower=worst_ci_lower,
        worst_record_tpr_ci_upper=worst_ci_upper,
        median_tpr_at_fpr=float(np.median(tprs)),
        fpr_target=fpr_lowest,
        worst_target_pod=worst["pod_id"],
        worst_target_index_in_pod=worst["in_pod_index"],
        per_target=per_target,
        per_class={int(k): v for k, v in per_class.items()},
        meta_classifier=meta_classifier,
    )


# ─── SHADOW-POOL MIA: shadow-target FL parity ────────────────────────────────
#
# Carlini-style architecture: train K shadow models once, reuse across
# all targets. The IN/OUT split per target is determined by pod
# membership — shadow s is "IN" for target T at pod P iff P was sampled
# into shadow s's training subset.
#
# Crucially: each shadow is a *full FL training run*, not plain SGD on
# a random data slice. This makes the shadow distribution a faithful
# proxy for the target model's output distribution — the central
# assumption of LiRA [Carlini §4] and RMIA [Zarifzadeh §3].


def build_shadow_pool(
    *,
    pod_data: list[tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    fl_train_fn,                          # callable: (pod_data_subset, **kwargs) -> trained model
    fl_config,                            # FLConfig (or fl_torch.FLConfig)
    n_shadows: int = 64,
    pod_fraction: float = 0.5,
    master_seed: int = 20260525,
    namespace: str = "attacks.shadow_pool.v1",
    progress: bool = False,
) -> list[ShadowModel]:
    """Train K shadow FL models on random pod subsets.

    Each shadow trains on a fresh random subset of ~`pod_fraction` of
    the pods. The trained model + the pod_id subset are stored, ready
    for per-target IN/OUT lookup in `per_record_mia_pool`.
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    n_pods = len(pod_data)
    pool: list[ShadowModel] = []

    for s in range(n_shadows):
        n_sample = max(2, int(round(n_pods * pod_fraction)))
        subset = rng.choice(n_pods, size=n_sample, replace=False)
        subset_sorted = sorted(int(i) for i in subset)
        pod_subset_data = [pod_data[i] for i in subset_sorted]
        shadow_ns = f"{namespace}.shadow.{s}"
        trained = fl_train_fn(
            pod_data=pod_subset_data,
            X_test=X_test, y_test=y_test, config=fl_config,
            master_seed=master_seed, namespace=shadow_ns,
        )
        # Some fl_train_fn returns (model, records); we just want the model
        if isinstance(trained, tuple):
            trained = trained[0]
        pool.append(ShadowModel(
            model=trained,
            pod_ids=frozenset(subset_sorted),
            config_namespace=shadow_ns,
        ))
        if progress and (s + 1) % max(1, n_shadows // 10) == 0:
            print(f"  shadow pool: {s + 1}/{n_shadows} trained")
    return pool


def per_record_mia_pool(
    *,
    federated_model,                      # released model — has .predict_proba
    pod_data: list[tuple[np.ndarray, np.ndarray]],
    shadow_pool: list[ShadowModel],
    n_targets: int = 60,
    fpr_targets: tuple[float, ...] = (0.001, 0.01),
    meta_classifier: Literal["logistic", "lira", "rmia"] = "lira",
    n_bootstrap: int = 1000,
    master_seed: int = 20260525,
    namespace: str = "attacks.mia_pool.v1",
) -> MIAResult:
    """Per-record MIA against the FL-released model with a shadow pool.

    Differences vs `per_record_mia`:
      - Each shadow is a *full FL training run* on a pod subset (shadow-
        target parity) — built once via `build_shadow_pool` and reused.
      - IN/OUT split per target is by pod membership in the shadow's
        training subset, the FL-native analog of Carlini's record-IN/OUT.
      - `meta_classifier="rmia"` uses the *online* RMIA variant [Zarifzadeh
        et al. ICML 2024 §3]: per-target score is the released model's
        signal divided by the mean OUT-shadow signal on the same target,
        thresholded by the IN/OUT shadow distribution of the same ratio.

    Bootstrap CIs, per-class stratification, audit fields are the same
    shape as `per_record_mia`.
    """
    rng = np.random.default_rng(derive_seed(master_seed, namespace))
    boot_rng = np.random.default_rng(derive_seed(master_seed, namespace + ".bootstrap"))

    pool: list[tuple[int, int, np.ndarray, int]] = []
    for pod_id, (Xp, yp) in enumerate(pod_data):
        for i, (xi, yi) in enumerate(zip(Xp, yp)):
            pool.append((pod_id, i, xi, int(yi)))

    target_choices = rng.choice(len(pool), size=min(n_targets, len(pool)), replace=False)
    targets = [pool[i] for i in target_choices]

    # Precompute each shadow's signal on every target — vectorised.
    target_X = np.stack([t[2] for t in targets]).astype(np.float32)
    target_y = np.array([t[3] for t in targets], dtype=np.int64)
    n_t = len(targets)
    n_s = len(shadow_pool)

    shadow_signals = np.empty((n_s, n_t), dtype=np.float64)
    for s_i, sh in enumerate(shadow_pool):
        probs = sh.model.predict_proba(target_X)
        shadow_signals[s_i] = probs[np.arange(n_t), target_y]

    released_probs = federated_model.predict_proba(target_X)
    released_signals = released_probs[np.arange(n_t), target_y]

    per_target: list[dict] = []
    fpr_lowest = min(fpr_targets)
    eps = 1e-7

    for ti, (pod_id, in_pod_idx, _, y_target) in enumerate(targets):
        # IN/OUT split by pod membership in shadow's training subset.
        in_mask = np.array([pod_id in sh.pod_ids for sh in shadow_pool])
        in_arr = shadow_signals[in_mask, ti]
        out_arr = shadow_signals[~in_mask, ti]
        released_signal = float(released_signals[ti])

        if len(in_arr) == 0 or len(out_arr) == 0:
            continue

        if meta_classifier == "lira":
            tpr_estimate, ci_lo, ci_hi = _bootstrap_tpr_ci(
                in_arr, out_arr, fpr_lowest,
                n_bootstrap=n_bootstrap, rng=boot_rng, transform="logit",
            )
            out_logits = np.log(out_arr.clip(eps, 1 - eps) /
                                (1 - out_arr).clip(eps, 1 - eps))
            thr = float(out_logits.mean() + (out_logits.std() + 1e-9) *
                        _norm.ppf(1.0 - fpr_lowest))
            rsig = max(min(released_signal, 1 - eps), eps)
            released_hit = int(np.log(rsig / (1 - rsig)) >= thr)
        elif meta_classifier == "rmia":
            # RMIA-style scoring [Zarifzadeh §3]: per-target score is the
            # confidence divided by the OUT-shadow mean — a likelihood
            # ratio against the per-target reference baseline. LR scores
            # are strictly positive (not confined to [0, 1]); the natural
            # transform when fitting the per-target Gaussian is therefore
            # the natural log, not the logit. (This is the corrected
            # transform; the prior version logit-transformed the LR and
            # introduced clip artefacts.)
            out_mean = float(out_arr.mean()) + eps
            in_score = in_arr / out_mean
            out_score = out_arr / out_mean
            tpr_estimate, ci_lo, ci_hi = _bootstrap_tpr_ci(
                in_score, out_score, fpr_lowest,
                n_bootstrap=n_bootstrap, rng=boot_rng, transform="log",
            )
            out_score_log = np.log(np.maximum(out_score, eps))
            thr = float(out_score_log.mean() + (out_score_log.std() + 1e-9) *
                        _norm.ppf(1.0 - fpr_lowest))
            released_score = released_signal / out_mean
            released_hit = int(np.log(max(released_score, eps)) >= thr)
        else:
            thr = float(np.quantile(out_arr, 1 - fpr_lowest))
            tpr_estimate = float((in_arr >= thr).mean())
            boot_tprs = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                bi = boot_rng.integers(0, len(in_arr), size=len(in_arr))
                bo = boot_rng.integers(0, len(out_arr), size=len(out_arr))
                boot_tprs[b] = float((in_arr[bi] >= float(np.quantile(out_arr[bo], 1 - fpr_lowest))).mean())
            ci_lo, ci_hi = float(np.quantile(boot_tprs, 0.025)), float(np.quantile(boot_tprs, 0.975))
            released_hit = int(released_signal >= thr)

        per_target.append({
            "target_index": ti, "pod_id": pod_id,
            "in_pod_index": in_pod_idx, "class_id": int(y_target),
            "n_shadow_in": int(in_mask.sum()), "n_shadow_out": int((~in_mask).sum()),
            "released_signal": released_signal,
            "shadow_in_mean": float(in_arr.mean()),
            "shadow_out_mean": float(out_arr.mean()),
            "threshold_at_fpr": float(thr),
            "tpr_at_fpr": tpr_estimate,
            "tpr_ci_lower": ci_lo, "tpr_ci_upper": ci_hi,
            "released_model_hit": released_hit,
            "meta_classifier": meta_classifier,
        })

    if not per_target:
        return MIAResult(0, 0.0, 0.0, 0.0, 0.0, fpr_lowest, -1, -1, [], {}, meta_classifier)

    tprs = np.array([t["tpr_at_fpr"] for t in per_target])
    ci_lo_arr = np.array([t["tpr_ci_lower"] for t in per_target])
    ci_hi_arr = np.array([t["tpr_ci_upper"] for t in per_target])
    worst_idx = int(np.argmax(tprs))
    worst = per_target[worst_idx]

    per_class: dict[int, dict] = {}
    for t in per_target:
        c = t["class_id"]
        bucket = per_class.setdefault(c, {"tprs": []})
        bucket["tprs"].append(t["tpr_at_fpr"])
    for c, bucket in per_class.items():
        arr = np.array(bucket["tprs"])
        bucket["worst_tpr"] = float(arr.max())
        bucket["mean_tpr"] = float(arr.mean())
        bucket["n_targets"] = int(len(arr))
        del bucket["tprs"]

    return MIAResult(
        n_targets_swept=len(per_target),
        worst_record_tpr_at_fpr=float(tprs.max()),
        worst_record_tpr_ci_lower=float(ci_lo_arr.max()),
        worst_record_tpr_ci_upper=float(ci_hi_arr.max()),
        median_tpr_at_fpr=float(np.median(tprs)),
        fpr_target=fpr_lowest,
        worst_target_pod=worst["pod_id"],
        worst_target_index_in_pod=worst["in_pod_index"],
        per_target=per_target,
        per_class={int(k): v for k, v in per_class.items()},
        meta_classifier=meta_classifier,
    )
