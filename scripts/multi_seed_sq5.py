"""SQ-5 multi-seed ordering stability.

The paper's central empirical claim is that scalar (ε, AUROC) ordering
and empirical worst-record TPR ordering *disagree* on three FL
configurations (Loose, Default, Tight). A reviewer will ask whether
that disagreement is robust to seed or a one-seed coincidence. This
script runs the SQ-5 pipeline under N seeds and records ordering
stability.

Outputs:
    results/sq5/multi_seed__N{N_SEEDS}__<timestamp>.json

The result record carries: per-seed (ε, AUROC, worst_TPR) per
configuration; per-seed scalar and empirical orderings; the fraction
of seeds on which the orderings disagree; and per-configuration mean +
standard error of worst-record TPR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flta_eval import attacks, datasets, fl
from flta_eval.audit import config_hash, git_commit, now_utc

CONFIGS = [
    ("Loose",   fl.FLConfig(n_rounds=15, client_lr=0.1, client_batch_size=32,
                            clip_norm=10.0, noise_multiplier=0.3)),
    ("Default", fl.FLConfig(n_rounds=15, client_lr=0.1, client_batch_size=32,
                            clip_norm=1.0,  noise_multiplier=1.1)),
    ("Tight",   fl.FLConfig(n_rounds=15, client_lr=0.1, client_batch_size=32,
                            clip_norm=1.0,  noise_multiplier=2.5)),
]


def _ordering(items: list[tuple[str, float]]) -> list[str]:
    """Ascending sort — first element is the lowest value (scalar most-private)."""
    return [lab for lab, _ in sorted(items, key=lambda t: t[1])]


def run_one_seed(seed: int, n_targets: int, n_shadow: int,
                 n_bootstrap: int = 500) -> dict:
    ds = datasets.load_bloodmnist(REPO_ROOT / "data")
    manifest = datasets.load_partition(REPO_ROOT / "data")
    pod_data = [
        (ds["X_train"][np.array(manifest["pod_indices"][i])],
         ds["y_train"][np.array(manifest["pod_indices"][i])])
        for i in range(manifest["n_pods"])
    ]

    rows = []
    for label, cfg in CONFIGS:
        model = fl.MLP(input_dim=ds["input_dim"], hidden_dim=64, n_classes=ds["n_classes"])
        model.init_from_seed(seed, f"fl.scalarvs.{label}")
        t0 = time.time()
        trained, _ = fl.federated_train(
            model=model, pod_data=pod_data,
            X_test=ds["X_test"], y_test=ds["y_test"], config=cfg,
            master_seed=seed, namespace=f"fl.scalarvs.{label}.train",
        )
        train_s = time.time() - t0
        acc = trained.accuracy(ds["X_test"], ds["y_test"])
        eps = fl.rdp_epsilon(noise_multiplier=cfg.noise_multiplier,
                             n_rounds=cfg.n_rounds, sample_rate=1.0, delta=1e-5)

        t0 = time.time()
        mia = attacks.per_record_mia(
            federated_model=trained, pod_data=pod_data,
            X_test=ds["X_test"], y_test=ds["y_test"],
            n_targets=n_targets, n_shadow_runs=n_shadow,
            shadow_steps=15, shadow_batch_size=32,
            fpr_targets=(0.001,),
            meta_classifier="lira",
            n_bootstrap=n_bootstrap,
            master_seed=seed, namespace=f"attacks.mia.scalarvs.{label}",
        )
        mia_s = time.time() - t0
        rows.append({
            "label": label, "sigma": cfg.noise_multiplier, "clip_norm": cfg.clip_norm,
            "rdp_epsilon": float(eps), "test_accuracy": float(acc),
            "worst_tpr": float(mia.worst_record_tpr_at_fpr),
            "worst_tpr_ci_lower": float(mia.worst_record_tpr_ci_lower),
            "worst_tpr_ci_upper": float(mia.worst_record_tpr_ci_upper),
            "median_tpr": float(mia.median_tpr_at_fpr),
            "train_s": round(train_s, 2), "mia_s": round(mia_s, 2),
        })

    scalar = _ordering([(r["label"], r["rdp_epsilon"]) for r in rows])
    empirical = _ordering([(r["label"], r["worst_tpr"]) for r in rows])
    return {
        "seed": seed,
        "rows": rows,
        "scalar_ordering": scalar,
        "empirical_ordering": empirical,
        "orderings_agree": scalar == empirical,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=20260525)
    ap.add_argument("--paper-scale", action="store_true",
                    help="Use 40 targets × 64 shadow models per config (slower).")
    args = ap.parse_args()

    n_targets, n_shadow = (40, 64) if args.paper_scale else (6, 12)

    print(f"Running SQ-5 across {args.seeds} seeds at "
          f"({'paper' if args.paper_scale else 'tutorial'} scale: "
          f"n_targets={n_targets}, n_shadow={n_shadow}).")

    per_seed = []
    t_start = time.time()
    for i in range(args.seeds):
        seed = args.base_seed + i
        t0 = time.time()
        out = run_one_seed(seed, n_targets, n_shadow)
        elapsed = time.time() - t0
        print(f"  seed {seed}: scalar={out['scalar_ordering']} "
              f"empirical={out['empirical_ordering']} "
              f"agree={out['orderings_agree']} ({elapsed:.1f}s)")
        per_seed.append(out)
    total_elapsed = time.time() - t_start

    # Aggregates per configuration.
    by_label: dict[str, list[float]] = {lab: [] for lab, _ in CONFIGS}
    for o in per_seed:
        for r in o["rows"]:
            by_label[r["label"]].append(r["worst_tpr"])
    aggregates = {}
    for lab, vals in by_label.items():
        arr = np.array(vals)
        aggregates[lab] = {
            "mean_worst_tpr": float(arr.mean()),
            "std_worst_tpr": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "sem_worst_tpr": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "min_worst_tpr": float(arr.min()),
            "max_worst_tpr": float(arr.max()),
        }

    # Ordering stability.
    n_agree = sum(1 for o in per_seed if o["orderings_agree"])
    canonical_empirical = ["Loose", "Tight", "Default"]
    n_canonical = sum(1 for o in per_seed if o["empirical_ordering"] == canonical_empirical)

    config = {
        "n_seeds": args.seeds, "base_seed": args.base_seed,
        "n_targets_per_config": n_targets, "n_shadow_runs_per_config": n_shadow,
        "paper_scale": args.paper_scale,
        "configs": [{"label": lab, "sigma": c.noise_multiplier,
                     "clip_norm": c.clip_norm, "rounds": c.n_rounds}
                    for lab, c in CONFIGS],
    }
    config["config_hash_sha256"] = config_hash(config)

    record = {
        "schema_version": "1.0",
        "attack": "scalar_vs_stepwise.multi_seed",
        "variant": "SQ-5-multi",
        "threat_profile": "R",
        "dataset": {"name": "bloodmnist"},
        "config": config,
        "seed_namespace": "sq5.multi_seed.bloodmnist.v1",
        "harness_commit": git_commit(),
        "result": {
            "per_seed": per_seed,
            "per_config_aggregates": aggregates,
            "scalar_ordering_canonical": ["Tight", "Default", "Loose"],
            "empirical_ordering_canonical": canonical_empirical,
            "n_seeds_orderings_disagree": args.seeds - n_agree,
            "n_seeds_matching_canonical_empirical": n_canonical,
            "ordering_stability_fraction": n_canonical / args.seeds,
            "total_elapsed_seconds": round(total_elapsed, 2),
        },
        "timestamp_utc": now_utc(),
    }

    out_dir = REPO_ROOT / "results" / "sq5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["timestamp_utc"].replace(":", "-")
    out = out_dir / f"multi_seed__N{args.seeds}__{stamp}.json"
    out.write_text(json.dumps(record, indent=2))

    print()
    print(f"== Multi-seed SQ-5 summary (N={args.seeds}) ==")
    print(f"  Orderings disagree on {args.seeds - n_agree}/{args.seeds} seeds")
    print(f"  Canonical empirical ordering (Loose ≺ Tight ≺ Default) held on "
          f"{n_canonical}/{args.seeds} seeds")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Per-configuration worst-TPR (mean ± SEM):")
    for lab in ["Loose", "Default", "Tight"]:
        a = aggregates[lab]
        print(f"    {lab:7s}  {a['mean_worst_tpr']:.4f} ± {a['sem_worst_tpr']:.4f}  "
              f"[min {a['min_worst_tpr']:.4f}, max {a['max_worst_tpr']:.4f}]")
    print(f"  Record: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
