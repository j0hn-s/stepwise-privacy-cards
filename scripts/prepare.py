#!/usr/bin/env python3
"""Prepare the evaluation: fetch BloodMNIST, build the partition manifest,
the pod federation, and the 16-chain battery.

Run after `pip install -e .` (or `make install`). Output:

- `data/bloodmnist.npz` + `data/_manifest.json`  (BloodMNIST + Dirichlet partition)
- `pods/persona_*/` + `pods/_manifest.json`     (200-persona federation)
- `card/chains/<chain_id>.json`                  (16 battery chains)

Bit-for-bit reproducible from `--seed`; default 20260525.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from flta_eval import chains, datasets, pods  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--n-partition-pods", type=int, default=50,
                        help="Number of pods carrying a data partition for FL training.")
    parser.add_argument("--base", type=Path, default=HERE.parent)
    args = parser.parse_args()

    base = args.base
    data_dir, pods_dir, card_dir = base / "data", base / "pods", base / "card"
    chains_dir = card_dir / "chains"

    print(f"[prepare] master seed: {args.seed}")
    print(f"[prepare] fetching BloodMNIST + building Dirichlet partition under {data_dir}")
    data_manifest = datasets.write_manifest(
        data_dir, n_pods=args.n_partition_pods, master_seed=args.seed,
    )
    print(f"  - dataset: {data_manifest['dataset']}; n_train={data_manifest['n_train']}, n_test={data_manifest['n_test']}, n_classes={data_manifest['n_classes']}")
    print(f"  - npz sha256: {data_manifest['npz_sha256'][:16]}…")
    print(f"  - {data_manifest['n_pods']} partition pods; α={data_manifest['partition_alpha']}")
    sizes = sorted(data_manifest['pod_sizes'])
    print(f"  - pod sizes: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}")

    print(f"[prepare] building pod federation in {pods_dir}")
    pods_manifest = pods.write_federation(
        pods_dir, master_seed=args.seed, n_partition_pods=args.n_partition_pods,
    )
    print(f"  - {pods_manifest['size']} pods "
          f"({pods_manifest['positive_count']} positive, {pods_manifest['negative_count']} negative)")
    print(f"  - negative distribution: {pods_manifest['negative_distribution']}")

    print(f"[prepare] building 16-chain battery in {chains_dir}")
    chains_dir.mkdir(parents=True, exist_ok=True)
    battery = chains.build_battery()
    battery_index: list[dict] = []
    for entry in battery:
        out = chains_dir / f"{entry['chain_id'].replace('/', '_')}.json"
        out.write_text(json.dumps(entry["card"], indent=2, sort_keys=True))
        battery_index.append({
            "chain_id": entry["chain_id"],
            "label": entry["label"],
            "card_path": str(out.relative_to(base)),
        })
    (card_dir / "battery_manifest.json").write_text(
        json.dumps({"version": 2, "size": len(battery), "entries": battery_index},
                   indent=2, sort_keys=True)
    )
    print(f"  - {len(battery)} chains (4 base stacks × 4 hazard variants)")

    print("[prepare] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
