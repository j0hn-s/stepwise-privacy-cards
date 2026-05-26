"""Audit-trail discipline.

Every result record carries four fields (harness commit, dataset checksum,
configuration hash, seed namespace). The discipline is inherited from the
companion repository's privacy-eval harness; the four fields are sufficient
to reproduce a metric from the record alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def derive_seed(master_seed: int, namespace: str) -> int:
    """Deterministic 64-bit seed for a namespace.

    Two callers with the same (master_seed, namespace) get the same seed.
    Different namespaces produce independent seeds. Keeps RNG values out of
    attack code where they could drift silently.
    """
    h = hashlib.blake2b(
        f"{master_seed}:{namespace}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo_root: Path | None = None) -> str:
    """HEAD commit hash; 'uncommitted' if not in a git tree."""
    cwd = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "uncommitted"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empirical_epsilon_lower_bound(
    *,
    n_positives: int,
    n_positive_trials: int,
    n_false_positives: int,
    n_negative_trials: int,
    delta: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Empirical lower bound on ε from a membership-inference audit.

    Closed-form bound: TPR ≤ exp(ε)·FPR + δ  ⇒  ε ≥ log((TPR − δ) / FPR)
    when TPR > δ. Replace the point TPR with its Clopper–Pearson lower
    confidence bound and the point FPR with its Clopper–Pearson upper
    confidence bound to obtain a one-sided (1 − α) lower bound on ε.

    Reference: M. Jagielski, J. Ullman, A. Oprea. *Auditing differentially
    private machine learning: how private is private SGD?* NeurIPS 2020,
    §4. Same form as M. Nasr *et al.* "Tight auditing of differentially
    private machine learning" USENIX Security 2023, Eq. (1).

    Args:
        n_positives: number of canary detections (true positives).
        n_positive_trials: total positive trials (number of canaries).
        n_false_positives: number of false detections on non-canaries.
        n_negative_trials: total negative trials.
        delta: the δ in the audited (ε, δ)-DP claim.
        confidence: one-sided confidence; default 0.95.

    Returns dict with `tpr_lower`, `fpr_upper`, `epsilon_lower_bound`,
    and `feasible` (True iff `tpr_lower > delta`, i.e. the bound binds).
    """
    from scipy.stats import beta  # imported locally to keep top-level deps clean

    alpha = 1.0 - confidence
    # Clopper–Pearson one-sided lower bound on a binomial proportion.
    tpr_lower = 0.0 if n_positives == 0 else float(
        beta.ppf(alpha, n_positives, n_positive_trials - n_positives + 1)
    )
    # Clopper–Pearson one-sided upper bound on a binomial proportion.
    fpr_upper = 1.0 if n_false_positives == n_negative_trials else float(
        beta.ppf(1.0 - alpha, n_false_positives + 1, n_negative_trials - n_false_positives)
    )

    if tpr_lower <= delta or fpr_upper <= 0:
        return {
            "tpr_lower": tpr_lower, "fpr_upper": fpr_upper,
            "epsilon_lower_bound": 0.0, "feasible": False,
        }

    import math
    eps_lb = math.log((tpr_lower - delta) / fpr_upper)
    return {
        "tpr_lower": tpr_lower, "fpr_upper": fpr_upper,
        "epsilon_lower_bound": max(0.0, float(eps_lb)),
        "feasible": True,
    }


def write_result_record(
    *,
    target_dir: Path,
    attack: str,
    variant: str,
    threat_profile: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    seed_namespace: str,
    result: dict[str, Any],
    tolerances: dict[str, Any] | None = None,
) -> Path:
    """Write a single result record honouring the audit-trail discipline."""
    target_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "attack": attack,
        "variant": variant,
        "threat_profile": threat_profile,
        "dataset": dataset,
        "config": {**config, "config_hash_sha256": config_hash(config)},
        "seed_namespace": seed_namespace,
        "harness_commit": git_commit(),
        "result": result,
        "tolerances": tolerances or {},
        "timestamp_utc": now_utc(),
    }
    stamp = record["timestamp_utc"].replace(":", "-")
    safe_attack = attack.replace(".", "_").replace("/", "_")
    out = target_dir / f"{safe_attack}__{variant}__{stamp}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    return out
