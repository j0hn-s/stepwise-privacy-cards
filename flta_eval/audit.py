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
