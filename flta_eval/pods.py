"""Solid pod federation mock.

Each pod represents one *data subject* — one FL participant contributing
a slice of training data. Per-pod resources mirror the five Solid-pod
resource classes that the paper claims as the substrate for per-subject
metadata (docs/SCHEMAS.md §2). The classes are FL-native; provenance graphs
(PROV-O) are not used in this evaluation (provenance is a broader topic
than this short paper covers — see paper §VI).

Pod resources:

1. `consent.jsonld` — Consent Receipt (purposes, lawful basis, validity)
2. `jurisdiction.jsonld` — data-subject and operator jurisdictions plus
   any cross-border mechanism
3. `withdrawal.jsonld` — withdrawal log; non-empty entries exclude the
   subject and trigger a lifecycle obligation
4. `participation.jsonld` — FL participation log: which training rounds
   this subject's data was used in (built by the FL training loop)
5. `data.json` — pointer to the subject's data slice (indices into the
   central NPZ; the data itself is *not* duplicated per pod)

Plus `_pod.json` — the SQ-2 ground-truth meta-record (expected
inclusion / expected firings). Would not exist in a real pod.

The federation is split 100 positive / 100 negative. Positive pods
have all checks passing; negative pods fail one of five FL-relevant
checks. See `NEGATIVE_DISTRIBUTION` for the breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from flta_eval.audit import derive_seed


CONTROLLER_URI = "https://federation.example/controller/fl-bloodmnist-2026"
PURPOSE_AUTHORISED = ["research", "model-training"]
LAWFUL_BASIS = "Art. 6(1)(a) GDPR — consent + Art. 9(2)(j) — research"

POD_FEDERATION_SIZE = 200
POSITIVE_SIZE = 100
NEGATIVE_SIZE = 100

# Failure-mode distribution for the negative set (sum = NEGATIVE_SIZE).
# Provenance failures (formerly `stale_provenance`) are replaced with
# `lifecycle_violation` — a participation log that exists past a withdrawal.
NEGATIVE_DISTRIBUTION = {
    "expired_consent": 25,
    "withdrawn": 20,
    "bad_signature": 15,
    "lifecycle_violation": 15,
    "missing_jurisdiction": 15,
    "controller_mismatch": 10,
}
assert sum(NEGATIVE_DISTRIBUTION.values()) == NEGATIVE_SIZE


# ─── Resource shapes ─────────────────────────────────────────────────────────


@dataclass
class ConsentReceipt:
    subject_webid: str
    controller: str
    purposes: list[str]
    lawful_basis: str
    issued: str
    not_after: str
    withdrawable: bool = True
    signature: str = ""
    signature_valid: bool = True

    def to_jsonld(self) -> dict:
        return {
            "@context": "https://www.w3.org/ns/solid/consent/v1",
            "@type": "ConsentReceipt",
            "subject_webid": self.subject_webid,
            "controller": self.controller,
            "purposes": list(self.purposes),
            "lawful_basis": self.lawful_basis,
            "issued": self.issued,
            "not_after": self.not_after,
            "withdrawable": self.withdrawable,
            "signature": self.signature,
        }


@dataclass
class WithdrawalLog:
    subject_webid: str
    entries: list[dict] = field(default_factory=list)

    def to_jsonld(self) -> dict:
        return {
            "@context": "https://www.w3.org/ns/solid/consent/v1",
            "@type": "WithdrawalLog",
            "subject_webid": self.subject_webid,
            "entries": list(self.entries),
        }


@dataclass
class JurisdictionalTag:
    """Single coherent regulatory regime (no cross-border by default).

    The base setup is: data subjects in the EU, controller in the EU,
    operator in the EU. The COMP-JURIS-001 rule fires when a chain
    declares an operator outside the EU without a cross-border
    mechanism — see chains.py for the negative case.
    """

    data_subject_jurisdiction: str = "EU"
    controller_jurisdiction: str = "EU"
    operator_jurisdiction: str = "EU"
    cross_border_mechanism: str | None = None

    def to_jsonld(self) -> dict:
        return {
            "@type": "JurisdictionalTag",
            "data_subject_jurisdiction": self.data_subject_jurisdiction,
            "controller_jurisdiction": self.controller_jurisdiction,
            "operator_jurisdiction": self.operator_jurisdiction,
            "cross_border_mechanism": self.cross_border_mechanism,
        }


@dataclass
class ParticipationLog:
    """FL participation log — which training rounds used this subject's data."""

    subject_webid: str
    entries: list[dict] = field(default_factory=list)
    last_round: int | None = None

    def add_round(self, *, round_index: int, model_version: str, timestamp: str) -> None:
        self.entries.append({
            "round_index": round_index,
            "model_version": model_version,
            "timestamp": timestamp,
        })
        self.last_round = round_index

    def to_jsonld(self) -> dict:
        return {
            "@context": "https://federation.example/ns/fl-participation/v1",
            "@type": "ParticipationLog",
            "subject_webid": self.subject_webid,
            "entries": list(self.entries),
            "last_round": self.last_round,
        }


@dataclass
class Pod:
    subject_id: str
    webid: str
    pod_index: int                   # index into the dataset partition
    consent: ConsentReceipt
    withdrawal: WithdrawalLog
    jurisdiction: JurisdictionalTag
    participation: ParticipationLog
    expected_inclusion: bool
    expected_firings: list[str]

    def write(self, base: Path) -> None:
        d = base / self.subject_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "consent.jsonld").write_text(
            json.dumps(self.consent.to_jsonld(), indent=2, sort_keys=True)
        )
        (d / "withdrawal.jsonld").write_text(
            json.dumps(self.withdrawal.to_jsonld(), indent=2, sort_keys=True)
        )
        (d / "jurisdiction.jsonld").write_text(
            json.dumps(self.jurisdiction.to_jsonld(), indent=2, sort_keys=True)
        )
        (d / "participation.jsonld").write_text(
            json.dumps(self.participation.to_jsonld(), indent=2, sort_keys=True)
        )
        (d / "data.json").write_text(
            json.dumps({"partition_index": self.pod_index}, indent=2, sort_keys=True)
        )
        (d / "_pod.json").write_text(
            json.dumps(
                {
                    "subject_id": self.subject_id,
                    "webid": self.webid,
                    "pod_index": self.pod_index,
                    "expected_inclusion": self.expected_inclusion,
                    "expected_firings": self.expected_firings,
                },
                indent=2, sort_keys=True,
            )
        )


# ─── Federation construction ─────────────────────────────────────────────────


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _positive_pod(
    subject_id: str, pod_index: int, rng: np.random.Generator, now: datetime,
) -> Pod:
    webid = f"https://example.solidcommunity.net/{subject_id}/profile/card#me"
    issued = now - timedelta(days=int(rng.integers(1, 200)))
    not_after = issued + timedelta(days=365)
    return Pod(
        subject_id=subject_id,
        webid=webid,
        pod_index=pod_index,
        consent=ConsentReceipt(
            subject_webid=webid,
            controller=CONTROLLER_URI,
            purposes=PURPOSE_AUTHORISED,
            lawful_basis=LAWFUL_BASIS,
            issued=_iso(issued),
            not_after=_iso(not_after),
            signature=f"sig:{subject_id}",
            signature_valid=True,
        ),
        withdrawal=WithdrawalLog(subject_webid=webid, entries=[]),
        jurisdiction=JurisdictionalTag(),  # all-EU default
        participation=ParticipationLog(subject_webid=webid),
        expected_inclusion=True,
        expected_firings=[],
    )


def _negative_pod(
    subject_id: str, pod_index: int, failure_mode: str,
    rng: np.random.Generator, now: datetime,
) -> Pod:
    pod = _positive_pod(subject_id, pod_index, rng, now)
    pod.expected_inclusion = False

    if failure_mode == "expired_consent":
        issued = now - timedelta(days=400)
        not_after = issued + timedelta(days=365)
        pod.consent.issued = _iso(issued)
        pod.consent.not_after = _iso(not_after)
        pod.expected_firings = ["SOLID-CONSENT-EXPIRED"]
    elif failure_mode == "withdrawn":
        pod.withdrawal.entries = [{
            "withdrawn_at": _iso(now - timedelta(days=int(rng.integers(1, 60)))),
            "reason": "subject request",
        }]
        pod.expected_firings = ["SOLID-WITHDRAW-001"]
    elif failure_mode == "bad_signature":
        pod.consent.signature_valid = False
        pod.consent.signature = f"sig:{subject_id}:invalid"
        pod.expected_firings = ["SOLID-CONSENT-SIGNATURE"]
    elif failure_mode == "lifecycle_violation":
        # Withdrawal recorded, but participation log shows a *later* round.
        withdrawal_at = now - timedelta(days=30)
        pod.withdrawal.entries = [{
            "withdrawn_at": _iso(withdrawal_at),
            "reason": "subject request",
        }]
        pod.participation.add_round(
            round_index=99,
            model_version="model@2026-05-20",
            timestamp=_iso(withdrawal_at + timedelta(days=10)),
        )
        # Both rules fire — withdrawal is RED, lifecycle is RED.
        pod.expected_firings = ["SOLID-LIFECYCLE-001", "SOLID-WITHDRAW-001"]
    elif failure_mode == "missing_jurisdiction":
        pod.jurisdiction.data_subject_jurisdiction = ""
        pod.expected_firings = ["SOLID-JURIS-001"]
    elif failure_mode == "controller_mismatch":
        pod.consent.controller = "https://federation.example/controller/other-study"
        pod.expected_firings = ["SOLID-CONSENT-CONTROLLER"]
    else:
        raise ValueError(f"unknown failure_mode: {failure_mode}")
    return pod


def build_federation(
    master_seed: int = 20260525,
    now: datetime | None = None,
    n_partition_pods: int = 50,
) -> list[Pod]:
    """Build the 200-persona federation.

    `n_partition_pods` is the size of the *data* partition the FL training
    loop uses. The federation has 200 personae for SQ-2 testing, but only
    the first `n_partition_pods` of them get assigned a partition index;
    the rest carry `pod_index = -1` (no data slice) and are exercised
    only by SQ-2's metadata rules.
    """
    now = now or datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(derive_seed(master_seed, "pods.federation.v1"))

    pods: list[Pod] = []
    for i in range(POSITIVE_SIZE):
        pi = i if i < n_partition_pods else -1
        pods.append(_positive_pod(f"persona_pos_{i:03d}", pi, rng, now))

    neg_modes: list[str] = []
    for mode, count in NEGATIVE_DISTRIBUTION.items():
        neg_modes.extend([mode] * count)
    rng.shuffle(neg_modes)
    for i, mode in enumerate(neg_modes):
        pods.append(_negative_pod(f"persona_neg_{i:03d}", -1, mode, rng, now))

    return pods


def write_federation(out_dir: Path, master_seed: int = 20260525,
                     n_partition_pods: int = 50) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pods = build_federation(master_seed=master_seed, n_partition_pods=n_partition_pods)
    entries: list[dict] = []
    for pod in pods:
        pod.write(out_dir)
        entries.append({
            "subject_id": pod.subject_id,
            "webid": pod.webid,
            "pod_index": pod.pod_index,
            "expected_inclusion": pod.expected_inclusion,
            "expected_firings": sorted(pod.expected_firings),
        })
    manifest = {
        "version": 2,
        "size": len(pods),
        "positive_count": POSITIVE_SIZE,
        "negative_count": NEGATIVE_SIZE,
        "negative_distribution": dict(NEGATIVE_DISTRIBUTION),
        "n_partition_pods": n_partition_pods,
        "master_seed": master_seed,
        "entries": entries,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_pod(pod_dir: Path) -> dict[str, Any]:
    """Read a pod's resources from disk for the rule engine."""
    return {
        "subject_id": pod_dir.name,
        "consent": json.loads((pod_dir / "consent.jsonld").read_text()),
        "consent_signature_valid": _consent_signature_valid(pod_dir),
        "withdrawal": json.loads((pod_dir / "withdrawal.jsonld").read_text()),
        "jurisdiction": json.loads((pod_dir / "jurisdiction.jsonld").read_text()),
        "participation": json.loads((pod_dir / "participation.jsonld").read_text()),
        "data": json.loads((pod_dir / "data.json").read_text()),
        "_pod": json.loads((pod_dir / "_pod.json").read_text()),
    }


def _consent_signature_valid(pod_dir: Path) -> bool:
    sig = json.loads((pod_dir / "consent.jsonld").read_text()).get("signature", "")
    return bool(sig) and not sig.endswith(":invalid")
