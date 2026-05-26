"""FLTA-specific rule engine.

Three rule families on top of the yaps base rules:

- `COMP-*`     : composability hazards
                 - COMP-SECAGG-001  [Pasquini, Francati & Ateniese, ACM CCS 2022]
                 - COMP-GRADINV-001 [Hatamizadeh et al. 2023; Boenisch et al. 2023]
                 - COMP-TEEMPC-001  [Mo et al., MobiSys 2021; OLIVE 2022]
                 - COMP-JURIS-001   [ICO Anon CoP 2022; EU AI Act 2024]
- `SOLID-*`    : per-subject metadata fidelity
                 - SOLID-CONSENT-001, SOLID-CONSENT-002
                 - SOLID-WITHDRAW-001
                 - SOLID-LIFECYCLE-001 (participation log past a withdrawal)
                 - SOLID-JURIS-001
- `RISKCAL-*`  : operational DP reporting alongside (ε, δ)

The base yaps rules are loaded separately from `yaps/rules/rules.yaml`
via `yaps/engine/risk_engine.py`; this module evaluates only the
FLTA-specific rules. `evaluate_card` returns a combined report.

## Threshold provenance

`GRADINV_EPSILON_THRESHOLD = 4.0` is the per-round configured ε above
which gradient inversion is *practically* successful against realistic
FL configurations *without* secure aggregation, per:

- Hatamizadeh *et al.* (CVPR 2023, "Do gradient inversion attacks make
  federated learning unsafe?") — Table 2, ε = 4 admits partial input
  reconstruction; ε ≥ 8 admits high-fidelity reconstruction.
- Boenisch *et al.* (USENIX Security 2023, "When the curious abandon
  honesty") — §6, the bound where curious aggregators can reconstruct
  inputs from a small number of training rounds.

Below the threshold gradient inversion may still succeed under
adversarial-server attack settings; the rule is conservative for the
honest-but-curious profile that this evaluation tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEVERITY_ORDER = {"RED": 3, "AMBER": 2, "GREEN": 1, "INFO": 0}

# Pinned per the citations above.
GRADINV_EPSILON_THRESHOLD = 4.0


@dataclass
class Finding:
    rule_id: str
    severity: str
    title: str
    detail: str
    source: str = "flta"

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "severity": self.severity,
                "title": self.title, "detail": self.detail, "source": self.source}


# ─── Card accessors ──────────────────────────────────────────────────────────


def _step_pets(card: dict) -> list[str]:
    return [s.get("pet_added", "") for s in card.get("stepwise_chain") or []]


def _primitive_ids(card: dict) -> list[str]:
    return [c.get("primitive_id", "") for c in card.get("pet_components") or []]


def _epsilon(card: dict) -> float | None:
    for c in card.get("pet_components") or []:
        if c.get("primitive_id") == "DP-C":
            eps = (c.get("parameters") or {}).get("epsilon")
            if eps is not None:
                return float(eps)
    return None


def _has_pet(card: dict, pid: str) -> bool:
    return pid in _primitive_ids(card) or pid in _step_pets(card)


_ACCEPTED = "accepted residual:"


def _accepts(card: dict, keyword: str) -> bool:
    needle = f"{_ACCEPTED} {keyword.lower()}"
    for s in card.get("stepwise_chain") or []:
        if needle in (s.get("residual_risk") or "").lower():
            return True
    return False


# ─── COMP-* composability rules ──────────────────────────────────────────────


def comp_secagg_001(card: dict) -> list[Finding]:
    """Secure-aggregation elusion [Pasquini, Francati & Ateniese, CCS 2022]."""
    if not _has_pet(card, "FL"):
        return []
    if _has_pet(card, "MPC"):
        return []
    if _accepts(card, "coordinator") or _accepts(card, "mpc"):
        return []
    return [Finding(
        rule_id="COMP-SECAGG-001",
        severity="RED",
        title="Secure-aggregation elusion is in scope",
        detail=("FL is declared without MPC and no step accepts the residual "
                "coordinator visibility. Model-inconsistency attacks against the "
                "coordinator are in scope (Pasquini et al., CCS 2022)."),
    )]


def comp_gradinv_001(card: dict) -> list[Finding]:
    """Gradient inversion at configured ε [Hatamizadeh 2023; Boenisch 2023]."""
    if not _has_pet(card, "FL"):
        return []
    eps = _epsilon(card)
    if eps is None or eps <= GRADINV_EPSILON_THRESHOLD:
        return []
    if _has_pet(card, "MPC") or _accepts(card, "gradient"):
        return []
    return [Finding(
        rule_id="COMP-GRADINV-001",
        severity="AMBER",
        title="Gradient inversion may remain practical at configured ε",
        detail=(f"DP-C ε={eps} exceeds the operational threshold (ε*={GRADINV_EPSILON_THRESHOLD}) "
                "pinned to Hatamizadeh et al. (CVPR 2023) §5 and Boenisch et al. "
                "(USENIX Security 2023) §6. Add MPC secure aggregation, tighten ε, "
                "or document the residual as accepted."),
    )]


def comp_teempc_001(card: dict) -> list[Finding]:
    """In-enclave coordinator visibility [Mo et al. 2021; OLIVE 2022]."""
    if not _has_pet(card, "TEE"):
        return []
    if _has_pet(card, "MPC"):
        return []
    if _accepts(card, "coordinator") or _accepts(card, "in-enclave"):
        return []
    return [Finding(
        rule_id="COMP-TEEMPC-001",
        severity="AMBER",
        title="In-enclave coordinator visibility is residual",
        detail=("TEE is declared without MPC; the in-enclave coordinator still "
                "observes per-client updates. Add MPC inside the enclave, or "
                "accept the residual explicitly in the chain."),
    )]


def comp_juris_001(card: dict) -> list[Finding]:
    """Cross-border mechanism missing [ICO Anon CoP 2022; EU AI Act 2024]."""
    j = card.get("jurisdictional_context") or {}
    subjects = set(j.get("data_subjects") or [])
    controllers = set(j.get("controllers") or [])
    operators = set(j.get("operators") or [])
    mechanism = j.get("cross_border_mechanism")
    if not subjects or not controllers:
        return []
    crosses = subjects != controllers or subjects != operators
    if crosses and not mechanism:
        return [Finding(
            rule_id="COMP-JURIS-001",
            severity="RED",
            title="Cross-border mechanism missing",
            detail=(f"Subjects {sorted(subjects)}, controllers {sorted(controllers)}, "
                    f"operators {sorted(operators)} span jurisdictions without a "
                    "`cross_border_mechanism` declared."),
        )]
    return []


COMP_RULES = [comp_secagg_001, comp_gradinv_001, comp_teempc_001, comp_juris_001]


# ─── RISKCAL-* operational DP reporting ──────────────────────────────────────


def riskcal_001(card: dict) -> list[Finding]:
    """DP declared but no `risk_calibration.attack_target` block."""
    if not _has_pet(card, "DP-C"):
        return []
    rc = card.get("risk_calibration") or {}
    if not rc.get("attack_target"):
        return [Finding(
            rule_id="RISKCAL-001",
            severity="GREEN",
            title="Consider an attack-rate calibration",
            detail=("DP is declared but no `risk_calibration.attack_target` block. "
                    "An operational attack-rate bound makes the privacy claim "
                    "legible without re-derivation from ε."),
        )]
    return []


def riskcal_002(card: dict, measured: dict | None = None) -> list[Finding]:
    """Measured advantage exceeds declared target."""
    if measured is None:
        return []
    rc = card.get("risk_calibration") or {}
    target = rc.get("attack_target") or {}
    target_adv = target.get("target_advantage")
    actual = measured.get("target_advantage")
    if target_adv is None or actual is None:
        return []
    if actual > target_adv:
        return [Finding(
            rule_id="RISKCAL-002",
            severity="AMBER",
            title="Measured advantage exceeds declared target",
            detail=(f"Measured target_advantage={actual:.4f} exceeds the declared "
                    f"target ({target_adv:.4f}) in `attack_target`."),
        )]
    return []


def riskcal_003(card: dict) -> list[Finding]:
    """`attack_target` declared but `evidence_ref` empty."""
    if not _has_pet(card, "DP-C"):
        return []
    rc = card.get("risk_calibration") or {}
    target = rc.get("attack_target") or {}
    if target and not target.get("evidence_ref"):
        return [Finding(
            rule_id="RISKCAL-003",
            severity="AMBER",
            title="Calibration evidence reference missing",
            detail=("`risk_calibration.attack_target` is populated but `evidence_ref` "
                    "is empty; the calibration is not reproducible from the card "
                    "alone — point it at the result record under results/sq1/."),
        )]
    return []


RISKCAL_RULES = [riskcal_001, riskcal_003]


# ─── SOLID-* per-subject metadata rules ──────────────────────────────────────


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def solid_consent_001(pod: dict, card: dict, *, now: datetime) -> list[Finding]:
    consent = pod.get("consent") or {}
    if not consent:
        return [Finding("SOLID-CONSENT-001", "RED", "Consent receipt absent",
                        f"Pod {pod.get('subject_id')} has no consent receipt.")]
    not_after = _parse_iso(consent.get("not_after", ""))
    if not_after is None or not_after < now:
        return [Finding("SOLID-CONSENT-001", "RED", "Consent receipt expired",
                        f"Pod {pod.get('subject_id')} not_after={consent.get('not_after')} "
                        f"is in the past.")]
    expected_controller = (card.get("regulatory_context") or {}).get("controller_uri") \
        or "https://federation.example/controller/fl-bloodmnist-2026"
    if consent.get("controller") != expected_controller:
        return [Finding("SOLID-CONSENT-001", "RED", "Controller mismatch",
                        f"Consent controller={consent.get('controller')} "
                        f"≠ deployment controller={expected_controller}.")]
    return []


def solid_consent_002(pod: dict, **_: Any) -> list[Finding]:
    if not pod.get("consent_signature_valid", True):
        return [Finding("SOLID-CONSENT-002", "AMBER", "Consent signature does not verify",
                        f"Pod {pod.get('subject_id')} signature failed mocked "
                        "verification against subject WebID.")]
    return []


def solid_withdraw_001(pod: dict, **_: Any) -> list[Finding]:
    entries = ((pod.get("withdrawal") or {}).get("entries")) or []
    if entries:
        return [Finding("SOLID-WITHDRAW-001", "RED", "Subject has withdrawn",
                        f"Pod {pod.get('subject_id')} withdrawal log has "
                        f"{len(entries)} entry/entries; subject is excluded.")]
    return []


def solid_lifecycle_001(pod: dict, **_: Any) -> list[Finding]:
    """Participation log shows a round timestamped *after* the most recent withdrawal."""
    entries = ((pod.get("withdrawal") or {}).get("entries")) or []
    if not entries:
        return []
    last_withdrawn = max(
        (_parse_iso(e.get("withdrawn_at", "")) for e in entries),
        key=lambda d: d or datetime.min.replace(tzinfo=timezone.utc),
        default=None,
    )
    if last_withdrawn is None:
        return []
    participation = (pod.get("participation") or {}).get("entries") or []
    for p in participation:
        ts = _parse_iso(p.get("timestamp", ""))
        if ts is not None and ts > last_withdrawn:
            return [Finding(
                "SOLID-LIFECYCLE-001", "RED",
                "FL participation recorded after withdrawal",
                f"Pod {pod.get('subject_id')} participation round "
                f"{p.get('round_index')} timestamped {p.get('timestamp')} is "
                f"after the most recent withdrawal at {last_withdrawn.isoformat()}. "
                "Lifecycle obligation: retrain or unlearn.",
            )]
    return []


def solid_juris_001(pod: dict, **_: Any) -> list[Finding]:
    j = pod.get("jurisdiction") or {}
    if not (j.get("data_subject_jurisdiction") or "").strip():
        return [Finding("SOLID-JURIS-001", "AMBER", "Jurisdictional tag missing",
                        f"Pod {pod.get('subject_id')} has no resolvable jurisdiction tag.")]
    return []


SOLID_RULES = [
    solid_consent_001, solid_consent_002, solid_withdraw_001,
    solid_lifecycle_001, solid_juris_001,
]


# ─── Composite evaluators ────────────────────────────────────────────────────


def evaluate_card(card: dict, *, measured: dict | None = None,
                  yaps_findings: list[Finding] | None = None) -> dict:
    findings: list[Finding] = list(yaps_findings or [])
    for rule in COMP_RULES + RISKCAL_RULES:
        findings.extend(rule(card))
    findings.extend(riskcal_002(card, measured))

    summary = {"RED": 0, "AMBER": 0, "GREEN": 0, "INFO": 0}
    for f in findings:
        summary[f.severity] = summary.get(f.severity, 0) + 1
    if summary["RED"]:
        top = "RED"
    elif summary["AMBER"]:
        top = "AMBER"
    elif summary["GREEN"]:
        top = "GREEN"
    else:
        top = "INFO"
    return {"findings": [f.to_dict() for f in findings],
            "summary": summary, "top_severity": top}


def project_residual_for_subject(card: dict, pod: dict, *, now: datetime) -> dict:
    findings: list[Finding] = []
    findings.extend(solid_consent_001(pod, card, now=now))
    findings.extend(solid_consent_002(pod))
    findings.extend(solid_withdraw_001(pod))
    findings.extend(solid_lifecycle_001(pod))
    findings.extend(solid_juris_001(pod))

    red = any(f.severity == "RED" for f in findings)
    amber = any(f.severity == "AMBER" for f in findings)
    included = not (red or amber)
    return {
        "subject_id": pod.get("subject_id"),
        "included": included,
        "firings": [f.rule_id for f in findings],
        "findings": [f.to_dict() for f in findings],
    }


def load_card(path: Path) -> dict:
    return json.loads(Path(path).read_text())
