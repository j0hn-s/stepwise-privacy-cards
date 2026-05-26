"""The 16-chain battery for SQ-3 (composability surfacing).

Four base FL stacks × four hazard variants. Each chain is a minimal
privacy card sufficient to exercise the FLTA-specific rule engine — not
a real deployment, only the smallest configuration that surfaces
(or does not surface) the composability hazard under test.

The *expected* rule firings are not derived from this module — they
are hand-authored in [`card/battery_expected.json`](../card/battery_expected.json) and read by the SQ-3 notebook
independently. This avoids the "the rule engine agrees with itself"
critique: the battery cards live here; the expected firings live there;
SQ-3 checks the rule engine's output against the hand-authored set.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# ─── Base stacks ─────────────────────────────────────────────────────────────


def _base(card_id: str, title: str, pets: list[str], steps: list[dict]) -> dict:
    """Minimal card shape sufficient for the rule engine.

    Note: this is *not* a fully schema-valid privacy card (missing role,
    implementation_status, etc.). It is a rule-engine test fixture only.
    """
    return {
        "card_id": card_id,
        "schema_version": "1.2",
        "title": title,
        "pet_components": [{"primitive_id": p, "parameters": {}} for p in pets],
        "stepwise_chain": steps,
        "jurisdictional_context": {
            "data_subjects": ["EU"], "controllers": ["EU"], "operators": ["EU"],
        },
    }


BASE_FL = _base(
    "battery-fl",
    "Battery: FL only",
    ["FL"],
    [{"step": 1, "name": "FL", "pet_added": "FL",
      "residual_risk": "per-update gradient leakage; coordinator visibility"}],
)

BASE_FL_DP = _base(
    "battery-fl-dp",
    "Battery: FL + DP-C",
    ["FL", "DP-C"],
    [
        {"step": 1, "name": "FL", "pet_added": "FL",
         "residual_risk": "per-update gradient leakage"},
        {"step": 2, "name": "DP-C", "pet_added": "DP-C",
         "residual_risk": "coordinator sees DP-noised updates"},
    ],
)
for c in BASE_FL_DP["pet_components"]:
    if c["primitive_id"] == "DP-C":
        c["parameters"] = {"epsilon": 6.0, "delta": 1.0e-5}  # above threshold

BASE_FL_TEE_DP = _base(
    "battery-fl-tee-dp",
    "Battery: FL + TEE + DP-C",
    ["FL", "TEE", "DP-C"],
    [
        {"step": 1, "name": "FL", "pet_added": "FL",
         "residual_risk": "per-update gradient leakage"},
        {"step": 2, "name": "DP-C", "pet_added": "DP-C",
         "residual_risk": "coordinator visibility (DP-noised)"},
        {"step": 3, "name": "TEE", "pet_added": "TEE",
         "residual_risk": "hardware vendor compromise; in-enclave coordinator observes updates"},
    ],
)
for c in BASE_FL_TEE_DP["pet_components"]:
    if c["primitive_id"] == "DP-C":
        c["parameters"] = {"epsilon": 6.0, "delta": 1.0e-5}

BASE_FL_MPC_DP = _base(
    "battery-fl-mpc-dp",
    "Battery: FL + MPC + DP-C",
    ["FL", "MPC", "DP-C"],
    [
        {"step": 1, "name": "FL", "pet_added": "FL",
         "residual_risk": "per-update gradient leakage"},
        {"step": 2, "name": "MPC", "pet_added": "MPC",
         "residual_risk": "encrypted intermediate state"},
        {"step": 3, "name": "DP-C", "pet_added": "DP-C",
         "residual_risk": "released model leakage bounded by epsilon"},
    ],
)
for c in BASE_FL_MPC_DP["pet_components"]:
    if c["primitive_id"] == "DP-C":
        c["parameters"] = {"epsilon": 6.0, "delta": 1.0e-5}


BASE_STACKS: list[tuple[str, dict]] = [
    ("FL", BASE_FL),
    ("FL+DP", BASE_FL_DP),
    ("FL+TEE+DP", BASE_FL_TEE_DP),
    ("FL+MPC+DP", BASE_FL_MPC_DP),
]


# ─── Hazard modifications ────────────────────────────────────────────────────


def _accept_residual(card: dict, keyword: str) -> dict:
    card = deepcopy(card)
    card["stepwise_chain"][-1]["residual_risk"] = (
        card["stepwise_chain"][-1].get("residual_risk", "") +
        f" — accepted residual: {keyword}"
    )
    return card


def _set_epsilon(card: dict, epsilon: float) -> dict:
    card = deepcopy(card)
    for c in card.get("pet_components", []):
        if c.get("primitive_id") == "DP-C":
            c.setdefault("parameters", {})["epsilon"] = epsilon
    return card


def _set_jurisdiction(card: dict, *, subjects: list[str], controllers: list[str],
                      operators: list[str], mechanism: str | None = None) -> dict:
    card = deepcopy(card)
    card["jurisdictional_context"] = {
        "data_subjects": subjects, "controllers": controllers, "operators": operators,
    }
    if mechanism is not None:
        card["jurisdictional_context"]["cross_border_mechanism"] = mechanism
    return card


# ─── Battery construction ───────────────────────────────────────────────────


def build_battery() -> list[dict[str, Any]]:
    """Build the 16 battery chains.

    Each entry: {chain_id, label, card}. *Expected* firings are NOT
    encoded here — they sit in `card/battery_expected.json` and are
    consumed independently by SQ-3.
    """
    battery: list[dict[str, Any]] = []
    for base_name, base in BASE_STACKS:
        # H1: default
        battery.append({
            "chain_id": f"{base_name}__H1_default",
            "label": f"{base_name} — default (no extra mitigations)",
            "card": deepcopy(base),
        })
        # H2: gradient inversion at permissive ε
        battery.append({
            "chain_id": f"{base_name}__H2_gradinv_eps6",
            "label": f"{base_name} — ε=6.0 (above gradient-inversion threshold)",
            "card": _set_epsilon(base, 6.0),
        })
        # H3: cross-jurisdictional with no mechanism
        battery.append({
            "chain_id": f"{base_name}__H3_juris_unmechanised",
            "label": f"{base_name} — cross-jurisdictional, no mechanism declared",
            "card": _set_jurisdiction(
                base, subjects=["EU"], controllers=["EU"], operators=["US"],
                mechanism=None,
            ),
        })
        # H4: well-configured (mitigations + accepted residuals + mechanism)
        well = _set_epsilon(base, 1.0)
        well = _set_jurisdiction(
            well, subjects=["EU"], controllers=["EU"], operators=["US"],
            mechanism="EU SCCs (2021/914)",
        )
        if "MPC" not in [c["primitive_id"] for c in well["pet_components"]]:
            well = _accept_residual(well, "coordinator")
        battery.append({
            "chain_id": f"{base_name}__H4_well_configured",
            "label": f"{base_name} — well-configured",
            "card": well,
        })
    return battery
