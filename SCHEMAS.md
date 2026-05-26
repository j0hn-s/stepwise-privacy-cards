# Schemas — FLTA 2026 evaluation companion artefact

Data structures the evaluation depends on. The privacy card schema is inherited from yaps schema 1.2 ([yaps/schemas/privacy_card.schema.json](../privacy-explorer/yaps/schemas/privacy_card.schema.json)); FLTA-specific additions and pod resource classes are documented here.

## 1. Privacy card — relevant fields

Schema 1.2 already provides `risk_calibration.attack_target` with `target_advantage`, `target_fpr`, `evidence_ref`. The example card uses these directly:

```jsonc
{
  "risk_calibration": {
    "mu_dp": 0.85,
    "attack_target": {
      "attack": "MIA",
      "target_advantage": 0.030,
      "target_fpr": 0.001,
      "evidence_ref": "results/sq1/mia_per_record__B__bloodmnist.json"
    },
    "operational_interpretation": "Worst-record MIA against the FL-released model bounded at TPR=0.030 at FPR=10⁻³ …",
    "source_library": "manual"
  }
}
```

## 2. Pod resource classes (FL-native)

Each pod hosts five resources. Provenance graphs (PROV-O) are **not** used by this evaluation — provenance scope is broader than this short paper covers.

### 2.1 Consent Receipt

```jsonc
{
  "@context": "https://www.w3.org/ns/solid/consent/v1",
  "@type": "ConsentReceipt",
  "subject_webid": "https://example.solidcommunity.net/persona_pos_000/profile/card#me",
  "controller": "https://federation.example/controller/fl-bloodmnist-2026",
  "purposes": ["research", "model-training"],
  "lawful_basis": "Art. 6(1)(a) GDPR — consent + Art. 9(2)(j) — research",
  "issued": "2026-04-01T09:30:00Z",
  "not_after": "2027-04-01T09:30:00Z",
  "withdrawable": true,
  "signature": "sig:persona_pos_000"
}
```

Rule engine reads: presence; not expired; controller match; signature validity (mocked).

### 2.2 Withdrawal Log

```jsonc
{
  "@context": "https://www.w3.org/ns/solid/consent/v1",
  "@type": "WithdrawalLog",
  "subject_webid": "...",
  "entries": []
}
```

Non-empty `entries` triggers `SOLID-WITHDRAW-001` (RED) and the subject is excluded.

### 2.3 Jurisdictional Tag

```jsonc
{
  "@type": "JurisdictionalTag",
  "data_subject_jurisdiction": "EU",
  "controller_jurisdiction": "EU",
  "operator_jurisdiction": "EU",
  "cross_border_mechanism": null
}
```

Empty `data_subject_jurisdiction` triggers `SOLID-JURIS-001` (AMBER). The chain-level `jurisdictional_context` block + `COMP-JURIS-001` handles cross-border-mechanism declarations.

### 2.4 FL Participation Log

```jsonc
{
  "@context": "https://federation.example/ns/fl-participation/v1",
  "@type": "ParticipationLog",
  "subject_webid": "...",
  "entries": [],
  "last_round": null
}
```

Each entry: `{round_index, model_version, timestamp}`. A round timestamped *after* a withdrawal entry triggers `SOLID-LIFECYCLE-001` (RED) — the lifecycle obligation that retraining / unlearning is required.

### 2.5 Data pointer

```jsonc
{ "partition_index": 0 }
```

Index into the central NPZ; the data itself is not duplicated per pod.

## 3. Rule catalogue (FLTA-specific)

### 3.1 COMP-* composability rules

| Rule ID | Severity | Fires when |
|---|---|---|
| COMP-SECAGG-001 | RED | FL declared, no MPC, no `accepted residual: coordinator`/`mpc` in the chain |
| COMP-GRADINV-001 | AMBER | FL + DP-C declared with ε > 4.0 (Hatamizadeh 2023, Boenisch 2023), no MPC, no `accepted residual: gradient` |
| COMP-TEEMPC-001 | AMBER | TEE declared, no MPC, no `accepted residual: coordinator`/`in-enclave` |
| COMP-JURIS-001 | RED | `jurisdictional_context` roles span jurisdictions without a `cross_border_mechanism` |

### 3.2 SOLID-* per-subject rules

| Rule ID | Severity | Fires when |
|---|---|---|
| SOLID-CONSENT-001 | RED | Consent receipt missing / expired / controller mismatch |
| SOLID-CONSENT-002 | AMBER | Consent receipt signature does not verify |
| SOLID-WITHDRAW-001 | RED | Withdrawal log non-empty |
| SOLID-LIFECYCLE-001 | RED | FL participation log shows a round timestamped after a withdrawal |
| SOLID-JURIS-001 | AMBER | Jurisdictional tag missing / unresolvable |

### 3.3 RISKCAL-* operational reporting rules

| Rule ID | Severity | Fires when |
|---|---|---|
| RISKCAL-001 | GREEN | DP declared but no `risk_calibration.attack_target` block |
| RISKCAL-002 | AMBER | Measured advantage exceeds declared target |
| RISKCAL-003 | AMBER | `attack_target` declared but `evidence_ref` is empty |

## 4. Result record format

```jsonc
{
  "schema_version": "1.0",
  "attack": "mia.per_record",
  "variant": "B",
  "threat_profile": "R",
  "dataset": {"name": "bloodmnist", "sha256": "...", "n_train": 11959},
  "config": {"n_targets": 8, ..., "config_hash_sha256": "..."},
  "seed_namespace": "attacks.mia.bloodmnist.v1",
  "harness_commit": "...",
  "result": {
    "worst_record_tpr_at_fpr_1e-3": 0.025,
    "median_tpr_at_fpr_1e-3": 0.005,
    "fpr_target": 0.001,
    "n_targets_swept": 8,
    "fl_test_accuracy": 0.55,
    "rdp_accountant_epsilon": 2.0
  },
  "tolerances": {"absolute_tpr": 0.01},
  "timestamp_utc": "2026-05-25T..."
}
```

## 5. Battery expected manifest

[`card/battery_expected.json`](card/battery_expected.json) — hand-authored expected firings for the 16-chain battery. Authored independently from [`flta_eval/chains.py`](flta_eval/chains.py) by reading each chain's pet_components, ε, jurisdictional_context, and applying the rule definitions from first principles. Re-author this file whenever rule semantics change; SQ-3 then becomes a meaningful cross-check.
