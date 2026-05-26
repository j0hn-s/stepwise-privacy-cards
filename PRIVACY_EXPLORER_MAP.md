# Link to the wider privacy-explorer

This FL-focused evaluation plugs into the parent privacy-explorer repository.

## Direct dependencies — what we reuse

| Evaluation artefact | What it reuses | From |
|---|---|---|
| [`flta_eval/audit.py`](flta_eval/audit.py) — seed derivation, file checksums | The same `derive_seed(master, namespace)` discipline | [`privacy-eval/attacks/common/seeding.py`](../privacy-explorer/privacy-eval/attacks/common/seeding.py) |
| [`card/example_chain.json`](card/example_chain.json) | Validates against yaps schema 1.2 | [`yaps/schemas/privacy_card.schema.json`](../privacy-explorer/yaps/schemas/privacy_card.schema.json) |
| Base rule firings (IFACE / COMP / ASSUR / GOV / SECTOR / REG) | Inherited unchanged | [`yaps/engine/risk_engine.py`](../privacy-explorer/yaps/engine/risk_engine.py) + [`yaps/rules/rules.yaml`](../privacy-explorer/yaps/rules/rules.yaml) |

## Extensions — what is net new here

| Evaluation artefact | What it adds |
|---|---|
| [`flta_eval/datasets.py`](flta_eval/datasets.py) | BloodMNIST loader + Dirichlet non-IID partitioning (Hsu et al. 2019) |
| [`flta_eval/fl.py`](flta_eval/fl.py) | Pure-numpy FedAvg + DP-SGD + RDP accountant |
| [`flta_eval/attacks.py`](flta_eval/attacks.py) | FL-native attacks: gradient inversion (Geiping); per-record MIA against released model (Carlini) |
| [`flta_eval/pods.py`](flta_eval/pods.py) | 200-persona Solid pod federation mock with FL participation logs |
| [`flta_eval/rules.py`](flta_eval/rules.py) | FLTA-specific rule families: `COMP-*`, `SOLID-*`, `RISKCAL-*` |
| [`flta_eval/chains.py`](flta_eval/chains.py) | 16-chain composability battery |
| [`card/battery_expected.json`](card/battery_expected.json) | Hand-authored expected firings (independent of rule engine) |
| [`solid_deploy/`](solid_deploy/) | Docker-compose CSS + populate.py for real Solid runtime testing |

## What we do NOT reuse from privacy-eval

The aggregate-query attack surface harness (`privacy-eval/attacks/mia/per_record_sweep.py`, `dinur_nissim`, `km_singling_out`) is **not** used by this evaluation. The earlier draft did; this revision focuses on FL-native attacks only — gradient inversion and per-record MIA against the *FL-released model*, not against an aggregate-query plugin. The privacy-eval harness remains valuable for its original purpose (auditing aggregate-query plugins like the OXFORDIA-node statistical surface); FLTA is a separate target.

## The card-construction loop

1. **Author a card** in [`yaps/frontend/index.html`](../privacy-explorer/yaps/frontend/index.html) or [`workbench/`](../privacy-explorer/workbench/).
2. **Run the yaps base rule engine** ([`yaps/engine/risk_engine.py`](../privacy-explorer/yaps/engine/risk_engine.py)).
3. **Run this evaluation** (`make eval`) — produces SQ-1 calibration records under `results/sq1/`.
4. **Paste the measured advantage** into the card's `risk_calibration.attack_target` block; point `evidence_ref` at the record.
5. **Re-run the FLTA rule engine** ([`flta_eval/rules.py`](flta_eval/rules.py) `evaluate_card(card, measured=…)`) to surface `COMP-*`, `RISKCAL-*` findings; pair with the pod federation to surface `SOLID-*` findings.
6. **(Optional) Deploy the pod federation against a real Solid server** via [`solid_deploy/`](solid_deploy/) to exercise the runtime path.

This is the same loop W1, W2, W3 workshops exercise with practitioners.
