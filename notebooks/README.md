# Notebooks — FLTA 2026 evaluation companion artefact

Six notebooks plus a top-level walkthrough. All FL-native; each runnable independently after `make prepare`.

| Path | Study question | Headline | Tutorial wall-clock |
|---|---|---|---|
| [`00_walkthrough.ipynb`](00_walkthrough.ipynb) | orientation | inspect data, pods, card, battery | < 10 s |
| [`sq1-calibration/01_mia_per_record_sweep.ipynb`](sq1-calibration/01_mia_per_record_sweep.ipynb) | SQ-1 (MIA) | worst-record TPR @ FPR=10⁻³ against FL-released model | ~ 60 s |
| [`sq1-calibration/02_gradient_inversion.ipynb`](sq1-calibration/02_gradient_inversion.ipynb) | SQ-1 (gradient inversion) | reconstruction PSNR vs σ | ~ 60–120 s |
| [`sq2-metadata/01_per_subject_fidelity.ipynb`](sq2-metadata/01_per_subject_fidelity.ipynb) | SQ-2 | 0 false inclusions / exclusions over 200-pod federation | ~ 1 s |
| [`sq3-composability/01_rule_battery.ipynb`](sq3-composability/01_rule_battery.ipynb) | SQ-3 | per-rule TPR + TNR vs **hand-authored** expectations | < 1 s |
| [`sq5-comparison/01_scalar_vs_stepwise.ipynb`](sq5-comparison/01_scalar_vs_stepwise.ipynb) | SQ-5 | does the scalar (ε) ordering match the empirical TPR ordering? | ~ 1–2 min |

SQ-4 (reviewer legibility) is a workshop study under [`workshops/`](../../privacy-explorer/workshops/) — not a notebook.

## Tutorial scale vs paper scale

Each notebook has a `PAPER_SCALE` (or `TUTORIAL_SCALE`) flag at the top of its config cell. Tutorial scale keeps wall-clock low for the video demo; paper scale converges the worst-record statistics. Set `PAPER_SCALE = True` and re-run for calibrated numbers.

## Conventions

- **Audit trail.** Result records under `../results/<sq>/` carry harness commit hash, dataset SHA-256, configuration hash, seed namespace.
- **Comparison cell.** The last code cell in SQ-1 notebooks compares the measured value against the card's declared claim and prints any RISKCAL rule firings.
- **No network calls inside the notebook.** Datasets are fetched once by `scripts/prepare.py`; notebooks only read from local files.
- **No interactive widgets.** Every notebook runs under `jupyter nbconvert --execute`.

## Running

```bash
make eval            # runs all notebooks via nbconvert
```

Or individually:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/sq1-calibration/01_mia_per_record_sweep.ipynb
```

## Re-using the harness on your own card

```python
import json, sys
from pathlib import Path
sys.path.insert(0, "papers/flta-2026/evaluation")
from flta_eval import rules

card = json.loads(Path("path/to/your/card.json").read_text())
measured = {"target_advantage": 0.025}      # from your own SQ-1 run
report = rules.evaluate_card(card, measured=measured)
for f in report["findings"]:
    print(f["severity"], f["rule_id"], f["title"])
```
