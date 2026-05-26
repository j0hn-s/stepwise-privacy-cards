"""flta_eval — evaluation harness for the FLTA 2026 short paper.

Six modules, all FL-native:

- `audit`     : audit-trail discipline (seed namespaces, file checksums, result records)
- `datasets`  : BloodMNIST (MedMNIST v2) loader + Dirichlet partitioning across pods
- `fl`        : minimal numpy FL training loop (FedAvg + DP-SGD + RDP accountant)
- `attacks`   : FL-native attacks (gradient inversion; per-record MIA against the released model)
- `pods`      : 200-persona Solid pod federation mock (consent, withdrawal, jurisdiction, FL participation log)
- `rules`     : FLTA-specific rule engine (COMP-*, SOLID-*, RISKCAL-*)
- `chains`    : the 16-chain battery for SQ-3

See `PRIVACY_EXPLORER_MAP.md` for what is reused from the parent
repository and what is net new here.
"""

__version__ = "0.1.0"
