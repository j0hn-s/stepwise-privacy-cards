# Threat model — FLTA 2026 evaluation companion artefact

Three FL-specific threat profiles. Structure follows NIST SP 800-188 [1] and the ICO motivated-intruder framing [2].

## 1. Profiles

### Profile R — semi-honest result consumer

**Position.** Outside the federation. Consumes only the FL-released model.

**Capabilities.** Query the released model; standard ML tooling; population-level auxiliary data.

**Attacks exercised.** Per-record MIA against the released model (SQ-1, SQ-5).

**Source mapping.** NIST SP 800-188 "membership disclosure" and "inference disclosure".

### Profile A — budget-bounded motivated intruder

**Position.** Outside the federation but with a stated query budget against any release surface; limited per-subject auxiliary data of bounded specificity.

**Attacks exercised.** Reserved for future linkage / quasi-identifier attacks against pod-resident metadata. Not currently used by any notebook in this iteration.

**Source mapping.** NIST SP 800-188 "identity disclosure"; ICO motivated-intruder canonical case.

### Profile I — authorised federation collaborator

**Position.** Inside the federation. Observes per-round client gradients before the TEE-internal aggregation (i.e., when secure aggregation is *not* declared, or when secure aggregation is bypassed via model-inconsistency [Pasquini et al. 2022]).

**Attacks exercised.** Gradient inversion against per-round client updates (SQ-1).

**Source mapping.** NIST SP 800-188 "identity disclosure" with insider context.

## 2. Out-of-scope adversaries

- **Byzantine FL participant** — model-poisoning attacks. The card is about confidentiality/disclosure, not training integrity.
- **Adversarial-server attacker** [Boenisch et al. 2023, §5] — a colluding server-side adversary that crafts gradients to exfiltrate inputs. The COMP-GRADINV-001 threshold (ε* = 4.0) is calibrated to the *honest-but-curious* profile; adversarial-server work suggests a tighter threshold but is out of scope here.
- **Supply-chain attacker** against the FL framework, the DP library, or CSS.
- **Hardware-side-channel attacker** against the TEE.

## 3. Trust assumptions on the example federation

- Participating data subjects are EU residents under GDPR.
- Federation coordinator is operated by an EU cloud provider; treated as honest-but-curious at the OS/hypervisor level. TEE isolates aggregation from the host.
- Released model is public output bounded by the declared DP guarantee.
- External analysts are untrusted (Profile R).

Any departure from these is recorded as a chain change and surfaces via the rule engine.

## 4. References

[1] NIST SP 800-188, 2023.
[2] ICO Anonymisation Code of Practice 2022.
