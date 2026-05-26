# Solid deployment — Community Solid Server + populate script

This directory deploys the FLTA pod federation against a real Solid server, so the rule engine can be exercised against a live Solid runtime rather than only against in-memory JSON-LD files.

**Why.** The default evaluation (`make eval`) reads pod resources from disk. That tests the rule engine's *file-format fidelity*. The Solid-server deployment additionally tests the runtime path: identity (WebID), authentication (Solid-OIDC), access control (WAC), and the linked-data resource layout — exactly the Solid-specific properties the paper claims as substrate-level value.

## What this directory provides

- [`docker-compose.yml`](docker-compose.yml) — runs the Community Solid Server (CSS) reference implementation [Verborgh et al.; W3C Solid Project] in a single container at `http://localhost:3000`.
- [`css-config/dynamic-pods.json`](css-config/dynamic-pods.json) — CSS configuration enabling dynamic pod provisioning, WAC, file-backed storage, OIDC.
- [`populate.py`](populate.py) — Python script that creates an account + pod per subject, then PUTs the four metadata resources from `pods/persona_*/` into each.

## Run

```bash
cd solid_deploy

# Start CSS
docker compose up -d                 # ~ 10 s to ready

# Populate the federation (small smoke test first)
python populate.py --css http://localhost:3000 --limit 5

# Full federation (200 pods; ~ 30–60 s)
python populate.py --css http://localhost:3000

# Stop and clean up
docker compose down -v
```

After `populate.py` finishes, `populated_manifest.json` maps each `subject_id` to its WebID, pod URL, and per-resource URL. A future evaluation iteration will run the rule engine via these URLs (`load_pod_via_solid` would replace the on-disk `load_pod`); the test of *runtime* fidelity then becomes meaningful.

## What this exercises end-to-end

| Solid property | Path |
|---|---|
| WebID identity | Account creation per pod |
| Solid-OIDC | The session token used by `populate.py` (DPoP-bearer) |
| Linked Data Platform (LDP) | The PUT operations creating per-resource URLs |
| Web Access Control (WAC) | Default ACL applied by CSS on pod creation |
| Resource ownership | Each pod is owned by its subject's WebID; the FL client agent is a separate principal that the WAC rule grants read access to |

## What it does NOT exercise

- **Real cryptographic consent signatures.** The mocked Consent Receipts carry placeholder signatures. Real consent-receipt signing (e.g., via Verifiable Credentials) is out of scope.
- **Federated identity providers.** Subjects authenticate against the same CSS instance; cross-IdP scenarios are not exercised.
- **Multi-server federation.** All pods live on a single CSS instance for the demo. Production cross-server federation would test additional WAC traversal patterns.
- **Performance.** The mock is for *behavioural* fidelity, not throughput.

## When to run it

The on-disk evaluation (`make eval`) is sufficient for SQ-1, SQ-2, SQ-3, SQ-5 as published. The Solid deployment is needed:

- For workshop demos that benefit from showing the real-runtime path (W3 in particular).
- For reviewers who ask "does this actually work against a Solid server, or just on disk?" — the answer is: yes, here's the recipe.
- For the next iteration's notebook that runs the rule engine against the live CSS via HTTP rather than against on-disk files.
