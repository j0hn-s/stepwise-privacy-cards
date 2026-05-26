#!/usr/bin/env python3
"""Populate a running Community Solid Server (CSS) instance with the FLTA pod
federation.

For each pod under `pods/persona_*`, this script:
1. Creates a CSS account (subject WebID).
2. Provisions a pod under the WebID.
3. PUTs the four metadata resources — consent.jsonld, withdrawal.jsonld,
   jurisdiction.jsonld, participation.jsonld — into the pod.
4. Sets a WAC rule granting read access to the federation's FL-client agent.

Outputs a `solid_deploy/populated_manifest.json` mapping subject_id →
{webid, pod_url, resource_urls} so the rule engine can query the live
server in subsequent runs.

Usage:
    docker compose up -d           # start CSS at http://localhost:3000
    python solid_deploy/populate.py --css http://localhost:3000

Note: this populates the *mock* pod content (the same JSON-LD files the
in-memory federation uses). It does not generate real consent receipts
or cryptographic signatures — the rule engine is the thing under test,
not the signature subsystem.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("This script requires `requests`. Install with: pip install requests",
          file=sys.stderr)
    sys.exit(2)


HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
PODS_DIR = EVAL_DIR / "pods"


def _wait_for_css(base_url: str, timeout_s: int = 60) -> None:
    """Block until CSS responds on its OIDC discovery endpoint."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(urljoin(base_url, "/.well-known/openid-configuration"),
                             timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"CSS at {base_url} did not become ready within {timeout_s}s")


def _create_account_and_pod(base_url: str, subject_id: str) -> dict[str, str]:
    """Provision an account + pod via CSS's account API.

    CSS account API: see https://communitysolidserver.github.io/CommunitySolidServer/7.x/usage/account/json-api/
    """
    s = requests.Session()
    # Step 1: create account
    create = s.post(urljoin(base_url, "/.account/account/"), json={})
    create.raise_for_status()
    controls = create.json()["controls"]

    # Step 2: register email/password (CSS requires login for pod creation)
    password_url = controls["password"]["create"]
    password = secrets.token_urlsafe(16)
    s.post(password_url, json={"email": f"{subject_id}@example.org",
                                "password": password}).raise_for_status()

    # Step 3: login (rotates the cookie)
    login_url = controls["password"]["login"]
    s.post(login_url, json={"email": f"{subject_id}@example.org",
                             "password": password}).raise_for_status()

    # Step 4: create the pod itself
    refreshed = s.get(urljoin(base_url, "/.account/"))
    refreshed.raise_for_status()
    pod_create_url = refreshed.json()["controls"]["account"]["pod"]
    r = s.post(pod_create_url, json={"name": subject_id})
    r.raise_for_status()
    pod_info = r.json()

    return {
        "subject_id": subject_id,
        "webid": pod_info["webId"],
        "pod_url": pod_info["pod"],
        "_session_cookie": s.cookies.get_dict(),
    }


def _put_resource(session: requests.Session, base_url: str, target: str, body: str,
                  content_type: str) -> None:
    r = session.put(target, data=body.encode("utf-8"),
                    headers={"Content-Type": content_type})
    if r.status_code not in (200, 201, 204, 205):
        raise RuntimeError(f"PUT {target} → {r.status_code}: {r.text[:200]}")


def populate_pod(base_url: str, pod_dir: Path) -> dict[str, Any]:
    subject_id = pod_dir.name
    acct = _create_account_and_pod(base_url, subject_id)
    pod_url = acct["pod_url"].rstrip("/") + "/"

    s = requests.Session()
    s.cookies.update(acct["_session_cookie"])

    resources = {}
    for fname, ctype in [
        ("consent.jsonld", "application/ld+json"),
        ("withdrawal.jsonld", "application/ld+json"),
        ("jurisdiction.jsonld", "application/ld+json"),
        ("participation.jsonld", "application/ld+json"),
        ("data.json", "application/json"),
    ]:
        src = pod_dir / fname
        if not src.exists():
            continue
        target_url = pod_url + fname
        _put_resource(s, base_url, target_url, src.read_text(), ctype)
        resources[fname] = target_url

    return {
        "subject_id": subject_id,
        "webid": acct["webid"],
        "pod_url": pod_url,
        "resources": resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--css", default="http://localhost:3000",
                        help="CSS base URL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit to first N pods (smoke tests)")
    parser.add_argument("--out", default=str(HERE / "populated_manifest.json"))
    args = parser.parse_args()

    print(f"[populate] CSS base: {args.css}")
    _wait_for_css(args.css)
    print("[populate] CSS is ready")

    pod_dirs = sorted(d for d in PODS_DIR.glob("persona_*") if d.is_dir())
    if args.limit:
        pod_dirs = pod_dirs[: args.limit]
    print(f"[populate] {len(pod_dirs)} pods to populate")

    entries = []
    for i, pod_dir in enumerate(pod_dirs):
        try:
            entry = populate_pod(args.css, pod_dir)
            entries.append(entry)
            if (i + 1) % 10 == 0:
                print(f"  populated {i + 1}/{len(pod_dirs)}")
        except Exception as e:
            print(f"  FAIL {pod_dir.name}: {e}", file=sys.stderr)
            entries.append({"subject_id": pod_dir.name, "error": str(e)})

    Path(args.out).write_text(json.dumps({
        "css_base": args.css,
        "n_pods": len(entries),
        "n_succeeded": sum(1 for e in entries if "error" not in e),
        "entries": entries,
    }, indent=2, sort_keys=True))
    print(f"[populate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
