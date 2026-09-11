"""Binding between a served artifact and the evaluation that approved it.

Training provenance says what data an artifact *claims* to have used.  It does
not say that its displayed policy was evaluated, nor that a person chose it for
serving.  The local deployment manifest supplies that last, deliberately
manual link.  It is data-local (and therefore gitignored), because promotion
is an operational decision, not a source-code default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import DATA_DIR


DEPLOYMENT_MANIFEST_PATH = DATA_DIR / "ml" / "deployment_manifest.json"
REQUIRED_PROVENANCE = (
    "manifest_sha256",
    "export_fingerprint",
    "eval_version",
    "gold_input",
    "training_split",
    "selection_split",
)


def sha256_file(path: Path) -> str:
    """Return a stable artifact/report digest without trusting its filename."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serving_eligibility(
    artifact: Path,
    provenance: dict[str, Any],
    manifest_path: Path = DEPLOYMENT_MANIFEST_PATH,
) -> dict[str, str | bool]:
    """Return an explainable, fail-closed production eligibility decision."""
    if provenance.get("status") == "legacy_unverified":
        return {
            "eligible": False,
            "reason": "artifact is legacy/unverified and has no frozen-split policy evaluation",
        }
    if provenance.get("probe") or provenance.get("status") == "probe_only":
        return {
            "eligible": False,
            "reason": "artifact came from an isolated early probe and is permanently ineligible for serving",
        }
    if not manifest_path.exists():
        return {
            "eligible": False,
            "reason": "no deployment manifest; evaluate a candidate then promote it explicitly",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"eligible": False, "reason": f"invalid deployment manifest: {exc}"}
    if manifest.get("schema_version") != 1:
        return {"eligible": False, "reason": "unsupported deployment manifest schema"}
    if manifest.get("artifact_sha256") != sha256_file(artifact):
        return {"eligible": False, "reason": "served artifact does not match the promoted artifact digest"}
    approved = manifest.get("artifact_provenance") or {}
    for key in REQUIRED_PROVENANCE:
        if approved.get(key) != provenance.get(key):
            return {"eligible": False, "reason": f"deployment provenance mismatch: {key}"}
    if manifest.get("evaluation_split") != "val":
        return {"eligible": False, "reason": "deployment was not approved from validation policy evaluation"}
    if manifest.get("unaffordable_recommendations") != 0:
        return {"eligible": False, "reason": "promotion report did not pass the exact-budget hard stop"}
    return {"eligible": True, "reason": "promoted after provenance-checked validation policy evaluation"}
