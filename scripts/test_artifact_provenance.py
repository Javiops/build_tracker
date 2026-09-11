"""Regression test for provenance required on future artifacts.

Run: .venv\\Scripts\\python.exe scripts\\test_artifact_provenance.py
"""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_prefix as tp
import eval_policy
from app.deployment import REQUIRED_PROVENANCE, serving_eligibility, sha256_file


manifest = {
    "eval_version": 3,
    "export_fingerprint": "frozen-example",
    "gold_est_versions": {"prequential-v2": 42},
    "reconstruction_versions": {tp.RECONSTRUCTION_VERSION: 42},
    "patch": "16.17",
    "ddragon_version": "16.17.1",
    "static_data_sha256": tp.dragon_for_patch("16.17").signature,
}
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = tp.SPLIT_MANIFEST_PATH
    tp.SPLIT_MANIFEST_PATH = path
    try:
        tp.apply_config({"gold_x": True})
        provenance = tp.training_provenance()
        assert provenance["training_split"] == "train"
        assert provenance["selection_split"] == "validation"
        assert provenance["gold_input"] == "prequential-v2"
        assert provenance["export_fingerprint"] == "frozen-example"

        for overrides in (
            {"reconstruction_versions": {}},
            {"reconstruction_versions": {tp.RECONSTRUCTION_VERSION: 41, "old": 1}},
            {"ddragon_version": "16.18.1"},
            {"static_data_sha256": "wrong"},
        ):
            path.write_text(json.dumps({**manifest, **overrides}), encoding="utf-8")
            try:
                tp.training_provenance()
            except SystemExit:
                pass
            else:
                raise AssertionError(f"Accepted incompatible provenance: {overrides}")

        path.write_text(json.dumps({**manifest, "gold_est_versions": {"untagged(leaky-v1)": 42}}), encoding="utf-8")
        try:
            tp.training_provenance()
        except SystemExit as exc:
            assert "prequential-v2" in str(exc)
        else:
            raise AssertionError("GOLDX accepted a mixed or untagged gold export")
    finally:
        tp.SPLIT_MANIFEST_PATH = original
        tp.apply_config({})

with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = eval_policy.SPLIT_MANIFEST_PATH
    eval_policy.SPLIT_MANIFEST_PATH = path
    try:
        class Candidate:
            artifact = Path("candidate.pt")
            config = {"gold_input": "prequential-v2", "gold_x": True}
            provenance = {
                "status": "declared",
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "export_fingerprint": "frozen-example",
                "eval_version": 3,
                "gold_input": "prequential-v2",
                "training_split": "train",
                "selection_split": "validation",
            }

        checked = eval_policy.verify_artifact_provenance(Candidate(), manifest, allow_legacy=False)
        assert checked["status"] == "verified"
        Candidate.provenance = {"status": "legacy_unverified", "reason": "missing"}
        try:
            eval_policy.verify_artifact_provenance(Candidate(), manifest, allow_legacy=False)
        except SystemExit as exc:
            assert "legacy/unverified" in str(exc)
        else:
            raise AssertionError("policy evaluator accepted an unbound legacy artifact")
    finally:
        eval_policy.SPLIT_MANIFEST_PATH = original

with tempfile.TemporaryDirectory() as temp:
    temp_path = Path(temp)
    artifact = temp_path / "served.pt"
    artifact.write_bytes(b"candidate-bytes")
    provenance = {
        "status": "declared",
        "manifest_sha256": "manifest",
        "export_fingerprint": "fingerprint",
        "eval_version": 3,
        "gold_input": "prequential-v2",
        "training_split": "train",
        "selection_split": "validation",
    }
    deployment = temp_path / "deployment_manifest.json"
    deployment.write_text(json.dumps({
        "schema_version": 1,
        "artifact_sha256": sha256_file(artifact),
        "artifact_provenance": {key: provenance[key] for key in REQUIRED_PROVENANCE},
        "evaluation_split": "val",
        "unaffordable_recommendations": 0,
    }), encoding="utf-8")
    assert serving_eligibility(artifact, provenance, deployment)["eligible"] is True
    artifact.write_bytes(b"different-bytes")
    assert serving_eligibility(artifact, provenance, deployment)["eligible"] is False
    assert serving_eligibility(artifact, {"status": "legacy_unverified"}, deployment)["eligible"] is False

print("ok artifact provenance: frozen split, causal gold, and deliberate serving promotion are mandatory")
