"""Early-probe artifacts must remain isolated and permanently unservable."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.deployment import serving_eligibility


with tempfile.TemporaryDirectory() as directory:
    artifact = Path(directory) / "probe.pt"
    artifact.write_bytes(b"diagnostic only")
    result = serving_eligibility(
        artifact,
        {
            "status": "probe_only",
            "probe": True,
            "manifest_sha256": "not-used",
            "export_fingerprint": "not-used",
            "eval_version": "probe",
            "gold_input": "prequential-v2",
            "training_split": "train",
            "selection_split": "validation",
        },
        Path(directory) / "deployment_manifest.json",
    )
    assert result["eligible"] is False
    assert "probe" in str(result["reason"])

print("test_probe_isolation: ok")
