"""Artifact/evaluator provenance must survive Windows newline translation."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_policy import manifest_sha256


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "split_manifest.json"
    path.write_bytes(b'{\r\n  "probe": true\r\n}\r\n')
    expected = hashlib.sha256('{\n  "probe": true\n}\n'.encode("utf-8")).hexdigest()
    assert manifest_sha256(path) == expected

print("test_manifest_digest: ok")
