"""Raw Match-V5 archival must be local, immutable and avoid repeat API calls.

Run: .venv\\Scripts\\python.exe scripts\\test_raw_match_archive.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.riot import RiotClient


MATCH_ID = "KR_1234567890"
MATCH = {"metadata": {"matchId": MATCH_ID}, "info": {"gameVersion": "16.17.1.1"}}
TIMELINE = {"metadata": {"matchId": MATCH_ID}, "info": {"frames": []}}


with tempfile.TemporaryDirectory() as directory:
    client = RiotClient(api_key="test-key", raw_cache_dir=Path(directory))
    calls: list[str] = []

    def fetch(url: str, params=None):
        calls.append(url)
        return TIMELINE if url.endswith("/timeline") else MATCH

    client._get = fetch  # type: ignore[method-assign]
    assert client.match("asia", MATCH_ID) == MATCH
    assert client.timeline("asia", MATCH_ID) == TIMELINE
    assert len(calls) == 2

    def no_network(*_args, **_kwargs):
        raise AssertionError("a cached Match-V5 response made a network call")

    client._get = no_network  # type: ignore[method-assign]
    assert client.match("asia", MATCH_ID) == MATCH
    assert client.timeline("asia", MATCH_ID) == TIMELINE
    assert (Path(directory) / "match" / f"{MATCH_ID}.json.gz").exists()
    assert (Path(directory) / "timeline" / f"{MATCH_ID}.json.gz").exists()
    client.close()

print("test_raw_match_archive: ok")
