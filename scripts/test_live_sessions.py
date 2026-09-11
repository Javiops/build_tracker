"""Regression test for explicit live shop sessions and local telemetry.

Run: .venv\\Scripts\\python.exe scripts\\test_live_sessions.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.telemetry import ShopDecisionSessions, TelemetryStore


def row(gold: int, inventory: list[int]) -> dict:
    return {
        "champion": "Ahri", "champion_id": 103, "role": "MIDDLE", "team_id": 100,
        "ts": 600_000, "level": 7, "gold": gold, "inventory": inventory,
        "kills": 2, "deaths": 1,
    }


OPTIONS = [{"kind": "buy", "tier": "best", "items": [{"item_id": 1058, "cost": 400}]}]
MODEL = {"artifact_sha256": "test", "provenance_status": "verified"}


with tempfile.TemporaryDirectory() as temp:
    sessions = ShopDecisionSessions(TelemetryStore(Path(temp)), window_s=10)
    started = sessions.start(row(700, [1056]), OPTIONS, MODEL, now=100.0)
    assert started["active"] and started["start_gold"] == 700
    active = sessions.observe(row(700, [1056]), now=105.0)
    assert active and active["active"], active
    resolved = sessions.observe(row(300, [1056, 1058]), now=106.5)
    assert resolved == {"active": False, "resolved": "purchase_observed"}

    sessions.start(row(500, [1056, 2003]), OPTIONS, MODEL, now=150.0)
    changed = sessions.observe(row(500, [1056]), now=151.0)
    assert changed == {"active": False, "resolved": "inventory_changed_unattributed"}

    sessions.start(row(500, [1056]), OPTIONS, MODEL, now=200.0)
    expired = sessions.observe(row(500, [1056]), now=210.0)
    assert expired == {"active": False, "resolved": "no_buy_timeout"}

    sessions.start(row(500, [1056]), OPTIONS, MODEL, now=300.0)
    assert sessions.cancel() is True
    assert sessions.cancel() is False

    records = []
    for path in Path(temp).glob("*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    assert [record["event"] for record in records] == [
        "shop_query", "purchase_observed", "shop_query", "inventory_changed_unattributed",
        "shop_query", "no_buy_timeout", "shop_query", "abandoned",
    ]
    purchase = records[1]
    assert purchase["outcome"]["bought"] == [1058]
    assert purchase["outcome"]["gold_exact"] == 300
    assert "riot_id" not in json.dumps(records).lower()

main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
predictor_src = (ROOT / "app" / "predictor.py").read_text(encoding="utf-8")
live_ui_src = (ROOT / "web" / "live.html").read_text(encoding="utf-8")
assert '@app.post("/api/live/session")' in main_src
assert "no recommendation outside an explicit shop decision" in live_ui_src
assert "def _target_probs" in predictor_src and "def _plan_probs" not in predictor_src

print("ok explicit shop sessions: query, purchase, no-buy timeout, abandonment, and local telemetry")
