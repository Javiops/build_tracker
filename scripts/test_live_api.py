"""Regression test: live advice exists only inside an explicit shop session.

Run: .venv\\Scripts\\python.exe scripts\\test_live_api.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.live as live
import app.main as main
from app.telemetry import ShopDecisionSessions, TelemetryStore


class Dragon:
    version = "test"

    @staticmethod
    def item_name(item_id: int) -> str:
        return f"item-{item_id}"


class Predictor:
    def __init__(self):
        self.dragon = Dragon()
        self.trained_at = "test"
        self.provenance = {"status": "declared", "gold_input": "prequential-v2", "scored_on": "validation"}
        self.deployment = {"eligible": True, "reason": "test promotion"}
        self.calls = 0

    def predict_options(self, row, budget_slack=0, allow_save=False):
        self.calls += 1
        assert budget_slack == 0 and allow_save is True
        assert row["gold"] == 700
        return [{"kind": "buy", "tier": "best", "items": [{"item_id": 1036, "name": "item-1036", "cost": 350}], "toward": None}]

    @staticmethod
    def telemetry_metadata():
        return {"artifact_sha256": "test", "provenance_status": "declared"}


row = {
    "champion": "Ahri", "role": "MIDDLE", "gold": 700, "level": 7, "ts": 600_000,
    "inventory": [1056], "champion_id": 103, "team_id": 100, "kills": 2, "deaths": 1,
}
predictor = Predictor()
old_fetch, old_row = live.fetch_snapshot, live.snapshot_to_row
old_predictor, old_dragon, old_sessions = main._predictor, main._live_dragon, main._shop_sessions
try:
    live.fetch_snapshot = lambda: {"fake": True}
    live.snapshot_to_row = lambda _snap, _dragon: dict(row)
    with tempfile.TemporaryDirectory() as temp:
        main._predictor = predictor
        main._live_dragon = predictor.dragon
        main._shop_sessions = ShopDecisionSessions(TelemetryStore(Path(temp)), window_s=90)

        idle = main.api_live()
        assert idle["in_game"] and idle["advice_ready"] is False and idle["options"] == []
        assert predictor.calls == 0, "idle polling must not invoke the model"

        started = main.start_live_shop_session()
        assert started["advice_ready"] is True and started["options"]
        assert predictor.calls == 1

        active = main.api_live()
        assert active["advice_ready"] is True and active["options"]
        assert predictor.calls == 1, "active polling reuses the decision-time options"

        row["inventory"] = [1056, 1036]
        resolved = main.api_live()
        assert resolved["advice_ready"] is False and resolved["options"] == []
finally:
    live.fetch_snapshot, live.snapshot_to_row = old_fetch, old_row
    main._predictor, main._live_dragon, main._shop_sessions = old_predictor, old_dragon, old_sessions

print("ok live API: no model call outside an explicit shop session; purchase resolves the session")
