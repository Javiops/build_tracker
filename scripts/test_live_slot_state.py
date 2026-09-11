"""Live ownership stays observable; absent role-slot items remain unknown.

Run: .venv\\Scripts\\python.exe scripts\\test_live_slot_state.py
Fixtures use Riot's documented itemID/count/slot fields and the ordinary slots
observed in the existing local captures, without requiring a running game.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.live import _player_items, _slot_state, snapshot_to_row


class Dragon:
    @staticmethod
    def classify(item_id):
        return {
            "skip": item_id in {0, 1202, 2003, 3340},
            "is_boots": item_id == 3006,
            "is_basic_boots": item_id == 1001,
        }

    @staticmethod
    def combine_recipes():
        return []

    @staticmethod
    def champion_id_by_name(name):
        return {"Ashe": 22, "Leona": 89}.get(name, 0)


dragon = Dragon()


def player(role, items):
    return {"position": role, "items": items}


def item(item_id, count=1, slot=0):
    return {"itemID": item_id, "count": count, "slot": slot}


# Preserve item copies (two components, two wards) and keep exposed boots once,
# even if their slot index is unfamiliar. No slot index establishes completion.
observed = player("BOTTOM", [
    item(1036, slot=0), item(1036, slot=1), item(2055, count=2, slot=2),
    item(3006, slot=12), item(3340, slot=6),
])
inventory = _player_items(observed, dragon)
assert Counter(inventory) == {1036: 2, 2055: 2, 3006: 1}
state = _slot_state(observed, inventory, dragon)
assert state["slot_state_version"] == "role-slots-v1"
assert state["bot_quest_complete"] is None
assert state["role_slot_inventory_unknown"] is False

# Missing role inventory has two possible meanings. Do not fabricate boots,
# wards, completed quests, or read guessed fields/future match summaries.
for role in ("BOTTOM", "UTILITY"):
    missing = player(role, [item(1036), item(3340, slot=6)])
    missing.update({"roleQuestComplete": True, "roleSlot": {"itemID": 3006}})
    assert _player_items(missing, dragon) == [1036]
    state = _slot_state(missing, [1036], dragon)
    assert state["role_slot_inventory_unknown"] is True
    assert state["bot_quest_complete"] is None

# A raw quest marker is context, not owned build inventory. Its presence or
# absence alone is not an established Live Client quest-completion protocol.
quest = player("BOTTOM", [item(1202, slot=8), item(1001, slot=1)])
assert _player_items(quest, dragon) == [1001]
assert _slot_state(quest, [1001], dragon)["bot_quest_complete"] is None
assert _slot_state(player("TOP", []), [], dragon)["role_slot_inventory_unknown"] is False
assert _slot_state(player("NONE", []), [], dragon)["role_slot_inventory_unknown"] is False

# Observe either Control Ward ID and its actual quantity. Do not add an item
# for a reported zero count, or turn a two-ward stack into two physical slots.
support = player("UTILITY", [item(772043, count=2, slot=11), item(2055, count=0)])
assert _player_items(support, dragon) == [772043, 772043]
assert _slot_state(support, [772043, 772043], dragon)["role_slot_inventory_unknown"] is False

# Skipped regular items still consume capacity. Trinkets and unknown slots do
# not count as regular slots; repeated entries for one slot cannot double it.
occupied = player("MIDDLE", [
    item(1036, slot=0), item(2003, count=3, slot=1), item(2003, count=3, slot=1),
    item(0, slot=2), item(3340, slot=6), item(1202, slot=8),
])
assert _slot_state(occupied, _player_items(occupied, dragon), dragon)["unmodeled_regular_slots"] == 1

# The active and board rows carry the same conservative state from this one
# snapshot. A later independent row must not backfill the earlier decision.
snap = {
    "activePlayer": {"riotId": "self#test", "currentGold": 777, "level": 9},
    "allPlayers": [
        dict(player("BOTTOM", [item(1036), item(2003, count=2, slot=1)]),
             riotId="self#test", championName="Ashe", team="ORDER"),
        dict(support, riotId="other#test", championName="Leona", team="ORDER"),
    ],
    "gameData": {"gameTime": 600},
}
row = snapshot_to_row(snap, dragon)
assert row["gold_est"] == 777 and row["gold_est_version"] == "live-exact-v2"
assert row["slot_state_version"] == "role-slots-v1"
assert row["bot_quest_complete"] is None and row["role_slot_inventory_unknown"] is True
assert row["unmodeled_regular_slots"] == 1
assert row["inventory"] == [1036]
assert row["others"][0]["items"] == [772043, 772043]
assert row["others"][0]["role_slot_inventory_unknown"] is False
later = deepcopy(snap)
later["allPlayers"][0]["items"].append(item(1001, slot=2))
assert snapshot_to_row(later, dragon)["role_slot_inventory_unknown"] is False
assert snapshot_to_row(snap, dragon) == row

print("ok live slot state: observed copies, unknown role ownership, skipped slot occupancy, snapshot isolation")
