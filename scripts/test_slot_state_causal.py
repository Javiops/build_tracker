"""Role-slot inputs follow the strict decision prefix through export/streaming."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from app.ddragon import dragon_for_patch
from app.reconstruct import SLOT_STATE_VERSION, _items_payload, reconstruct_game
from baseline import _example, _unmodeled_regular_slots
from eval_conditional_policy import legal
from train_prefix import stream_rows

dragon = dragon_for_patch("16.18")
MATCH = {
    "metadata": {"matchId": "KR_SLOT_STATE_CAUSAL"},
    "info": {
        "gameCreation": 1, "gameDuration": 1800, "gameVersion": "16.18.1.1", "queueId": 420,
        "participants": [
            {"participantId": pid, "puuid": str(pid), "championId": champion,
             "championName": name, "teamId": team, "teamPosition": "BOTTOM", "win": pid == 1}
            for pid, champion, name, team in ((1, 22, "Ashe", 100), (2, 51, "Caitlyn", 200))
        ],
    },
}


def item(kind, ts, item_id=1036, pid=1):
    return {"type": kind, "participantId": pid, "itemId": item_id, "timestamp": ts}


def quest(ts, pid=1):
    return item("ITEM_DESTROYED", ts, 1202, pid)


def reconstruct(actions, match=MATCH):
    # Every fixture has a prior inventory and a strictly earlier wallet frame.
    actions = [item("ITEM_PURCHASED", 20000, 1001)] + actions
    frames = []
    for ts in (0, 60000, 120000, 180000, 240000):
        frames.append({
            "timestamp": ts,
            "events": [a for a in actions if ts - 60000 < a["timestamp"] <= ts],
            "participantFrames": {
                str(pid): {"participantId": pid, "currentGold": 1000, "totalGold": 3000,
                           "level": 8, "minionsKilled": 50, "jungleMinionsKilled": 0}
                for pid in (1, 2)
            },
        })
    return reconstruct_game(match, {"info": {"frames": frames}}, dragon)[1]


def visit(actions, ts=150000, match=MATCH):
    return next(e for e in reconstruct(actions, match) if e["participant_id"] == 1 and e["ts"] == ts)


def assert_state(event, expected):
    assert event["bot_quest_complete"] is expected, event
    assert event["slot_state_version"] == SLOT_STATE_VERSION
    self_row = next(p for p in event["board"] if p["is_self"])
    assert self_row["bot_quest_complete"] is expected
    assert self_row["slot_state_version"] == SLOT_STATE_VERSION
    exported = _example(event, dragon)
    assert exported["bot_quest_complete"] is expected
    assert exported["slot_state_version"] == SLOT_STATE_VERSION
    return exported


buy = item("ITEM_PURCHASED", 150000)
assert_state(visit([buy]), False)
assert_state(visit([quest(149999), buy]), True)

# Ties are excluded before and after the purchase, including separate batches
# interrupted by another player's event at that timestamp.
interruption = item("ITEM_PURCHASED", 150000, 1055, pid=2)
for actions in ([quest(150000), buy], [buy, quest(150000)],
                [quest(150000), interruption, buy], [buy, interruption, quest(150000)]):
    assert_state(visit(actions), False)

# Completing after the first purchase cannot change that visit's inputs. A
# later decision does see it; repeated markers cannot overwrite the first one.
actions = [buy, quest(151000), item("ITEM_PURCHASED", 152000),
           item("ITEM_PURCHASED", 170000), quest(171000)]
assert_state(visit(actions), False)
assert_state(visit(actions, 170000), True)

# Role and final DTO do not establish completion, even with final role boots.
changed_match = copy.deepcopy(MATCH)
changed_match["info"]["participants"][0].update(
    teamPosition="BOTTOM", item0=3006, roleQuestComplete=True, roleQuestCompletionTime=1)
assert_state(visit([buy], match=changed_match), False)
changed_match["info"]["participants"][0]["teamPosition"] = "MIDDLE"
assert_state(visit([quest(149999), buy], match=changed_match), True)

# No-buy inputs are captured at the death snapshot, together with its inventory
# and board, not at closeout or from the future +15s synthetic save timestamp.
death = {"type": "CHAMPION_KILL", "timestamp": 150000, "victimId": 1, "killerId": 2}
assert_state(visit([quest(149999), death], 165000), True)
for actions in ([death], [quest(150000), death], [death, quest(150000)], [death, quest(150001)]):
    saved = visit(actions, 165000)
    assert saved["is_save"] is True
    assert_state(saved, False)

# The other nine board tokens retain the same causal state through row adapters.
exported = assert_state(visit([quest(149000, pid=2), quest(149999), buy]), True)
assert exported["others"][0]["bot_quest_complete"] is True
assert exported["others"][0]["slot_state_version"] == SLOT_STATE_VERSION
legacy = copy.deepcopy(exported)
for row in [legacy, *legacy["others"]]:
    row.pop("bot_quest_complete")
    row.pop("slot_state_version")
with tempfile.TemporaryDirectory(prefix="slot-state-causal-") as directory:
    path = Path(directory) / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (exported, legacy)), encoding="utf-8")
    current, old = list(stream_rows(path))
assert current["bot_quest_complete"] is True and current["slot_state_version"] == SLOT_STATE_VERSION
assert current["others"][0]["bot_quest_complete"] is True
assert current["others"][0]["slot_state_version"] == SLOT_STATE_VERSION
assert old["bot_quest_complete"] is None and old["slot_state_version"] is None
assert old["others"][0]["bot_quest_complete"] is None

legacy_event = copy.deepcopy(visit([buy]))
for row in [legacy_event, *legacy_event["board"]]:
    row.pop("bot_quest_complete")
    row.pop("slot_state_version")
old_export = _example(legacy_event, dragon)
assert old_export["bot_quest_complete"] is None and old_export["slot_state_version"] is None
assert old_export["others"][0]["bot_quest_complete"] is None

# Skipped items are absent from model tokens but still occupy regular slots.
# Five potions share their pinned stack, a biscuit stack occupies another slot,
# and the suppressed spent Armguard remains a regular inventory item.
def omitted_slots(ids):
    return _unmodeled_regular_slots(_items_payload(ids, dragon), dragon)


assert omitted_slots([2003] * 5) == 1
assert omitted_slots([2003] * 6) == 2  # no silent clipping of malformed excess copies
assert omitted_slots([2003] * 3 + [2010] * 2 + [2421, 2031]) == 4
assert omitted_slots([2138]) == 1, "consumeOnFull does not prove a held item was consumed"
assert omitted_slots([3340, 3363, 3364, 1200, 1201, 1202, 1203, 1204]) == 0
assert omitted_slots([1036, 2055, 2055, 3006]) == 0
assert _unmodeled_regular_slots([{"item_id": 2419, "skip": True}], dragon) == 1
assert _example(visit([buy]), dragon)["unmodeled_regular_slots"] == 0

# A future potion consumption cannot free a slot at the first purchase. The
# later decision sees it, and the actual baseline legality uses that occupancy.
potion_actions = [item("ITEM_PURCHASED", 100000, i) for i in [1036] * 4 + [2003] * 3]
potion_actions += [item("ITEM_DESTROYED", 150000, 1036) for _ in range(2)]
potion_actions += [item("ITEM_PURCHASED", 150000, 3133)]
potion_actions += [item("ITEM_DESTROYED", 151000, 2003) for _ in range(3)]
potion_actions += [item("ITEM_PURCHASED", 170000, 1029)]
potion_row = _example(visit(potion_actions), dragon)
assert len(potion_row["inventory"]) == 5 and potion_row["unmodeled_regular_slots"] == 1
assert not legal((1029,), potion_row, dragon), "hidden potion permitted a seventh regular item"
assert legal((1029,), {**potion_row, "unmodeled_regular_slots": 0}, dragon), "capacity control is dead"
assert _example(visit(potion_actions, 170000), dragon)["unmodeled_regular_slots"] == 0
without_consumption = [a for a in potion_actions if a["timestamp"] != 151000]
assert _example(visit(without_consumption), dragon)["unmodeled_regular_slots"] == 1

# Stream adapters retain occupancy and live uncertainty without inventing either
# one for a legacy row that omitted the fields.
with tempfile.TemporaryDirectory(prefix="slot-occupancy-stream-") as directory:
    path = Path(directory) / "rows.jsonl"
    path.write_text(json.dumps({**potion_row, "role_slot_inventory_unknown": True}) + "\n{}", encoding="utf-8")
    occupied, legacy = list(stream_rows(path))
assert occupied["unmodeled_regular_slots"] == 1 and occupied["role_slot_inventory_unknown"] is True
assert legacy["unmodeled_regular_slots"] is None and legacy["role_slot_inventory_unknown"] is None
print("ok role-slot strict prefix, tied markers, snapshot/stream parity, and causal omitted-slot occupancy")
