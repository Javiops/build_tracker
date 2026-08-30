import json

from app.ddragon import DataDragon
from app.reconstruct import reconstruct_visits

dragon = DataDragon()
match = {
    "metadata": {"matchId": "KR_TEST"},
    "info": {
        "gameCreation": 1,
        "gameDuration": 1200,
        "gameVersion": "16.17.1.1",
        "queueId": 420,
        "participants": [
            {
                "participantId": 8,
                "puuid": "faker",
                "championId": 805,
                "championName": "Locke",
                "teamId": 200,
                "teamPosition": "MIDDLE",
                "riotIdGameName": "Hide on bush",
                "riotIdTagline": "KR1",
                "win": True,
                "kills": 6,
                "deaths": 2,
                "assists": 5,
            },
            {
                "participantId": 1,
                "puuid": "other",
                "championId": 103,
                "championName": "Ahri",
                "teamId": 100,
                "teamPosition": "MIDDLE",
                "riotIdGameName": "x",
                "riotIdTagline": "KR1",
                "win": False,
                "kills": 0,
                "deaths": 0,
                "assists": 0,
            },
        ],
    },
}
timeline = {
    "info": {
        "frames": [
            {
                "timestamp": 370000,
                "participantFrames": {},
                "events": [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1056, "timestamp": 4600},
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 3020, "timestamp": 355200},
                    {"type": "ITEM_UNDO", "participantId": 8, "beforeId": 3020, "afterId": 0, "timestamp": 357200},
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1001, "timestamp": 359500},
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 3113, "timestamp": 462800},
                ],
            }
        ]
    }
}
_, visits = reconstruct_visits(match, timeline, "faker", dragon)
assert len(visits) == 3, [ (v["ts_start"], v["ts_end"]) for v in visits ]
shop = visits[1]
after_ids = [i["item_id"] for i in json.loads(shop["inventory_after_json"])]
before_ids = [i["item_id"] for i in json.loads(shop["inventory_before_json"])]
assert 3020 not in after_ids, after_ids
assert 1001 in after_ids, after_ids
assert 1056 in before_ids
wisp = visits[2]
assert [i["item_id"] for i in json.loads(wisp["bought_json"])] == [3113]
print("ok", len(visits), "visits; mid-shop after", after_ids)

complete_timeline = {
    "info": {
        "frames": [
            {
                "timestamp": 210000,
                "participantFrames": {},
                "events": [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 3145, "timestamp": 100000},
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 3113, "timestamp": 101000},
                    {
                        "type": "ITEM_DESTROYED",
                        "participantId": 8,
                        "itemId": 3145,
                        "timestamp": 200000,
                    },
                    {
                        "type": "ITEM_DESTROYED",
                        "participantId": 8,
                        "itemId": 3113,
                        "timestamp": 200000,
                    },
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 4646, "timestamp": 200000},
                ],
            }
        ]
    }
}
_, finished = reconstruct_visits(match, complete_timeline, "faker", dragon)
assert len(finished) == 2, len(finished)
before = [i["item_id"] for i in json.loads(finished[1]["inventory_before_json"])]
after = [i["item_id"] for i in json.loads(finished[1]["inventory_after_json"])]
assert 3145 in before and 3113 in before, before
assert 4646 in after, after
assert 3145 not in after and 3113 not in after, after
print("ok complete-item before", before, "after", after)

support_match = {
    "metadata": {"matchId": "KR_SUPPORT"},
    "info": {
        "gameCreation": 1,
        "gameDuration": 1200,
        "gameVersion": "16.17.1.1",
        "queueId": 420,
        "participants": [
            {
                "participantId": 1,
                "puuid": "faker",
                "championId": 103,
                "championName": "Ahri",
                "teamId": 100,
                "teamPosition": "MIDDLE",
                "riotIdGameName": "Hide on bush",
                "riotIdTagline": "KR1",
                "win": True,
                "kills": 1,
                "deaths": 0,
                "assists": 0,
                "item0": 1056,
                "item1": 3020,
            },
            {
                "participantId": 5,
                "puuid": "bard",
                "championId": 432,
                "championName": "Bard",
                "teamId": 100,
                "teamPosition": "UTILITY",
                "riotIdGameName": "b",
                "riotIdTagline": "KR1",
                "win": True,
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "item0": 3190,
                "item1": 3869,
            },
        ],
    },
}
support_timeline = {
    "info": {
        "frames": [
            {
                "timestamp": 700000,
                "participantFrames": {},
                "events": [
                    {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 1056, "timestamp": 1000},
                    {"type": "ITEM_DESTROYED", "participantId": 5, "itemId": 3865, "timestamp": 600000},
                    {"type": "ITEM_DESTROYED", "participantId": 5, "itemId": 3866, "timestamp": 650000},
                    {"type": "ITEM_DESTROYED", "participantId": 5, "itemId": 3867, "timestamp": 660000},
                    {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 3020, "timestamp": 700000},
                ],
            }
        ]
    }
}
_, support_visits = reconstruct_visits(support_match, support_timeline, "faker", dragon)
early = json.loads(support_visits[0]["board_json"])
late = json.loads(support_visits[-1]["board_json"])
early_bard = next(p for p in early if p["champion_name"] == "Bard")
late_bard = next(p for p in late if p["champion_name"] == "Bard")
assert [i["item_id"] for i in early_bard["items"] if not i["skip"]] == [3865], early_bard["items"]
assert 3869 in [i["item_id"] for i in late_bard["items"] if not i["skip"]], late_bard["items"]
print("ok support chain atlas then celestial")

from app.reconstruct import reconstruct_game

narrator_timeline = {
    "info": {
        "frames": [
            {
                "timestamp": 400000,
                "participantFrames": {},
                "events": [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1056, "timestamp": 4000},
                    {"type": "ELITE_MONSTER_KILL", "killerId": 8, "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON", "timestamp": 180000},
                    {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 1055, "timestamp": 200000},
                ],
            }
        ]
    }
}
game, events = reconstruct_game(match, narrator_timeline, dragon)
kinds = [(e["type"], e.get("champion_name")) for e in events]
assert ("shop", "Locke") in kinds, kinds
assert ("shop", "Ahri") in kinds, kinds
assert all(e["type"] == "shop" for e in events), kinds
print("ok narrator", kinds)
