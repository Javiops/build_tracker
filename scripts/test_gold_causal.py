"""gold_est must be causal: nothing after the decision may change it.

This is stronger than the wallet identity, which only checks that reconstructed
spending reconciles two frames — it cannot detect that the estimate READ one of
those frames from the future. Version "leaky-v1" passed the identity check and
still leaked: it prorated a residual computed from the frame AFTER the visit.

Run: .venv\\Scripts\\python.exe scripts\\test_gold_causal.py
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ddragon import DataDragon
from app.reconstruct import GOLD_EST_VERSION, reconstruct_game

dragon = DataDragon()

MATCH = {
    "metadata": {"matchId": "KR_GOLD_CAUSAL"},
    "info": {
        "gameCreation": 1,
        "gameDuration": 1800,
        "gameVersion": "16.17.1.1",
        "queueId": 420,
        "participants": [
            {
                "participantId": 8, "puuid": "shopper", "championId": 103,
                "championName": "Ahri", "teamId": 200, "teamPosition": "MIDDLE",
                "riotIdGameName": "a", "riotIdTagline": "KR1", "win": True,
                "kills": 2, "deaths": 0, "assists": 0,
            },
            {
                "participantId": 1, "puuid": "other", "championId": 64,
                "championName": "LeeSin", "teamId": 100, "teamPosition": "JUNGLE",
                "riotIdGameName": "b", "riotIdTagline": "KR1", "win": False,
                "kills": 0, "deaths": 2, "assists": 0,
            },
        ],
    },
}

VISIT_TS = 150_000


def frame(ts: int, current: int, total: int, events: list) -> dict:
    return {
        "timestamp": ts,
        "events": events,
        "participantFrames": {
            "8": {
                "participantId": 8, "currentGold": current, "totalGold": total,
                "level": 8, "minionsKilled": 60, "jungleMinionsKilled": 0,
            },
            "1": {
                "participantId": 1, "currentGold": 100, "totalGold": 1200,
                "level": 8, "minionsKilled": 10, "jungleMinionsKilled": 40,
            },
        },
    }


def timeline() -> dict:
    return {
        "info": {
            "frames": [
                frame(60_000, 300, 1000, [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1056, "timestamp": 20_000},
                ]),
                frame(120_000, 500, 1600, []),
                # the visit itself lives in this frame's event list
                frame(180_000, 120, 2100, [
                    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1026, "timestamp": VISIT_TS},
                ]),
                frame(240_000, 400, 2600, []),
            ]
        }
    }


def visit_estimate(tl: dict) -> tuple[int, dict]:
    _game, events = reconstruct_game(MATCH, tl, dragon)
    shop = next(
        e for e in events
        if e["type"] == "shop" and e["participant_id"] == 8 and e["ts"] == VISIT_TS
    )
    return shop["gold_est"], shop


base_est, base_event = visit_estimate(timeline())
assert base_event["gold_est_version"] == GOLD_EST_VERSION, base_event["gold_est_version"]
assert base_event["gold_gap"]["frame_ts"] == 120_000, base_event["gold_gap"]
assert base_event["gold_gap"]["prior_residual"] is not None, "prior-window forecast not applied"
print(f"baseline gold_est = {base_est} ({GOLD_EST_VERSION}), gap {base_event['gold_gap']}")

# 1. The frame AFTER the visit may not influence the estimate.
future_frame = timeline()
future_frame["info"]["frames"][3] = frame(240_000, 9_999, 99_999, [])
est, _ = visit_estimate(future_frame)
assert est == base_est, f"post-visit frame changed gold_est: {est} != {base_est}"

# 2. Neither may the frame the visit's own events were listed under, when its
#    stats move (that frame is stamped at 180s, i.e. after t).
own_frame = timeline()
own_frame["info"]["frames"][2]["participantFrames"]["8"]["currentGold"] = 4_242
own_frame["info"]["frames"][2]["participantFrames"]["8"]["totalGold"] = 42_424
est, _ = visit_estimate(own_frame)
assert est == base_est, f"the visit's enclosing (later) frame changed gold_est: {est} != {base_est}"

# 3. Events after t may not influence it: a fat bounty and a later purchase.
after_events = timeline()
after_events["info"]["frames"][2]["events"].extend([
    {"type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 170_000,
     "bounty": 300, "shutdownBounty": 600},
    {"type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 175_000},
])
est, _ = visit_estimate(after_events)
assert est == base_est, f"post-visit events changed gold_est: {est} != {base_est}"

# 4. Control — the test must be able to fail. Pre-decision information DOES
#    move the estimate, otherwise assertions 1-3 would be vacuous.
prior = timeline()
prior["info"]["frames"][0]["participantFrames"]["8"]["totalGold"] = 1_450  # smaller prior income
est_prior, _ = visit_estimate(prior)
assert est_prior != base_est, "prior-window income does not affect the estimate (term is dead)"

pre_bounty = timeline()
pre_bounty["info"]["frames"][2]["events"].insert(0, {
    "type": "CHAMPION_KILL", "killerId": 8, "victimId": 1, "timestamp": 130_000,
    "bounty": 300, "shutdownBounty": 0,
})
est_bounty, _ = visit_estimate(pre_bounty)
assert est_bounty == base_est + 300, f"pre-visit bounty not credited: {est_bounty} vs {base_est}"

pre_spend = timeline()
pre_spend["info"]["frames"][2]["events"].insert(0, {
    "type": "ITEM_PURCHASED", "participantId": 8, "itemId": 1036, "timestamp": 130_000,
})
est_spend, _ = visit_estimate(pre_spend)
assert est_spend < base_est, f"pre-visit spend not debited: {est_spend} vs {base_est}"

# 5. The visit's own purchase (same timestamp) is never debited.
assert base_event["gold_gap"]["spent"] == 0, base_event["gold_gap"]

print("ok gold_est is causal: future frames and events cannot move it")
print(f"   controls: prior income {est_prior}, +bounty {est_bounty}, -spend {est_spend}")
