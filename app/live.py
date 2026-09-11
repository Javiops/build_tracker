"""Riot Live Client Data API: poll the running game and shape it like a visit row.

The League client serves https://127.0.0.1:2999 during any game (Practice Tool
included) with a self-signed cert. This is the sanctioned local API — no memory
reading. Items on the scoreboard are public, so all ten builds are available;
only other players' gold is not (those features are zeroed, measured neutral).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from app.config import DATA_DIR
from app.ddragon import DataDragon
from app.shop_econ import PINK, inventory_state

LIVE_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
TEAM_IDS = {"ORDER": 100, "CHAOS": 200}
SNAPSHOT_DIR = DATA_DIR / "live_snapshots"

# The live API names summoner spells; match-v5 (and the model) uses ids.
SUMMONER_SPELL_IDS = {
    "SummonerBoost": 1,       # Cleanse
    "SummonerExhaust": 3,
    "SummonerFlash": 4,
    "SummonerHaste": 6,       # Ghost
    "SummonerHeal": 7,
    "SummonerSmite": 11,
    "SummonerTeleport": 12,
    "SummonerDot": 14,        # Ignite
    "SummonerBarrier": 21,
}


def _summ_id(spell: dict | None) -> int:
    raw = (spell or {}).get("rawDescription") or ""
    for name, sid in SUMMONER_SPELL_IDS.items():
        if name in raw:
            return sid
    return 0


def fetch_snapshot(timeout: float = 2.0) -> dict | None:
    """None when no game is running (connection refused) or the API errors."""
    try:
        response = httpx.get(LIVE_URL, verify=False, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except httpx.HTTPError:
        return None


def _player_name(player: dict) -> str:
    return player.get("riotId") or player.get("summonerName") or ""


def _position(player: dict) -> str:
    pos = (player.get("position") or "").upper()
    return "" if pos in ("", "NONE") else pos


def _player_items(player: dict, dragon: DataDragon) -> list[int]:
    """Keep the observed item multiset, including any exposed role-slot items.

    Read the documented items array once. A slot index is not a second item,
    and no undocumented role-inventory field or earlier snapshot is merged in.
    """
    out: list[int] = []
    for item in player.get("items") or []:
        item_id = int(item.get("itemID") or 0)
        if not item_id or dragon.classify(item_id)["skip"]:
            continue
        count = item.get("count")
        out.extend([item_id] * (int(count) if count is not None else 1))
    return out


def _slot_state(player: dict, inventory: list[int], dragon: DataDragon) -> dict:
    """Expose observed capacity and preserve missing role-slot information.

    Reviewed 2026-09-11: Riot's Live Client docs and the local September 2--4
    captures expose items/count/slot, but no verified bot-quest completion or
    separate role inventory. The captures use slots 0--5 for ordinary items
    and 6 for trinkets. An absent boot/ward may be unowned OR omitted from the
    role slot; neither time, role, nor an unfamiliar slot number resolves it.
    https://developer.riotgames.com/docs/lol#game-client-api_live-client-data-api

    Do not infer completion from missing quest items or invent owned items.
    Consumers must withhold that role's boot/ward buys when ownership is
    unknown. The source items still count wherever the API actually lists them.
    """
    role = _position(player)
    boots = any(
        dragon.classify(item_id).get("is_boots")
        or dragon.classify(item_id).get("is_basic_boots")
        for item_id in inventory
    )
    wards = any(item_id in PINK for item_id in inventory)
    unmodeled_slots = set()
    for item in player.get("items") or []:
        item_id = int(item.get("itemID") or 0)
        slot = item.get("slot")
        if (
            item_id
            and type(slot) is int
            and 0 <= slot < 6
            and dragon.classify(item_id)["skip"]
        ):
            # A potion stack occupies one physical slot, not count slots.
            unmodeled_slots.add(slot)
    return {
        "slot_state_version": "role-slots-v1",
        "bot_quest_complete": None,
        "unmodeled_regular_slots": len(unmodeled_slots),
        "role_slot_inventory_unknown": (
            (role == "BOTTOM" and not boots) or (role == "UTILITY" and not wards)
        ),
    }


def _objectives(snap: dict, players_by_name: dict[str, dict]) -> dict[int, dict]:
    score = {
        100: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
        200: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
    }
    for event in (snap.get("events") or {}).get("Events") or []:
        name = event.get("EventName")
        killer = players_by_name.get(event.get("KillerName") or "")
        killer_team = TEAM_IDS.get((killer or {}).get("team") or "")
        if name == "DragonKill" and killer_team:
            score[killer_team]["dragons"] += 1
        elif name == "BaronKill" and killer_team:
            score[killer_team]["barons"] += 1
        elif name == "HeraldKill" and killer_team:
            score[killer_team]["heralds"] += 1
        elif name == "HordeKill" and killer_team:
            score[killer_team]["voidgrubs"] += 1
        elif name == "TurretKilled":
            turret = event.get("TurretKilled") or ""
            # Turret_T1_* belongs to ORDER, so its death scores for CHAOS.
            if "_T1_" in turret:
                score[200]["towers"] += 1
            elif "_T2_" in turret:
                score[100]["towers"] += 1
    return score


def snapshot_to_row(snap: dict, dragon: DataDragon) -> dict | None:
    """Shape a live snapshot like one exported shop visit (baseline._example)."""
    active = snap.get("activePlayer") or {}
    players = snap.get("allPlayers") or []
    my_name = _player_name(active) or active.get("summonerName") or ""
    players_by_name = {_player_name(p): p for p in players}
    me = players_by_name.get(my_name)
    if me is None:
        return None

    team = TEAM_IDS.get(me.get("team") or "", 100)
    gold = int(active.get("currentGold") or 0)
    inventory = _player_items(me, dragon)
    state = inventory_state(inventory, gold, dragon)
    score = _objectives(snap, players_by_name)
    my_scores = me.get("scores") or {}

    others = []
    for player in players:
        if player is me:
            continue
        champ_id = dragon.champion_id_by_name(player.get("championName") or "")
        player_items = _player_items(player, dragon)
        others.append(
            {
                "champion": player.get("championName") or "",
                "champion_id": champ_id,
                "team_id": TEAM_IDS.get(player.get("team") or "", 200),
                "role": _position(player),
                "level": player.get("level"),
                "gold": None,  # not exposed by the live API
                "items": player_items,
                **_slot_state(player, player_items, dragon),
            }
        )

    return {
        "champion": me.get("championName") or "",
        "champion_id": dragon.champion_id_by_name(me.get("championName") or ""),
        "role": _position(me),
        "team_id": team,
        "gold": gold,  # exact, unlike the frame-stale training value
        "gold_exact": True,  # tells gold_est featurization not to add income drift
        # GOLDX models read the budget from gold_est. Live is the ONLY place the
        # value is genuinely exact — offline rows carry an approximation — so it
        # gets its own version tag, and the trainer/predictor accept both.
        "gold_est": gold,
        "gold_est_version": "live-exact-v2",
        "total_gold": None,
        "keystone_id": int(((active.get("fullRunes") or {}).get("keystone") or {}).get("id") or 0),
        "sub_style": int(((active.get("fullRunes") or {}).get("secondaryRuneTree") or {}).get("id") or 0),
        "summ1": _summ_id((me.get("summonerSpells") or {}).get("summonerSpellOne")),
        "summ2": _summ_id((me.get("summonerSpells") or {}).get("summonerSpellTwo")),
        "ally_obj": score.get(team) or {},
        "enemy_obj": score.get(100 if team == 200 else 200) or {},
        "level": int(active.get("level") or me.get("level") or 0),
        "ts": int(float((snap.get("gameData") or {}).get("gameTime") or 0) * 1000),
        "kills": my_scores.get("kills") or 0,
        "deaths": my_scores.get("deaths") or 0,
        "inventory": inventory,
        **_slot_state(me, inventory, dragon),
        "can_complete": state["can_complete"],
        "n_completable": state["n_completable"],
        "cheapest_complete": state["cheapest_complete"],
        "gold_after_complete": state["gold_after_complete"],
        "n_inventory": state["n_inventory"],
        "others": others,
        "label_id": None,
        "label_ids": [],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Poll the live client API.")
    parser.add_argument("--dump", action="store_true", help="Save raw snapshots as JSON fixtures")
    parser.add_argument("--once", action="store_true", help="Fetch one snapshot, print the row, exit")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    dragon = DataDragon()
    if args.once:
        snap = fetch_snapshot()
        if not snap:
            raise SystemExit("No live game found (is a game or Practice Tool running?)")
        print(json.dumps(snapshot_to_row(snap, dragon), indent=2))
        return

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    print("Polling live client… ctrl-c to stop.")
    while True:
        snap = fetch_snapshot()
        if snap:
            row = snapshot_to_row(snap, dragon)
            champ = (row or {}).get("champion")
            gold = (row or {}).get("gold")
            print(f"{datetime.now():%H:%M:%S}  {champ}  gold={gold}")
            if args.dump:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                (SNAPSHOT_DIR / f"snap_{stamp}.json").write_text(
                    json.dumps(snap), encoding="utf-8"
                )
        else:
            print(f"{datetime.now():%H:%M:%S}  no game")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
