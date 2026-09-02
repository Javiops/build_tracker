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
from app.shop_econ import inventory_state

LIVE_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
TEAM_IDS = {"ORDER": 100, "CHAOS": 200}
SNAPSHOT_DIR = DATA_DIR / "live_snapshots"


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


def _player_items(player: dict, dragon: DataDragon) -> list[int]:
    out: list[int] = []
    for item in player.get("items") or []:
        item_id = int(item.get("itemID") or 0)
        if not item_id or dragon.classify(item_id)["skip"]:
            continue
        out.extend([item_id] * int(item.get("count") or 1))
    return out


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
        others.append(
            {
                "champion": player.get("championName") or "",
                "champion_id": champ_id,
                "team_id": TEAM_IDS.get(player.get("team") or "", 200),
                "role": player.get("position") or "",
                "level": player.get("level"),
                "gold": None,  # not exposed by the live API
                "items": _player_items(player, dragon),
            }
        )

    return {
        "champion": me.get("championName") or "",
        "champion_id": dragon.champion_id_by_name(me.get("championName") or ""),
        "role": me.get("position") or "",
        "team_id": team,
        "gold": gold,  # exact, unlike the frame-stale training value
        "total_gold": None,
        "ally_obj": score.get(team) or {},
        "enemy_obj": score.get(100 if team == 200 else 200) or {},
        "level": int(active.get("level") or me.get("level") or 0),
        "ts": int(float((snap.get("gameData") or {}).get("gameTime") or 0) * 1000),
        "kills": my_scores.get("kills") or 0,
        "deaths": my_scores.get("deaths") or 0,
        "inventory": inventory,
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
