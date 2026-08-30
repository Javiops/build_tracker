from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

from app.config import SHOP_IDLE_MS
from app.ddragon import DataDragon

SHOP_ACTIONS = {"ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO"}
ITEM_EVENTS = SHOP_ACTIONS | {"ITEM_DESTROYED"}

# Match-v5 never PURCHASES the support gold line. It is granted at start, then
# only ITEM_DESTROYED fires as it upgrades: Atlas → Compass → Bounty → finished.
SUPPORT_ATLAS = 3865
SUPPORT_COMPASS = 3866
SUPPORT_BOUNTY = 3867
SUPPORT_FINISHED = {3869, 3870, 3871, 3876, 3877}
SUPPORT_LINE = {SUPPORT_ATLAS, SUPPORT_COMPASS, SUPPORT_BOUNTY} | SUPPORT_FINISHED


def patch_from_version(game_version: str) -> str:
    parts = (game_version or "").split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return game_version or "unknown"


def reconstruct_game(match: dict, timeline: dict, dragon: DataDragon) -> tuple[dict, list[dict]]:
    info = match["info"]
    participants = []
    by_pid: dict[int, dict] = {}
    win_team = 100
    for p in info["participants"]:
        row = {
            "participant_id": p["participantId"],
            "puuid": p["puuid"],
            "champion_id": p["championId"],
            "champion_name": p.get("championName") or dragon.champion_id_name(p["championId"]),
            "team_id": p["teamId"],
            "team_position": p.get("teamPosition") or p.get("individualPosition") or "",
            "riot_id": f"{p.get('riotIdGameName') or ''}#{p.get('riotIdTagline') or ''}".strip("#"),
            "win": bool(p.get("win")),
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
        }
        participants.append(row)
        by_pid[row["participant_id"]] = row
        if row["win"]:
            win_team = row["team_id"]

    game = {
        "match_id": match["metadata"]["matchId"],
        "game_creation": info.get("gameCreation") or 0,
        "game_duration": info.get("gameDuration") or 0,
        "game_version": info.get("gameVersion") or "",
        "patch": patch_from_version(info.get("gameVersion") or ""),
        "queue_id": info.get("queueId") or 0,
        "win_team": win_team,
        "participants_json": json.dumps(participants),
        "source": "reconstructed",
    }

    pid_by_key = {str(p["participant_id"]): p for p in participants}
    inventories: dict[int, list[int]] = {p["participant_id"]: [] for p in participants}
    endgame_items: dict[int, list[int]] = {}
    for p in info["participants"]:
        pid = p["participantId"]
        owned = [p.get(f"item{i}") or 0 for i in range(7)]
        endgame_items[pid] = owned
        role = p.get("teamPosition") or p.get("individualPosition") or ""
        if any(item_id in SUPPORT_LINE for item_id in owned) or role == "UTILITY":
            inventories[pid].append(SUPPORT_ATLAS)
    kda = {p["participant_id"]: {"kills": 0, "deaths": 0, "assists": 0} for p in participants}
    objectives = {
        100: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
        200: {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0},
    }
    last_frames: dict[int, dict] = {}
    events: list[dict] = []
    open_visits: dict[int, dict] = {}

    def close_visit(pid: int) -> None:
        ov = open_visits.pop(pid, None)
        if not ov:
            return
        before = ov["inventory_before"]
        after = ov["inventory_after"]
        before_build = [i for i in before if not dragon.classify(i)["skip"]]
        after_build = [i for i in after if not dragon.classify(i)["skip"]]
        if before_build == after_build:
            return
        bought, consumed = _diff(before, after)
        owner = by_pid[pid]
        stats = ov["end_frames"].get(pid) or {}
        end_kda = ov["end_kda"][pid]
        events.append(
            {
                "type": "shop",
                "ts": ov["ts_start"],
                "ts_end": ov["ts_end"],
                "puuid": owner["puuid"],
                "participant_id": pid,
                "champion_name": owner["champion_name"],
                "champion_id": owner["champion_id"],
                "team_id": owner["team_id"],
                "team_position": owner["team_position"],
                "riot_id": owner["riot_id"],
                "gold": stats.get("currentGold", 500),
                "level": stats.get("level"),
                "cs": (stats.get("minionsKilled") or 0) + (stats.get("jungleMinionsKilled") or 0),
                "kills": end_kda["kills"],
                "deaths": end_kda["deaths"],
                "assists": end_kda["assists"],
                "inventory_before": _items_payload(before, dragon),
                "inventory_after": _items_payload(after, dragon),
                "bought": _items_payload(bought, dragon),
                "consumed": _items_payload(consumed, dragon),
                "board": _board_snapshot(
                    participants,
                    ov["start_inventories"],
                    ov["start_frames"],
                    ov["start_kda"],
                    dragon,
                    pid,
                ),
                "score": deepcopy(ov["end_objectives"]),
            }
        )

    def maybe_close(ts: int) -> None:
        for pid, ov in list(open_visits.items()):
            if ts - ov["last_shop_ts"] > SHOP_IDLE_MS:
                close_visit(pid)

    def stamp_visit(pid: int, ts: int) -> None:
        ov = open_visits.get(pid)
        if not ov:
            return
        ov["ts_end"] = ts
        ov["inventory_after"] = list(inventories[pid])
        ov["end_frames"] = dict(last_frames)
        ov["end_kda"] = {k: dict(v) for k, v in kda.items()}
        ov["end_objectives"] = deepcopy(objectives)

    def start_visit(pid: int, ts: int) -> None:
        open_visits[pid] = {
            "ts_start": ts,
            "ts_end": ts,
            "last_shop_ts": ts,
            "inventory_before": list(inventories[pid]),
            "inventory_after": list(inventories[pid]),
            "start_inventories": {k: list(v) for k, v in inventories.items()},
            "start_frames": dict(last_frames),
            "start_kda": {k: dict(v) for k, v in kda.items()},
            "end_frames": dict(last_frames),
            "end_kda": {k: dict(v) for k, v in kda.items()},
            "end_objectives": deepcopy(objectives),
        }

    def handle_item_batch(owner: int, ts: int, batch: list[dict]) -> None:
        is_shop = any(e.get("type") in SHOP_ACTIONS for e in batch)
        if is_shop:
            if owner not in open_visits:
                start_visit(owner, ts)
            for event in batch:
                _apply_item_event(inventories, event, endgame_items)
            open_visits[owner]["last_shop_ts"] = ts
            stamp_visit(owner, ts)
            return
        for event in batch:
            _apply_item_event(inventories, event, endgame_items)
            if owner in open_visits:
                stamp_visit(owner, ts)

    frames = timeline.get("info", {}).get("frames") or []
    for frame in frames:
        raw_events = frame.get("events") or []
        i = 0
        while i < len(raw_events):
            event = raw_events[i]
            etype = event.get("type")
            ts = event.get("timestamp") or 0
            maybe_close(ts)

            if etype in ITEM_EVENTS:
                owner = event.get("participantId")
                batch = [event]
                while i + 1 < len(raw_events):
                    nxt = raw_events[i + 1]
                    if (nxt.get("timestamp") or 0) != ts:
                        break
                    if nxt.get("participantId") != owner:
                        break
                    if nxt.get("type") not in ITEM_EVENTS:
                        break
                    batch.append(nxt)
                    i += 1
                if owner:
                    handle_item_batch(owner, ts, batch)
                i += 1
                continue

            if etype == "CHAMPION_KILL":
                killer = event.get("killerId") or 0
                victim = event.get("victimId") or 0
                if killer in kda:
                    kda[killer]["kills"] += 1
                if victim in kda:
                    kda[victim]["deaths"] += 1
                for aid in event.get("assistingParticipantIds") or []:
                    if aid in kda:
                        kda[aid]["assists"] += 1

            elif etype == "BUILDING_KILL":
                team = event.get("teamId")
                if team == 100:
                    objectives[200]["towers"] += 1
                elif team == 200:
                    objectives[100]["towers"] += 1

            elif etype == "ELITE_MONSTER_KILL":
                killer_id = event.get("killerId") or 0
                killer_team = pid_by_key.get(str(killer_id), {}).get("team_id")
                if not killer_team:
                    killer_team = event.get("killerTeamId")
                monster = (event.get("monsterType") or "").upper()
                sub = (event.get("monsterSubType") or "").upper()
                if killer_team in objectives:
                    if monster == "DRAGON":
                        objectives[killer_team]["dragons"] += 1
                    elif monster == "BARON_NASHOR":
                        objectives[killer_team]["barons"] += 1
                    elif monster == "RIFTHERALD":
                        objectives[killer_team]["heralds"] += 1
                    elif "HORDE" in monster or "GRUB" in monster or "VOID" in sub:
                        objectives[killer_team]["voidgrubs"] += 1

            i += 1

        for key, stats in (frame.get("participantFrames") or {}).items():
            pid = int(stats.get("participantId") or key)
            last_frames[pid] = stats

    for pid in list(open_visits):
        close_visit(pid)
    events.sort(key=lambda e: (e.get("ts") or 0, e.get("ts_end") or 0))
    for idx, event in enumerate(events, start=1):
        event["event_index"] = idx
    return game, events


def reconstruct_visits(
    match: dict,
    timeline: dict,
    puuid: str,
    dragon: DataDragon,
) -> tuple[dict, list[dict]]:
    game, events = reconstruct_game(match, timeline, dragon)
    participants = json.loads(game["participants_json"])
    target = next((p for p in participants if p["puuid"] == puuid), None)
    if not target:
        raise ValueError("Target player not in this match")
    match_row = {
        "match_id": game["match_id"],
        "puuid": puuid,
        "game_creation": game["game_creation"],
        "game_duration": game["game_duration"],
        "game_version": game["game_version"],
        "patch": game["patch"],
        "queue_id": game["queue_id"],
        "champion_id": target["champion_id"],
        "champion_name": target["champion_name"],
        "team_position": target["team_position"],
        "team_id": target["team_id"],
        "win": int(target["win"]),
        "kills": target["kills"],
        "deaths": target["deaths"],
        "assists": target["assists"],
        "participants_json": game["participants_json"],
    }
    visits = []
    for event in events:
        if event["type"] != "shop" or event["puuid"] != puuid:
            continue
        visits.append(_shop_event_to_visit(event, game["match_id"], puuid, target["team_id"]))
    for i, visit in enumerate(visits, start=1):
        visit["visit_index"] = i
    return match_row, visits


def _shop_event_to_visit(event: dict, match_id: str, puuid: str, team_id: int) -> dict:
    score = event.get("score") or {}
    return {
        "match_id": match_id,
        "puuid": puuid,
        "visit_index": event.get("event_index") or 0,
        "ts_start": event["ts"],
        "ts_end": event.get("ts_end") or event["ts"],
        "gold": event.get("gold"),
        "level": event.get("level"),
        "cs": event.get("cs"),
        "kills": event.get("kills") or 0,
        "deaths": event.get("deaths") or 0,
        "assists": event.get("assists") or 0,
        "inventory_before_json": json.dumps(event.get("inventory_before") or []),
        "inventory_after_json": json.dumps(event.get("inventory_after") or []),
        "bought_json": json.dumps(event.get("bought") or []),
        "consumed_json": json.dumps(event.get("consumed") or []),
        "board_json": json.dumps(event.get("board") or []),
        "allies_json": json.dumps([]),
        "enemies_json": json.dumps([]),
        "objectives_json": json.dumps(
            {
                "ally": score.get(team_id) or {},
                "enemy": score.get(100 if team_id == 200 else 200) or {},
            }
        ),
    }


def reconstruct_decisions(match: dict, timeline: dict, puuid: str, dragon: DataDragon):
    return reconstruct_visits(match, timeline, puuid, dragon)



def _apply_item_event(
    inventories: dict[int, list[int]],
    event: dict,
    endgame_items: dict[int, list[int]] | None = None,
) -> None:
    etype = event.get("type")
    owner = event.get("participantId")
    if not owner:
        return
    inv = inventories.setdefault(owner, [])
    if etype == "ITEM_PURCHASED":
        item_id = event.get("itemId") or 0
        if item_id in SUPPORT_LINE and item_id in inv:
            return
        inv.append(item_id)
    elif etype in {"ITEM_SOLD", "ITEM_DESTROYED"}:
        item_id = event.get("itemId") or 0
        if etype == "ITEM_DESTROYED" and item_id in SUPPORT_LINE:
            _upgrade_support_item(inv, item_id, (endgame_items or {}).get(owner) or [])
            return
        _remove_item(inv, item_id)
    elif etype == "ITEM_UNDO":
        before = event.get("beforeId") or 0
        after = event.get("afterId") or 0
        if before:
            _remove_item(inv, before)
        if after:
            inv.append(after)


def _upgrade_support_item(inventory: list[int], destroyed_id: int, endgame: list[int]) -> None:
    if destroyed_id not in inventory:
        inventory.append(destroyed_id)
    _remove_item(inventory, destroyed_id)
    nxt = None
    if destroyed_id == SUPPORT_ATLAS:
        nxt = SUPPORT_COMPASS
    elif destroyed_id == SUPPORT_COMPASS:
        nxt = SUPPORT_BOUNTY
    elif destroyed_id == SUPPORT_BOUNTY:
        nxt = next((item_id for item_id in SUPPORT_FINISHED if item_id in endgame), None)
    if nxt and nxt not in inventory:
        inventory.append(nxt)


def _diff(before: list[int], after: list[int]) -> tuple[list[int], list[int]]:
    bought = list((Counter(after) - Counter(before)).elements())
    consumed = list((Counter(before) - Counter(after)).elements())
    return bought, consumed


def _items_payload(ids: list[int], dragon: DataDragon) -> list[dict]:
    out = []
    for item_id in ids:
        cls = dragon.classify(item_id)
        out.append(
            {
                "item_id": item_id,
                "item_name": cls["name"],
                "skip": bool(cls["skip"]),
            }
        )
    return out


def _remove_item(inventory: list[int], item_id: int) -> None:
    try:
        inventory.remove(item_id)
    except ValueError:
        return


def _board_snapshot(
    participants: list[dict],
    inventories: dict[int, list[int]],
    frames: dict[int, dict],
    kda: dict[int, dict],
    dragon: DataDragon,
    self_pid: int,
) -> list[dict]:
    out = []
    for p in participants:
        items = inventories.get(p["participant_id"], [])
        stats = frames.get(p["participant_id"]) or {}
        out.append(
            {
                "participant_id": p["participant_id"],
                "champion_id": p["champion_id"],
                "champion_name": p["champion_name"],
                "team_id": p["team_id"],
                "team_position": p["team_position"],
                "is_self": p["participant_id"] == self_pid,
                "items": _items_payload(items, dragon),
                "gold": stats.get("totalGold"),
                "level": stats.get("level"),
                **kda[p["participant_id"]],
            }
        )
    out.sort(key=lambda row: (row["team_id"], row["participant_id"]))
    return out
