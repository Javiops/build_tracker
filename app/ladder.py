from __future__ import annotations

from collections.abc import Callable

from app.config import (
    FAKER,
    LADDER_REGIONS,
    LADDER_SIZE,
    RANKED_SOLO_QUEUE,
    SOLO_QUEUE_NAME,
    patch_start_unix,
)
from app.db import (
    game_exists,
    init_db,
    insert_game,
    insert_perspectives,
    match_participants,
    match_puuids,
    now_iso,
    set_meta,
    upsert_player,
)
from app.ddragon import DataDragon, latest_version
from app.reconstruct import patch_from_version, reconstruct_game, reconstruct_visits
from app.riot import RiotClient, RiotError

Progress = Callable[[dict], None]
TIER_RANK = {"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}


def ingest_ladder(
    size: int = LADDER_SIZE,
    patch: str | None = None,
    progress: Progress | None = None,
    max_games_per_player: int = 200,
    region: str = "kr",
) -> dict:
    init_db()
    emit = progress or (lambda _event: None)
    patch = patch or patch_from_version(latest_version())
    start_time = patch_start_unix(patch)
    spec = LADDER_REGIONS.get(region)
    if not spec:
        raise RiotError(f"Unknown ladder region {region}. Use kr or euw.")
    platform = spec["platform"]
    regional = spec["regional"]
    label = spec["label"]
    ingested = 0
    skipped = 0
    errors: list[str] = []
    client = RiotClient()
    dragon = DataDragon()

    try:
        emit({"step": "ladder", "message": f"Fetching {label} solo ladder (top {size})"})
        ladder = top_solo_ladder(client, platform, size)
        ladder_set = {row["puuid"] for row in ladder if row.get("puuid")}
        set_meta(f"ladder_size_{region}", str(len(ladder_set)))
        set_meta(f"ladder_patch_{region}", patch)
        emit(
            {
                "step": "ladder",
                "message": f"{len(ladder_set)} players · patch {patch}",
                "players": len(ladder_set),
            }
        )

        match_ids: set[str] = set()
        for i, row in enumerate(ladder, start=1):
            puuid = row.get("puuid")
            if not puuid:
                continue
            emit(
                {
                    "step": "list",
                    "message": f"Match list {i}/{len(ladder)}",
                    "index": i,
                    "total": len(ladder),
                }
            )
            try:
                ids = client.match_ids(
                    regional,
                    puuid,
                    RANKED_SOLO_QUEUE,
                    max_games_per_player,
                    start_time=start_time,
                )
                match_ids.update(ids)
            except RiotError as exc:
                errors.append(f"list {puuid[:8]}: {exc}")
                emit({"step": "error", "message": str(exc)})
                if exc.status in (401, 403):
                    raise

        ordered = sorted(match_ids, reverse=True)
        emit(
            {
                "step": "list",
                "message": f"{len(ordered)} unique ranked games since patch {patch} start",
                "total": len(ordered),
            }
        )
        set_meta(f"ladder_unique_ids_{region}", str(len(ordered)))

        for i, match_id in enumerate(ordered, start=1):
            try:
                added, skipped_here = _ingest_match_perspectives(
                    client,
                    dragon,
                    regional,
                    platform,
                    match_id,
                    patch,
                    ladder_set,
                )
                ingested += added
                skipped += skipped_here
                emit(
                    {
                        "step": "stored" if added else "skip",
                        "match_id": match_id,
                        "index": i,
                        "total": len(ordered),
                        "perspectives": added,
                        "message": (
                            f"{match_id} ({i}/{len(ordered)}) +{added} perspectives"
                            if added
                            else f"{match_id} ({i}/{len(ordered)}) already stored"
                        ),
                    }
                )
            except RiotError as exc:
                skipped += 1
                errors.append(f"{match_id}: {exc}")
                emit({"step": "error", "match_id": match_id, "message": str(exc)})
                if exc.status in (401, 403):
                    raise
            except Exception as exc:
                skipped += 1
                errors.append(f"{match_id}: {exc}")
                emit({"step": "error", "match_id": match_id, "message": str(exc)})

        set_meta("last_sync", now_iso())
        set_meta(f"last_ladder_sync_{region}", now_iso())
        result = {
            "step": "done",
            "ingested": ingested,
            "skipped": skipped,
            "unique_games": len(ordered),
            "players": len(ladder_set),
            "patch": patch,
            "errors": errors[:12],
            "message": (
                f"Done. {ingested} new perspectives from {len(ordered)} unique games "
                f"({len(ladder_set)} {label} ladder players, patch {patch})."
            ),
        }
        emit(result)
        return result
    except RiotError as exc:
        emit({"step": "error", "message": str(exc)})
        raise
    finally:
        client.close()


def top_solo_ladder(client: RiotClient, platform: str, size: int) -> list[dict]:
    rows: list[dict] = []
    for kind, tier in (
        ("challenger", "CHALLENGER"),
        ("grandmaster", "GRANDMASTER"),
        ("master", "MASTER"),
    ):
        for entry in client.league_entries(platform, kind, SOLO_QUEUE_NAME):
            puuid = entry.get("puuid")
            if not puuid:
                continue
            rows.append(
                {
                    "puuid": puuid,
                    "tier": tier,
                    "rank": entry.get("rank") or "I",
                    "league_points": int(entry.get("leaguePoints") or 0),
                    "wins": entry.get("wins") or 0,
                    "losses": entry.get("losses") or 0,
                }
            )
    rows.sort(key=lambda r: (TIER_RANK.get(r["tier"], 9), -r["league_points"]))
    return rows[:size]


def routing_for_match_id(match_id: str) -> tuple[str, str]:
    host = (match_id.split("_", 1)[0] or "").upper()
    return {
        "KR": ("asia", "kr"),
        "EUW1": ("europe", "euw1"),
        "EUN1": ("europe", "eun1"),
        "NA1": ("americas", "na1"),
    }.get(host, ("asia", "kr"))


def _ingest_match_perspectives(
    client: RiotClient,
    dragon: DataDragon,
    regional: str,
    platform: str,
    match_id: str,
    patch: str,
    ladder_set: set[str],
) -> tuple[int, int]:
    regional, platform = routing_for_match_id(match_id)
    if game_exists(match_id):
        return 0, 1
    stored = match_puuids(match_id)
    if stored:
        wanted = [p["puuid"] for p in match_participants(match_id) if p.get("puuid") in ladder_set]
        missing = [puuid for puuid in wanted if puuid not in stored]
        if wanted and not missing:
            return 0, 1
    match = client.match(regional, match_id)
    if not match:
        return 0, 1
    info = match.get("info") or {}
    if info.get("queueId") != RANKED_SOLO_QUEUE:
        return 0, 1
    if patch_from_version(info.get("gameVersion") or "") != patch:
        return 0, 1
    participants = info.get("participants") or []
    wanted = [p["puuid"] for p in participants if p.get("puuid") in ladder_set]
    missing = [puuid for puuid in wanted if puuid not in stored]
    if not missing:
        return 0, 1
    timeline = client.timeline(regional, match_id)
    if not timeline:
        return 0, 1
    for participant in participants:
        upsert_player(_player_from_match(participant, platform, regional))
    game, events = reconstruct_game(match, timeline, dragon)
    insert_game(game, events)
    entries: list[tuple[dict, list[dict]]] = []
    for puuid in missing:
        match_row, visits = reconstruct_visits(match, timeline, puuid, dragon)
        entries.append((match_row, visits))
    if entries:
        insert_perspectives(entries)
    return max(len(entries), 1), 0


def ingest_player_patch(
    puuid: str,
    patch: str | None = None,
    progress: Progress | None = None,
    max_games: int = 200,
) -> dict:
    init_db()
    emit = progress or (lambda _event: None)
    patch = patch or patch_from_version(latest_version())
    start_time = patch_start_unix(patch)
    platform = FAKER["platform"]
    regional = FAKER["regional"]
    client = RiotClient()
    dragon = DataDragon()
    ingested = 0
    skipped = 0
    try:
        ids = client.match_ids(
            regional, puuid, RANKED_SOLO_QUEUE, max_games, start_time=start_time
        )
        emit({"step": "list", "message": f"{len(ids)} ranked games this patch", "total": len(ids)})
        for i, match_id in enumerate(ids, start=1):
            added, skipped_here = _ingest_match_perspectives(
                client, dragon, regional, platform, match_id, patch, {puuid}
            )
            ingested += added
            skipped += skipped_here
            emit(
                {
                    "step": "stored" if added else "skip",
                    "match_id": match_id,
                    "index": i,
                    "total": len(ids),
                    "message": f"{match_id} ({i}/{len(ids)}) +{added}",
                }
            )
        set_meta("last_sync", now_iso())
        result = {
            "step": "done",
            "puuid": puuid,
            "ingested": ingested,
            "skipped": skipped,
            "message": f"Done. {ingested} new perspectives, {skipped} skipped.",
        }
        emit(result)
        return result
    finally:
        client.close()


def current_rank1(client: RiotClient, platform: str) -> dict:
    ladder = top_solo_ladder(client, platform, 1)
    if not ladder:
        raise RiotError("Could not load KR solo ladder.")
    return ladder[0]


def _player_from_match(participant: dict, platform: str, regional: str) -> dict:
    game_name = participant.get("riotIdGameName") or "unknown"
    tag = participant.get("riotIdTagline") or ""
    return {
        "puuid": participant["puuid"],
        "game_name": game_name,
        "tag_line": tag,
        "platform": platform,
        "regional": regional,
        "label": f"{game_name}#{tag}".strip("#"),
        "profile_icon_id": None,
        "summoner_level": None,
    }
