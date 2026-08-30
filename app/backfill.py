from __future__ import annotations

import json
from collections.abc import Callable

from app.db import db, init_db, insert_game, insert_perspectives, now_iso, set_meta, upsert_player
from app.ddragon import DataDragon
from app.ladder import _player_from_match, routing_for_match_id
from app.reconstruct import reconstruct_game, reconstruct_visits
from app.riot import RiotClient, RiotError

Progress = Callable[[dict], None]


def games_missing_full_lobby() -> list[str]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT match_id FROM games
            WHERE source != 'reconstructed'
            ORDER BY game_creation DESC
            """
        ).fetchall()
        return [row["match_id"] for row in rows]


def _fill_match(client: RiotClient, dragon: DataDragon, match_id: str) -> int:
    regional, platform = routing_for_match_id(match_id)
    match = client.match(regional, match_id)
    timeline = client.timeline(regional, match_id) if match else None
    if not match or not timeline:
        raise RiotError(f"{match_id}: missing match or timeline", 404)
    participants = (match.get("info") or {}).get("participants") or []
    for participant in participants:
        upsert_player(_player_from_match(participant, platform, regional))
    game, events = reconstruct_game(match, timeline, dragon)
    insert_game(game, events)
    entries = []
    for participant in participants:
        puuid = participant.get("puuid")
        if not puuid:
            continue
        match_row, visits = reconstruct_visits(match, timeline, puuid, dragon)
        entries.append((match_row, visits))
    if entries:
        insert_perspectives(entries)
    shoppers = len({e.get("puuid") for e in events if e.get("type") == "shop" and e.get("puuid")})
    return shoppers


def ingest_backfill(progress: Progress | None = None) -> dict:
    init_db()
    emit = progress or (lambda _event: None)
    todo = games_missing_full_lobby()
    emit({"step": "list", "message": f"{len(todo)} games need a full 10-player timeline"})
    client = RiotClient()
    dragon = DataDragon()
    filled = 0
    skipped = 0
    errors: list[str] = []
    try:
        for i, match_id in enumerate(todo, start=1):
            try:
                shoppers = _fill_match(client, dragon, match_id)
                filled += 1
                emit(
                    {
                        "step": "stored",
                        "match_id": match_id,
                        "index": i,
                        "total": len(todo),
                        "message": f"{match_id} ({i}/{len(todo)}) {shoppers} buyers",
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
        set_meta("last_backfill", now_iso())
        left = len(games_missing_full_lobby())
        result = {
            "step": "done",
            "filled": filled,
            "skipped": skipped,
            "left": left,
            "errors": errors[:12],
            "message": f"Done. {filled} games rewritten with full lobbies, {skipped} failed, {left} still thin.",
        }
        emit(result)
        return result
    finally:
        client.close()
