from __future__ import annotations

import json
from collections.abc import Callable

from app.config import DATA_DIR
from app.db import db, init_db, insert_reconstruction, now_iso, set_meta, upsert_player
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
    entries = []
    for participant in participants:
        puuid = participant.get("puuid")
        if not puuid:
            continue
        match_row, visits = reconstruct_visits(match, timeline, puuid, dragon)
        entries.append((match_row, visits))
    insert_reconstruction(game, events, entries)
    shoppers = len({e.get("puuid") for e in events if e.get("type") == "shop" and e.get("puuid")})
    return shoppers


def all_game_ids() -> list[str]:
    with db() as conn:
        rows = conn.execute("SELECT match_id FROM games ORDER BY game_creation DESC").fetchall()
        return [row["match_id"] for row in rows]


REFRESH_DONE = DATA_DIR / "refresh_done.txt"


def ingest_backfill(progress: Progress | None = None, refresh: bool = False) -> dict:
    """refresh=True re-fetches EVERY stored game so reconstruction changes
    (e.g. save events) apply to the whole corpus, not just new ingests.
    Progress is checkpointed to data/refresh_done.txt so an expired API key or
    interruption resumes instead of refetching from the start."""
    init_db()
    emit = progress or (lambda _event: None)
    todo = all_game_ids() if refresh else games_missing_full_lobby()
    done: set[str] = set()
    if refresh and REFRESH_DONE.exists():
        done = set(REFRESH_DONE.read_text(encoding="utf-8").split())
        todo = [m for m in todo if m not in done]
    emit({"step": "list", "message": f"{len(todo)} games to re-reconstruct (refresh={refresh}, {len(done)} already done)"})
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
                if refresh:
                    with REFRESH_DONE.open("a", encoding="utf-8") as handle:
                        handle.write(match_id + "\n")
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
