from __future__ import annotations

from collections.abc import Callable

from app.config import DEFAULT_MATCH_COUNT, FAKER, RANKED_SOLO_QUEUE, RIOT_API_KEY
from app.db import (
    init_db,
    insert_match_and_visits,
    match_exists,
    now_iso,
    set_meta,
    upsert_player,
)
from app.ddragon import DataDragon
from app.reconstruct import reconstruct_visits
from app.riot import RiotClient, RiotError


Progress = Callable[[dict], None]


def has_api_key() -> bool:
    key = RIOT_API_KEY
    return bool(key) and not key.startswith("RGAPI-xxxx")


def ingest_faker(
    count: int = DEFAULT_MATCH_COUNT,
    progress: Progress | None = None,
    force: bool = False,
) -> dict:
    init_db()
    emit = progress or (lambda _event: None)
    player = FAKER
    ingested = 0
    skipped = 0
    errors: list[str] = []
    client = RiotClient()
    dragon = DataDragon()

    try:
        emit({"step": "account", "message": f"Looking up {player['game_name']}#{player['tag_line']}"})
        account = client.account_by_riot_id(player["regional"], player["game_name"], player["tag_line"])
        puuid = account["puuid"]
        summoner = client.summoner_by_puuid(player["platform"], puuid) or {}
        upsert_player(
            {
                "puuid": puuid,
                "game_name": account.get("gameName") or player["game_name"],
                "tag_line": account.get("tagLine") or player["tag_line"],
                "platform": player["platform"],
                "regional": player["regional"],
                "label": player["label"],
                "profile_icon_id": summoner.get("profileIconId"),
                "summoner_level": summoner.get("summonerLevel"),
            }
        )
        emit(
            {
                "step": "account",
                "message": f"Resolved {account.get('gameName')}#{account.get('tagLine')}",
                "puuid": puuid,
            }
        )

        match_ids = client.match_ids(player["regional"], puuid, RANKED_SOLO_QUEUE, count)
        emit(
            {
                "step": "list",
                "message": f"Found {len(match_ids)} ranked solo games",
                "total": len(match_ids),
            }
        )

        for i, match_id in enumerate(match_ids, start=1):
            if match_exists(match_id, puuid) and not force:
                skipped += 1
                emit(
                    {
                        "step": "skip",
                        "match_id": match_id,
                        "index": i,
                        "total": len(match_ids),
                        "message": f"{match_id} already stored",
                    }
                )
                continue

            emit(
                {
                    "step": "fetch",
                    "match_id": match_id,
                    "index": i,
                    "total": len(match_ids),
                    "message": f"Fetching {match_id} ({i}/{len(match_ids)})",
                }
            )
            try:
                match = client.match(player["regional"], match_id)
                timeline = client.timeline(player["regional"], match_id) if match else None
                if not match or not timeline:
                    skipped += 1
                    errors.append(f"{match_id}: missing match or timeline")
                    continue
                if match.get("info", {}).get("queueId") != RANKED_SOLO_QUEUE:
                    skipped += 1
                    continue
                match_row, visits = reconstruct_visits(match, timeline, puuid, dragon)
                insert_match_and_visits(match_row, visits)
                ingested += 1
                emit(
                    {
                        "step": "stored",
                        "match_id": match_id,
                        "index": i,
                        "total": len(match_ids),
                        "champion": match_row["champion_name"],
                        "visits": len(visits),
                        "message": f"Stored {match_row['champion_name']} — {len(visits)} shop visits",
                    }
                )
            except Exception as exc:
                skipped += 1
                errors.append(f"{match_id}: {exc}")
                emit({"step": "error", "match_id": match_id, "message": str(exc)})

        set_meta("last_sync", now_iso())
        set_meta("faker_puuid", puuid)
        result = {
            "step": "done",
            "puuid": puuid,
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors[:8],
            "message": f"Done. {ingested} new games, {skipped} skipped.",
        }
        emit(result)
        return result
    except RiotError as exc:
        emit({"step": "error", "message": str(exc)})
        raise
    finally:
        client.close()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pull ranked solo games into SQLite.")
    parser.add_argument("--count", type=int, default=DEFAULT_MATCH_COUNT)
    parser.add_argument("--force", action="store_true", help="Re-fetch games already stored")
    parser.add_argument("--ladder", action="store_true", help="Pull top solo ladder for a region this patch")
    parser.add_argument("--region", default="kr", choices=("kr", "euw", "both"), help="Ladder server: kr, euw, or both")
    parser.add_argument("--size", type=int, default=500, help="Ladder size (0 = all of the selected tiers)")
    parser.add_argument("--patch", default=None, help="Patch like 16.17 (default: latest Data Dragon)")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Only games since local midnight; all Challenger + Grandmaster (faster incremental pull)",
    )
    parser.add_argument("--pros", action="store_true", help="Resolve LCK/LEC accounts and append their patch games")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Re-fetch timelines so every stored game has all 10 players' shops",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="With --backfill: re-fetch ALL stored games so reconstruction changes apply everywhere",
    )
    args = parser.parse_args()

    from app.config import DATA_DIR

    log_path = DATA_DIR / (
        "backfill.log"
        if args.backfill
        else "pros.log"
        if args.pros
        else f"ladder-{args.region}.log"
        if args.ladder
        else "ingest.log"
    )
    if args.ladder and args.region == "both":
        log_path = DATA_DIR / "ladder-daily.log"

    def log(event: dict) -> None:
        msg = event.get("message") or json.dumps(event)
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

    if args.backfill:
        from app.backfill import ingest_backfill

        ingest_backfill(progress=log, refresh=args.refresh)
    elif args.pros:
        from app.pros import ingest_pros

        ingest_pros(patch=args.patch, progress=log)
    elif args.ladder:
        from app.config import local_day_start_unix
        from app.ladder import ingest_ladder

        regions = ("kr", "euw") if args.region == "both" else (args.region,)
        daily = args.daily
        kwargs = {
            "patch": args.patch,
            "progress": log,
            "start_time": local_day_start_unix() if daily else None,
            "tiers": ("challenger", "grandmaster") if daily else ("challenger", "grandmaster", "master"),
            "max_games_per_player": 30 if daily else 200,
            "size": 0 if daily else args.size,
        }
        for region in regions:
            ingest_ladder(region=region, **kwargs)
    else:
        ingest_faker(count=args.count, progress=log, force=args.force)

