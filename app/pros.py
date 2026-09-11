from __future__ import annotations

import json
from collections.abc import Callable

from app.config import DATA_DIR, RANKED_SOLO_QUEUE, patch_start_unix
from app.db import db, init_db, now_iso, set_meta, upsert_player
from app.ddragon import DataDragon
from app.ladder import _ingest_match_perspectives
from app.reconstruct import patch_from_version
from app.riot import RiotClient, RiotError

Progress = Callable[[dict], None]
RESOLVE_PATH = DATA_DIR / "pros_resolved.json"

# Current 2026 starters. Riot IDs are guesses plus accounts already in this DB.
# account-v1 is the source of truth; 404s are skipped.
PROS: list[dict] = [
    # LCK playoffs
    {"id": "Faker", "team": "T1", "role": "MID", "league": "LCK", "also": ["Hide on bush"], "ids": [("Hide on bush", "KR1"), ("Faker", "구라티")]},
    {"id": "Oner", "team": "T1", "role": "JNG", "league": "LCK", "also": ["오 너"], "ids": [("오 너", "111"), ("Oner", "KR1")]},
    {"id": "Painter", "team": "T1", "role": "JNG", "league": "LCK", "also": ["T1 Painter"], "ids": [("T1 Painter", "KR3"), ("Painter", "KR1")]},
    {"id": "Doran", "team": "T1", "role": "TOP", "league": "LCK", "also": ["도란"], "ids": [("Doran", "KR1"), ("T1 Doran", "KR1")]},
    {"id": "Peyz", "team": "T1", "role": "BOT", "league": "LCK", "ids": [("Peyz", "KR11"), ("Peyz", "KR1")]},
    {"id": "Keria", "team": "T1", "role": "SUP", "league": "LCK", "also": ["케리아"], "ids": [("케리아", "KR1"), ("Keria", "KR1")]},
    {"id": "Kiin", "team": "GEN", "role": "TOP", "league": "LCK", "also": ["기인"], "ids": [("kiin", "KR1"), ("Kiin", "KR1")]},
    {"id": "Canyon", "team": "GEN", "role": "JNG", "league": "LCK", "also": ["캐니언"], "ids": [("Canyon", "KR1"), ("캐니언", "KR1")]},
    {"id": "Chovy", "team": "GEN", "role": "MID", "league": "LCK", "also": ["쵸비"], "ids": [("Chovy", "KR1"), ("쵸비", "KR1")]},
    {"id": "Ruler", "team": "GEN", "role": "BOT", "league": "LCK", "also": ["룰러"], "ids": [("Ruler", "KR1"), ("GEN Ruler", "KR1")]},
    {"id": "Duro", "team": "GEN", "role": "SUP", "league": "LCK", "ids": [("Duro", "Gen"), ("Duro", "KR1")]},
    {"id": "Zeus", "team": "HLE", "role": "TOP", "league": "LCK", "also": ["제우스"], "ids": [("제우스", "KR1"), ("Zeus", "KR1")]},
    {"id": "Kanavi", "team": "HLE", "role": "JNG", "league": "LCK", "also": ["카나비"], "ids": [("Kanavi", "KR1"), ("HLE Kanavi", "KR1")]},
    {"id": "Zeka", "team": "HLE", "role": "MID", "league": "LCK", "also": ["제카"], "ids": [("Zeka", "KR1"), ("제카", "KR1")]},
    {"id": "Gumayusi", "team": "HLE", "role": "BOT", "league": "LCK", "also": ["구마유시"], "ids": [("HLE Gumayusi", "0298"), ("구마유시", "KR1")]},
    {"id": "Delight", "team": "HLE", "role": "SUP", "league": "LCK", "also": ["딜라이트"], "ids": [("Delight", "KR1"), ("HLE Delight", "KR1")]},
    {"id": "Siwoo", "team": "DK", "role": "TOP", "league": "LCK", "ids": [("Siwoo", "KR1"), ("DK Siwoo", "KR1")]},
    {"id": "Lucid", "team": "DK", "role": "JNG", "league": "LCK", "ids": [("DK Lucid", "KR1"), ("Lucid", "KR1")]},
    {"id": "ShowMaker", "team": "DK", "role": "MID", "league": "LCK", "also": ["쇼메이커"], "ids": [("DK ShowMaker", "KR1"), ("ShowMaker", "KR1")]},
    {"id": "Smash", "team": "DK", "role": "BOT", "league": "LCK", "ids": [("DK Smash", "KR7"), ("Smash", "KR1")]},
    {"id": "Career", "team": "DK", "role": "SUP", "league": "LCK", "ids": [("Career", "KR1"), ("DK Career", "KR1")]},
    {"id": "PerfecT", "team": "KT", "role": "TOP", "league": "LCK", "ids": [("PerfecT", "132"), ("PerfecT", "KR1")]},
    {"id": "Cuzz", "team": "KT", "role": "JNG", "league": "LCK", "ids": [("Cuzz", "KR1")]},
    {"id": "Bdd", "team": "KT", "role": "MID", "league": "LCK", "also": ["비디디"], "ids": [("Bdd", "KR1"), ("비디디", "KR1")]},
    {"id": "Jiwoo", "team": "KT", "role": "BOT", "league": "LCK", "ids": [("Jiwoo", "KR1"), ("KT Jiwoo", "KR1")]},
    {"id": "FenRir", "team": "KT", "role": "BOT", "league": "LCK", "ids": [("FenRir", "KR1"), ("Fenrir", "KR1")]},
    {"id": "Effort", "team": "KT", "role": "SUP", "league": "LCK", "ids": [("Effort", "KR1"), ("KT Effort", "KR1")]},
    {"id": "Clear", "team": "BFX", "role": "TOP", "league": "LCK", "ids": [("Clear", "KR1"), ("BFX Clear", "KR1")]},
    {"id": "Raptor", "team": "BFX", "role": "JNG", "league": "LCK", "ids": [("Raptor", "KR1"), ("BFX Raptor", "KR1")]},
    {"id": "VicLa", "team": "BFX", "role": "MID", "league": "LCK", "ids": [("VicLa", "KR1"), ("BFX VicLa", "KR1")]},
    {"id": "Taeyoon", "team": "BFX", "role": "BOT", "league": "LCK", "ids": [("Taeyoon", "KR1"), ("BFX Taeyoon", "KR1")]},
    {"id": "Kellin", "team": "BFX", "role": "SUP", "league": "LCK", "ids": [("Kellin", "KR1"), ("BFX Kellin", "KR1")]},
    # LCK remainder
    {"id": "Kingen", "team": "NS", "role": "TOP", "league": "LCK", "ids": [("Kingen", "KR1")]},
    {"id": "Sponge", "team": "NS", "role": "JNG", "league": "LCK", "ids": [("Sponge", "KR1")]},
    {"id": "Scout", "team": "NS", "role": "MID", "league": "LCK", "ids": [("Scout", "KR1")]},
    {"id": "Lehends", "team": "NS", "role": "SUP", "league": "LCK", "ids": [("Lehends", "KR1")]},
    {"id": "Rich", "team": "DRX", "role": "TOP", "league": "LCK", "ids": [("Rich", "KR1")]},
    {"id": "Vincenzo", "team": "DRX", "role": "JNG", "league": "LCK", "ids": [("Vincenzo", "KR1")]},
    {"id": "Ucal", "team": "DRX", "role": "MID", "league": "LCK", "ids": [("Ucal", "KR1"), ("ucal", "KR1")]},
    {"id": "Andil", "team": "DRX", "role": "SUP", "league": "LCK", "ids": [("Andil", "KR1")]},
    {"id": "Casting", "team": "BRO", "role": "TOP", "league": "LCK", "ids": [("Casting", "KR1")]},
    {"id": "GIDEON", "team": "BRO", "role": "JNG", "league": "LCK", "ids": [("GIDEON", "KR1"), ("Gideon", "KR1")]},
    {"id": "Fisher", "team": "BRO", "role": "MID", "league": "LCK", "ids": [("Fisher", "KR1")]},
    {"id": "Teddy", "team": "BRO", "role": "BOT", "league": "LCK", "ids": [("Teddy", "KR1")]},
    {"id": "Namgung", "team": "BRO", "role": "SUP", "league": "LCK", "ids": [("Namgung", "KR1")]},
    {"id": "DuDu", "team": "DNF", "role": "TOP", "league": "LCK", "ids": [("DuDu", "KR1")]},
    {"id": "Pyosik", "team": "DNF", "role": "JNG", "league": "LCK", "ids": [("Pyosik", "KR1")]},
    {"id": "Clozer", "team": "DNF", "role": "MID", "league": "LCK", "ids": [("Clozer", "KR1")]},
    {"id": "deokdam", "team": "DNF", "role": "BOT", "league": "LCK", "ids": [("deokdam", "KR1")]},
    {"id": "Peter", "team": "DNF", "role": "SUP", "league": "LCK", "ids": [("Peter", "KR1")]},
    # LEC
    {"id": "BrokenBlade", "team": "G2", "role": "TOP", "league": "LEC", "ids": [("BrokenBlade", "EUW"), ("BrokenBlade", "G2")]},
    {"id": "SkewMond", "team": "G2", "role": "JNG", "league": "LEC", "ids": [("SkewMond", "EUW"), ("SkewMond", "G2")]},
    {"id": "Caps", "team": "G2", "role": "MID", "league": "LEC", "ids": [("Caps", "EUW"), ("Caps", "G2"), ("Caps", "2112")]},
    {"id": "Hans Sama", "team": "G2", "role": "BOT", "league": "LEC", "ids": [("Hans Sama", "EUW"), ("Hans Sama", "G2")]},
    {"id": "Labrov", "team": "G2", "role": "SUP", "league": "LEC", "ids": [("Labrov", "EUW"), ("Labrov", "G2")]},
    {"id": "Canna", "team": "KC", "role": "TOP", "league": "LEC", "ids": [("Canna", "EUW"), ("Canna", "KR1"), ("Canna", "KC")]},
    {"id": "Yike", "team": "KC", "role": "JNG", "league": "LEC", "ids": [("Yike", "EUW"), ("Yike", "KC")]},
    {"id": "kyeahoo", "team": "KC", "role": "MID", "league": "LEC", "ids": [("kyeahoo", "EUW"), ("kyeahoo", "KR1"), ("kyeahoo", "KC")]},
    {"id": "Caliste", "team": "KC", "role": "BOT", "league": "LEC", "ids": [("Caliste", "EUW"), ("Caliste", "KC")]},
    {"id": "Busio", "team": "KC", "role": "SUP", "league": "LEC", "ids": [("Busio", "EUW"), ("Busio", "KC")]},
    {"id": "Soboro", "team": "FNC", "role": "TOP", "league": "LEC", "ids": [("Soboro", "EUW"), ("Soboro", "KR1")]},
    {"id": "Razork", "team": "FNC", "role": "JNG", "league": "LEC", "ids": [("Razork", "EUW"), ("Razork", "FNC")]},
    {"id": "Vladi", "team": "FNC", "role": "MID", "league": "LEC", "ids": [("Vladi", "EUW"), ("Vladi", "FNC")]},
    {"id": "Upset", "team": "FNC", "role": "BOT", "league": "LEC", "ids": [("Upset", "EUW"), ("Upset", "FNC")]},
    {"id": "Lospa", "team": "FNC", "role": "SUP", "league": "LEC", "ids": [("Lospa", "EUW"), ("Lospa", "KR1")]},
    {"id": "Oscarinin", "team": "GX", "role": "TOP", "league": "LEC", "ids": [("Oscarinin", "EUW")]},
    {"id": "Isma", "team": "GX", "role": "JNG", "league": "LEC", "ids": [("Isma", "EUW")]},
    {"id": "Jackies", "team": "GX", "role": "MID", "league": "LEC", "ids": [("Jackies", "EUW")]},
    {"id": "Flakked", "team": "GX", "role": "BOT", "league": "LEC", "ids": [("Flakked", "EUW")]},
    {"id": "Jun", "team": "GX", "role": "SUP", "league": "LEC", "ids": [("Jun", "EUW"), ("Jun", "KR1")]},
    {"id": "Myrwn", "team": "MKOI", "role": "TOP", "league": "LEC", "ids": [("Myrwn", "EUW")]},
    {"id": "Elyoya", "team": "MKOI", "role": "JNG", "league": "LEC", "ids": [("Elyoya", "EUW")]},
    {"id": "Jojopyun", "team": "MKOI", "role": "MID", "league": "LEC", "ids": [("Jojopyun", "EUW")]},
    {"id": "Supa", "team": "MKOI", "role": "BOT", "league": "LEC", "ids": [("Supa", "EUW")]},
    {"id": "Alvaro", "team": "MKOI", "role": "SUP", "league": "LEC", "ids": [("Alvaro", "EUW")]},
    {"id": "Maynter", "team": "NAVI", "role": "TOP", "league": "LEC", "ids": [("Maynter", "EUW")]},
    {"id": "Rhilech", "team": "NAVI", "role": "JNG", "league": "LEC", "ids": [("Rhilech", "EUW")]},
    {"id": "Poby", "team": "NAVI", "role": "MID", "league": "LEC", "ids": [("Poby", "EUW"), ("Poby", "KR1")]},
    {"id": "SamD", "team": "NAVI", "role": "BOT", "league": "LEC", "ids": [("SamD", "EUW"), ("SamD", "KR1")]},
    {"id": "Parus", "team": "NAVI", "role": "SUP", "league": "LEC", "ids": [("Parus", "EUW")]},
    {"id": "Rooster", "team": "SHF", "role": "TOP", "league": "LEC", "ids": [("Rooster", "EUW")]},
    {"id": "Boukada", "team": "SHF", "role": "JNG", "league": "LEC", "ids": [("Boukada", "EUW")]},
    {"id": "nuc", "team": "SHF", "role": "MID", "league": "LEC", "ids": [("nuc", "EUW")]},
    {"id": "Paduck", "team": "SHF", "role": "BOT", "league": "LEC", "ids": [("Paduck", "EUW"), ("Paduck", "KR1")]},
    {"id": "Stend", "team": "SHF", "role": "SUP", "league": "LEC", "ids": [("Stend", "EUW")]},
    {"id": "Wunder", "team": "SK", "role": "TOP", "league": "LEC", "ids": [("Wunder", "EUW")]},
    {"id": "Skeanz", "team": "SK", "role": "JNG", "league": "LEC", "ids": [("Skeanz", "EUW")]},
    {"id": "SlowQ", "team": "SK", "role": "MID", "league": "LEC", "ids": [("SlowQ", "EUW"), ("SlowQ", "KR1")]},
    {"id": "Jopa", "team": "SK", "role": "BOT", "league": "LEC", "ids": [("Jopa", "EUW")]},
    {"id": "Mikyx", "team": "SK", "role": "SUP", "league": "LEC", "ids": [("Mikyx", "EUW")]},
    {"id": "Tracyn", "team": "TH", "role": "TOP", "league": "LEC", "ids": [("Tracyn", "EUW")]},
    {"id": "Daglas", "team": "TH", "role": "JNG", "league": "LEC", "ids": [("Daglas", "EUW")]},
    {"id": "Serin", "team": "TH", "role": "MID", "league": "LEC", "ids": [("Serin", "EUW")]},
    {"id": "Hype", "team": "TH", "role": "BOT", "league": "LEC", "ids": [("Hype", "EUW")]},
    {"id": "Way", "team": "TH", "role": "SUP", "league": "LEC", "ids": [("Way", "EUW")]},
    {"id": "Naak Nako", "team": "VIT", "role": "TOP", "league": "LEC", "ids": [("Naak Nako", "EUW")]},
    {"id": "Lyncas", "team": "VIT", "role": "JNG", "league": "LEC", "ids": [("Lyncas", "EUW")]},
    {"id": "FIESTA", "team": "VIT", "role": "MID", "league": "LEC", "ids": [("FIESTA", "EUW"), ("FIESTA", "KR1")]},
    {"id": "Carzzy", "team": "VIT", "role": "BOT", "league": "LEC", "ids": [("Carzzy", "EUW")]},
    {"id": "Fleshy", "team": "VIT", "role": "SUP", "league": "LEC", "ids": [("Fleshy", "EUW")]},
]


def regionals_for(league: str) -> list[str]:
    if league == "LEC":
        return ["europe", "asia"]
    return ["asia"]


def routing_for_regional(regional: str) -> tuple[str, str]:
    if regional == "europe":
        return "europe", "euw1"
    return "asia", "kr"


def _find_in_db(pro: dict) -> dict | None:
    needles = [pro["id"], *(pro.get("also") or []), *[name for name, _tag in pro.get("ids") or []]]
    label = f"{pro['team']} {pro['id']}"
    with db() as conn:
        for needle in needles:
            row = conn.execute(
                """
                SELECT puuid, game_name, tag_line, platform, regional
                FROM players
                WHERE game_name = ? COLLATE NOCASE
                   OR label = ? COLLATE NOCASE
                   OR label = ? COLLATE NOCASE
                LIMIT 1
                """,
                (needle, needle, label),
            ).fetchone()
            if row:
                return dict(row)
    return None


def _lookup_account(client: RiotClient, game_name: str, tag: str) -> dict | None:
    for regional in ("asia", "europe"):
        try:
            return client.account_by_riot_id(regional, game_name, tag)
        except RiotError as exc:
            if exc.status in (401, 403):
                raise
            continue
    return None


def resolve_pros(client: RiotClient, progress: Progress | None = None) -> list[dict]:
    emit = progress or (lambda _event: None)
    resolved: list[dict] = []
    for i, pro in enumerate(PROS, start=1):
        label = f"{pro['team']} {pro['id']}"
        hit = _find_in_db(pro)
        account = None
        source = "db"
        if hit:
            account = {
                "puuid": hit["puuid"],
                "gameName": hit["game_name"],
                "tagLine": hit["tag_line"],
            }
            regional, platform = routing_for_regional(
                "europe" if (hit.get("regional") == "europe" or (hit.get("platform") or "").startswith("euw")) else "asia"
            )
            if pro["league"] == "LEC":
                regional, platform = "europe", "euw1"
            else:
                regional, platform = "asia", "kr"
        else:
            source = "riot"
            for game_name, tag in pro.get("ids") or []:
                account = _lookup_account(client, game_name, tag)
                if account:
                    break
            regional, platform = ("europe", "euw1") if pro["league"] == "LEC" else ("asia", "kr")
        if not account:
            emit({"step": "miss", "message": f"{i}/{len(PROS)} {label} — no Riot ID"})
            resolved.append({**pro, "label": label, "ok": False})
            continue
        row = {
            **pro,
            "ok": True,
            "label": label,
            "puuid": account["puuid"],
            "game_name": account.get("gameName") or "",
            "tag_line": account.get("tagLine") or "",
            "platform": platform,
            "regional": regional,
            "source": source,
        }
        resolved.append(row)
        emit(
            {
                "step": "resolve",
                "message": f"{i}/{len(PROS)} {label} → {row['game_name']}#{row['tag_line']} ({source})",
            }
        )
    RESOLVE_PATH.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def ingest_pros(
    patch: str | None = None,
    progress: Progress | None = None,
    max_games: int = 200,
) -> dict:
    init_db()
    emit = progress or (lambda _event: None)
    from app.config import selected_patches
    patch = ",".join(selected_patches(patch))
    start_time = patch_start_unix(patch)
    client = RiotClient()
    dragon = DataDragon()
    ingested = 0
    skipped = 0
    errors: list[str] = []
    try:
        emit({"step": "resolve", "message": f"Resolving {len(PROS)} LCK/LEC accounts"})
        roster = resolve_pros(client, emit)
        ok = [p for p in roster if p.get("ok")]
        emit({"step": "resolve", "message": f"{len(ok)}/{len(roster)} accounts resolved"})
        match_jobs: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for player in ok:
            upsert_player(
                {
                    "puuid": player["puuid"],
                    "game_name": player["game_name"],
                    "tag_line": player["tag_line"],
                    "platform": player["platform"],
                    "regional": player["regional"],
                    "label": player["label"],
                    "profile_icon_id": None,
                    "summoner_level": None,
                }
            )
            for regional in regionals_for(player["league"]):
                platform = routing_for_regional(regional)[1]
                try:
                    ids = client.match_ids(
                        regional,
                        player["puuid"],
                        RANKED_SOLO_QUEUE,
                        max_games,
                        start_time=start_time,
                    )
                except RiotError as exc:
                    errors.append(f"{player['label']} {regional}: {exc}")
                    emit({"step": "error", "message": f"{player['label']} {regional}: {exc}"})
                    if exc.status in (401, 403):
                        raise
                    continue
                emit(
                    {
                        "step": "list",
                        "message": f"{player['label']} {regional}: {len(ids)} ranked this patch",
                    }
                )
                for match_id in ids:
                    key = (match_id, player["puuid"])
                    if key in seen:
                        continue
                    seen.add(key)
                    match_jobs.append((match_id, player["puuid"], platform))

        emit({"step": "list", "message": f"{len(match_jobs)} pro-game pulls queued"})
        for i, (match_id, puuid, platform) in enumerate(match_jobs, start=1):
            regional = "europe" if platform == "euw1" else "asia"
            try:
                added, skipped_here = _ingest_match_perspectives(
                    client,
                    dragon,
                    regional,
                    platform,
                    match_id,
                    patch,
                    {puuid},
                )
                ingested += added
                skipped += skipped_here
                emit(
                    {
                        "step": "stored" if added else "skip",
                        "match_id": match_id,
                        "index": i,
                        "total": len(match_jobs),
                        "message": (
                            f"{match_id} ({i}/{len(match_jobs)}) +{added}"
                            if added
                            else f"{match_id} ({i}/{len(match_jobs)}) already stored"
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
        set_meta("last_pro_sync", now_iso())
        result = {
            "step": "done",
            "resolved": len(ok),
            "missing": len(roster) - len(ok),
            "ingested": ingested,
            "skipped": skipped,
            "queued": len(match_jobs),
            "patch": patch,
            "errors": errors[:12],
            "message": (
                f"Done. {len(ok)} accounts, {ingested} new perspectives, "
                f"{skipped} skipped, {len(roster) - len(ok)} IDs unresolved."
            ),
        }
        emit(result)
        return result
    except RiotError as exc:
        emit({"step": "error", "message": str(exc)})
        raise
    finally:
        client.close()
