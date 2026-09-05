"""Backfill runes + summoner spells into games.participants_json.

Older ingests stored only champion/position/KDA per participant; the rune
features need keystone, secondary tree, and summoner spells from the match-v5
DTO. One request per stored game (~4.5h for the full DB at the dev-key rate
limit). Resumable: games whose participants already carry keystone_id are
skipped, so re-running after an interruption continues where it left off.
Vanished matches (404) are stamped with keystone_id=0 so they are not refetched.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import db
from app.reconstruct import perk_fields
from app.riot import RiotClient, RiotError

REGIONAL_BY_PREFIX = {"KR": "asia", "EUW1": "europe", "EUN1": "europe", "NA1": "americas"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def pending_games(conn) -> list[str]:
    rows = conn.execute(
        """
        SELECT match_id FROM games
        WHERE participants_json NOT LIKE '%"keystone_id"%'
        ORDER BY game_creation DESC
        """
    ).fetchall()
    return [r["match_id"] for r in rows]


def apply_perks(participants: list[dict], dto: dict | None) -> None:
    by_puuid = {}
    if dto:
        for p in (dto.get("info") or {}).get("participants") or []:
            by_puuid[p.get("puuid")] = perk_fields(p)
    for row in participants:
        row.update(by_puuid.get(row.get("puuid")) or {"keystone_id": 0, "sub_style": 0, "summ1": 0, "summ2": 0})


def main() -> int:
    client = RiotClient()
    with db() as conn:
        todo = pending_games(conn)
    log(f"{len(todo)} games missing runes (~{len(todo) * 1.3 / 3600:.1f}h at the rate limit)")
    done = failed = 0
    started = time.monotonic()
    try:
        for match_id in todo:
            regional = REGIONAL_BY_PREFIX.get(match_id.split("_")[0], "europe")
            try:
                dto = client.match(regional, match_id)
            except RiotError as exc:
                if exc.status in (401, 403):
                    log(f"API key rejected after {done} games — renew RIOT_API_KEY and re-run to resume.")
                    return 1
                failed += 1
                log(f"{match_id} FAILED: {exc}")
                continue
            with db() as conn:
                row = conn.execute(
                    "SELECT participants_json FROM games WHERE match_id = ?", (match_id,)
                ).fetchone()
                if not row:
                    continue
                participants = json.loads(row["participants_json"])
                apply_perks(participants, dto)
                conn.execute(
                    "UPDATE games SET participants_json = ? WHERE match_id = ?",
                    (json.dumps(participants), match_id),
                )
            done += 1
            if done % 200 == 0:
                rate = done / max(time.monotonic() - started, 1)
                eta_h = (len(todo) - done) / max(rate, 1e-6) / 3600
                log(f"{done}/{len(todo)} backfilled ({failed} failed), eta {eta_h:.1f}h")
    finally:
        client.close()
    log(f"done: {done} backfilled, {failed} failed, {(time.monotonic() - started) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
