from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def db():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS players (
                puuid TEXT PRIMARY KEY,
                game_name TEXT NOT NULL,
                tag_line TEXT NOT NULL,
                platform TEXT NOT NULL,
                regional TEXT NOT NULL,
                label TEXT NOT NULL,
                profile_icon_id INTEGER,
                summoner_level INTEGER
            );
            """
        )
        _ensure_matches_schema(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(shop_visits)").fetchall()}
        if cols and "board_json" not in cols:
            conn.execute("ALTER TABLE shop_visits ADD COLUMN board_json TEXT NOT NULL DEFAULT '[]'")
        _ensure_games_schema(conn)
        _unify_perspectives_into_games(conn)


def _matches_pk_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(matches)").fetchall()
    ranked = sorted((row[5], row[1]) for row in rows if row[5])
    return [name for _pk, name in ranked]


def _ensure_matches_schema(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='matches'"
    ).fetchone()
    if not exists:
        conn.executescript(_MATCHES_SQL + _VISITS_SQL + _MATCH_INDEXES)
        return
    pk = _matches_pk_columns(conn)
    if pk == ["match_id", "puuid"]:
        conn.executescript(_MATCH_INDEXES)
        return
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("ALTER TABLE matches RENAME TO matches_old")
    visits_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='shop_visits'"
    ).fetchone()
    if visits_exist:
        conn.execute("ALTER TABLE shop_visits RENAME TO shop_visits_old")
    conn.executescript(_MATCHES_SQL + _VISITS_SQL)
    conn.execute("INSERT INTO matches SELECT * FROM matches_old")
    if visits_exist:
        conn.execute("INSERT INTO shop_visits SELECT * FROM shop_visits_old")
        conn.execute("DROP TABLE shop_visits_old")
    conn.execute("DROP TABLE matches_old")
    conn.executescript(_MATCH_INDEXES)
    conn.execute("PRAGMA foreign_keys = ON")


_MATCHES_SQL = """
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT NOT NULL,
                puuid TEXT NOT NULL,
                game_creation INTEGER NOT NULL,
                game_duration INTEGER NOT NULL,
                game_version TEXT NOT NULL,
                patch TEXT NOT NULL,
                queue_id INTEGER NOT NULL,
                champion_id INTEGER NOT NULL,
                champion_name TEXT NOT NULL,
                team_position TEXT,
                team_id INTEGER NOT NULL,
                win INTEGER NOT NULL,
                kills INTEGER NOT NULL,
                deaths INTEGER NOT NULL,
                assists INTEGER NOT NULL,
                participants_json TEXT NOT NULL,
                PRIMARY KEY (match_id, puuid),
                FOREIGN KEY (puuid) REFERENCES players(puuid)
            );
"""

_VISITS_SQL = """
            CREATE TABLE IF NOT EXISTS shop_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                puuid TEXT NOT NULL,
                visit_index INTEGER NOT NULL,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER NOT NULL,
                gold INTEGER,
                level INTEGER,
                cs INTEGER,
                kills INTEGER NOT NULL DEFAULT 0,
                deaths INTEGER NOT NULL DEFAULT 0,
                assists INTEGER NOT NULL DEFAULT 0,
                inventory_before_json TEXT NOT NULL,
                inventory_after_json TEXT NOT NULL,
                bought_json TEXT NOT NULL,
                consumed_json TEXT NOT NULL,
                board_json TEXT NOT NULL DEFAULT '[]',
                allies_json TEXT NOT NULL,
                enemies_json TEXT NOT NULL,
                objectives_json TEXT NOT NULL,
                FOREIGN KEY (match_id, puuid) REFERENCES matches(match_id, puuid)
            );
"""

_MATCH_INDEXES = """
            CREATE INDEX IF NOT EXISTS idx_matches_champ ON matches(champion_id);
            CREATE INDEX IF NOT EXISTS idx_matches_puuid ON matches(puuid);
            CREATE INDEX IF NOT EXISTS idx_matches_patch ON matches(patch);
            CREATE INDEX IF NOT EXISTS idx_visits_match ON shop_visits(match_id);
            CREATE INDEX IF NOT EXISTS idx_visits_player ON shop_visits(match_id, puuid);
"""

_GAMES_SQL = """
            CREATE TABLE IF NOT EXISTS games (
                match_id TEXT PRIMARY KEY,
                game_creation INTEGER NOT NULL,
                game_duration INTEGER NOT NULL,
                game_version TEXT NOT NULL,
                patch TEXT NOT NULL,
                queue_id INTEGER NOT NULL,
                win_team INTEGER NOT NULL,
                participants_json TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'unified'
            );
            CREATE TABLE IF NOT EXISTS game_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                ts INTEGER NOT NULL,
                ts_end INTEGER,
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (match_id) REFERENCES games(match_id)
            );
            CREATE TABLE IF NOT EXISTS game_players (
                match_id TEXT NOT NULL,
                puuid TEXT NOT NULL,
                champion_id INTEGER,
                champion_name TEXT,
                team_id INTEGER,
                team_position TEXT,
                win INTEGER,
                PRIMARY KEY (match_id, puuid)
            );
            CREATE INDEX IF NOT EXISTS idx_events_match ON game_events(match_id, ts);
            CREATE INDEX IF NOT EXISTS idx_game_players_puuid ON game_players(puuid);
"""


def _ensure_games_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_GAMES_SQL)


def _unify_perspectives_into_games(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'narrator_unified'").fetchone()
    if row:
        return
    match_ids = [r["match_id"] for r in conn.execute("SELECT DISTINCT match_id FROM matches")]
    for match_id in match_ids:
        if conn.execute("SELECT 1 FROM games WHERE match_id = ?", (match_id,)).fetchone():
            continue
        game, events = _narrator_from_perspectives(conn, match_id)
        if game:
            _write_game(conn, game, events)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('narrator_unified', ?)",
        (now_iso(),),
    )


def _narrator_from_perspectives(conn: sqlite3.Connection, match_id: str) -> tuple[dict | None, list[dict]]:
    base = conn.execute("SELECT * FROM matches WHERE match_id = ? LIMIT 1", (match_id,)).fetchone()
    if not base:
        return None, []
    participants = json.loads(base["participants_json"])
    win_team = next((p["team_id"] for p in participants if p.get("win")), 100)
    game = {
        "match_id": match_id,
        "game_creation": base["game_creation"],
        "game_duration": base["game_duration"],
        "game_version": base["game_version"],
        "patch": base["patch"],
        "queue_id": base["queue_id"],
        "win_team": win_team,
        "participants_json": base["participants_json"],
        "source": "unified",
    }
    seen: set[tuple] = set()
    shops: list[dict] = []
    for row in conn.execute(
        "SELECT * FROM shop_visits WHERE match_id = ? ORDER BY ts_start ASC, id ASC",
        (match_id,),
    ):
        key = (row["puuid"], row["ts_start"], row["ts_end"])
        if key in seen:
            continue
        seen.add(key)
        owner = next((p for p in participants if p["puuid"] == row["puuid"]), {})
        shops.append(
            {
                "type": "shop",
                "ts": row["ts_start"],
                "ts_end": row["ts_end"],
                "puuid": row["puuid"],
                "participant_id": owner.get("participant_id"),
                "champion_name": owner.get("champion_name") or "",
                "champion_id": owner.get("champion_id"),
                "team_id": owner.get("team_id"),
                "team_position": owner.get("team_position") or "",
                "riot_id": owner.get("riot_id") or "",
                "gold": row["gold"],
                "level": row["level"],
                "cs": row["cs"],
                "kills": row["kills"],
                "deaths": row["deaths"],
                "assists": row["assists"],
                "inventory_before": json.loads(row["inventory_before_json"] or "[]"),
                "inventory_after": json.loads(row["inventory_after_json"] or "[]"),
                "bought": json.loads(row["bought_json"] or "[]"),
                "consumed": json.loads(row["consumed_json"] or "[]"),
                "board": json.loads(row["board_json"] or "[]"),
                "score": _absolute_score(json.loads(row["objectives_json"] or "{}"), owner.get("team_id")),
            }
        )
    events = shops
    for idx, event in enumerate(events, start=1):
        event["event_index"] = idx
    return game, events


def _absolute_score(relative: dict, team_id: int | None) -> dict:
    empty = {"towers": 0, "dragons": 0, "barons": 0, "heralds": 0, "voidgrubs": 0}
    ally = relative.get("ally") or empty
    enemy = relative.get("enemy") or empty
    if team_id == 200:
        return {100: enemy, 200: ally}
    return {100: ally, 200: enemy}


def _write_game(conn: sqlite3.Connection, game: dict, events: list[dict]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO games(
            match_id, game_creation, game_duration, game_version, patch, queue_id,
            win_team, participants_json, source
        ) VALUES(
            :match_id, :game_creation, :game_duration, :game_version, :patch, :queue_id,
            :win_team, :participants_json, :source
        )
        """,
        game,
    )
    conn.execute("DELETE FROM game_events WHERE match_id = ?", (game["match_id"],))
    conn.execute("DELETE FROM game_players WHERE match_id = ?", (game["match_id"],))
    conn.executemany(
        """
        INSERT INTO game_events(match_id, event_index, ts, ts_end, type, payload_json)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        [
            (
                game["match_id"],
                ev.get("event_index") or i,
                ev.get("ts") or 0,
                ev.get("ts_end"),
                ev["type"],
                json.dumps(ev),
            )
            for i, ev in enumerate(events, start=1)
            if ev.get("type") == "shop"
        ],
    )
    players = json.loads(game["participants_json"])
    conn.executemany(
        """
        INSERT OR REPLACE INTO game_players(
            match_id, puuid, champion_id, champion_name, team_id, team_position, win
        ) VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                game["match_id"],
                p["puuid"],
                p.get("champion_id"),
                p.get("champion_name"),
                p.get("team_id"),
                p.get("team_position"),
                int(bool(p.get("win"))),
            )
            for p in players
        ],
    )


def insert_game(game: dict, events: list[dict]) -> None:
    with db() as conn:
        _write_game(conn, game, events)


def game_exists(match_id: str) -> bool:
    with db() as conn:
        row = conn.execute("SELECT 1 FROM games WHERE match_id = ?", (match_id,)).fetchone()
        return row is not None


def set_meta(key: str, value: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_meta(key: str) -> str | None:
    with db() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def upsert_player(player: dict) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO players(puuid, game_name, tag_line, platform, regional, label, profile_icon_id, summoner_level)
            VALUES(:puuid, :game_name, :tag_line, :platform, :regional, :label, :profile_icon_id, :summoner_level)
            ON CONFLICT(puuid) DO UPDATE SET
                game_name=excluded.game_name,
                tag_line=excluded.tag_line,
                label=excluded.label,
                profile_icon_id=excluded.profile_icon_id,
                summoner_level=excluded.summoner_level
            """,
            player,
        )


def match_exists(match_id: str, puuid: str | None = None) -> bool:
    with db() as conn:
        if puuid:
            row = conn.execute(
                "SELECT 1 FROM matches WHERE match_id = ? AND puuid = ?",
                (match_id, puuid),
            ).fetchone()
        else:
            row = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,)).fetchone()
        return row is not None


def match_puuids(match_id: str) -> set[str]:
    with db() as conn:
        rows = conn.execute("SELECT puuid FROM matches WHERE match_id = ?", (match_id,)).fetchall()
        return {row["puuid"] for row in rows}


def match_participants(match_id: str) -> list[dict]:
    with db() as conn:
        row = conn.execute(
            "SELECT participants_json FROM matches WHERE match_id = ? LIMIT 1",
            (match_id,),
        ).fetchone()
        if not row:
            return []
        return json.loads(row["participants_json"])


def insert_match_and_visits(match: dict, visits: list[dict]) -> None:
    insert_perspectives([(match, visits)])


def insert_reconstruction(game: dict, events: list[dict], entries: list[tuple[dict, list[dict]]]) -> None:
    """Publish a reconstructed game and its requested perspectives atomically."""
    if any(m["match_id"] != game["match_id"] or m["patch"] != game["patch"] for m, _ in entries):
        raise ValueError("Reconstruction contains mismatched game IDs or patches")
    with db() as conn:
        _write_game(conn, game, events)
        insert_perspectives(entries, _conn=conn)


def insert_perspectives(entries: list[tuple[dict, list[dict]]], *, _conn=None) -> None:
    if not entries:
        return
    with (nullcontext(_conn) if _conn is not None else db()) as conn:
        for match, visits in entries:
            conn.execute(
                """
                INSERT OR REPLACE INTO matches(
                    match_id, puuid, game_creation, game_duration, game_version, patch, queue_id,
                    champion_id, champion_name, team_position, team_id, win, kills, deaths, assists, participants_json
                ) VALUES(
                    :match_id, :puuid, :game_creation, :game_duration, :game_version, :patch, :queue_id,
                    :champion_id, :champion_name, :team_position, :team_id, :win, :kills, :deaths, :assists, :participants_json
                )
                """,
                match,
            )
            conn.execute(
                "DELETE FROM shop_visits WHERE match_id = ? AND puuid = ?",
                (match["match_id"], match["puuid"]),
            )
            if visits:
                conn.executemany(
                    """
                    INSERT INTO shop_visits(
                        match_id, puuid, visit_index, ts_start, ts_end, gold, level, cs,
                        kills, deaths, assists, inventory_before_json, inventory_after_json,
                        bought_json, consumed_json, board_json, allies_json, enemies_json, objectives_json
                    ) VALUES(
                        :match_id, :puuid, :visit_index, :ts_start, :ts_end, :gold, :level, :cs,
                        :kills, :deaths, :assists, :inventory_before_json, :inventory_after_json,
                        :bought_json, :consumed_json, :board_json, :allies_json, :enemies_json, :objectives_json
                    )
                    """,
                    visits,
                )


def insert_match_and_decisions(match: dict, visits: list[dict]) -> None:
    insert_match_and_visits(match, visits)


PLAYER_ALIASES = {
    "oner": "오 너",
    "faker": "Hide on bush",
    "zeus": "제우스",
    "gumayusi": "구마유시",
    "keria": "케리아",
    "chovy": "쵸비",
    "canyon": "캐니언",
    "doran": "도란",
    "showmaker": "쇼메이커",
    "bdd": "비디디",
}


def list_players(query: str | None = None, limit: int = 60) -> list[dict]:
    raw = (query or "").strip()
    q = PLAYER_ALIASES.get(raw.lower(), raw)
    like = f"%{q}%"
    with db() as conn:
        rows = conn.execute(
            """
            SELECT p.puuid, p.game_name, p.tag_line, p.label, p.profile_icon_id,
                   COUNT(DISTINCT COALESCE(gp.match_id, m.match_id)) AS games
            FROM players p
            LEFT JOIN game_players gp ON gp.puuid = p.puuid
            LEFT JOIN matches m ON m.puuid = p.puuid
            WHERE ? = ''
               OR p.game_name LIKE ?
               OR p.tag_line LIKE ?
               OR p.label LIKE ?
            GROUP BY p.puuid
            HAVING games > 0
            ORDER BY games DESC, p.game_name ASC
            LIMIT ?
            """,
            (q, like, like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def list_matches(puuid: str | None = None) -> list[dict]:
    with db() as conn:
        has_games = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
        ).fetchone()
        if has_games and conn.execute("SELECT 1 FROM games LIMIT 1").fetchone():
            if puuid:
                rows = conn.execute(
                    """
                    SELECT g.match_id, g.game_creation, g.game_duration, g.patch, g.win_team,
                           gp.champion_id, gp.champion_name, gp.team_position, gp.team_id,
                           gp.win, g.participants_json
                    FROM games g
                    JOIN game_players gp ON gp.match_id = g.match_id
                    WHERE gp.puuid = ?
                    ORDER BY g.game_creation DESC
                    """,
                    (puuid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT match_id, game_creation, game_duration, patch, win_team, participants_json
                    FROM games
                    ORDER BY game_creation DESC
                    """
                ).fetchall()
            rows = [dict(r) for r in rows]
            counts = {
                r["match_id"]: r["n"]
                for r in conn.execute(
                    """
                    SELECT match_id, COUNT(DISTINCT json_extract(payload_json, '$.puuid')) AS n
                    FROM game_events
                    WHERE type = 'shop'
                    GROUP BY match_id
                    """
                )
            }
            for row in rows:
                row["shoppers"] = counts.get(row["match_id"], 0)
            if not puuid:
                rows.sort(key=lambda r: (-(r.get("shoppers") or 0), -(r.get("game_creation") or 0)))
            return rows
        if not puuid:
            return []
        rows = conn.execute(
            """
            SELECT match_id, game_creation, game_duration, patch, champion_id, champion_name,
                   team_position, win, kills, deaths, assists
            FROM matches
            WHERE puuid = ?
            ORDER BY game_creation DESC
            """,
            (puuid,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_game(match_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM games WHERE match_id = ?", (match_id,)).fetchone()
        if not row:
            return None
        game = dict(row)
        game["participants"] = json.loads(game.pop("participants_json"))
        events = []
        for ev in conn.execute(
            "SELECT payload_json FROM game_events WHERE match_id = ? AND type = 'shop' ORDER BY event_index ASC, ts ASC",
            (match_id,),
        ):
            events.append(json.loads(ev["payload_json"]))
        game["events"] = events
        game["visits"] = [e for e in events if e.get("type") == "shop"]
        game["shoppers"] = len({e.get("puuid") for e in game["visits"] if e.get("puuid")})
        _refresh_item_flags(game["visits"])
        return game


def get_match(match_id: str, puuid: str | None = None) -> dict | None:
    game = get_game(match_id)
    if game:
        if puuid:
            me = next((p for p in game["participants"] if p["puuid"] == puuid), None)
            if me:
                game["champion_name"] = me["champion_name"]
                game["champion_id"] = me["champion_id"]
                game["team_id"] = me["team_id"]
                game["win"] = int(bool(me.get("win")))
                game["kills"] = me.get("kills") or 0
                game["deaths"] = me.get("deaths") or 0
                game["assists"] = me.get("assists") or 0
        return game
    return _get_match_legacy(match_id, puuid)


def _get_match_legacy(match_id: str, puuid: str | None = None) -> dict | None:
    with db() as conn:
        if puuid:
            row = conn.execute(
                "SELECT * FROM matches WHERE match_id = ? AND puuid = ?",
                (match_id, puuid),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM matches WHERE match_id = ? LIMIT 1", (match_id,)).fetchone()
        if not row:
            return None
        match = dict(row)
        match["participants"] = json.loads(match.pop("participants_json"))
        rows = conn.execute(
            """
            SELECT * FROM shop_visits WHERE match_id = ? AND puuid = ? ORDER BY ts_end ASC, id ASC
            """,
            (match_id, match["puuid"]),
        ).fetchall()
        visits = []
        for row in rows:
            item = dict(row)
            for key in (
                "inventory_before_json",
                "inventory_after_json",
                "bought_json",
                "consumed_json",
                "board_json",
                "allies_json",
                "enemies_json",
                "objectives_json",
            ):
                if key not in item or item[key] is None:
                    item[key.replace("_json", "")] = []
                    item.pop(key, None)
                    continue
                item[key.replace("_json", "")] = json.loads(item.pop(key))
            visits.append(item)
        match["visits"] = visits
        _refresh_item_flags(match["visits"])
        return match


def _refresh_item_flags(visits: list[dict]) -> None:
    from app.ddragon import default_dragon

    dragon = default_dragon()

    def touch(items):
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            if "item_id" in it:
                it["skip"] = dragon.classify(it["item_id"])["skip"]
            if "items" in it:
                touch(it["items"])

    for visit in visits:
        for key in ("inventory_before", "inventory_after", "bought", "consumed", "board"):
            touch(visit.get(key))
            if key == "board" and isinstance(visit.get(key), list):
                for row in visit[key]:
                    if isinstance(row, dict):
                        touch(row.get("items"))


def champion_stats(puuid: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT champion_id, champion_name,
                   COUNT(*) AS games,
                   SUM(win) AS wins,
                   ROUND(AVG(kills), 1) AS avg_kills,
                   ROUND(AVG(deaths), 1) AS avg_deaths,
                   ROUND(AVG(assists), 1) AS avg_assists
            FROM matches
            WHERE puuid = ?
            GROUP BY champion_id
            ORDER BY games DESC, champion_name ASC
            """,
            (puuid,),
        ).fetchall()
        return [dict(r) for r in rows]


def build_for_champion(puuid: str, champion_id: int, vs_champion_id: int | None = None) -> dict:
    with db() as conn:
        matches = conn.execute(
            "SELECT * FROM matches WHERE puuid = ? AND champion_id = ? ORDER BY game_creation DESC",
            (puuid, champion_id),
        ).fetchall()
        filtered = []
        for m in matches:
            match = dict(m)
            match["participants"] = json.loads(match["participants_json"])
            if vs_champion_id:
                enemies = [
                    p
                    for p in match["participants"]
                    if p["team_id"] != match["team_id"]
                ]
                if not any(p["champion_id"] == vs_champion_id for p in enemies):
                    continue
            filtered.append(match)

        match_ids = [m["match_id"] for m in filtered]
        if not match_ids:
            return {
                "champion_id": champion_id,
                "games": 0,
                "wins": 0,
                "first_completed": [],
                "second_completed": [],
                "boots": [],
                "matches": [],
            }

        placeholders = ",".join("?" * len(match_ids))
        decisions = conn.execute(
            f"""
            SELECT match_id, item_id, item_name, is_completed, is_boots, completed_index, ts_ms
            FROM decisions
            WHERE puuid = ? AND match_id IN ({placeholders})
            ORDER BY ts_ms ASC
            """,
            [puuid, *match_ids],
        ).fetchall()

        first: dict[tuple[int, str], int] = {}
        second: dict[tuple[int, str], int] = {}
        boots: dict[tuple[int, str], int] = {}
        for d in decisions:
            key = (d["item_id"], d["item_name"])
            if d["is_boots"]:
                boots[key] = boots.get(key, 0) + 1
            if d["is_completed"] and d["completed_index"] == 1:
                first[key] = first.get(key, 0) + 1
            if d["is_completed"] and d["completed_index"] == 2:
                second[key] = second.get(key, 0) + 1

        games = len(filtered)
        wins = sum(m["win"] for m in filtered)

        def rank(counter: dict[tuple[int, str], int]) -> list[dict]:
            ranked = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
            return [
                {
                    "item_id": k[0],
                    "item_name": k[1],
                    "count": n,
                    "pct": round(100 * n / games, 1),
                }
                for k, n in ranked
            ]

        return {
            "champion_id": champion_id,
            "champion_name": filtered[0]["champion_name"] if filtered else "",
            "games": games,
            "wins": wins,
            "first_completed": rank(first),
            "second_completed": rank(second),
            "boots": rank(boots),
            "matches": [
                {
                    "match_id": m["match_id"],
                    "game_creation": m["game_creation"],
                    "patch": m["patch"],
                    "win": m["win"],
                    "kills": m["kills"],
                    "deaths": m["deaths"],
                    "assists": m["assists"],
                    "team_position": m["team_position"],
                }
                for m in filtered
            ],
        }


def summary(puuid: str | None) -> dict:
    with db() as conn:
        player = None
        if puuid:
            row = conn.execute("SELECT * FROM players WHERE puuid = ?", (puuid,)).fetchone()
            player = dict(row) if row else None
        try:
            unique_games = conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"]
            event_count = conn.execute(
                "SELECT COUNT(*) AS n FROM game_events WHERE type = 'shop'"
            ).fetchone()["n"]
        except sqlite3.OperationalError:
            unique_games = conn.execute("SELECT COUNT(DISTINCT match_id) AS n FROM matches").fetchone()["n"]
            event_count = 0
        match_count = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
        player_count = conn.execute("SELECT COUNT(*) AS n FROM players").fetchone()["n"]
        try:
            visit_count = event_count or conn.execute("SELECT COUNT(*) AS n FROM shop_visits").fetchone()["n"]
        except sqlite3.OperationalError:
            visit_count = event_count
        last_sync = get_meta("last_sync")
        return {
            "player": player,
            "match_count": match_count,
            "unique_games": unique_games,
            "player_count": player_count,
            "visit_count": visit_count,
            "decision_count": visit_count,
            "last_sync": last_sync,
        }


def export_all(puuid: str | None) -> dict:
    data = summary(puuid)
    if not puuid:
        data["matches"] = []
        return data
    data["matches"] = [get_match(m["match_id"], puuid) for m in list_matches(puuid)]
    return data


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_path() -> Path:
    return DB_PATH
