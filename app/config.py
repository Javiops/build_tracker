from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import os
import sys

# Frozen (PyInstaller beta bundle): code lives under _internal, but web/, data/
# and scripts/ ship next to the exe — anchor ROOT there so paths keep working.
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_MATCH_V5_DIR = DATA_DIR / "raw_match_v5"
DB_PATH = DATA_DIR / "tracker.db"
WEB_DIR = ROOT / "web"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "").strip()
# Retain immutable Match-V5 source responses locally so a future replay change
# does not require an avoidable second API crawl. Set to 0 only for a
# deliberately ephemeral/debug run; cache files are never shipped or committed.
RAW_MATCH_V5_CACHE = os.getenv("RAW_MATCH_V5_CACHE", "1").strip() not in {"0", "false", "False"}

# Faker's main ranked account (KR).
FAKER = {
    "game_name": "Hide on bush",
    "tag_line": "KR1",
    "platform": "kr",
    "regional": "asia",
    "label": "Faker",
}

RANKED_SOLO_QUEUE = 420
SOLO_QUEUE_NAME = "RANKED_SOLO_5x5"
DEFAULT_MATCH_COUNT = 20
LADDER_SIZE = 500
LADDER_REGIONS = {
    "kr": {"platform": "kr", "regional": "asia", "label": "KR"},
    "euw": {"platform": "euw1", "regional": "europe", "label": "EUW"},
}
MIN_REQUEST_GAP_S = 1.3
MAX_REQUESTS_PER_2MIN = 95
# League shop visits are short; a gap longer than this starts a new visit.
SHOP_IDLE_MS = 10_000

# Ranked solo patch go-live (UTC). Used as match-v5 startTime; DTO patch still filtered.
PATCH_STARTS_UTC = {
    "16.17": datetime(2026, 8, 26, tzinfo=timezone.utc),
    # Conservative request lower bound for the regional rollout, not a claim
    # about the exact KR/EUW activation hour. DTO gameVersion decides inclusion.
    "16.18": datetime(2026, 9, 9, tzinfo=timezone.utc),
}
CURRENT_INGEST_PATCH = "16.18"
# Reviewed against Riot's versions.json on 2026-09-10. Never use latest static
# data to reinterpret an older match. Review this mapping on each patch.
PATCH_DATA_VERSIONS = {"16.17": "16.17.1", "16.18": "16.18.1"}


def selected_patches(patch: str | None = None) -> tuple[str, ...]:
    patches = tuple(dict.fromkeys((patch or CURRENT_INGEST_PATCH).split(",")))
    if not patches or any(p not in PATCH_DATA_VERSIONS or p not in PATCH_STARTS_UTC for p in patches):
        raise ValueError(f"Unreviewed ingest patch selection: {patch!r}")
    return patches


def patch_start_unix(patch: str) -> int:
    return min(int(PATCH_STARTS_UTC[p].timestamp()) for p in selected_patches(patch))


def local_day_start_unix() -> int:
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp())
