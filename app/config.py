from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "tracker.db"
WEB_DIR = ROOT / "web"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "").strip()

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
}


def patch_start_unix(patch: str) -> int:
    start = PATCH_STARTS_UTC.get(patch)
    if start:
        return int(start.timestamp())
    return int((datetime.now(timezone.utc) - timedelta(days=16)).timestamp())


def local_day_start_unix() -> int:
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp())
