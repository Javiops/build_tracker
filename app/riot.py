from __future__ import annotations

import gzip
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.config import (
    MAX_REQUESTS_PER_2MIN,
    MIN_REQUEST_GAP_S,
    RAW_MATCH_V5_CACHE,
    RAW_MATCH_V5_DIR,
    RIOT_API_KEY,
)


RAW_ARCHIVE_SCHEMA = 1


class RawMatchArchive:
    """Local, immutable cache for raw Match-V5 source responses.

    Reconstructed rows are lossy by design.  Retaining the raw match and
    timeline payloads means a parser/feature correction can be replayed
    without making another rate-limited API crawl.  Each endpoint is cached
    independently because callers sometimes need only a match DTO.
    """

    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled

    @staticmethod
    def _safe_match_id(match_id: str) -> str:
        if not match_id or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in match_id):
            raise ValueError(f"unsafe Match-V5 match id: {match_id!r}")
        return match_id

    def _path(self, endpoint: str, match_id: str) -> Path:
        return self.root / endpoint / f"{self._safe_match_id(match_id)}.json.gz"

    def read(self, endpoint: str, match_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            with gzip.open(self._path(endpoint, match_id), "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)
            if (
                envelope.get("schema") != RAW_ARCHIVE_SCHEMA
                or envelope.get("endpoint") != endpoint
                or envelope.get("match_id") != match_id
                or not isinstance(envelope.get("payload"), dict)
            ):
                return None
            return envelope["payload"]
        except (OSError, ValueError, json.JSONDecodeError):
            # A partial/corrupt cache entry is never authoritative: fall back
            # to the Riot endpoint, then replace it atomically.
            return None

    def write(self, endpoint: str, match_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(endpoint, match_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        envelope = {
            "schema": RAW_ARCHIVE_SCHEMA,
            "endpoint": endpoint,
            "match_id": match_id,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        try:
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(envelope, handle, separators=(",", ":"), ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


class RiotError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RiotClient:
    def __init__(
        self,
        api_key: str | None = None,
        raw_cache_dir: Path | None = None,
        raw_cache_enabled: bool | None = None,
    ):
        self.api_key = (api_key or RIOT_API_KEY).strip()
        if not self.api_key or self.api_key.startswith("RGAPI-xxxx"):
            raise RiotError("Missing RIOT_API_KEY. Put your key in a .env file.")
        self._last_request = 0.0
        self._window: deque[float] = deque()
        self._client = httpx.Client(timeout=20.0, headers={"X-Riot-Token": self.api_key})
        self._raw_archive = RawMatchArchive(
            raw_cache_dir or RAW_MATCH_V5_DIR,
            RAW_MATCH_V5_CACHE if raw_cache_enabled is None else raw_cache_enabled,
        )

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        gap = MIN_REQUEST_GAP_S - (time.monotonic() - self._last_request)
        if gap > 0:
            time.sleep(gap)
        now = time.monotonic()
        while self._window and now - self._window[0] >= 120:
            self._window.popleft()
        if len(self._window) >= MAX_REQUESTS_PER_2MIN:
            time.sleep(120 - (now - self._window[0]) + 0.05)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(8):
            self._throttle()
            response = self._client.get(url, params=params)
            self._last_request = time.monotonic()
            self._window.append(self._last_request)
            if response.status_code == 429:
                retry = float(response.headers.get("Retry-After", "2"))
                time.sleep(max(retry, 1.5))
                continue
            if response.status_code in (401, 403):
                raise RiotError(
                    "Riot rejected the API key. Dev keys expire every 24 hours.",
                    response.status_code,
                )
            if response.status_code == 404:
                return None
            if response.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise RiotError(
                    f"Riot API {response.status_code}: {response.text[:200]}",
                    response.status_code,
                )
            return response.json()
        raise RiotError("Riot API kept failing after retries.")

    def account_by_riot_id(self, regional: str, game_name: str, tag_line: str) -> dict:
        name = quote(game_name, safe="")
        tag = quote(tag_line, safe="")
        url = f"https://{regional}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
        data = self._get(url)
        if not data:
            raise RiotError(f"Account {game_name}#{tag_line} not found.", 404)
        return data

    def account_by_puuid(self, regional: str, puuid: str) -> dict | None:
        url = f"https://{regional}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"
        return self._get(url)

    def summoner_by_puuid(self, platform: str, puuid: str) -> dict | None:
        url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._get(url)

    def match_ids(
        self,
        regional: str,
        puuid: str,
        queue: int,
        count: int,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        url = f"https://{regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        ids: list[str] = []
        start = 0
        # count <= 0 means "all matches in the window": page until a short chunk.
        remaining = count if count > 0 else None
        while remaining is None or remaining > 0:
            batch = 100 if remaining is None else min(remaining, 100)
            params: dict[str, Any] = {"queue": queue, "start": start, "count": batch}
            if start_time:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
            chunk = self._get(url, params=params) or []
            ids.extend(chunk)
            if len(chunk) < batch:
                break
            start += batch
            if remaining is not None:
                remaining -= batch
        return ids

    def league_entries(self, platform: str, kind: str, queue: str) -> list[dict]:
        path = {
            "challenger": "challengerleagues",
            "grandmaster": "grandmasterleagues",
            "master": "masterleagues",
        }[kind]
        url = f"https://{platform}.api.riotgames.com/lol/league/v4/{path}/by-queue/{queue}"
        data = self._get(url) or {}
        return data.get("entries") or []

    def match(self, regional: str, match_id: str) -> dict | None:
        cached = self._raw_archive.read("match", match_id)
        if cached is not None:
            return cached
        url = f"https://{regional}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        payload = self._get(url)
        if isinstance(payload, dict):
            self._raw_archive.write("match", match_id, payload)
        return payload

    def timeline(self, regional: str, match_id: str) -> dict | None:
        cached = self._raw_archive.read("timeline", match_id)
        if cached is not None:
            return cached
        url = f"https://{regional}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        payload = self._get(url)
        if isinstance(payload, dict):
            self._raw_archive.write("timeline", match_id, payload)
        return payload
