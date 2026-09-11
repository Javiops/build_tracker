"""Prospective, local-only evidence collection for explicit shop decisions.

Match-V5 cannot see a manual recall where a player buys nothing, and the live
client API cannot tell us whether the player is standing in shop.  We therefore
do not infer either.  The user starts a short decision session; the server logs
the exact wallet and options shown, then observes an inventory change or a
timeout.  This is the first dataset in the project where a no-buy means an
actual queried shop decision rather than a post-death proxy.

Records are append-only JSONL under data/beta_telemetry/.  They stay local and
contain no Riot ID, summoner name, raw client snapshot, or opponent identifiers.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

TELEMETRY_DIR = DATA_DIR / "beta_telemetry"
DEFAULT_SESSION_WINDOW_S = 90


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _inventory(row: dict[str, Any]) -> list[int]:
    return [int(item_id) for item_id in (row.get("inventory") or []) if item_id]


def _item_delta(before: list[int], after: list[int]) -> tuple[list[int], list[int]]:
    """Return positive and negative inventory deltas, preserving duplicates."""
    start, end = Counter(before), Counter(after)
    bought = list((end - start).elements())
    removed = list((start - end).elements())
    return bought, removed


def _safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """The decision context needed for later analysis, with no player identity."""
    return {
        "champion": row.get("champion") or "",
        "champion_id": int(row.get("champion_id") or 0),
        "role": row.get("role") or "",
        "team_id": int(row.get("team_id") or 0),
        "game_time_ms": int(row.get("ts") or 0),
        "level": int(row.get("level") or 0),
        "gold_exact": int(row.get("gold") or 0),
        "inventory": _inventory(row),
        "kills": int(row.get("kills") or 0),
        "deaths": int(row.get("deaths") or 0),
    }


def _safe_options(options: list[dict]) -> list[dict]:
    """Persist exactly the user-visible action, never logits or label fields."""
    return [
        {
            "kind": option.get("kind"),
            "tier": option.get("tier"),
            "items": [
                {"item_id": int(item.get("item_id") or 0), "cost": int(item.get("cost") or 0)}
                for item in (option.get("items") or [])
            ],
            "toward_item_id": int(((option.get("toward") or {}).get("item_id") or 0)),
        }
        for option in options
    ]


class TelemetryStore:
    """Append local JSONL records atomically enough for this single-process app."""

    def __init__(self, directory: Path = TELEMETRY_DIR):
        self.directory = directory
        self._lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"shop_sessions_{datetime.now():%Y%m%d}.jsonl"
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        return path


@dataclass
class ShopDecisionSession:
    session_id: str
    started_monotonic: float
    expires_monotonic: float
    started_at: str
    start_row: dict[str, Any]
    options: list[dict]
    model: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "active": True,
            "session_id": self.session_id,
            "expires_in_s": max(0, int(round(self.expires_monotonic - time.monotonic()))),
            "options": self.options,
            "start_gold": self.start_row["gold_exact"],
            "started_at": self.started_at,
        }


class ShopDecisionSessions:
    """One explicit decision session for the local overlay process.

    A later request supersedes an unresolved session rather than pretending it
    was a no-buy.  That distinction keeps abandonment out of the training label.
    """

    def __init__(self, store: TelemetryStore | None = None, window_s: int = DEFAULT_SESSION_WINDOW_S):
        self.store = store or TelemetryStore()
        self.window_s = window_s
        self._active: ShopDecisionSession | None = None
        self._lock = threading.Lock()

    def _write(self, event: str, session: ShopDecisionSession, **extra: Any) -> None:
        self.store.append(
            {
                "schema_version": 1,
                "event": event,
                "recorded_at": _utc_now(),
                "session_id": session.session_id,
                "started_at": session.started_at,
                "model": session.model,
                **extra,
            }
        )

    def start(self, row: dict[str, Any], options: list[dict], model: dict[str, Any], now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._active is not None:
                self._write("abandoned", self._active, reason="superseded")
            session = ShopDecisionSession(
                session_id=uuid.uuid4().hex,
                started_monotonic=now,
                expires_monotonic=now + self.window_s,
                started_at=_utc_now(),
                start_row=_safe_row(row),
                options=options,
                model=model,
            )
            self._active = session
            self._write("shop_query", session, context=session.start_row, options=_safe_options(session.options))
            return session.public()

    def observe(self, row: dict[str, Any], now: float | None = None) -> dict | None:
        """Resolve only a real inventory change or the explicit no-buy timeout."""
        now = time.monotonic() if now is None else now
        with self._lock:
            session = self._active
            if session is None:
                return None
            current = _safe_row(row)
            bought, removed = _item_delta(session.start_row["inventory"], current["inventory"])
            elapsed_s = max(0.0, now - session.started_monotonic)
            # A net addition is the strongest signal exposed by the local
            # client.  A removal alone could be a consumed item, an automatic
            # upgrade, or an item sold; calling it a purchase would poison the
            # prospective label.  Keep it as an explicitly unscored event.
            if bought:
                self._write(
                    "purchase_observed",
                    session,
                    elapsed_s=round(elapsed_s, 3),
                    outcome={**current, "bought": bought, "removed": removed},
                )
                self._active = None
                return {"active": False, "resolved": "purchase_observed"}
            if removed:
                self._write(
                    "inventory_changed_unattributed",
                    session,
                    elapsed_s=round(elapsed_s, 3),
                    outcome={**current, "bought": bought, "removed": removed},
                )
                self._active = None
                return {"active": False, "resolved": "inventory_changed_unattributed"}
            if now >= session.expires_monotonic:
                self._write(
                    "no_buy_timeout",
                    session,
                    elapsed_s=round(elapsed_s, 3),
                    outcome=current,
                )
                self._active = None
                return {"active": False, "resolved": "no_buy_timeout"}
            return session.public()

    def cancel(self, reason: str = "user_cancelled") -> bool:
        with self._lock:
            if self._active is None:
                return False
            self._write("abandoned", self._active, reason=reason)
            self._active = None
            return True

    def public(self) -> dict[str, Any] | None:
        with self._lock:
            return self._active.public() if self._active else None
