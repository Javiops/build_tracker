from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import httpx

from app.config import CACHE_DIR

DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"


class DataDragon:
    def __init__(self, version: str | None = None):
        self.version = version or latest_version()
        self._items = load_json(
            CACHE_DIR / f"item-{self.version}.json",
            f"https://ddragon.leagueoflegends.com/cdn/{self.version}/data/en_US/item.json",
        )
        self._champions = load_json(
            CACHE_DIR / f"champion-{self.version}.json",
            f"https://ddragon.leagueoflegends.com/cdn/{self.version}/data/en_US/champion.json",
        )
        self._champ_by_id = {
            int(c["key"]): c for c in self._champions.get("data", {}).values()
        }
        self._combine_recipes: list[tuple[int, tuple[int, ...], int]] | None = None
        # Static data: memoize the hot accessors — the ML mask calls them
        # hundreds of millions of times per export.
        self._gold_cache: dict[int, dict] = {}
        self._from_cache: dict[int, list[int]] = {}
        self._classify_cache: dict[int, dict] = {}

    def item(self, item_id: int) -> dict | None:
        return self._items.get("data", {}).get(str(item_id))

    def item_name(self, item_id: int) -> str:
        data = self.item(item_id)
        return data["name"] if data else f"Item {item_id}"

    def champion_name(self, champion_id: int) -> str:
        champ = self._champ_by_id.get(champion_id)
        return champ["name"] if champ else f"Champion {champion_id}"

    def champion_id_name(self, champion_id: int) -> str:
        champ = self._champ_by_id.get(champion_id)
        return champ["id"] if champ else str(champion_id)

    def champion_info(self, champion_id: int) -> dict:
        """Data Dragon 0-10 ratings (attack, magic, defense, difficulty)."""
        champ = self._champ_by_id.get(champion_id) or {}
        return champ.get("info") or {}

    def champion_stats(self, champion_id: int) -> dict:
        """Base stats block (attackrange, hp, armor, …) from champion.json."""
        champ = self._champ_by_id.get(champion_id) or {}
        return champ.get("stats") or {}

    def champion_id_by_name(self, name: str) -> int:
        """Numeric id from a display name ('Lee Sin') or internal id ('LeeSin')."""
        if not hasattr(self, "_champ_by_name"):
            self._champ_by_name = {}
            for key, champ in self._champ_by_id.items():
                self._champ_by_name[champ["name"].lower()] = key
                self._champ_by_name[champ["id"].lower()] = key
        return self._champ_by_name.get((name or "").lower(), 0)

    def gold_block(self, item_id: int) -> dict:
        cached = self._gold_cache.get(item_id)
        if cached is None:
            data = self.item(item_id) or {}
            gold = data.get("gold") or {}
            cached = {
                "base": int(gold.get("base") or 0),
                "total": int(gold.get("total") or gold.get("base") or 0),
                "sell": int(gold.get("sell") or 0),
                "purchasable": bool(gold.get("purchasable", True)),
            }
            self._gold_cache[item_id] = cached
        return cached

    def from_ids(self, item_id: int) -> list[int]:
        cached = self._from_cache.get(item_id)
        if cached is None:
            data = self.item(item_id) or {}
            cached = [int(x) for x in (data.get("from") or []) if int(x)]
            self._from_cache[item_id] = cached
        return cached

    def combine_recipes(self) -> list[tuple[int, tuple[int, ...], int]]:
        if self._combine_recipes is None:
            recipes = []
            for raw_id, data in (self._items.get("data") or {}).items():
                from_ids = tuple(int(x) for x in (data.get("from") or []) if int(x))
                if not from_ids:
                    continue
                item_id = int(raw_id)
                if self.classify(item_id)["skip"]:
                    continue
                gold = data.get("gold") or {}
                if gold.get("purchasable", True) is False:
                    continue
                recipes.append((item_id, from_ids, int(gold.get("base") or 0)))
            self._combine_recipes = recipes
        return self._combine_recipes

    def classify(self, item_id: int) -> dict:
        cached = self._classify_cache.get(item_id)
        if cached is not None:
            return cached
        data = self.item(item_id) or {}
        tags = set(data.get("tags") or [])
        depth = data.get("depth") or 0
        gold = (data.get("gold") or {}).get("total") or 0
        purchasable = (data.get("gold") or {}).get("purchasable", True)
        from_items = data.get("from") or []
        is_trinket = "Trinket" in tags
        is_consumable = "Consumable" in tags
        is_boots = "Boots" in tags
        # Final items either sit deep in the tree OR combine straight from
        # basics and build into nothing (Rabadon's, Infinity Edge are depth 2).
        builds_into = data.get("into") or []
        is_completed = depth >= 3 or (bool(from_items) and not builds_into)
        is_component = bool(from_items) and bool(builds_into) and not is_boots
        is_support_gold = "GoldPer" in tags
        is_pink_ward = item_id in {2055, 772043}
        skip = (
            not data
            or item_id in {0, 2419, 2421}
            or is_trinket
            or (is_consumable and not is_pink_ward)
            or (
                not is_support_gold
                and not is_pink_ward
                and (gold == 0 or purchasable is False)
            )
        )
        cached = {
            "name": data.get("name") or f"Item {item_id}",
            "tags": sorted(tags),
            "depth": depth,
            "gold": gold,
            "from": from_items,
            "is_boots": is_boots and bool(from_items),
            "is_basic_boots": is_boots and not from_items,
            "is_completed": is_completed,
            "is_component": is_component,
            "skip": skip,
        }
        self._classify_cache[item_id] = cached
        return cached


def latest_version() -> str:
    versions = load_json(CACHE_DIR / "versions.json", DDRAGON_VERSIONS)
    return versions[0]


def load_json(path: Path, url: str) -> dict | list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        return response.json()


@lru_cache(maxsize=1)
def default_dragon() -> DataDragon:
    return DataDragon()
