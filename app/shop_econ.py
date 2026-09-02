"""Gold on arrival and what the inventory can combine."""

from __future__ import annotations

from collections import Counter

from app.ddragon import DataDragon

PINK = {2055, 772043}


def payload_ids(items: list | None) -> list[int]:
    out = []
    for item in items or []:
        if isinstance(item, dict):
            item_id = item.get("item_id")
        else:
            item_id = item
        if item_id:
            out.append(int(item_id))
    return out


def leftover_gold(event: dict) -> int:
    if event.get("gold_left") is not None:
        return int(event.get("gold_left") or 0)
    return int(event.get("gold") or 0)


def damage_profile(items: list | None, dragon: DataDragon) -> dict:
    """Gold-weighted stat profile of an item set. Hybrid items count in both buckets."""
    out = {"ad": 0, "ap": 0, "armor": 0, "mr": 0}
    for item_id in payload_ids(items):
        data = dragon.item(item_id) or {}
        tags = set(data.get("tags") or [])
        gold = int((data.get("gold") or {}).get("total") or 0)
        if tags & {"Damage", "CriticalStrike", "AttackSpeed"}:
            out["ad"] += gold
        if "SpellDamage" in tags:
            out["ap"] += gold
        if "Armor" in tags:
            out["armor"] += gold
        if "SpellBlock" in tags:
            out["mr"] += gold
    return out


def net_spent(
    before: list | None,
    bought: list | None,
    consumed: list | None,
    dragon: DataDragon,
) -> int:
    consumed_c = Counter(payload_ids(consumed))
    bought_ids = payload_ids(bought)
    spent = 0

    def depth(item_id: int) -> int:
        return int(dragon.classify(item_id).get("depth") or 0)

    for item_id in sorted(bought_ids, key=depth, reverse=True):
        from_ids = dragon.from_ids(item_id)
        from_c = Counter(from_ids)
        gold = dragon.gold_block(item_id)
        if from_ids and from_c <= consumed_c:
            spent += gold["base"]
            consumed_c -= from_c
        else:
            spent += gold["total"] or gold["base"]
    gained = 0
    for item_id, count in consumed_c.items():
        gained += dragon.gold_block(item_id)["sell"] * count
    return spent - gained


def arrival_from_parts(
    leftover: int,
    before: list | None,
    bought: list | None,
    consumed: list | None,
    dragon: DataDragon,
) -> int:
    return max(0, int(leftover or 0) + net_spent(before, bought, consumed, dragon))


def event_arrival_gold(event: dict, dragon: DataDragon) -> int:
    if event.get("gold_left") is not None:
        return int(event.get("gold") or 0)
    leftover = leftover_gold(event)
    return arrival_from_parts(
        leftover,
        event.get("inventory_before"),
        event.get("bought"),
        event.get("consumed"),
        dragon,
    )


def buy_cost(item_id: int, inventory: list[int], dragon: DataDragon) -> int:
    from_ids = dragon.from_ids(item_id)
    gold = dragon.gold_block(item_id)
    if from_ids and Counter(from_ids) <= Counter(int(i) for i in inventory if i):
        return gold["base"]
    return gold["total"] or gold["base"]


def inventory_state(inventory: list[int], gold: int, dragon: DataDragon) -> dict:
    inv = [int(i) for i in inventory if i]
    inv_c = Counter(inv)
    affordable: list[tuple[int, int]] = []
    for item_id, from_ids, base in dragon.combine_recipes():
        if Counter(from_ids) <= inv_c and base <= gold:
            affordable.append((base, item_id))
    cheapest = min((row[0] for row in affordable), default=0)
    return {
        "can_complete": 1 if affordable else 0,
        "n_completable": len(affordable),
        "cheapest_complete": cheapest,
        "gold_after_complete": gold - cheapest if cheapest else gold,
        "n_inventory": len(inv),
    }


def decision_kind(bought: list[int], inventory: list[int], dragon: DataDragon) -> str:
    ids = [int(i) for i in bought if i]
    if not ids or all(item_id in PINK for item_id in ids):
        return "save"
    inv_c = Counter(int(i) for i in inventory if i)
    for item_id in ids:
        info = dragon.classify(item_id)
        if info.get("is_completed"):
            return "complete"
        from_ids = dragon.from_ids(item_id)
        if from_ids and Counter(from_ids) <= inv_c:
            return "complete"
    for item_id in ids:
        if dragon.from_ids(item_id):
            return "component"
    return "start"
