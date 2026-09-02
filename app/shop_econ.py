"""Gold on arrival and what the inventory can combine."""

from __future__ import annotations

from collections import Counter

from app.ddragon import DataDragon

PINK = {2055, 772043}
# Timeline frames snapshot gold up to 60s BEFORE a shop visit, so the honest
# budget is that pre-visit gold plus what a player can earn before buying.
GOLD_DRIFT = 500
# Sentinel label for the save/no-buy action: a Challenger at the fountain with
# meaningful gold who bought nothing. First-class action in the model's output.
SAVE_ITEM = 999999


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


def combine_cost(item_id: int, owned: Counter, dragon: DataDragon, consume: bool = False) -> int:
    """Gold to buy item_id given owned components, mirroring the in-game combine
    discount recursively (partial component ownership included). `owned` is left
    unchanged unless consume=True, in which case used components are removed."""
    taken: list[int] = []

    def rec(iid: int) -> int:
        from_ids = dragon.from_ids(iid)
        gold = dragon.gold_block(iid)
        if not from_ids:
            return gold["total"] or gold["base"]
        cost = gold["base"]
        for comp in from_ids:
            if owned[comp] > 0:
                owned[comp] -= 1
                taken.append(comp)
            else:
                cost += rec(comp)
        return cost

    cost = rec(item_id)
    if not consume:
        for comp in taken:
            owned[comp] += 1
    return cost


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
        spent += combine_cost(item_id, consumed_c, dragon, consume=True)
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
    return combine_cost(item_id, Counter(int(i) for i in inventory if i), dragon)


def components_of(item_id: int, dragon: DataDragon) -> set[int]:
    """Every item id anywhere in the recipe tree below item_id."""
    out: set[int] = set()
    stack = list(dragon.from_ids(item_id))
    while stack:
        comp = stack.pop()
        if comp in out:
            continue
        out.add(comp)
        stack.extend(dragon.from_ids(comp))
    return out


def is_blocked(item_id: int, inventory: list[int], dragon: DataDragon) -> bool:
    """Purchase restrictions the recipe math doesn't capture: completed
    (legendary) items are Limited to 1, and only one boots line is allowed
    unless the owned boots build into the candidate."""
    inv = [int(i) for i in inventory if i]
    if dragon.classify(item_id).get("is_completed") and item_id in inv:
        return True
    tags = set((dragon.item(item_id) or {}).get("tags") or [])
    if "Boots" in tags:
        comps = components_of(item_id, dragon)
        for owned in inv:
            owned_tags = (dragon.item(owned) or {}).get("tags") or []
            if "Boots" in owned_tags and owned not in comps:
                return True
    return False


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
