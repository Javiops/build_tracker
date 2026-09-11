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


def feature_inventory(items: list[int]) -> list[int]:
    """Use the same support-tier information offline and live.

    Historical timelines can omit the final support choice. Preserve the slot
    and quest tier without giving live inference a distinct, untrained token.
    This only shapes model input; raw observed inventory stays available.
    """
    return [3867 if int(i) in {3869, 3870, 3871, 3876, 3877} else int(i) for i in items if i]


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
    candidate = dragon.classify(item_id)
    if candidate.get("is_boots") or candidate.get("is_basic_boots"):
        comps = components_of(item_id, dragon)
        for owned in inv:
            info = dragon.classify(owned)
            if (info.get("is_boots") or info.get("is_basic_boots")) and owned not in comps:
                return True
    return False


def inventory_shape(
    inventory: list[int], dragon: DataDragon, *, role: str = "",
    bot_quest_complete: bool | None = None,
) -> dict:
    """Physical capacity of the supplied owned items, not their tensor length.

    Control wards stack; supports have a dedicated ward slot. Only an observed
    completed bottom quest frees the boot slot. Unknown/legacy completion never
    grants extra capacity. Callers supply modeled inventory; omitted consumables
    are outside this check's scope.
    """
    owned = Counter(int(i) for i in inventory if i)
    boots = sum(n for i, n in owned.items()
                if dragon.classify(i).get("is_boots") or dragon.classify(i).get("is_basic_boots"))
    wards = sum(owned[i] for i in PINK)
    role_boots = min(1, boots) if role == "BOTTOM" and bot_quest_complete is True else 0
    regular = sum(n for i, n in owned.items() if i not in PINK) - role_boots
    if role != "UTILITY":
        regular += sum(1 for i in PINK if owned[i])
    return {
        "modeled_item_copies": sum(owned.values()), "regular_slots": regular,
        "role_slot_item_copies": wards if role == "UTILITY" else role_boots,
        "ward_copies": wards, "boot_copies": boots,
    }


def try_purchase(
    item_id: int, inventory: list[int], budget: float, dragon: DataDragon, *,
    role: str = "", bot_quest_complete: bool | None = None,
    unmodeled_regular_slots: int = 0, role_slot_inventory_unknown: bool = False,
    protected: Counter | None = None,
) -> tuple[int, Counter] | None:
    """Simulate one purchase under shared budget, recipe and capacity rules.

    The caller's inventory is never mutated. Restrictions are rechecked after
    each copy, so repeated wards cannot overflow their stack and a combine can
    free slots before its result occupies one.
    """
    if item_id == SAVE_ITEM or not dragon.item(item_id) or is_blocked(item_id, inventory, dragon):
        return None
    candidate = dragon.classify(item_id)
    if role_slot_inventory_unknown and (
        (role == "UTILITY" and item_id in PINK)
        or (role == "BOTTOM" and (candidate.get("is_boots") or candidate.get("is_basic_boots")))
    ):
        # An absent live role item is not proof of empty ownership. The public
        # snapshot may omit it, so do not risk duplicate boots or extra wards.
        return None
    owned = Counter(int(i) for i in inventory if i)
    reserved = protected or Counter()
    if not reserved <= owned:
        raise ValueError("Previously displayed purchases are missing from simulated inventory")
    # Basket entries represent retained purchases. A later combine may use
    # pre-existing components, but must not eat an earlier displayed entry.
    owned.subtract(reserved)
    cost = combine_cost(item_id, owned, dragon, consume=True)
    owned.update(reserved)
    owned[item_id] += 1
    if cost > budget:
        return None
    shape = inventory_shape(list(owned.elements()), dragon, role=role,
                            bot_quest_complete=bot_quest_complete)
    if shape["regular_slots"] + max(0, unmodeled_regular_slots) > 6 or shape["boot_copies"] > 1:
        return None
    for ward in PINK:
        if owned[ward] > int((dragon.item(ward) or {}).get("stacks") or 1):
            return None
    return cost, +owned


def basket_is_legal(items: tuple[int, ...], row: dict, budget: float, dragon: DataDragon) -> bool:
    """Does an unordered retained multiset have a legal purchase sequence?

    Try combines first, but search alternatives when competing recipes or full
    slots make that ordering insufficient. Memoization collapses permutations
    of repeated items; baskets contain only the handful of retained purchases.
    """
    from functools import lru_cache

    target = Counter(items)

    @lru_cache(maxsize=None)
    def search(remaining, inventory, wallet):
        if not remaining:
            return True
        pending = Counter(dict(remaining))
        protected = target - pending
        for item_id in sorted(pending, key=lambda i: (dragon.classify(i).get("depth", 0), i), reverse=True):
            purchase = try_purchase(item_id, list(inventory), wallet, dragon,
                                    role=row.get("role") or "",
                                    bot_quest_complete=row.get("bot_quest_complete"),
                                    unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                                    role_slot_inventory_unknown=bool(row.get("role_slot_inventory_unknown")),
                                    protected=protected)
            if purchase is None:
                continue
            cost, owned = purchase
            rest = pending.copy()
            rest[item_id] -= 1
            if not rest[item_id]:
                del rest[item_id]
            if search(tuple(sorted(rest.items())), tuple(sorted(owned.elements())), wallet - cost):
                return True
        return False

    return search(tuple(sorted(target.items())), tuple(sorted(row.get("inventory") or [])), budget)


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
