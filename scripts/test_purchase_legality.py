"""The actual purchase decoders respect role slots, stacks, recipes and wallets."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

from app.ddragon import dragon_for_patch
from app.predictor import Predictor
from app.shop_econ import inventory_shape, try_purchase
from eval_conditional_policy import action_of, legal
from eval_policy import check_displayed_purchases


dragon = dragon_for_patch("16.18")
MISSING = object()


def row_for(inventory, budget, role="MIDDLE", quest=MISSING):
    row = {"inventory": list(inventory), "gold": budget, "gold_est": budget,
           "role": role, "label_ids": [], "others": []}
    if quest is not MISSING:
        row["bot_quest_complete"] = quest
    return row


def scored_policy(proposals):
    """Replace only learned scores; all real filtering and simulation still run."""
    policy = Predictor.__new__(Predictor)
    policy.config = {"max_basket_items": 8}
    policy.dragon = dragon
    policy.threshold = 0.8
    policy.label_ids = [item for item, _copies in proposals]
    policy.label_index = {item: idx for idx, item in enumerate(policy.label_ids)}
    policy._standalone = set(policy.label_ids)
    policy._final_components = {}
    scores = torch.tensor([0.99 - idx * 0.005 for idx in range(len(proposals))])
    copies = torch.tensor([count for _item, count in proposals])
    policy._probs = lambda _row: (scores, scores, copies, None)
    policy._target_probs = lambda _row: torch.zeros(len(proposals))
    return policy


def check_decoders(name, row, proposals, expected, expected_cost, *, full_legal):
    """Expected actions are fixture facts, independent of the shared helper."""
    original = deepcopy(row)
    policy = scored_policy(proposals)
    options = policy.predict_options(row, n_options=1, budget_slack=0)
    assert len(options) == bool(expected), (name, options)
    displayed = options[0]["items"] if options else []
    assert Counter(item["item_id"] for item in displayed) == Counter(expected), (name, displayed)
    assert sum(item["cost"] for item in displayed) == expected_cost, (name, displayed)
    assert not check_displayed_purchases(options, row, row["gold"], dragon), name
    assert row == original, (name, "displayed decoder mutated its input")

    prediction = policy.predict(row, max_items=8, budget_slack=0)
    basket = prediction["basket"]
    assert Counter(item["item_id"] for item in basket) == Counter(expected), (name, basket)
    assert sum(item["cost"] for item in basket) == expected_cost, (name, basket)
    assert row == original, (name, "predict decoder mutated its input")

    proposed = tuple(item for item, count in proposals for _ in range(count))
    assert legal(proposed, row, dragon) is full_legal, (name, "conditional full basket")
    if expected:
        assert legal(tuple(expected), row, dragon), (name, "conditional retained basket")
    assert row == original, (name, "conditional policy mutated its input")


# Six regular support items can coexist with two role-slot Control Wards.
# Higher scores and excess wallet must never admit a third ward or seventh item.
support_full = [3111, 3867, 3190, 1029, 1028, 2022]
check_decoders("support separate ward slot", row_for(support_full, 5000, "UTILITY"),
               [(2055, 3), (1036, 1)], [2055, 2055], 150, full_legal=False)
check_decoders("support ward stack already full", row_for(support_full + [2055, 2055], 5000, "UTILITY"),
               [(2055, 1)], [], 0, full_legal=False)
check_decoders("support second ward", row_for(support_full + [2055], 75, "UTILITY"),
               [(2055, 1)], [2055], 75, full_legal=True)

# Away from support, two copies share one regular slot; a third is still illegal.
check_decoders("ordinary ward stack", row_for([1036] * 5, 5000),
               [(2055, 3), (1029, 1)], [2055, 2055], 150, full_legal=False)
check_decoders("ordinary second ward", row_for([1036] * 5 + [2055], 75),
               [(2055, 1)], [2055], 75, full_legal=True)
check_decoders("ordinary no ward slot", row_for([1036] * 6, 5000),
               [(2055, 1)], [], 0, full_legal=False)

# The same observed boot inventory needs causal completion information.
# Missing or unknown completion cannot grant an additional regular slot.
for state in (False, None, MISSING):
    check_decoders(f"bottom uncompleted or unknown {state!r}",
                   row_for([3006] + [1036] * 5, 5000, "BOTTOM", state),
                   [(1029, 1)], [], 0, full_legal=False)
check_decoders("bottom completed boot slot", row_for([3006] + [1036] * 5, 5000, "BOTTOM", True),
               [(1029, 2)], [1029], 300, full_legal=False)
check_decoders("bottom completed legal single", row_for([3006] + [1036] * 5, 300, "BOTTOM", True),
               [(1029, 1)], [1029], 300, full_legal=True)
check_decoders("completion cannot grant another role boot slot", row_for([3006] + [1036] * 5, 5000, "MIDDLE", True),
               [(1029, 1)], [], 0, full_legal=False)

# Gunmetal Greaves has no literal Boots tag in this pinned patch. Its recipe
# identifies the boots family, both when held and when proposed for purchase.
assert "Boots" not in dragon.item(3172)["tags"]
check_decoders("upgraded boots block basic boots", row_for([3172], 5000),
               [(1001, 1)], [], 0, full_legal=False)
check_decoders("upgraded boots block other finished boots", row_for([3172], 5000),
               [(3006, 1)], [], 0, full_legal=False)
check_decoders("missing-tag candidate is still boots", row_for([3158], 5000),
               [(3172, 1)], [], 0, full_legal=False)

# This full inventory releases two regular slots when buying the Warhammer.
# Its 100g combine and the two 75g wards fit exactly within the wallet.
ingredients = [1036, 1036, 2022, 1029, 1028, 1052]
check_decoders("combine releases slots", row_for(ingredients, 250),
               [(3133, 1), (2055, 2)], [3133, 2055, 2055], 250, full_legal=True)
check_decoders("combine then insufficient wallet", row_for(ingredients, 249),
               [(3133, 1), (2055, 2)], [3133, 2055], 175, full_legal=False)

# Basket targets have no purchase order. Sorting their IDs puts wards before
# the combine that frees their slot, so the baseline must find a legal order.
for labels in set(permutations((3133, 2055, 2055))):
    multiset_row = {**row_for(ingredients, 250), "label_ids": list(labels)}
    original_multiset = deepcopy(multiset_row)
    assert legal(action_of(multiset_row), multiset_row, dragon), labels
    assert legal(labels, multiset_row, dragon), labels
    assert multiset_row == original_multiset

# Displayed basket entries are retained purchases. A later combine cannot eat
# the Sword just displayed and falsely claim both items for the Warhammer price.
# Pinned 16.18 costs: Sword 350; Warhammer 100 + 350 + 250 + 350 = 1050.
assert dragon.gold_block(3133)["total"] == 1050
check_decoders("new Sword cannot discount later retained Warhammer", row_for([], 1050),
               [(1036, 1), (3133, 1)], [1036], 350, full_legal=False)
check_decoders("both retained purchases need full combined budget", row_for([], 1400),
               [(1036, 1), (3133, 1)], [1036, 3133], 1400, full_legal=True)
# One pre-existing Sword can still contribute 350g to the recipe. The newly
# displayed Sword remains separate, requiring 350 + 700 = 1050g in total.
check_decoders("pre-existing Sword still discounts Warhammer", row_for([1036], 1050),
               [(1036, 1), (3133, 1)], [1036, 3133], 1050, full_legal=True)
check_decoders("protected Sword cannot supplement old component discount", row_for([1036], 1049),
               [(1036, 1), (3133, 1)], [1036], 350, full_legal=False)

# No implicit gold drift is allowed in an explicit exact-wallet decision.
for wallet, quantity in ((349, 0), (350, 1), (699, 1), (700, 2)):
    check_decoders(f"exact wallet {wallet}", row_for([], wallet),
                   [(1036, 2)], [1036] * quantity, 350 * quantity,
                   full_legal=quantity == 2)
check_decoders("five Long Swords survive decoding", row_for([1086], 1750, "TOP"),
               [(1036, 5)], [1036] * 5, 1750, full_legal=True)

# Missing live role-slot contents restrict the affected item family only.
check_decoders("unknown support wards preserve ordinary purchases",
               {**row_for([1036] * 5, 5000, "UTILITY"), "role_slot_inventory_unknown": True},
               [(2055, 1), (1029, 1)], [1029], 300, full_legal=False)
check_decoders("unknown bot boots preserve ordinary purchases",
               {**row_for([], 5000, "BOTTOM", True), "role_slot_inventory_unknown": True},
               [(1001, 1), (1036, 1)], [1036], 350, full_legal=False)
check_decoders("unknown role slot does not restrict mid purchases",
               {**row_for([], 5000), "role_slot_inventory_unknown": True},
               [(2055, 1), (1001, 1)], [2055, 1001], 375, full_legal=True)

# Live consumables omitted from model features still occupy physical slots.
check_decoders("unmodeled occupied slot blocks seventh physical item",
               {**row_for([1036] * 5, 5000), "unmodeled_regular_slots": 1},
               [(1029, 1)], [], 0, full_legal=False)
check_decoders("unmodeled slot permits only one remaining item",
               {**row_for([1036] * 4, 5000), "unmodeled_regular_slots": 1},
               [(1029, 2)], [1029], 300, full_legal=False)
check_decoders("combine frees space beside unmodeled slot",
               {**row_for([1036, 1036, 2022, 1029, 1028], 250), "unmodeled_regular_slots": 1},
               [(3133, 1), (2055, 2)], [3133, 2055, 2055], 250, full_legal=True)

# Inject faulty displayed baskets independently of either decoder. The checker
# must reject illegal inventories and forged prices even with ample gold.
def buy_option(entries):
    return {"kind": "buy", "items": [{"item_id": item, "cost": cost} for item, cost in entries]}


def rejected_display(name, row, entries):
    original = deepcopy(row)
    try:
        check_displayed_purchases([buy_option(entries)], row, row["gold"], dragon)
    except ValueError:
        assert row == original, (name, "checker mutated its input")
    else:
        raise AssertionError(f"Invalid displayed basket accepted: {name}")


rejected_display("third support ward", row_for(support_full, 5000, "UTILITY"), [(2055, 75)] * 3)
rejected_display("duplicate upgraded boots", row_for([], 5000), [(3172, 1100)] * 2)
rejected_display("seven regular items", row_for([], 5000), [(1036, 350)] * 7)
rejected_display("forged free purchase", row_for([], 5000), [(1036, 0)])
rejected_display("forged discount consumes displayed Sword", row_for([], 5000),
                 [(1036, 350), (3133, 700)])
rejected_display("forged discount consumes both old and displayed Swords", row_for([1036], 5000),
                 [(1036, 350), (3133, 350)])
rejected_display("unmodeled slot overflow", {**row_for([1036] * 5, 5000), "unmodeled_regular_slots": 1},
                 [(1029, 300)])
rejected_display("unknown support ward inventory",
                 {**row_for([], 5000, "UTILITY"), "role_slot_inventory_unknown": True}, [(2055, 75)])
assert check_displayed_purchases([buy_option([(1036, 350)])], row_for([], 349), 349, dragon) == [
    {"cost": 350, "budget": 349.0}], "One-gold wallet violation was tolerated"

# Alternatives each start from the observed inventory and wallet, so either of
# these can fill the same last slot. They must not consume each other's state.
alternative_row = row_for([1036] * 5, 350)
original_alternatives = deepcopy(alternative_row)
assert check_displayed_purchases([buy_option([(1029, 300)]), buy_option([(1036, 350)])],
                                 alternative_row, 350, dragon) == []
assert alternative_row == original_alternatives

# The shared transaction returns a detached inventory and preserves components
# even when a proposed combine fails. It must not mutate the caller's list.
original = list(ingredients)
transaction = try_purchase(3133, ingredients, 100, dragon, role="MIDDLE")
assert transaction is not None
cost, after = transaction
assert cost == 100
assert after == Counter([3133, 1029, 1028, 1052])
assert ingredients == original
after[1036] += 10
assert ingredients == original
assert try_purchase(3133, ingredients, 99, dragon, role="MIDDLE") is None
assert ingredients == original
held_swords = [1036, 1036]
protected_sword = Counter({1036: 1})
retained = try_purchase(3133, held_swords, 700, dragon, protected=protected_sword)
assert retained == (700, Counter({1036: 1, 3133: 1}))
assert held_swords == [1036, 1036] and protected_sword == Counter({1036: 1})

support_shape = inventory_shape(support_full + [2055, 2055], dragon, role="UTILITY")
assert support_shape["regular_slots"] == 6
assert support_shape["role_slot_item_copies"] == 2
assert support_shape["ward_copies"] == 2
ordinary_shape = inventory_shape([1036] * 5 + [2055, 2055], dragon, role="MIDDLE")
assert ordinary_shape["regular_slots"] == 6
assert ordinary_shape["role_slot_item_copies"] == 0
bot_shape = inventory_shape([3006] + [1036] * 6, dragon, role="BOTTOM", bot_quest_complete=True)
assert bot_shape["regular_slots"] == 6
assert bot_shape["role_slot_item_copies"] == 1
assert bot_shape["boot_copies"] == 1

print("ok purchase legality: actual decoders, conditional policy, role slots, live uncertainty, hidden slot occupancy, independent displayed-basket checks, exact wallets and five-copy targets")
