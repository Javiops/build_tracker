import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from app.ddragon import DataDragon
from app.shop_econ import (
    arrival_from_parts,
    buy_cost,
    combine_cost,
    decision_kind,
    inventory_state,
    net_spent,
)

dragon = DataDragon()

# Buy boots, leftover 50 → arrived with 350.
assert net_spent([], [1001], [], dragon) == 300
assert arrival_from_parts(50, [], [1001], [], dragon) == 350

# Combine Stormsurge from Alternator + Wisp, leftover 120 → spent 800.
spent = net_spent([3145, 3113], [4646], [3145, 3113], dragon)
assert spent == 800, spent
assert arrival_from_parts(120, [3145, 3113], [4646], [3145, 3113], dragon) == 920

# Sell Long Sword (245) and buy Pickaxe (875).
spent = net_spent([1036], [1037], [1036], dragon)
assert spent == 875 - 245, spent

# Completing Stormsurge is affordable; Infinity Edge is not.
state = inventory_state([3145, 3113], 900, dragon)
assert state["can_complete"] == 1
assert state["cheapest_complete"] == 800
assert buy_cost(4646, [3145, 3113], dragon) == 800
assert buy_cost(4646, [], dragon) == 2800
assert decision_kind([4646], [3145, 3113], dragon) == "complete"
assert decision_kind([1036], [], dragon) == "start"
assert decision_kind([2055], [1038], dragon) == "save"

# Partial components: own Alternator only, buy Stormsurge. In-game cost is
# total minus Alternator's value, not the full price.
storm_total = dragon.gold_block(4646)["total"]
alt_total = dragon.gold_block(3145)["total"]
owned = Counter([3145])
assert combine_cost(4646, owned, dragon) == storm_total - alt_total
assert owned[3145] == 1, "non-consuming call must not mutate owned"
assert buy_cost(4646, [3145], dragon) == storm_total - alt_total
# net_spent with partial ownership: Alternator destroyed on combine, wisp paid.
spent = net_spent([3145], [4646], [3145], dragon)
assert spent == storm_total - alt_total, spent
# consume=True removes used components
combine_cost(4646, owned, dragon, consume=True)
assert owned[3145] == 0

# Purchase blocks: legendaries limited to 1, one boots line only.
from app.shop_econ import is_blocked

assert is_blocked(4646, [4646], dragon), "owned Stormsurge blocks a second one"
assert not is_blocked(4646, [3145], dragon)
assert not is_blocked(1036, [1036], dragon), "components stack freely"
assert is_blocked(1001, [1001], dragon), "second Boots blocked"
assert not is_blocked(3047, [1001], dragon), "Tabi upgrade from owned Boots allowed"
assert is_blocked(3047, [3006], dragon), "Tabi blocked when owning Berserker's"
assert is_blocked(3047, [3047], dragon), "second Tabi blocked"

print("ok shop_econ")
