import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ddragon import DataDragon
from app.shop_econ import (
    arrival_from_parts,
    buy_cost,
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

print("ok shop_econ")
