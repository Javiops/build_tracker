"""Regression test for the conditional multiset policy's action semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.ddragon import default_dragon
from app.shop_econ import SAVE_ITEM
from eval_conditional_policy import action_of, legal

dragon = default_dragon()
base = {"gold": 800, "gold_est": None, "inventory": [1056], "label_ids": [1036], "save_kind": "build_spend"}
assert action_of({**base, "save_kind": "no_buy_death", "label_ids": [SAVE_ITEM]}) == (SAVE_ITEM,)
assert action_of({**base, "save_kind": "ward_only", "label_ids": [2055]}) == (2055,)
assert action_of({**base, "label_ids": [1036, 1036]}) == (1036, 1036)
assert legal((1036,), base, dragon)
assert not legal((1036,), {**base, "gold": 100}, dragon)
assert legal((SAVE_ITEM,), {**base, "gold": 0}, dragon)
print("ok conditional policy: no-buy, wards, duplicates, and affordability stay distinct")
