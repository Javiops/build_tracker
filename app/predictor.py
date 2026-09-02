"""Load the trained board model and score a single live game state.

Reuses the training featurization (ShopDataset on a one-row list) so live
inference can never drift from what the model saw in training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from app.config import DATA_DIR
from app.ddragon import DataDragon

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_prefix as tp  # noqa: E402

ARTIFACT = DATA_DIR / "ml" / "prefix_model.pt"


class Predictor:
    def __init__(self, artifact: Path = ARTIFACT):
        if not artifact.exists():
            raise FileNotFoundError(f"No trained model at {artifact}. Run scripts/train_prefix.py.")
        blob = torch.load(artifact, map_location="cpu", weights_only=False)
        config = blob.get("config") or {}
        # featurize exactly like the artifact was trained
        tp.USE_EXTRAS = bool(config.get("extras", True))
        tp.USE_HISTORY = bool(config.get("history", False))
        self.threshold = float(config.get("basket_threshold", tp.BASKET_THRESHOLD))
        self.champ_index = blob["champ_index"]
        self.item_index = blob["item_index"]
        self.label_ids = blob["label_ids"]
        self.label_index = {item_id: i for i, item_id in enumerate(self.label_ids)}
        self.metrics = blob.get("metrics") or {}
        self.trained_at = blob.get("trained_at")
        self.dragon = DataDragon()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tp.PrefixModel(
            len(self.champ_index), len(self.item_index), len(self.label_ids), len(self.item_index) + 1
        ).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def _probs(self, row: dict) -> torch.Tensor:
        ds = tp.ShopDataset([row], self.champ_index, self.item_index, self.label_index, self.dragon)
        batch = next(ds.batches(1, shuffle=False))
        with torch.no_grad():
            logits = self.model(
                batch["champs"].to(self.device),
                batch["sides"].to(self.device),
                batch["items"].to(self.device),
                batch["query"].to(self.device),
                batch["inv"].to(self.device),
                batch["hist"].to(self.device),
            )
            logits = logits.masked_fill(~batch["legal"].to(self.device), -1e4)
        return torch.sigmoid(logits)[0].cpu()

    @staticmethod
    def _restate(row: dict, inventory: list[int], gold: float, dragon) -> dict:
        from app.shop_econ import inventory_state

        state = inventory_state(inventory, gold, dragon)
        return {
            **row,
            "inventory": inventory,
            "gold": gold,
            "can_complete": state["can_complete"],
            "n_completable": state["n_completable"],
            "cheapest_complete": state["cheapest_complete"],
            "gold_after_complete": state["gold_after_complete"],
            "n_inventory": state["n_inventory"],
        }

    def predict(self, row: dict, top_k: int = 3, max_items: int = 5) -> dict[str, Any]:
        """row must be shaped like one exported visit (see baseline._example).

        The basket is decoded iteratively: simulate each purchase (consume
        components, pay the combine cost, drop the gold) and ask the model
        again — so multi-buys like double Cloth Armor or boots + armor emerge
        from the model, and in-game purchase blocks are enforced.
        """
        from collections import Counter

        from app.shop_econ import combine_cost, is_blocked

        first_probs = self._probs(row)
        inventory = [int(i) for i in (row.get("inventory") or []) if i]
        top = []
        for idx in torch.argsort(first_probs, descending=True).tolist():
            item_id = self.label_ids[idx]
            if is_blocked(item_id, inventory, self.dragon):
                continue
            top.append(self._item_payload(item_id, float(first_probs[idx]), inventory))
            if len(top) >= top_k:
                break

        basket = []
        sim_inv = list(inventory)
        budget = float(row.get("gold") or 0)
        sim_row = row
        probs = first_probs
        for _step in range(max_items):
            picked = None
            for idx in torch.argsort(probs, descending=True).tolist():
                prob = float(probs[idx])
                if prob < self.threshold:
                    break
                item_id = self.label_ids[idx]
                if is_blocked(item_id, sim_inv, self.dragon):
                    continue
                inv_c = Counter(sim_inv)
                cost = combine_cost(item_id, inv_c, self.dragon)
                if cost > budget:
                    continue
                picked = (item_id, prob, cost, inv_c)
                break
            if not picked:
                break
            item_id, prob, cost, inv_c = picked
            basket.append({"item_id": item_id, "name": self.dragon.item_name(item_id), "prob": round(prob, 3), "cost": cost})
            combine_cost(item_id, inv_c, self.dragon, consume=True)
            inv_c[item_id] += 1
            sim_inv = list(inv_c.elements())
            budget -= cost
            sim_row = self._restate(sim_row, sim_inv, budget, self.dragon)
            probs = self._probs(sim_row)

        return {"top": top, "basket": basket, "threshold": self.threshold}

    def _item_payload(self, item_id: int, prob: float, inventory: list[int]) -> dict:
        from app.shop_econ import buy_cost

        return {
            "item_id": item_id,
            "name": self.dragon.item_name(item_id),
            "prob": round(prob, 3),
            "cost": buy_cost(item_id, inventory, self.dragon),
        }
