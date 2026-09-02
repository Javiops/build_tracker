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
        # Final items (and finished boots) a purchase can be "building toward",
        # with their full recursive component sets for arrow resolution.
        from app.shop_econ import components_of

        self._final_components: dict[int, set[int]] = {}
        for item_id in self.label_ids:
            cls = self.dragon.classify(item_id)
            if cls.get("is_completed") or cls.get("is_boots"):
                self._final_components[item_id] = components_of(item_id, self.dragon)
        # Items that are sensible buys outside any recipe plan: starter/value
        # items (no recipe in either direction), Dark Seal, control wards.
        self._standalone: set[int] = {1082, 2055, 772043}
        for item_id in self.label_ids:
            data = self.dragon.item(item_id) or {}
            if not data.get("from") and not data.get("into"):
                self._standalone.add(item_id)

    def _probs(self, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(affordability-masked probs, raw probs). The raw ones rank what the
        model wants regardless of current gold — needed for save/target logic,
        where the wanted item is often exactly the one the mask zeroes out."""
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
            raw = torch.sigmoid(logits)[0].cpu()
            masked = torch.sigmoid(logits.masked_fill(~batch["legal"].to(self.device), -1e4))[0].cpu()
        return masked, raw

    def _plan_probs(self, row: dict) -> torch.Tensor:
        """Masked probs for a counterfactual 'when you can afford it' state:
        same board and matchup, gold lifted into final-item range. Raw logits
        for unaffordable finals are untrained (the loss masks them), so this
        probe is the only training-consistent way to rank what the player is
        building toward."""
        from app.shop_econ import inventory_state

        inventory = [int(i) for i in (row.get("inventory") or []) if i]
        gold = max(float(row.get("gold") or 0), 3500.0)
        state = inventory_state(inventory, gold, self.dragon)
        probe = {
            **row,
            "gold": gold,
            "can_complete": state["can_complete"],
            "n_completable": state["n_completable"],
            "cheapest_complete": state["cheapest_complete"],
            "gold_after_complete": state["gold_after_complete"],
            "n_inventory": state["n_inventory"],
        }
        masked, _raw = self._probs(probe)
        return masked

    def predict(
        self,
        row: dict,
        top_k: int = 3,
        max_items: int = 5,
        budget_slack: float | None = None,
    ) -> dict[str, Any]:
        """row must be shaped like one exported visit (see baseline._example).

        budget_slack defaults to GOLD_DRIFT (offline gold is up to 60s stale);
        pass 0 for live states where gold is exact and every recommended buy
        must be affordable right now.
        """
        from collections import Counter

        from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, combine_cost, is_blocked

        first_probs, _ = self._probs(row)
        inventory = [int(i) for i in (row.get("inventory") or []) if i]
        gold = float(row.get("gold") or 0)
        start_budget = gold + (GOLD_DRIFT if budget_slack is None else budget_slack)

        def buyable(item_id: int, inv: list[int], budget: float) -> tuple[int, Counter] | None:
            """Cost and post-purchase inventory if the buy is possible: not
            blocked, affordable, and fits the 6 item slots after combining."""
            if is_blocked(item_id, inv, self.dragon):
                return None
            inv_c = Counter(inv)
            cost = combine_cost(item_id, inv_c, self.dragon, consume=True)
            inv_c[item_id] += 1
            if cost > budget or sum(inv_c.values()) > 6:
                return None
            return cost, inv_c

        # The plan: which final items the model wants for this board/matchup,
        # asked at a can-afford-it state (raw low-gold logits are untrained).
        plan = self._plan_probs(row)
        finals = []
        for fid, comps in self._final_components.items():
            p = float(plan[self.label_index[fid]])
            if p >= 0.15 and not is_blocked(fid, inventory, self.dragon):
                finals.append((p, fid, comps))
        finals.sort(reverse=True)
        # Plan-consistency: an optimal buy must advance a wanted final (or be
        # a standalone value item) — otherwise 300g gets burned on a Dagger
        # while the player is saving toward Black Cleaver.
        allowed: set[int] = set(self._standalone)
        for p, fid, comps in finals[:4]:
            if p >= 0.3:
                allowed |= comps | {fid}

        top = []
        for idx in torch.argsort(first_probs, descending=True).tolist():
            item_id = self.label_ids[idx]
            if item_id == SAVE_ITEM:
                continue
            if buyable(item_id, inventory, start_budget) is None:
                continue
            top.append(self._item_payload(item_id, float(first_probs[idx]), inventory))
            if len(top) >= top_k:
                break

        # Basket: one forward pass, greedy down the ranked list under the
        # running constraints (budget, slots, purchase blocks, plan).
        basket = []
        model_save_prob = 0.0
        sim_inv = list(inventory)
        budget = start_budget
        for idx in torch.argsort(first_probs, descending=True).tolist():
            prob = float(first_probs[idx])
            if prob < self.threshold or len(basket) >= max_items:
                break
            item_id = self.label_ids[idx]
            if item_id == SAVE_ITEM:
                # the model says stop buying here; only a primary save counts
                if not basket:
                    model_save_prob = prob
                break
            if item_id not in allowed and prob < 0.95:
                continue
            buy = buyable(item_id, sim_inv, budget)
            if buy is None:
                continue
            cost, inv_c = buy
            basket.append({"item_id": item_id, "name": self.dragon.item_name(item_id), "prob": round(prob, 3), "cost": cost})
            sim_inv = list(inv_c.elements())
            budget -= cost

        # "Building toward" arrows: for each non-final buy, the most probable
        # wanted final item whose recipe contains it.
        for entry in basket:
            entry["target"] = None
            if entry["item_id"] in self._final_components:
                continue  # already a full item
            for p, fid, comps in finals:
                if entry["item_id"] in comps:
                    entry["target"] = {"item_id": fid, "name": self.dragon.item_name(fid), "prob": round(p, 3)}
                    break

        # Save option: the model predicting the SAVE action outright, or nothing
        # affordable clearing the bar while a wanted final is out of gold reach.
        save = None
        if not basket and finals:
            p, fid, _comps = finals[0]
            cost = combine_cost(fid, Counter(inventory), self.dragon)
            if model_save_prob >= self.threshold or (p >= 0.4 and cost > gold):
                save = {
                    "item_id": fid,
                    "name": self.dragon.item_name(fid),
                    "prob": round(max(p, model_save_prob), 3),
                    "need": max(0, int(cost - gold)),
                }

        return {"top": top, "basket": basket, "save": save, "threshold": self.threshold}

    def _item_payload(self, item_id: int, prob: float, inventory: list[int]) -> dict:
        from app.shop_econ import buy_cost

        return {
            "item_id": item_id,
            "name": self.dragon.item_name(item_id),
            "prob": round(prob, 3),
            "cost": buy_cost(item_id, inventory, self.dragon),
        }
