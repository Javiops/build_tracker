"""Load the trained board model and score a single live game state.

Reuses the training featurization (ShopDataset on a one-row list) so live
inference can never drift from what the model saw in training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from app.config import DATA_DIR, ROOT
from app.ddragon import DataDragon

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_prefix as tp  # noqa: E402
from app.deployment import serving_eligibility, sha256_file

ARTIFACT = DATA_DIR / "ml" / "prefix_model.pt"


class Predictor:
    def __init__(self, artifact: Path = ARTIFACT):
        if not artifact.exists():
            raise FileNotFoundError(f"No trained model at {artifact}. Run scripts/train_prefix.py.")
        self.artifact = artifact
        self.artifact_sha256 = sha256_file(artifact)
        blob = torch.load(artifact, map_location="cpu", weights_only=False)
        config = blob.get("config") or {}
        self.config = config
        # featurize exactly like the artifact was trained (single source of
        # truth — eval harnesses call the same function, see tp.apply_config)
        tp.apply_config(config)
        self.threshold = float(config.get("basket_threshold", tp.BASKET_THRESHOLD))
        # multiset artifacts carry a trained count head (copies per item);
        # older ones don't — their count outputs are untrained noise to ignore
        self.has_counts = bool(config.get("counts"))
        # The save head's historical target is post-death no-buy only. It is
        # not a calibrated probability of every manual-recall no-buy.
        self.has_save = bool(config.get("save_head"))
        # plan-head artifacts predict the next completed final; the blend lifts
        # items by the plan mass they advance (component-altitude fix)
        self.has_target = bool(config.get("target_head"))
        self.target_blend = float(config.get("target_blend") or 0)
        self.champ_index = blob["champ_index"]
        self.item_index = blob["item_index"]
        self.label_ids = blob["label_ids"]
        self.label_index = {item_id: i for i, item_id in enumerate(self.label_ids)}
        self.metrics = blob.get("metrics") or {}
        self.trained_at = blob.get("trained_at")
        # Older artifacts predate split/provenance recording. They can still be
        # inspected for backwards compatibility, but must never masquerade as
        # validated causal-gold or displayed-policy candidates.
        self.provenance = dict(blob.get("provenance") or {})
        if not self.provenance:
            self.provenance = {
                "status": "legacy_unverified",
                "reason": "artifact predates required split, gold-input, and policy-evaluation provenance",
                "gold_input": config.get("gold_input") or "unknown",
                "scored_on": blob.get("scored_on") or "unknown",
            }
        else:
            self.provenance.setdefault("status", "declared")
            self.provenance.setdefault("gold_input", config.get("gold_input") or "unknown")
            self.provenance.setdefault("scored_on", blob.get("scored_on") or "validation")
        # A candidate remains non-serving until a validation policy report has
        # passed the budget hard stop and a human explicitly promotes its exact
        # bytes.  Loading it is still useful for offline/forensic tools.
        self.deployment = serving_eligibility(self.artifact, self.provenance)
        self.dragon = DataDragon(config.get("ddragon_version"))
        if config.get("static_data_sha256") and config["static_data_sha256"] != self.dragon.signature:
            raise ValueError("Artifact static-data digest mismatch")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tp.PrefixModel(
            len(self.champ_index),
            len(self.item_index),
            len(self.label_ids),
            len(self.item_index) + 1,
            d_model=int(config.get("d_model", 64)),
            layers=int(config.get("layers", 2)),
            ff_dim=int(config.get("ff_dim", 128)),
            heads=int(config.get("heads", 4)),
            query_dim=tp.QUERY_DIM,
        ).to(self.device)
        missing, unexpected = self.model.load_state_dict(blob["state_dict"], strict=False)
        assert not unexpected, f"artifact has unknown weights: {unexpected}"
        # heads the artifact predates may be missing (their outputs are ignored);
        # anything the config claims to have trained must be present
        allowed_missing = set()
        if not self.has_counts:
            allowed_missing.update(k for k in missing if k.startswith("count_head."))
        if not self.has_save:
            allowed_missing.update(k for k in missing if k.startswith("save_head."))
        if not self.has_target:
            allowed_missing.update(k for k in missing if k.startswith("target_head."))
        hard_missing = [k for k in missing if k not in allowed_missing]
        assert not hard_missing, f"artifact missing trained weights: {hard_missing}"
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

    def _probs(self, row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(affordability-masked probs, raw probs, copies per item or None).
        The raw ones rank what the model wants regardless of current gold —
        needed for save/target logic, where the wanted item is often exactly
        the one the mask zeroes out. Copies come from the count head (multiset
        artifacts only): how many of an item this visit should buy."""
        from app.shop_econ import SAVE_ITEM

        # Re-assert this artifact's featurization before every forward pass.
        # The knobs ShopDataset reads are module globals, so a second Predictor
        # loaded in the same process (comparing two candidates, say) would
        # otherwise leave the first one featurizing rows for the wrong model —
        # silently, with plausible numbers.
        tp.apply_config(self.config)
        ds = tp.ShopDataset([row], self.champ_index, self.item_index, self.label_index, self.dragon)
        batch = next(ds.batches(1, shuffle=False))
        with torch.no_grad():
            out = self.model(
                batch["champs"].to(self.device),
                batch["sides"].to(self.device),
                batch["items"].to(self.device),
                batch["query"].to(self.device),
                batch["inv"].to(self.device),
                batch["hist"].to(self.device),
            )
            logits = out["items"]
            save_q = None
            if self.has_target and self.target_blend > 0:
                if not hasattr(self, "_tmatrix"):
                    self._tmatrix = tp.target_matrix(self.label_ids, self.dragon).to(self.device)
                mass = torch.softmax(out["target"], dim=1) @ self._tmatrix
                logits = logits + self.target_blend * torch.log(mass + 1e-4)
            # splice AFTER the blend: SAVE has no recipe, plan mass must not punish it
            if self.has_save:
                save_q = float(torch.sigmoid(out["save"])[0, 0])  # post-death no-buy score
                # save_splice False keeps the item head's own SAVE ranking (it
                # handles the ward-vs-no-buy mix better, measured); the head
                # still supplies save_q for gating and honest display
                if self.config.get("save_splice", True):
                    save_idx = self.label_index.get(SAVE_ITEM)
                    tilt = float(self.config.get("save_tilt") or tp.SAVE_TILT)
                    logits = tp.splice_save(logits, out["save"], save_idx, tilt=tilt)
            raw = torch.sigmoid(logits)[0].cpu()
            masked = torch.sigmoid(logits.masked_fill(~batch["legal"].to(self.device), -1e4))[0].cpu()
            want = (out["counts"].argmax(dim=2) + 1)[0].cpu() if self.has_counts else None
        return masked, raw, want, save_q

    def _target_probs(self, row: dict) -> torch.Tensor:
        """Next-final probabilities at the observed state.

        The previous decoder changed gold to 3,500 while leaving minute, CS,
        board and history frozen, then called the result a plan.  That was an
        off-manifold counterfactual.  The target head is trained directly for
        the next completed final and has no affordability mask, so it is the
        only defensible source for a plan arrow at the current state.
        """
        if not self.has_target:
            return torch.zeros(len(self.label_ids))
        tp.apply_config(self.config)
        ds = tp.ShopDataset([row], self.champ_index, self.item_index, self.label_index, self.dragon)
        batch = next(ds.batches(1, shuffle=False))
        with torch.no_grad():
            out = self.model(
                batch["champs"].to(self.device),
                batch["sides"].to(self.device),
                batch["items"].to(self.device),
                batch["query"].to(self.device),
                batch["inv"].to(self.device),
                batch["hist"].to(self.device),
            )
        return torch.softmax(out["target"], dim=1)[0].cpu()

    def telemetry_metadata(self) -> dict[str, Any]:
        """Stable, non-marketing provenance attached to every live query."""
        return {
            "artifact_name": self.artifact.name,
            "artifact_sha256": self.artifact_sha256,
            "trained_at": self.trained_at,
            "provenance_status": self.provenance.get("status"),
            "gold_input": self.provenance.get("gold_input"),
            "scored_on": self.provenance.get("scored_on"),
            "save_target": self.config.get("save_target") or "legacy_unknown",
            "deployment_eligible": bool(self.deployment.get("eligible")),
        }

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

        from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, combine_cost, is_blocked, try_purchase

        first_probs, _, want, save_q = self._probs(row)
        inventory = [int(i) for i in (row.get("inventory") or []) if i]
        gold = float(row.get("gold") or 0)
        start_budget = gold + (GOLD_DRIFT if budget_slack is None else budget_slack)

        def buyable(item_id: int, inv: list[int], budget: float, protected=None) -> tuple[int, Counter] | None:
            return try_purchase(item_id, inv, budget, self.dragon,
                                role=row.get("role") or "",
                                bot_quest_complete=row.get("bot_quest_complete"),
                                unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                                role_slot_inventory_unknown=bool(row.get("role_slot_inventory_unknown")),
                                protected=protected)

        # Plan arrows come only from the directly supervised next-final head at
        # the observed state, never a fabricated high-gold probe.
        save_floor = self.threshold * 0.55
        override = min(0.97, self.threshold + 0.18)

        plan = self._target_probs(row)
        # Target-head scores are a softmax over finals, whereas the old plan
        # probe returned independent sigmoids. Ranking is therefore the only
        # scale-stable way to select candidate plans; reusing the sigmoid
        # threshold would regularly leave no final at all.
        finals = []
        for fid, comps in self._final_components.items():
            p = float(plan[self.label_index[fid]])
            if not is_blocked(fid, inventory, self.dragon):
                finals.append((p, fid, comps))
        finals = sorted(finals, reverse=True)[:4]
        # Recipe consistency only permits buys that advance a ranked final (or
        # a standalone value item), preventing a disconnected component from
        # spending a small budget.
        allowed: set[int] = set(self._standalone)
        for _p, fid, comps in finals:
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
        protected = Counter()
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
            if item_id not in allowed and prob < override:
                continue
            buy = buyable(item_id, sim_inv, budget, protected)
            if buy is None:
                continue
            cost, inv_c = buy
            basket.append({"item_id": item_id, "name": self.dragon.item_name(item_id), "score": round(prob, 3), "cost": cost})
            sim_inv = list(inv_c.elements())
            budget -= cost
            protected[item_id] += 1
            # Multiset: the count head says how many copies this visit buys
            # (double Long Sword, second control ward — ~7% of real visits).
            copies = int(want[idx]) if want is not None else 1
            for _extra in range(copies - 1):
                if len(basket) >= max_items:
                    break
                buy = buyable(item_id, sim_inv, budget, protected)
                if buy is None:
                    break
                cost, inv_c = buy
                basket.append({"item_id": item_id, "name": self.dragon.item_name(item_id), "score": round(prob, 3), "cost": cost})
                sim_inv = list(inv_c.elements())
                budget -= cost
                protected[item_id] += 1

        # "Building toward" arrows: for each non-final buy, the most probable
        # wanted final item whose recipe contains it.
        for entry in basket:
            entry["target"] = None
            if entry["item_id"] in self._final_components:
                continue  # already a full item
            for p, fid, comps in finals:
                if entry["item_id"] in comps:
                    entry["target"] = {"item_id": fid, "name": self.dragon.item_name(fid), "score": round(p, 3)}
                    break

        # Save option: the model predicting the SAVE action outright, or nothing
        # affordable clearing the bar while a wanted final is out of gold reach.
        save = None
        if not basket and finals:
            p, fid, _comps = finals[0]
            cost = combine_cost(fid, Counter(inventory), self.dragon)
            if (
                model_save_prob >= self.threshold
                or (save_q is not None and save_q >= 0.5)  # historical post-death score only
                or (p >= save_floor and cost > gold)
            ):
                save = {
                    "item_id": fid,
                    "name": self.dragon.item_name(fid),
            # This value is a historical post-death no-buy score, not a
            # probability of any generic manual recall.
                    "score": round(save_q if save_q is not None else max(p, model_save_prob), 3),
                    "need": max(0, int(cost - gold)),
                }

        return {"top": top, "basket": basket, "save": save, "threshold": self.threshold}

    def predict_options(
        self,
        row: dict,
        n_options: int = 3,
        budget_slack: float | None = None,
        allow_save: bool = False,
    ) -> list[dict]:
        """The 3-basket view: distinct shopping plans, no probabilities exposed.

        Option 1 is the model's basket; option 2 re-plans with option 1's lead
        item banned (a genuinely different line). A hold card is opt-in: the
        historical save head is trained only on post-death no-buys, so it must
        never silently appear outside an explicit decision session.
        """
        from collections import Counter

        from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, combine_cost, is_blocked, try_purchase

        first_probs, _, want, save_q = self._probs(row)
        inventory = [int(i) for i in (row.get("inventory") or []) if i]
        gold = float(row.get("gold") or 0)
        start_budget = gold + (GOLD_DRIFT if budget_slack is None else budget_slack)

        plan = self._target_probs(row)
        # See predict(): target-head values are a softmax, so retain the top
        # legal plans by rank rather than applying an item-sigmoid cutoff.
        finals = []
        for fid, comps in self._final_components.items():
            p = float(plan[self.label_index[fid]])
            if not is_blocked(fid, inventory, self.dragon):
                finals.append((p, fid, comps))
        finals = sorted(finals, reverse=True)[:4]
        allowed: set[int] = set(self._standalone)
        for _p, fid, comps in finals:
            allowed |= comps | {fid}

        def greedy(banned: set[int], bar: float = 0.7) -> list[dict]:
            basket: list[dict] = []
            sim_inv = list(inventory)
            budget = start_budget
            protected = Counter()
            for idx in torch.argsort(first_probs, descending=True).tolist():
                prob = float(first_probs[idx])
                if prob < self.threshold * bar or len(basket) >= int(self.config.get('max_basket_items',4)):
                    break
                item_id = self.label_ids[idx]
                if item_id == SAVE_ITEM or item_id in banned:
                    continue
                if item_id not in allowed and prob < min(0.97, self.threshold + 0.18):
                    continue
                copies = int(want[idx]) if want is not None else 1
                for _c in range(copies):
                    if len(basket) >= int(self.config.get('max_basket_items',4)):
                        break
                    buy = try_purchase(item_id, sim_inv, budget, self.dragon,
                                       role=row.get("role") or "",
                                       bot_quest_complete=row.get("bot_quest_complete"),
                                       unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                                       role_slot_inventory_unknown=bool(row.get("role_slot_inventory_unknown")),
                                       protected=protected)
                    if buy is None:
                        break
                    cost, inv_c = buy
                    entry = self._item_payload(item_id, prob, sim_inv)
                    entry["cost"] = cost
                    for p, fid, comps in finals:
                        if item_id in comps and item_id != fid:
                            entry["target"] = {"item_id": fid, "name": self.dragon.item_name(fid)}
                            break
                    basket.append(entry)
                    sim_inv = list(inv_c.elements())
                    budget -= cost
                    protected[item_id] += 1
            return basket

        options: list[dict] = []
        banned: set[int] = set()
        tiers = ["best", "alternative", "situational"]
        save_used = False
        bar = 0.7
        for slot in range(n_options):
            # the last slot goes to the save card when the head says it's live
            if allow_save and not save_used and save_q is not None and (
                (save_q >= 0.5 and not options) or (slot == n_options - 1 and save_q >= 0.3)
            ):
                toward = None
                if finals:
                    p, fid, _c = finals[0]
                    need = combine_cost(fid, Counter(inventory), self.dragon) - gold
                    toward = {"item_id": fid, "name": self.dragon.item_name(fid), "need": max(0, int(need))}
                options.append({"kind": "save", "tier": tiers[min(slot, 2)], "items": [], "toward": toward})
                save_used = True
                continue
            basket = greedy(banned, bar)
            if not basket:
                break
            options.append({"kind": "buy", "tier": tiers[min(slot, 2)], "items": basket, "toward": None})
            banned.add(basket[0]["item_id"])
        if not options:
            # a visit almost always deserves advice: relax the bar once, then
            # fall back to the save card (low gold usually IS a hold spot)
            basket = greedy(set(), bar=0.4)
            if basket:
                options.append({"kind": "buy", "tier": "situational", "items": basket, "toward": None})
            elif allow_save and save_q is not None:
                toward = None
                if finals:
                    p, fid, _c = finals[0]
                    need = combine_cost(fid, Counter(inventory), self.dragon) - gold
                    toward = {"item_id": fid, "name": self.dragon.item_name(fid), "need": max(0, int(need))}
                options.append({"kind": "save", "tier": "best", "items": [], "toward": toward})
        # Scores are ranking signals from weighted BCE, not probabilities; the
        # user-facing options deliberately expose neither values nor labels.
        for opt in options:
            for it in opt["items"]:
                it.pop("score", None)
        return options

    def _item_payload(self, item_id: int, score: float, inventory: list[int]) -> dict:
        from app.shop_econ import buy_cost

        return {
            "item_id": item_id,
            "name": self.dragon.item_name(item_id),
            "score": round(score, 3),
            "cost": buy_cost(item_id, inventory, self.dragon),
        }
