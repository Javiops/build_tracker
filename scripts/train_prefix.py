"""Train a small sequence model over earlier shops (the prefix) and score it like the forest."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import GOLD_DRIFT, combine_cost, damage_profile

ML_DIR = DATA_DIR / "ml"
MAX_OTHERS = 9
BOARD = MAX_OTHERS + 1  # token 0 is the shopper, then the 9 others
MAX_ITEMS = 6
D_MODEL = 64
BATCH = 256
# Experiment knobs (env): PREFIX_EPOCHS, PREFIX_COSINE=1, PREFIX_EXTRAS=0 to
# zero out the game-state/damage-profile features while keeping QUERY_DIM fixed,
# PREFIX_HISTORY=1 to feed the player's prior purchases as extra tokens,
# PREFIX_OUT to name the saved model file.
EPOCHS = int(os.environ.get("PREFIX_EPOCHS", "4"))
USE_COSINE = os.environ.get("PREFIX_COSINE") == "1"
USE_EXTRAS = os.environ.get("PREFIX_EXTRAS", "1") != "0"
USE_HISTORY = os.environ.get("PREFIX_HISTORY") == "1"
HIST_LEN = 12
SEED = 16
QUERY_DIM = 22
BASKET_THRESHOLD = 0.5
CACHE_VERSION = "v2"
ROLES = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "UTILITY": 5}
# side ids: 0 pad, 1 ally, 2 enemy, 3 self, 4 lane opponent, 5 history token
SIDE_ALLY, SIDE_ENEMY, SIDE_SELF, SIDE_LANE_OPP, SIDE_HISTORY = 1, 2, 3, 4, 5


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    keep = (
        "match_id",
        "champion",
        "champion_id",
        "role",
        "team_id",
        "gold",
        "total_gold",
        "ally_obj",
        "enemy_obj",
        "level",
        "ts",
        "kills",
        "deaths",
        "inventory",
        "label_id",
        "label_ids",
        "can_complete",
        "n_completable",
        "cheapest_complete",
        "gold_after_complete",
        "n_inventory",
        "decision",
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            others = []
            for player in (raw.get("others") or [])[:MAX_OTHERS]:
                others.append(
                    {
                        "champion": player.get("champion") or "",
                        "champion_id": player.get("champion_id"),
                        "team_id": player.get("team_id"),
                        "role": player.get("role") or "",
                        "level": player.get("level"),
                        "gold": player.get("gold"),
                        "items": (player.get("items") or [])[:MAX_ITEMS],
                    }
                )
            row = {key: raw.get(key) for key in keep}
            row["others"] = others
            ids = raw.get("label_ids")
            if not ids:
                ids = [raw["label_id"]] if raw.get("label_id") else []
            row["label_ids"] = [int(i) for i in ids if i]
            rows.append(row)
    return rows


def attach_history(rows: list[dict]) -> None:
    """Prior purchases of the same player in the same game, oldest first.

    Rows are exported in chronological order per match, so a single pass works.
    """
    seen: dict[tuple, list[int]] = {}
    for row in rows:
        hist = seen.setdefault((row.get("match_id"), row.get("champion_id")), [])
        row["prefix_items"] = list(hist[-HIST_LEN:])
        hist.extend(row.get("label_ids") or [])


def score_guesses(rows: list[dict], guesses: list[list[int]]) -> tuple[float, float]:
    hits1 = hits3 = 0
    for row, guess in zip(rows, guesses):
        if guess and row["label_id"] == guess[0]:
            hits1 += 1
        if row["label_id"] in guess:
            hits3 += 1
    n = len(rows) or 1
    return hits1 / n, hits3 / n


def score_basket(rows: list[dict], guesses: list[list[int]]) -> None:
    rec = prec = full = 0.0
    n = 0
    for row, guess in zip(rows, guesses):
        actual = {int(i) for i in (row.get("label_ids") or [row["label_id"]]) if i}
        pred = {int(i) for i in guess if i}
        if not actual:
            continue
        n += 1
        hit = len(actual & pred)
        rec += hit / len(actual)
        prec += hit / max(len(pred), 1)
        full += float(actual <= pred)
    n = n or 1
    print("  basket  (all items they bought this back, vs our top 3)")
    print(f"    cover   {rec / n:.2f}   of their buys are in our top 3")
    print(f"    precision {prec / n:.2f}   of our top 3 were actually bought")
    print(f"    full set {full / n:.2f}   we named every item they bought")


def baseline_guesses(train: list[dict], test: list[dict]) -> list[list[int]]:
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for row in train:
        by_champ[row["champion"]][row["label_id"]] += 1
        global_counts[row["label_id"]] += 1
    fallback = [item_id for item_id, _n in global_counts.most_common(3)]
    out = []
    for row in test:
        local = [item_id for item_id, _n in by_champ[row["champion"]].most_common(3)]
        extra = [item_id for item_id in fallback if item_id not in local]
        out.append((local + extra)[:3])
    return out


def _cache_path(split: str, source: Path) -> Path:
    stat = source.stat()
    key = (
        f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}:{QUERY_DIM}:{HIST_LEN}:"
        f"{int(USE_EXTRAS)}:{int(USE_HISTORY)}:{GOLD_DRIFT}:{CACHE_VERSION}"
    )
    return ML_DIR / f"tensors_{split}_{hashlib.md5(key.encode()).hexdigest()[:12]}.pt"


def _dataset(split: str, source: Path, rows, champ_index, item_index, label_index, dragon):
    cache = _cache_path(split, source)
    if cache.exists():
        print(f"loaded cached tensors {cache.name}", flush=True)
        return ShopDataset.from_cache(cache, champ_index, item_index, label_index, dragon)
    ds = ShopDataset(rows, champ_index, item_index, label_index, dragon)
    for old in ML_DIR.glob(f"tensors_{split}_*.pt"):
        if old != cache:
            old.unlink()
    ds.save_cache(cache)
    return ds


def _index_map(values: list[int]) -> dict[int, int]:
    unique = sorted({int(v) for v in values if v})
    return {item_id: i + 1 for i, item_id in enumerate(unique)}


class ShopDataset:
    """Precomputes every tensor once so training is GPU-bound, not Python-bound."""

    def __init__(
        self,
        rows: list[dict],
        champ_index: dict[int, int],
        item_index: dict[int, int],
        label_index: dict[int, int],
        dragon,
    ):
        self.champ_index = champ_index
        self.item_index = item_index
        self.label_index = label_index
        self.n_label = len(label_index)
        self.dragon = dragon
        n = len(rows)
        self.legal = torch.ones((n, self.n_label), dtype=torch.bool)
        self.champs = torch.zeros((n, BOARD), dtype=torch.int16)
        self.sides = torch.zeros((n, BOARD), dtype=torch.int8)
        self.items = torch.zeros((n, BOARD, MAX_ITEMS), dtype=torch.int16)
        self.hist = torch.zeros((n, HIST_LEN), dtype=torch.int16)
        self.query = torch.zeros((n, QUERY_DIM), dtype=torch.float32)
        self.inv = torch.zeros((n, len(item_index) + 1), dtype=torch.uint8)
        self.targets = torch.zeros((n, self.n_label), dtype=torch.uint8)
        self.budget = torch.zeros(n, dtype=torch.float32)
        self.inventories: list[list[int]] = [[] for _ in range(n)]
        for i, row in enumerate(rows):
            self._fill_row(i, row)

    TENSOR_ATTRS = (
        "legal",
        "champs",
        "sides",
        "items",
        "hist",
        "query",
        "inv",
        "targets",
        "budget",
        "inventories",
    )

    def save_cache(self, path: Path) -> None:
        torch.save({name: getattr(self, name) for name in self.TENSOR_ATTRS}, path)

    @classmethod
    def from_cache(
        cls,
        path: Path,
        champ_index: dict[int, int],
        item_index: dict[int, int],
        label_index: dict[int, int],
        dragon,
    ) -> "ShopDataset":
        ds = cls.__new__(cls)
        ds.champ_index = champ_index
        ds.item_index = item_index
        ds.label_index = label_index
        ds.n_label = len(label_index)
        ds.dragon = dragon
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for name in cls.TENSOR_ATTRS:
            setattr(ds, name, blob[name])
        return ds

    def batches(self, batch_size: int, shuffle: bool):
        n = self.champs.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for s in range(0, n, batch_size):
            idx = order[s : s + batch_size]
            yield {
                "champs": self.champs[idx].long(),
                "sides": self.sides[idx].long(),
                "items": self.items[idx].long(),
                "hist": self.hist[idx].long(),
                "query": self.query[idx],
                "inv": self.inv[idx].float(),
                "legal": self.legal[idx],
                "targets": self.targets[idx].float(),
            }

    def _context_extras(self, row: dict) -> dict:
        team = row.get("team_id")
        role = row.get("role") or ""
        attack = magic = 0.0
        opp_pos = -1
        opp_level = None
        opp_gold = None
        enemy_items: list[int] = []
        ally_gold = float(row.get("total_gold") or 0)
        enemy_gold = 0.0
        gold_known = 0
        for pos, player in enumerate((row.get("others") or [])[:MAX_OTHERS]):
            gold = player.get("gold")
            if player.get("team_id") == team:
                if gold is not None:
                    ally_gold += float(gold)
                    gold_known += 1
                continue
            if gold is not None:
                enemy_gold += float(gold)
                gold_known += 1
            enemy_items.extend(int(i) for i in (player.get("items") or []) if i)
            info = self.dragon.champion_info(int(player.get("champion_id") or 0))
            attack += float(info.get("attack") or 0)
            magic += float(info.get("magic") or 0)
            if role and player.get("role") == role and opp_pos < 0:
                opp_pos = pos
                opp_level = player.get("level")
                opp_gold = gold
        ap_share = magic / (attack + magic) if attack + magic > 0 else 0.5
        level_diff = 0.0
        if opp_level is not None and row.get("level") is not None:
            level_diff = float(row["level"]) - float(opp_level)
        profile = damage_profile(enemy_items, self.dragon)
        dmg_gold = profile["ad"] + profile["ap"]
        self_total = row.get("total_gold")
        return {
            "ap_share": ap_share,
            "opp_pos": opp_pos,
            "level_diff": level_diff,
            "has_opp": 1.0 if opp_pos >= 0 else 0.0,
            "ap_item_share": profile["ap"] / dmg_gold if dmg_gold > 0 else 0.5,
            "resist_gold": float(profile["armor"] + profile["mr"]),
            "team_gold_diff": ally_gold - enemy_gold if gold_known >= 4 else 0.0,
            "opp_gold_diff": (
                float(self_total) - float(opp_gold)
                if self_total is not None and opp_gold is not None
                else 0.0
            ),
        }

    def _fill_row(self, i: int, row: dict) -> None:
        team = row.get("team_id")
        gold = float(row.get("gold") or 0)
        inventory = [int(item_id) for item_id in (row.get("inventory") or []) if item_id]
        inv_c = Counter(inventory)
        self.budget[i] = gold
        self.inventories[i] = inventory
        allowed = gold + GOLD_DRIFT
        for item_id, idx in self.label_index.items():
            if combine_cost(item_id, inv_c, self.dragon) > allowed:
                self.legal[i, idx] = False
        for item_id in row.get("label_ids") or [row.get("label_id")]:
            idx = self.label_index.get(int(item_id or 0))
            if idx is not None:
                self.legal[i, idx] = True

        ex = self._context_extras(row)
        opp_pos = ex["opp_pos"]
        self.champs[i, 0] = self.champ_index.get(int(row.get("champion_id") or 0), 0)
        self.sides[i, 0] = SIDE_SELF
        for j, item_id in enumerate(inventory[:MAX_ITEMS]):
            self.items[i, 0, j] = self.item_index.get(item_id, 0)
        others = (row.get("others") or [])[:MAX_OTHERS]
        for pos, player in enumerate(others):
            slot = pos + 1
            champ = self.champ_index.get(int(player.get("champion_id") or 0), 0)
            if champ == 0:
                champ = self.champ_index.get(hash(player.get("champion") or "") % 100000, 0)
            self.champs[i, slot] = champ
            if player.get("team_id") == team:
                self.sides[i, slot] = SIDE_ALLY
            else:
                self.sides[i, slot] = SIDE_LANE_OPP if pos == opp_pos else SIDE_ENEMY
            for j, item_id in enumerate((player.get("items") or [])[:MAX_ITEMS]):
                self.items[i, slot, j] = self.item_index.get(int(item_id), 0)

        if USE_HISTORY:
            prefix = (row.get("prefix_items") or [])[-HIST_LEN:]
            for j, item_id in enumerate(prefix):
                self.hist[i, j] = self.item_index.get(int(item_id), 0)

        ally_obj = row.get("ally_obj") or {}
        enemy_obj = row.get("enemy_obj") or {}
        if USE_EXTRAS:
            extras = [
                ex["ap_item_share"],
                ex["resist_gold"] / 15000.0,
                ex["team_gold_diff"] / 5000.0,
                ex["opp_gold_diff"] / 3000.0,
                (float(ally_obj.get("towers") or 0) - float(enemy_obj.get("towers") or 0)) / 11.0,
                (float(ally_obj.get("dragons") or 0) - float(enemy_obj.get("dragons") or 0)) / 4.0,
                (float(ally_obj.get("barons") or 0) - float(enemy_obj.get("barons") or 0)) / 2.0,
            ]
        else:
            extras = [0.0] * 7
        self.query[i] = torch.tensor(
            [
                float(ROLES.get(row.get("role") or "", 0)),
                gold / 3000.0,
                float(row.get("level") or 0) / 18.0,
                float((row.get("ts") or 0) / 60000.0),
                float(row.get("kills") or 0) / 10.0,
                float(row.get("deaths") or 0) / 10.0,
                float(len(others)) / MAX_OTHERS,
                float(row.get("n_inventory") or len(row.get("inventory") or [])) / 6.0,
                float(row.get("can_complete") or 0),
                float(row.get("n_completable") or 0) / 8.0,
                float(row.get("cheapest_complete") or 0) / 3000.0,
                float(row.get("gold_after_complete") or gold) / 3000.0,
                ex["ap_share"],
                ex["level_diff"] / 5.0,
                ex["has_opp"],
                *extras,
            ],
            dtype=torch.float32,
        )
        for item_id in inventory:
            idx = self.item_index.get(item_id)
            if idx:
                self.inv[i, idx] = min(int(self.inv[i, idx]) + 1, 3)
        hit = False
        for item_id in row.get("label_ids") or []:
            idx = self.label_index.get(int(item_id))
            if idx is not None:
                self.targets[i, idx] = 1
                hit = True
        if not hit:
            self.targets[i, self.label_index.get(int(row.get("label_id") or 0), 0)] = 1


class PrefixModel(nn.Module):
    def __init__(self, n_champ: int, n_item: int, n_label: int, inv_dim: int):
        super().__init__()
        self.champ_emb = nn.Embedding(n_champ + 1, D_MODEL, padding_idx=0)
        self.item_emb = nn.Embedding(n_item + 1, D_MODEL, padding_idx=0)
        self.side_emb = nn.Embedding(6, D_MODEL, padding_idx=0)
        self.hist_pos = nn.Embedding(HIST_LEN, D_MODEL)
        self.item_mix = nn.Sequential(
            nn.Linear(MAX_ITEMS * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.query_proj = nn.Sequential(
            nn.Linear(QUERY_DIM + inv_dim, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.head = nn.Linear(D_MODEL, n_label)

    def forward(self, champs, sides, items, query, inv, hist):
        item_e = self.item_emb(items)
        empty = items == 0
        item_e = item_e.masked_fill(empty.unsqueeze(-1), 0)
        piled = self.item_mix(item_e.flatten(2))
        board = self.champ_emb(champs) + self.side_emb(sides) + piled
        q = self.query_proj(torch.cat([query, inv], dim=1)).unsqueeze(1)
        positions = torch.arange(HIST_LEN, device=hist.device).unsqueeze(0)
        hist_side = torch.full_like(hist, SIDE_HISTORY)
        hist_tok = self.item_emb(hist) + self.side_emb(hist_side) + self.hist_pos(positions)
        seq = torch.cat([q, board, hist_tok], dim=1)
        never = torch.zeros(sides.size(0), 1, dtype=torch.bool, device=sides.device)
        key_pad = torch.cat([never, sides == 0, hist == 0], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=key_pad)
        return self.head(encoded[:, 0])


def predict_top3(model, dataset: ShopDataset, device, labels: list[int]) -> list[list[int]]:
    model.eval()
    guesses: list[list[int]] = []
    with torch.no_grad():
        for batch in dataset.batches(BATCH, shuffle=False):
            logits = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            legal = batch["legal"].to(device)
            logits = logits.masked_fill(~legal, -1e4)
            top = logits.topk(k=min(3, logits.size(1)), dim=1).indices.cpu().tolist()
            for idxs in top:
                guesses.append([labels[i] for i in idxs])
    return guesses


def predict_baskets(
    model,
    dataset: ShopDataset,
    device,
    labels: list[int],
    dragon,
    threshold: float = 0.5,
    max_items: int = 5,
) -> list[list[int]]:
    """Variable-size basket: greedily take confident items while the gold lasts,
    pricing each pick with the recipe-aware combine cost given the inventory."""
    model.eval()
    baskets: list[list[int]] = []
    r = 0
    with torch.no_grad():
        for batch in dataset.batches(BATCH, shuffle=False):
            logits = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = logits.masked_fill(~batch["legal"].to(device), -1e4)
            probs = torch.sigmoid(logits).cpu()
            for b in range(probs.size(0)):
                row_p = probs[b]
                budget = float(dataset.budget[r]) + GOLD_DRIFT
                inv_c = Counter(dataset.inventories[r])
                basket: list[int] = []
                for idx in torch.argsort(row_p, descending=True).tolist():
                    if float(row_p[idx]) < threshold or len(basket) >= max_items:
                        break
                    item_id = labels[idx]
                    cost = combine_cost(item_id, inv_c, dragon)
                    if cost > budget:
                        continue
                    combine_cost(item_id, inv_c, dragon, consume=True)
                    inv_c[item_id] += 1
                    budget -= cost
                    basket.append(item_id)
                baskets.append(basket)
                r += 1
    return baskets


def score_pred_baskets(rows: list[dict], baskets: list[list[int]], threshold: float) -> None:
    prec = rec = exact = 0.0
    pred_sizes = actual_sizes = 0
    n = 0
    for row, basket in zip(rows, baskets):
        actual = {int(i) for i in (row.get("label_ids") or [row["label_id"]]) if i}
        pred = set(basket)
        if not actual:
            continue
        n += 1
        hit = len(actual & pred)
        rec += hit / len(actual)
        prec += hit / len(pred) if pred else 0.0
        exact += float(pred == actual)
        pred_sizes += len(pred)
        actual_sizes += len(actual)
    n = n or 1
    print(f"  predicted basket  (variable size, threshold {threshold}, budget-constrained)")
    print(f"    recall    {rec / n:.2f}   of their buys are in our basket")
    print(f"    precision {prec / n:.2f}   of our basket was actually bought")
    print(f"    exact set {exact / n:.2f}   basket matches exactly")
    print(f"    avg size  {pred_sizes / n:.2f} predicted vs {actual_sizes / n:.2f} actual")


def per_decision(test: list[dict], guesses: list[list[int]]) -> None:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row, guess in zip(test, guesses):
        kind = row.get("decision") or "?"
        per[kind][2] += 1
        if guess and row["label_id"] == guess[0]:
            per[kind][0] += 1
        if row["label_id"] in guess:
            per[kind][1] += 1
    print("  by decision")
    for kind in ("complete", "component", "start", "save"):
        if kind not in per:
            continue
        c1, c3, count = per[kind]
        print(f"    {kind:<12} top-1 {c1 / count:.2f}  top-3 {c3 / count:.2f}   n={count}")


def per_champ(test: list[dict], guesses: list[list[int]], limit: int = 8) -> None:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row, guess in zip(test, guesses):
        champ = row["champion"] or "?"
        per[champ][2] += 1
        if guess and row["label_id"] == guess[0]:
            per[champ][0] += 1
        if row["label_id"] in guess:
            per[champ][1] += 1
    ranked = sorted(per.items(), key=lambda kv: -kv[1][2])[:limit]
    for champ, (c1, c3, count) in ranked:
        print(f"  {champ:<14} top-1 {c1 / count:.2f}  top-3 {c3 / count:.2f}   n={count}")


def main() -> None:
    train_path = ML_DIR / "visits_train.jsonl"
    test_path = ML_DIR / "visits_test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Run scripts/baseline.py first so the jsonl files exist.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Loading rows…", flush=True)
    train = load_jsonl(train_path)
    print(f"loaded train {len(train)}", flush=True)
    test = load_jsonl(test_path)
    print(f"train {len(train)} shops  test {len(test)} shops", flush=True)
    attach_history(train)
    attach_history(test)

    base = baseline_guesses(train, test)
    b1, b3 = score_guesses(test, base)
    print(flush=True)
    print("baseline  (champion frequency only)", flush=True)
    print(f"  top-1  {b1:.2f}", flush=True)
    print(f"  top-3  {b3:.2f}", flush=True)
    per_decision(test, base)

    champ_ids = [int(row.get("champion_id") or 0) for row in train]
    item_ids = [int(row["label_id"]) for row in train]
    for row in train:
        item_ids.extend(int(i) for i in (row.get("inventory") or []))
        for player in row.get("others") or []:
            champ_ids.append(int(player.get("champion_id") or 0))
            item_ids.extend(int(i) for i in (player.get("items") or []))
    champ_index = _index_map(champ_ids)
    item_index = _index_map(item_ids)
    label_ids = sorted(
        {
            int(item_id)
            for row in train
            for item_id in (row.get("label_ids") or [row["label_id"]])
            if item_id
        }
    )
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}
    counts = torch.zeros(len(label_ids))
    for row in train:
        for item_id in row.get("label_ids") or [row["label_id"]]:
            idx = label_index.get(int(item_id or 0))
            if idx is not None:
                counts[idx] += 1
    pos_weight = ((len(train) - counts) / counts.clamp(min=1.0)).clamp(max=40.0)

    for row in train + test:
        for player in row.get("others") or []:
            if not player.get("champion_id"):
                player["champion_id"] = abs(hash(player.get("champion") or "")) % 10000 + 1
                if player["champion_id"] not in champ_index:
                    champ_index[player["champion_id"]] = len(champ_index) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dragon = default_dragon()
    print(flush=True)
    print(
        f"Board model  basket (all buys)  9 others x {MAX_ITEMS} slots  device={device}  "
        f"epochs={EPOCHS} extras={'on' if USE_EXTRAS else 'off'} cosine={'on' if USE_COSINE else 'off'} "
        f"history={'on' if USE_HISTORY else 'off'}",
        flush=True,
    )
    prep_started = time.monotonic()
    train_ds = _dataset("train", train_path, train, champ_index, item_index, label_index, dragon)
    test_ds = _dataset("test", test_path, test, champ_index, item_index, label_index, dragon)
    print(f"tensors ready in {time.monotonic() - prep_started:.1f}s", flush=True)

    model = PrefixModel(len(champ_index), len(item_index), len(label_ids), len(item_index) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if USE_COSINE else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_ds.batches(BATCH, shuffle=True):
            opt.zero_grad()
            logits = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = logits.masked_fill(~batch["legal"].to(device), -1e4)
            loss = loss_fn(logits, batch["targets"].to(device))
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(batch["targets"])
            n += len(batch["targets"])
        if sched:
            sched.step()
        print(f"  epoch {epoch}/{EPOCHS}  loss {total / max(n, 1):.3f}", flush=True)
    print(f"trained in {time.monotonic() - started:.1f}s", flush=True)

    guesses = predict_top3(model, test_ds, device, label_ids)
    m1, m3 = score_guesses(test, guesses)
    print(flush=True)
    print("board model  (gold + your inventory + lobby builds + full shop basket)", flush=True)
    print(f"  main item top-1  {m1:.2f}   vs baseline {b1:.2f}", flush=True)
    print(f"  main item top-3  {m3:.2f}   vs baseline {b3:.2f}", flush=True)
    print(flush=True)
    score_basket(test, guesses)
    print(flush=True)
    per_decision(test, guesses)
    print(flush=True)
    per_champ(test, guesses)
    print(flush=True)
    baskets = predict_baskets(model, test_ds, device, label_ids, dragon, threshold=BASKET_THRESHOLD)
    score_pred_baskets(test, baskets, BASKET_THRESHOLD)

    out_path = ML_DIR / os.environ.get("PREFIX_OUT", "prefix_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "champ_index": champ_index,
            "item_index": item_index,
            "label_ids": label_ids,
            "config": {
                "d_model": D_MODEL,
                "max_others": MAX_OTHERS,
                "max_items": MAX_ITEMS,
                "query_dim": QUERY_DIM,
                "epochs": EPOCHS,
                "extras": USE_EXTRAS,
                "cosine": USE_COSINE,
                "history": USE_HISTORY,
                "hist_len": HIST_LEN,
                "gold_drift": GOLD_DRIFT,
                "basket_threshold": BASKET_THRESHOLD,
            },
            "metrics": {"top1": m1, "top3": m3, "baseline_top1": b1, "baseline_top3": b3},
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_rows": len(train),
            "test_rows": len(test),
        },
        out_path,
    )
    print(flush=True)
    print(f"saved model -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
