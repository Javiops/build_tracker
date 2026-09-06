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

import math

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import GOLD_DRIFT, PINK, SAVE_ITEM, combine_cost, damage_profile

ML_DIR = DATA_DIR / "ml"
MAX_OTHERS = 9
BOARD = MAX_OTHERS + 1  # token 0 is the shopper, then the 9 others
MAX_ITEMS = 6
BATCH = 256
# Experiment knobs (env): PREFIX_EPOCHS, PREFIX_COSINE=1, PREFIX_EXTRAS=0 to
# zero out the game-state/damage-profile features while keeping QUERY_DIM fixed,
# PREFIX_HISTORY=1 to feed the player's prior purchases as extra tokens,
# PREFIX_OUT to name the saved model file.
# Architecture knobs: PREFIX_DMODEL, PREFIX_LAYERS, PREFIX_FF, PREFIX_HEADS.
# Feature knobs: PREFIX_GOLDEST=1 (interpolated arrival gold replaces the
# frame-stale value as the budget feature), PREFIX_RUNES=1 (keystone/secondary
# tree/summoner spells in the query — requires rune-backfilled export).
D_MODEL = int(os.environ.get("PREFIX_DMODEL", "64"))
N_LAYERS = int(os.environ.get("PREFIX_LAYERS", "2"))
FF_DIM = int(os.environ.get("PREFIX_FF", "128"))
N_HEADS = int(os.environ.get("PREFIX_HEADS", "4"))
EPOCHS = int(os.environ.get("PREFIX_EPOCHS", "4"))
USE_COSINE = os.environ.get("PREFIX_COSINE") == "1"
USE_EXTRAS = os.environ.get("PREFIX_EXTRAS", "1") != "0"
USE_HISTORY = os.environ.get("PREFIX_HISTORY") == "1"
USE_GOLDEST = os.environ.get("PREFIX_GOLDEST") == "1"
USE_RUNES = os.environ.get("PREFIX_RUNES") == "1"
# PREFIX_POSW=1 trains plain unweighted BCE (no rare-item emphasis) — the
# control experiment for confidence trustworthiness; default keeps the
# engraved low cap of 8.
POS_WEIGHT_CAP = float(os.environ.get("PREFIX_POSW", "8"))
# PREFIX_INIT_FROM=<artifact in data/ml> warm-starts from an existing model;
# PREFIX_FREEZE=1 then trains ONLY the save/target heads on the frozen trunk —
# auxiliary heads measurably drag the trunk when trained jointly (models E/F),
# so heads are grafted onto a finished trunk instead.
INIT_FROM = os.environ.get("PREFIX_INIT_FROM", "")
FREEZE_TRUNK = os.environ.get("PREFIX_FREEZE") == "1"
HIST_LEN = 12
SEED = 16
# Fixed one-hot vocabularies for the rune features (index 0 = unknown bucket).
KEYSTONE_IDS = [8005, 8008, 8010, 8021, 8112, 8128, 9923, 8214, 8229, 8230,
                8437, 8439, 8465, 8351, 8360, 8369, 8992]
STYLE_IDS = [8000, 8100, 8200, 8300, 8400]
SUMM_IDS = [1, 3, 4, 6, 7, 11, 12, 14, 21]
RUNE_DIM = (len(KEYSTONE_IDS) + 1) + (len(STYLE_IDS) + 1) + (len(SUMM_IDS) + 1)
BASE_QUERY_DIM = 26
QUERY_DIM = BASE_QUERY_DIM + (RUNE_DIM if USE_RUNES else 0)
BASKET_THRESHOLD = 0.8  # swept 0.5-0.8 on the honest model: best F1/exact-set
CACHE_VERSION = "v7"  # v7: next-completed-target labels for the plan head
COUNT_LOSS_W = 0.25  # weight of the copy-count CE next to the basket BCE
# Aux-head loss weights. Joint training taxes the trunk 4-5pts at ANY weight
# (models E/F), so the canonical pipeline is two-stage: trunk with these at 0,
# then PREFIX_INIT_FROM + PREFIX_FREEZE=1 with them non-zero (the graft).
SAVE_LOSS_W = float(os.environ.get("PREFIX_SAVEW", "0"))
TARGET_LOSS_W = float(os.environ.get("PREFIX_TARGETW", "0"))
# Decode blend: lift each item's logit by log of the target-head probability
# mass of the finals its purchase advances. Measured net-negative on the strict
# metric (ward confound removed) — default off; the head still ships for UI.
TARGET_BLEND = float(os.environ.get("PREFIX_TBLEND", "0"))
# Save splice replaces SAVE's pseudo-item ranking with the head — measured
# worse on the mixed ward/no-buy save class, so default off (head feeds the
# calibrated gate + display instead).
SAVE_SPLICE = os.environ.get("PREFIX_SPLICE") == "1"
MAX_COPIES = 3
# The item head trains with pos_weight, which tilts its logits by +ln(w)
# relative to true probabilities. The save head trains unweighted (honest);
# to let it compete in the item ranking, splice it in with the same tilt.
SAVE_TILT = math.log(max(POS_WEIGHT_CAP, 1.0))


def splice_save(logits, save_logits, save_idx, tilt: float | None = None):
    """Replace SAVE's pseudo-item logit with the save head, lifted onto the
    item head's pos_weight-tilted scale. ln(w) is the honest offset; the
    deployed offset is a decision-theoretic dial calibrated per artifact
    (E's sweep: ~3.5 trades ~1pt elsewhere for +15pts save top-1)."""
    if save_idx is not None:
        logits[:, save_idx] = save_logits.squeeze(1) + (SAVE_TILT if tilt is None else tilt)
    return logits


def target_matrix(labels: list[int], dragon) -> torch.Tensor:
    """M[f, c] = 1 when buying item c advances final f (c is in f's recursive
    recipe, or is f itself). Used to convert the target head's plan probability
    into per-item mass at decode time."""
    from app.shop_econ import components_of

    n = len(labels)
    M = torch.zeros((n, n))
    pos = {item_id: k for k, item_id in enumerate(labels)}
    for k, fid in enumerate(labels):
        M[k, k] = 1.0
        cls = dragon.classify(fid)
        if cls.get("is_completed") or cls.get("is_boots"):
            for c in components_of(fid, dragon):
                j = pos.get(c)
                if j is not None:
                    M[k, j] = 1.0
    # Standalone value items (no recipe in either direction: control wards,
    # Dark Seal, SAVE sentinel) are plan-neutral — mass 1 so the blend never
    # penalizes them. Without this the blend crushed ward-only save visits
    # (Control Ward advances no final → mass ~0 → logit −4.6).
    for j, item_id in enumerate(labels):
        data = dragon.item(item_id) or {}
        if not data.get("from") and not data.get("into"):
            M[:, j] = 1.0
    return M
ROLES = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "UTILITY": 5}
# side ids: 0 pad, 1 ally, 2 enemy, 3 self, 4 lane opponent, 5 history token
SIDE_ALLY, SIDE_ENEMY, SIDE_SELF, SIDE_LANE_OPP, SIDE_HISTORY = 1, 2, 3, 4, 5


def stream_rows(path: Path):
    """Yield visit rows one at a time. The full export no longer fits in RAM as
    Python dicts (1.19M rows ≈ 13 GB on a 16 GB machine — it thrashes), so no
    code path may materialize the whole file; scan it in passes instead."""
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
        "keystone_id",
        "sub_style",
        "summ1",
        "summ2",
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
            yield row


SLIM_KEYS = ("match_id", "champion", "label_id", "label_ids", "decision")


def with_history(rows):
    """Attach prior purchases of the same player in the same game, oldest first.

    Rows are exported in chronological order per match, so a single pass works.
    """
    seen: dict[tuple, list[int]] = {}
    for row in rows:
        hist = seen.setdefault((row.get("match_id"), row.get("champion_id")), [])
        row["prefix_items"] = list(hist[-HIST_LEN:])
        yield row
        hist.extend(row.get("label_ids") or [])


def with_targets(rows, dragon):
    """Attach the player's next completed item (their real plan) to each visit.

    Rows arrive grouped by match and chronological within it, so buffering one
    match at a time gives lookahead without holding the export in RAM. A visit
    that itself completes a final targets that final; a visit with no later
    completion in the game gets target_id 0 (masked from the plan loss).
    """
    final_cache: dict[int, bool] = {}

    def is_final(item_id: int) -> bool:
        if item_id not in final_cache:
            cls = dragon.classify(item_id)
            final_cache[item_id] = bool(cls.get("is_completed") or cls.get("is_boots"))
        return final_cache[item_id]

    def flush(group: list[dict]):
        completions: dict = {}
        for row in group:
            champ = row.get("champion_id")
            for item in row.get("label_ids") or []:
                if is_final(int(item)):
                    completions.setdefault(champ, []).append((row.get("ts") or 0, int(item)))
        pointers: dict = {}
        for row in group:
            champ = row.get("champion_id")
            comps = completions.get(champ) or []
            p = pointers.get(champ, 0)
            ts = row.get("ts") or 0
            while p < len(comps) and comps[p][0] < ts:
                p += 1
            pointers[champ] = p
            row["target_id"] = comps[p][1] if p < len(comps) else 0
            yield row

    group: list[dict] = []
    current = None
    for row in rows:
        if row.get("match_id") != current and group:
            yield from flush(group)
            group = []
        current = row.get("match_id")
        group.append(row)
    if group:
        yield from flush(group)


def fix_champ_ids(rows, champ_index: dict[int, int]):
    """Deterministic-ish fallback ids for board players missing a champion_id
    (mirrors the old pre-dataset fixup loop)."""
    for row in rows:
        for player in row.get("others") or []:
            if not player.get("champion_id"):
                player["champion_id"] = abs(hash(player.get("champion") or "")) % 10000 + 1
                if player["champion_id"] not in champ_index:
                    champ_index[player["champion_id"]] = len(champ_index) + 1
        yield row


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


def baseline_guesses(
    by_champ: dict[str, Counter], global_counts: Counter, test: list[dict]
) -> list[list[int]]:
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
        f"{int(USE_EXTRAS)}:{int(USE_HISTORY)}:{int(USE_GOLDEST)}:{int(USE_RUNES)}:"
        f"{GOLD_DRIFT}:{CACHE_VERSION}"
    )
    return ML_DIR / f"tensors_{split}_{hashlib.md5(key.encode()).hexdigest()[:12]}.pt"


def _dataset(split: str, source: Path, rows_factory, n: int, champ_index, item_index, label_index, dragon):
    cache = _cache_path(split, source)
    if cache.exists():
        print(f"loaded cached tensors {cache.name}", flush=True)
        return ShopDataset.from_cache(cache, champ_index, item_index, label_index, dragon)
    ds = ShopDataset.from_stream(rows_factory(), n, champ_index, item_index, label_index, dragon)
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
        self._init_meta(champ_index, item_index, label_index, dragon)
        self._alloc(len(rows))
        for i, row in enumerate(rows):
            self._fill_row(i, row)

    @classmethod
    def from_stream(cls, rows_iter, n: int, champ_index, item_index, label_index, dragon):
        """Fill tensors from a row generator — never holds the rows in RAM."""
        ds = cls.__new__(cls)
        ds._init_meta(champ_index, item_index, label_index, dragon)
        ds._alloc(n)
        i = -1
        for i, row in enumerate(rows_iter):
            ds._fill_row(i, row)
        assert i + 1 == n, f"stream yielded {i + 1} rows, expected {n}"
        return ds

    def _init_meta(self, champ_index, item_index, label_index, dragon) -> None:
        self.champ_index = champ_index
        self.item_index = item_index
        self.label_index = label_index
        self.n_label = len(label_index)
        self.dragon = dragon
        # Per-label metadata hoisted out of the row loop: prices plus the
        # purchase-block facts (completed flag, boots component set). The fill
        # loop must never call classify/is_blocked per row x label — that is
        # ~200M calls over a full export.
        from app.shop_econ import components_of

        self._label_meta = []
        self._boots_ids: set[int] = set()
        for item_id, idx in label_index.items():
            gold = dragon.gold_block(item_id)
            base = gold["base"]
            total = gold["total"] or base
            cls = dragon.classify(item_id)
            is_completed = bool(cls.get("is_completed"))
            boots_comps = None
            if "Boots" in (cls.get("tags") or []):
                boots_comps = components_of(item_id, dragon)
                self._boots_ids.add(item_id)
            self._label_meta.append((item_id, idx, base, total, is_completed, boots_comps))
        for item_id in item_index:
            if "Boots" in ((dragon.item(item_id) or {}).get("tags") or []):
                self._boots_ids.add(item_id)

    def _alloc(self, n: int) -> None:
        self.legal = torch.ones((n, self.n_label), dtype=torch.bool)
        self.champs = torch.zeros((n, BOARD), dtype=torch.int16)
        self.sides = torch.zeros((n, BOARD), dtype=torch.int8)
        self.items = torch.zeros((n, BOARD, MAX_ITEMS), dtype=torch.int16)
        self.hist = torch.zeros((n, HIST_LEN), dtype=torch.int16)
        self.query = torch.zeros((n, QUERY_DIM), dtype=torch.float32)
        self.inv = torch.zeros((n, len(self.item_index) + 1), dtype=torch.uint8)
        self.targets = torch.zeros((n, self.n_label), dtype=torch.uint8)
        self.budget = torch.zeros(n, dtype=torch.float32)
        self.is_save = torch.zeros(n, dtype=torch.uint8)
        self.target = torch.full((n,), -1, dtype=torch.int16)  # next completed final
        self.inventories: list[list[int]] = [[] for _ in range(n)]

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
        "is_save",
        "target",
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

    def to_device(self, device: torch.device) -> None:
        """Park the precomputed tensors on the device (GPU) so batches index
        there directly — the per-batch CPU gather + copy was the bottleneck."""
        for name in self.TENSOR_ATTRS:
            if name == "budget":  # read scalar-by-scalar in the basket loops — stays CPU
                continue
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def nbytes(self) -> int:
        return sum(
            v.element_size() * v.nelement()
            for v in (getattr(self, name) for name in self.TENSOR_ATTRS)
            if isinstance(v, torch.Tensor)
        )

    def batches(self, batch_size: int, shuffle: bool):
        n = self.champs.size(0)
        dev = self.champs.device
        order = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
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
                "targets": (self.targets[idx] > 0).float(),  # binary: was it bought
                "counts": self.targets[idx].long(),  # copies bought (0..MAX_COPIES)
                "is_save": (self.is_save[idx] > 0).float(),
                "target": self.target[idx].long(),  # next completed final (-1 unknown)
            }

    def _context_extras(self, row: dict) -> dict:
        team = row.get("team_id")
        role = row.get("role") or ""
        attack = magic = 0.0
        opp_pos = -1
        opp_level = None
        opp_gold = None
        opp_cid = 0
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
                opp_cid = int(player.get("champion_id") or 0)
        ap_share = magic / (attack + magic) if attack + magic > 0 else 0.5
        level_diff = 0.0
        if opp_level is not None and row.get("level") is not None:
            level_diff = float(row["level"]) - float(opp_level)
        profile = damage_profile(enemy_items, self.dragon)
        dmg_gold = profile["ad"] + profile["ap"]
        self_total = row.get("total_gold")
        # class-level opponent traits so adaptation generalizes across matchups
        opp_info = self.dragon.champion_info(opp_cid) if opp_cid else {}
        opp_range = float((self.dragon.champion_stats(opp_cid) or {}).get("attackrange") or 0)
        return {
            "opp_is_ranged": 1.0 if opp_range >= 300 else 0.0,
            "opp_attack": float(opp_info.get("attack") or 0) / 10.0,
            "opp_magic": float(opp_info.get("magic") or 0) / 10.0,
            "opp_defense": float(opp_info.get("defense") or 0) / 10.0,
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
        inv_set = set(inventory)
        owned_boots = [b for b in inventory if b in self._boots_ids]
        self.budget[i] = gold
        self.inventories[i] = inventory
        allowed = gold + GOLD_DRIFT
        for item_id, idx, base, total, is_completed, boots_comps in self._label_meta:
            # purchase blocks from precomputed metadata — no per-row classify
            if is_completed and item_id in inv_set:
                self.legal[i, idx] = False
                continue
            if boots_comps is not None and any(b not in boots_comps for b in owned_boots):
                self.legal[i, idx] = False
                continue
            # cost is always in [base, total]: only the band in between needs the
            # recursive combine walk, which keeps this loop fast.
            if total <= allowed:
                continue
            if base > allowed:
                self.legal[i, idx] = False
            elif combine_cost(item_id, inv_c, self.dragon) > allowed:
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

        # Tighter arrival gold (leak-free): the stored gold is from the frame
        # BEFORE the visit, up to 60s stale. Add estimated income for the gap —
        # average earn rate so far (totalGold / frame time, spending-invariant)
        # times seconds since the frame. Live rows carry exact gold (gold_exact)
        # and skip this. Feature only: legality masks stay on gold + GOLD_DRIFT.
        gold_feat = gold
        if USE_GOLDEST and not row.get("gold_exact"):
            ts_ms = float(row.get("ts") or 0)
            delta_s = (ts_ms % 60000.0) / 1000.0
            frame_s = (ts_ms - ts_ms % 60000.0) / 1000.0
            total = row.get("total_gold")
            rate = float(total) / frame_s if total and frame_s >= 60.0 else 2.2
            rate = min(max(rate, 1.8), 8.0)
            gold_feat = gold + rate * delta_s

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
        if USE_RUNES:
            keystone_vec = [0.0] * (len(KEYSTONE_IDS) + 1)
            k = int(row.get("keystone_id") or 0)
            keystone_vec[KEYSTONE_IDS.index(k) + 1 if k in KEYSTONE_IDS else 0] = 1.0
            style_vec = [0.0] * (len(STYLE_IDS) + 1)
            s = int(row.get("sub_style") or 0)
            style_vec[STYLE_IDS.index(s) + 1 if s in STYLE_IDS else 0] = 1.0
            summ_vec = [0.0] * (len(SUMM_IDS) + 1)
            for sid in (row.get("summ1"), row.get("summ2")):
                sid = int(sid or 0)
                summ_vec[SUMM_IDS.index(sid) + 1 if sid in SUMM_IDS else 0] = 1.0
            extras = extras + keystone_vec + style_vec + summ_vec
        self.query[i] = torch.tensor(
            [
                float(ROLES.get(row.get("role") or "", 0)),
                gold_feat / 3000.0,
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
                ex["opp_is_ranged"],
                ex["opp_attack"],
                ex["opp_magic"],
                ex["opp_defense"],
                *extras,
            ],
            dtype=torch.float32,
        )
        for item_id in inventory:
            idx = self.item_index.get(item_id)
            if idx:
                self.inv[i, idx] = min(int(self.inv[i, idx]) + 1, 3)
        hit = False
        self.is_save[i] = int(
            row.get("decision") == "save"
            or any(int(x) == SAVE_ITEM for x in (row.get("label_ids") or []))
        )
        tgt = self.label_index.get(int(row.get("target_id") or 0))
        if tgt is not None:
            self.target[i] = tgt
        for item_id in row.get("label_ids") or []:
            idx = self.label_index.get(int(item_id))
            if idx is not None:
                # multiset target: copy count, capped (duplicates are real buys)
                self.targets[i, idx] = min(int(self.targets[i, idx]) + 1, MAX_COPIES)
                hit = True
        if not hit:
            self.targets[i, self.label_index.get(int(row.get("label_id") or 0), 0)] = 1


class PrefixModel(nn.Module):
    def __init__(
        self,
        n_champ: int,
        n_item: int,
        n_label: int,
        inv_dim: int,
        d_model: int = D_MODEL,
        layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        heads: int = N_HEADS,
        query_dim: int | None = None,
    ):
        super().__init__()
        query_dim = QUERY_DIM if query_dim is None else query_dim
        self.champ_emb = nn.Embedding(n_champ + 1, d_model, padding_idx=0)
        self.item_emb = nn.Embedding(n_item + 1, d_model, padding_idx=0)
        self.side_emb = nn.Embedding(6, d_model, padding_idx=0)
        self.hist_pos = nn.Embedding(HIST_LEN, d_model)
        self.item_mix = nn.Sequential(
            nn.Linear(MAX_ITEMS * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim + inv_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # head sees the query token AND the encoded lane-opponent token directly,
        # so matchup adaptation doesn't depend on one attention hop being learned
        self.head = nn.Linear(2 * d_model, n_label)
        # multiset: per item, how many copies this visit buys (1..MAX_COPIES),
        # trained only on bought items — the binary head decides *whether*
        self.count_head = nn.Linear(2 * d_model, n_label * MAX_COPIES)
        # save: "this visit buys nothing" as its own calibrated binary — SAVE
        # as a pseudo-item can't calibrate across states (measured AUC 0.58)
        self.save_head = nn.Linear(2 * d_model, 1)
        # plan: softmax over which final the player completes next — explicit
        # supervision for the component-altitude problem (right plan, wrong piece)
        self.target_head = nn.Linear(2 * d_model, n_label)
        self.n_label = n_label

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
        board_enc = encoded[:, 1 : BOARD + 1]
        opp_mask = (sides == SIDE_LANE_OPP).unsqueeze(-1)
        opp_vec = (board_enc * opp_mask).sum(dim=1)  # zeros when no lane opponent
        feats = torch.cat([encoded[:, 0], opp_vec], dim=1)
        return {
            "items": self.head(feats),
            "counts": self.count_head(feats).view(-1, self.n_label, MAX_COPIES),
            "save": self.save_head(feats),
            "target": self.target_head(feats),
        }


def predict_top3(
    model, dataset: ShopDataset, device, labels: list[int], blend: float = 0.0, tmatrix=None
) -> list[list[int]]:
    model.eval()
    guesses: list[list[int]] = []
    save_idx = labels.index(SAVE_ITEM) if SAVE_ITEM in labels else None
    with torch.no_grad():
        for batch in dataset.batches(BATCH, shuffle=False):
            out = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = out["items"]
            if blend > 0 and tmatrix is not None:
                mass = torch.softmax(out["target"], dim=1) @ tmatrix
                logits = logits + blend * torch.log(mass + 1e-4)
            # splice AFTER the blend: SAVE has no recipe, so plan mass must
            # never penalize it (blend-before-splice zeroed save top-1)
            if SAVE_SPLICE:
                logits = splice_save(logits, out["save"], save_idx)
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
    """Variable-size multiset basket: greedily take confident items while the
    gold lasts (recipe-aware pricing); the count head decides how many copies
    of each taken item this visit buys."""
    model.eval()
    baskets: list[list[int]] = []
    save_idx = labels.index(SAVE_ITEM) if SAVE_ITEM in labels else None
    r = 0
    with torch.no_grad():
        for batch in dataset.batches(BATCH, shuffle=False):
            out = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = splice_save(out["items"], out["save"], save_idx) if SAVE_SPLICE else out["items"]
            logits = logits.masked_fill(~batch["legal"].to(device), -1e4)
            probs = torch.sigmoid(logits).cpu()
            want = (out["counts"].argmax(dim=2) + 1).cpu()  # copies per item, 1..MAX_COPIES
            for b in range(probs.size(0)):
                row_p = probs[b]
                budget = float(dataset.budget[r]) + GOLD_DRIFT
                inv_c = Counter(dataset.inventories[r])
                basket: list[int] = []
                for idx in torch.argsort(row_p, descending=True).tolist():
                    if float(row_p[idx]) < threshold or len(basket) >= max_items:
                        break
                    item_id = labels[idx]
                    for _copy in range(int(want[b, idx])):
                        if len(basket) >= max_items:
                            break
                        cost = combine_cost(item_id, inv_c, dragon)
                        if cost > budget:
                            break
                        combine_cost(item_id, inv_c, dragon, consume=True)
                        inv_c[item_id] += 1
                        budget -= cost
                        basket.append(item_id)
                baskets.append(basket)
                r += 1
    return baskets


def calibrate_threshold(
    model, dataset: ShopDataset, rows: list[dict], labels: list[int], dragon, sample: int = 20000
) -> float:
    """Pick the basket threshold by F1 on a test sample. Retrains that change
    the probability scale (pos_weight, new classes) recalibrate automatically."""
    n = min(sample, len(rows))
    probs_all = []
    device = next(model.parameters()).device
    model.eval()
    seen = 0
    save_idx = labels.index(SAVE_ITEM) if SAVE_ITEM in labels else None
    with torch.no_grad():
        for batch in dataset.batches(BATCH, shuffle=False):
            out = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = splice_save(out["items"], out["save"], save_idx) if SAVE_SPLICE else out["items"]
            logits = logits.masked_fill(~batch["legal"].to(device), -1e4)
            probs_all.append(torch.sigmoid(logits).cpu())
            seen += logits.size(0)
            if seen >= n:
                break
    probs_all = torch.cat(probs_all)[:n]
    order_all = torch.argsort(probs_all, dim=1, descending=True)
    best_tau, best_f1 = 0.5, -1.0
    for tau in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        f1 = 0.0
        scored = 0
        for r in range(n):
            actual = {int(i) for i in (rows[r].get("label_ids") or []) if i}
            if not actual:
                continue
            row_p = probs_all[r]
            budget = float(dataset.budget[r]) + GOLD_DRIFT
            inv_c = Counter(dataset.inventories[r])
            basket: set[int] = set()
            for idx in order_all[r].tolist():
                if float(row_p[idx]) < tau or len(basket) >= 5:
                    break
                item_id = labels[idx]
                cost = combine_cost(item_id, inv_c, dragon)
                if cost > budget:
                    continue
                combine_cost(item_id, inv_c, dragon, consume=True)
                inv_c[item_id] += 1
                budget -= cost
                basket.add(item_id)
            hit = len(actual & basket)
            p = hit / len(basket) if basket else 0.0
            q = hit / len(actual)
            f1 += 2 * p * q / (p + q) if p + q > 0 else 0.0
            scored += 1
        f1 /= max(scored, 1)
        print(f"  calibration tau {tau:.2f}  f1 {f1:.3f}", flush=True)
        if f1 > best_f1:
            best_tau, best_f1 = tau, f1
    print(f"  calibrated basket threshold: {best_tau:.2f} (f1 {best_f1:.3f})", flush=True)
    return best_tau


def score_pred_baskets(rows: list[dict], baskets: list[list[int]], threshold: float) -> None:
    """Multiset scoring: Long Sword x2 is only fully covered by predicting both
    copies; recall/precision count copies via Counter intersection."""
    prec = rec = exact = 0.0
    pred_sizes = actual_sizes = 0
    n = 0
    dup_n = dup_exact = 0.0
    for row, basket in zip(rows, baskets):
        actual = Counter(int(i) for i in (row.get("label_ids") or [row["label_id"]]) if i)
        pred = Counter(basket)
        if not actual:
            continue
        n += 1
        hit = sum((actual & pred).values())
        rec += hit / sum(actual.values())
        prec += hit / sum(pred.values()) if pred else 0.0
        exact += float(pred == actual)
        pred_sizes += sum(pred.values())
        actual_sizes += sum(actual.values())
        if any(c > 1 for c in actual.values()):
            dup_n += 1
            dup_exact += float(pred == actual)
    n = n or 1
    print(f"  predicted basket  (multiset, threshold {threshold}, budget-constrained)")
    print(f"    recall    {rec / n:.2f}   of their buys (copies counted) are in our basket")
    print(f"    precision {prec / n:.2f}   of our basket was actually bought")
    print(f"    exact     {exact / n:.2f}   multiset matches exactly")
    print(f"    avg size  {pred_sizes / n:.2f} predicted vs {actual_sizes / n:.2f} actual")
    if dup_n:
        print(f"    duplicate-visits {int(dup_n)}  exact-on-duplicates {dup_exact / dup_n:.2f}")


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
    # save_auxiliary: "save" visits are no-buys AND ward-only buys; recommending
    # either action there is the same advice (hold gold, ward up), so both count.
    aux_ok = set(PINK) | {SAVE_ITEM}
    a1 = a3 = n_aux = 0
    for row, guess in zip(test, guesses):
        if row.get("decision") != "save":
            continue
        n_aux += 1
        a1 += bool(guess) and guess[0] in aux_ok
        a3 += any(g in aux_ok for g in guess)
    if n_aux:
        print(f"    {'save_aux':<12} top-1 {a1 / n_aux:.2f}  top-3 {a3 / n_aux:.2f}   n={n_aux}  (ward or save both correct)")


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


def _windows_full_speed() -> None:
    """Background processes get EcoQoS/low memory priority on Windows 11 and
    run several times slower; opt this process out from the inside."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.GetCurrentProcess()

        class PPTS(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_uint32),
                ("ControlMask", ctypes.c_uint32),
                ("StateMask", ctypes.c_uint32),
            ]

        state = PPTS(1, 1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED off
        k32.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))

        class MPI(ctypes.Structure):
            _fields_ = [("MemoryPriority", ctypes.c_uint32)]

        mem = MPI(5)  # MEMORY_PRIORITY_NORMAL
        k32.SetProcessInformation(handle, 0, ctypes.byref(mem), ctypes.sizeof(mem))
        k32.SetPriorityClass(handle, 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def main() -> None:
    _windows_full_speed()
    train_path = ML_DIR / "visits_train.jsonl"
    test_path = ML_DIR / "visits_test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Run scripts/baseline.py first so the jsonl files exist.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Pass 1 (streaming): vocabulary, label counts, and baseline counters from
    # the train export; only slim scoring rows are retained for the test set.
    # Full rows are never materialized (they do not fit in RAM).
    print("Scanning train export…", flush=True)
    champ_ids: set[int] = set()
    item_ids: set[int] = set()
    label_counts: Counter = Counter()
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    n_train = 0
    for row in stream_rows(train_path):
        n_train += 1
        champ_ids.add(int(row.get("champion_id") or 0))
        item_ids.add(int(row["label_id"]))
        for i in row.get("inventory") or []:
            item_ids.add(int(i))
        for player in row.get("others") or []:
            cid = int(player.get("champion_id") or 0)
            if not cid:
                cid = abs(hash(player.get("champion") or "")) % 10000 + 1
            champ_ids.add(cid)
            for i in player.get("items") or []:
                item_ids.add(int(i))
        by_champ[row["champion"]][row["label_id"]] += 1
        global_counts[row["label_id"]] += 1
        for item_id in row.get("label_ids") or [row["label_id"]]:
            if item_id:
                label_counts[int(item_id)] += 1
    print(f"scanned train {n_train}", flush=True)
    test: list[dict] = []
    for row in stream_rows(test_path):
        test.append({key: row.get(key) for key in SLIM_KEYS})
        for player in row.get("others") or []:
            if not player.get("champion_id"):
                champ_ids.add(abs(hash(player.get("champion") or "")) % 10000 + 1)
    print(f"train {n_train} shops  test {len(test)} shops", flush=True)

    base = baseline_guesses(by_champ, global_counts, test)
    b1, b3 = score_guesses(test, base)
    print(flush=True)
    print("baseline  (champion frequency only)", flush=True)
    print(f"  top-1  {b1:.2f}", flush=True)
    print(f"  top-3  {b3:.2f}", flush=True)
    per_decision(test, base)

    champ_index = _index_map(champ_ids)
    item_index = _index_map(item_ids)
    label_ids = sorted(label_counts)
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}
    counts = torch.zeros(len(label_ids))
    for item_id, c in label_counts.items():
        counts[label_index[item_id]] = c
    # Low cap: trustworthy confidence over rare-item recall (owner's framing —
    # inflated rare-item probabilities "boost great matchup choices but also
    # highlight mistakes").
    pos_weight = ((n_train - counts) / counts.clamp(min=1.0)).clamp(max=POS_WEIGHT_CAP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dragon = default_dragon()
    print(flush=True)
    print(
        f"Board model  basket (all buys)  9 others x {MAX_ITEMS} slots  device={device}  "
        f"d={D_MODEL}x{N_LAYERS}L ff={FF_DIM} heads={N_HEADS}  "
        f"epochs={EPOCHS} extras={'on' if USE_EXTRAS else 'off'} cosine={'on' if USE_COSINE else 'off'} "
        f"history={'on' if USE_HISTORY else 'off'} goldest={'on' if USE_GOLDEST else 'off'} "
        f"runes={'on' if USE_RUNES else 'off'}",
        flush=True,
    )
    prep_started = time.monotonic()
    def _stream(path: Path):
        return fix_champ_ids(with_targets(with_history(stream_rows(path)), dragon), champ_index)

    train_ds = _dataset(
        "train", train_path, lambda: _stream(train_path), n_train,
        champ_index, item_index, label_index, dragon,
    )
    test_ds = _dataset(
        "test", test_path, lambda: _stream(test_path), len(test),
        champ_index, item_index, label_index, dragon,
    )
    print(f"tensors ready in {time.monotonic() - prep_started:.1f}s", flush=True)

    if device.type == "cuda":
        need = train_ds.nbytes() + test_ds.nbytes()
        free, _total = torch.cuda.mem_get_info()
        if need < free * 0.7:
            train_ds.to_device(device)
            test_ds.to_device(device)
            print(f"datasets resident on GPU ({need / 1e9:.2f} GB)", flush=True)
        else:
            print(f"datasets too big for GPU ({need / 1e9:.2f} GB), streaming from CPU", flush=True)

    model = PrefixModel(len(champ_index), len(item_index), len(label_ids), len(item_index) + 1).to(device)
    if INIT_FROM:
        blob0 = torch.load(ML_DIR / INIT_FROM, map_location="cpu", weights_only=False)
        missing0, unexpected0 = model.load_state_dict(blob0["state_dict"], strict=False)
        assert not unexpected0, unexpected0
        print(f"warm-started from {INIT_FROM} ({len(missing0)} fresh head tensors)", flush=True)
    if FREEZE_TRUNK:
        trainable_heads = ("save_head.", "target_head.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(trainable_heads)
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trunk frozen: training save+target heads only ({n_train_p} params)", flush=True)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if USE_COSINE else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    count_loss_fn = nn.CrossEntropyLoss()
    save_loss_fn = nn.BCEWithLogitsLoss()  # unweighted on purpose: calibrated save prob
    target_loss_fn = nn.CrossEntropyLoss()
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = torch.zeros(1, device=device)  # accumulate on-device; .item() syncs per step
        n = 0
        for batch in train_ds.batches(BATCH, shuffle=True):
            opt.zero_grad()
            out = model(
                batch["champs"].to(device),
                batch["sides"].to(device),
                batch["items"].to(device),
                batch["query"].to(device),
                batch["inv"].to(device),
                batch["hist"].to(device),
            )
            logits = out["items"].masked_fill(~batch["legal"].to(device), -1e4)
            loss = loss_fn(logits, batch["targets"].to(device))
            counts = batch["counts"].to(device)
            bought = counts > 0
            if bool(bought.any()):
                loss = loss + COUNT_LOSS_W * count_loss_fn(
                    out["counts"][bought], (counts[bought] - 1).clamp(max=MAX_COPIES - 1)
                )
            loss = loss + SAVE_LOSS_W * save_loss_fn(
                out["save"].squeeze(1), batch["is_save"].to(device)
            )
            tgt = batch["target"].to(device)
            known = tgt >= 0
            if bool(known.any()):
                loss = loss + TARGET_LOSS_W * target_loss_fn(out["target"][known], tgt[known])
            loss.backward()
            opt.step()
            total += loss.detach() * len(batch["targets"])
            n += len(batch["targets"])
        if sched:
            sched.step()
        print(f"  epoch {epoch}/{EPOCHS}  loss {float(total) / max(n, 1):.3f}", flush=True)
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
    if TARGET_BLEND > 0:
        tmatrix = target_matrix(label_ids, dragon).to(device)
        guesses_b = predict_top3(model, test_ds, device, label_ids, blend=TARGET_BLEND, tmatrix=tmatrix)
        tb1, tb3 = score_guesses(test, guesses_b)
        print(flush=True)
        print(f"with target blend {TARGET_BLEND}  top-1 {tb1:.2f}  top-3 {tb3:.2f}", flush=True)
        per_decision(test, guesses_b)
    print(flush=True)
    per_champ(test, guesses)
    print(flush=True)
    tau = calibrate_threshold(model, test_ds, test, label_ids, dragon)
    baskets = predict_baskets(model, test_ds, device, label_ids, dragon, threshold=tau)
    score_pred_baskets(test, baskets, tau)

    out_path = ML_DIR / os.environ.get("PREFIX_OUT", "prefix_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "champ_index": champ_index,
            "item_index": item_index,
            "label_ids": label_ids,
            "config": {
                "d_model": D_MODEL,
                "layers": N_LAYERS,
                "ff_dim": FF_DIM,
                "heads": N_HEADS,
                "max_others": MAX_OTHERS,
                "max_items": MAX_ITEMS,
                "query_dim": QUERY_DIM,
                "epochs": EPOCHS,
                "extras": USE_EXTRAS,
                "cosine": USE_COSINE,
                "history": USE_HISTORY,
                "gold_est": USE_GOLDEST,
                "runes": USE_RUNES,
                "counts": True,  # multiset baskets: count head is trained
                "max_copies": MAX_COPIES,
                "pos_weight_cap": POS_WEIGHT_CAP,
                "save_head": True,  # calibrated buy-nothing binary (gate + display)
                "save_splice": SAVE_SPLICE,  # measured: keep item head's SAVE ranking
                "target_head": True,  # next-completed-final plan supervision
                "target_blend": TARGET_BLEND,
                "init_from": INIT_FROM or None,
                "frozen_trunk": FREEZE_TRUNK,
                "hist_len": HIST_LEN,
                "gold_drift": GOLD_DRIFT,
                "basket_threshold": tau,
            },
            "metrics": {"top1": m1, "top3": m3, "baseline_top1": b1, "baseline_top3": b3},
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_rows": n_train,
            "test_rows": len(test),
        },
        out_path,
    )
    print(flush=True)
    print(f"saved model -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
