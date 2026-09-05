"""Export shop rows from the narrator timeline and score a champion-frequency baseline."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.db import db
from app.ddragon import default_dragon
from app.shop_econ import GOLD_DRIFT, SAVE_ITEM, decision_kind, inventory_state, leftover_gold

SEED = 16
TRAIN_FRAC = 0.8
OUT_DIR = DATA_DIR / "ml"
# Remakes/aborts: no ranked game legitimately ends this early (unanimous FF
# unlocks at 15:00), but a remake where all 10 made their opening buy would
# otherwise pass the 10-shopper filter. Owner call (2026-09-04): exclude them.
MIN_DURATION_S = 600


def _item_ids(items: list) -> list[int]:
    out = []
    for item in items or []:
        if isinstance(item, dict):
            if item.get("skip"):
                continue
            item_id = item.get("item_id")
        else:
            item_id = item
        if item_id:
            out.append(int(item_id))
    return out


def _label(bought: list[int], dragon) -> tuple[int, str, bool] | None:
    if not bought:
        return None
    completed = []
    rest = []
    for item_id in bought:
        info = dragon.classify(item_id)
        if info.get("skip"):
            continue
        row = (info.get("gold") or 0, item_id, info["name"], bool(info.get("is_completed")))
        if row[3]:
            completed.append(row)
        else:
            rest.append(row)
    pool = completed or rest
    if not pool:
        return None
    _gold, item_id, name, is_completed = max(pool)
    return item_id, name, is_completed


def _others(event: dict) -> list[dict]:
    out = []
    for player in event.get("board") or []:
        if player.get("is_self"):
            continue
        items = _item_ids(player.get("items") or [])
        out.append(
            {
                "champion": player.get("champion_name") or "",
                "champion_id": player.get("champion_id"),
                "team_id": player.get("team_id"),
                "role": player.get("team_position") or "",
                "gold": player.get("gold"),
                "level": player.get("level"),
                "items": items,
            }
        )
    return out


def _example(event: dict, dragon) -> dict | None:
    is_save = bool(event.get("is_save"))
    bought = _item_ids(event.get("bought") or [])
    if is_save:
        label_id, label_name, label_completed = SAVE_ITEM, "SAVE", False
    else:
        label = _label(bought, dragon)
        if not label:
            return None
        label_id, label_name, label_completed = label
    # Pre-visit frame gold: snapshotted BEFORE the buys, so it cannot leak the
    # label. It understates arrival by up to ~60s of income, hence GOLD_DRIFT
    # in the affordability features.
    inventory = _item_ids(event.get("inventory_before") or [])
    gold = leftover_gold(event)
    state = inventory_state(inventory, gold + GOLD_DRIFT, dragon)
    team = event.get("team_id")
    self_row = next((p for p in event.get("board") or [] if p.get("is_self")), None) or {}
    score = event.get("score") or {}
    ally_obj = score.get(str(team)) or score.get(team) or {}
    enemy_key = 100 if team == 200 else 200
    enemy_obj = score.get(str(enemy_key)) or score.get(enemy_key) or {}
    return {
        "match_id": event.get("match_id"),
        "server": "euw" if str(event.get("match_id") or "").startswith("EUW") else "kr",
        "ts": event.get("ts") or 0,
        "champion": event.get("champion_name") or "",
        "champion_id": event.get("champion_id"),
        "role": event.get("team_position") or "",
        "team_id": event.get("team_id"),
        "gold": gold,
        "gold_left": leftover_gold(event),
        "total_gold": self_row.get("gold"),
        "ally_obj": ally_obj,
        "enemy_obj": enemy_obj,
        "level": event.get("level"),
        "cs": event.get("cs"),
        "kills": event.get("kills") or 0,
        "deaths": event.get("deaths") or 0,
        "assists": event.get("assists") or 0,
        "inventory": inventory,
        "can_complete": state["can_complete"],
        "n_completable": state["n_completable"],
        "cheapest_complete": state["cheapest_complete"],
        "gold_after_complete": state["gold_after_complete"],
        "n_inventory": state["n_inventory"],
        "decision": "save" if is_save else decision_kind(bought, inventory, dragon),
        "keystone_id": (event.get("perks") or {}).get("keystone_id") or 0,
        "sub_style": (event.get("perks") or {}).get("sub_style") or 0,
        "summ1": (event.get("perks") or {}).get("summ1") or 0,
        "summ2": (event.get("perks") or {}).get("summ2") or 0,
        "others": _others(event),
        "label_id": label_id,
        # Multiset: duplicates preserved (double Long Sword is a real, distinct
        # decision — ~7% of buying visits repeat an item). Order kept as bought.
        "label_ids": [SAVE_ITEM]
        if is_save
        else [item_id for item_id in bought if not dragon.classify(item_id).get("skip")],
        "label_name": label_name,
        "label_completed": label_completed,
    }


def load_examples() -> list[dict]:
    dragon = default_dragon()
    examples: list[dict] = []
    with db() as conn:
        full = {
            row["match_id"]
            for row in conn.execute(
                """
                SELECT match_id
                FROM game_events
                WHERE type = 'shop'
                GROUP BY match_id
                HAVING COUNT(DISTINCT json_extract(payload_json, '$.puuid')) = 10
                """
            )
        }
        # Shopper runes/summoner spells, backfilled into games.participants_json
        # by scripts/backfill_runes.py (zeros for games not yet backfilled).
        # Remakes (short games) are dropped from the eligible set here.
        perks_by_match: dict[str, dict[str, dict]] = {}
        for row in conn.execute("SELECT match_id, participants_json, game_duration FROM games"):
            if row["match_id"] not in full:
                continue
            if (row["game_duration"] or 0) < MIN_DURATION_S:
                full.discard(row["match_id"])
                continue
            perks_by_match[row["match_id"]] = {
                p["puuid"]: {
                    "keystone_id": p.get("keystone_id") or 0,
                    "sub_style": p.get("sub_style") or 0,
                    "summ1": p.get("summ1") or 0,
                    "summ2": p.get("summ2") or 0,
                }
                for p in json.loads(row["participants_json"])
            }
        rows = conn.execute(
            """
            SELECT match_id, payload_json
            FROM game_events
            WHERE type = 'shop'
            ORDER BY match_id, event_index ASC, ts ASC
            """
        )
        for row in rows:
            match_id = row["match_id"]
            if match_id not in full:
                continue
            event = json.loads(row["payload_json"])
            event["match_id"] = match_id
            event["perks"] = (perks_by_match.get(match_id) or {}).get(event.get("puuid"))
            example = _example(event, dragon)
            if example:
                examples.append(example)
    return examples


def split_by_match(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    match_ids = sorted({row["match_id"] for row in examples})
    rng = random.Random(SEED)
    rng.shuffle(match_ids)
    cut = max(1, int(len(match_ids) * TRAIN_FRAC))
    train_ids = set(match_ids[:cut])
    train = [row for row in examples if row["match_id"] in train_ids]
    test = [row for row in examples if row["match_id"] not in train_ids]
    return train, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_baseline(train: list[dict], test: list[dict]) -> None:
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for row in train:
        by_champ[row["champion"]][row["label_id"]] += 1
        global_counts[row["label_id"]] += 1
    fallback = [item_id for item_id, _n in global_counts.most_common(3)]

    def topk(champion: str) -> list[int]:
        local = [item_id for item_id, _n in by_champ[champion].most_common(3)]
        if len(local) >= 3:
            return local[:3]
        extra = [item_id for item_id in fallback if item_id not in local]
        return (local + extra)[:3]

    hits1 = hits3 = 0
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in test:
        guess = topk(row["champion"])
        champ = row["champion"] or "?"
        per[champ][2] += 1
        if guess and row["label_id"] == guess[0]:
            hits1 += 1
            per[champ][0] += 1
        if row["label_id"] in guess:
            hits3 += 1
            per[champ][1] += 1

    n = len(test)
    print()
    print("baseline  (most common completed/buy item for this champion)")
    print(f"  top-1  {hits1 / n:.2f}" if n else "  top-1  n/a")
    print(f"  top-3  {hits3 / n:.2f}" if n else "  top-3  n/a")
    print()
    ranked = sorted(per.items(), key=lambda kv: -kv[1][2])[:8]
    for champ, (c1, c3, count) in ranked:
        print(f"  {champ:<14} top-1 {c1 / count:.2f}  top-3 {c3 / count:.2f}   n={count}")


def main() -> None:
    print("Loading shop events…")
    examples = load_examples()
    labels = {row["label_id"] for row in examples}
    completed = sum(1 for row in examples if row["label_completed"])
    games = {row["match_id"] for row in examples}
    decisions = Counter(row["decision"] for row in examples)
    print(
        f"games {len(games)}  usable shops {len(examples)}  labels {len(labels)}  "
        f"completed {completed}/{len(examples)}"
    )
    basket = sum(len(row.get("label_ids") or [row["label_id"]]) for row in examples) / max(len(examples), 1)
    print(
        "decisions  "
        + "  ".join(f"{name} {decisions.get(name, 0)}" for name in ("complete", "component", "start", "save"))
    )
    print(f"basket  avg {basket:.2f} items per shop")
    train, test = split_by_match(examples)
    train_games = {row["match_id"] for row in train}
    test_games = {row["match_id"] for row in test}
    print(f"split  train {len(train_games)} games / {len(train)} shops")
    print(f"       test  {len(test_games)} games / {len(test)} shops")
    write_jsonl(OUT_DIR / "visits_train.jsonl", train)
    write_jsonl(OUT_DIR / "visits_test.jsonl", test)
    print(f"wrote {OUT_DIR / 'visits_train.jsonl'}")
    print(f"wrote {OUT_DIR / 'visits_test.jsonl'}")
    if examples:
        sample = examples[min(12, len(examples) - 1)]
        print()
        print("sample row")
        print(
            f"  {sample['champion']} @ {sample['ts'] // 1000}s  "
            f"gold_in={sample['gold']} leftover={sample.get('gold_left')}  "
            f"can_complete={sample['can_complete']}  inv={sample['inventory']}"
        )
        print(f"  others ({len(sample.get('others') or [])} players)")
        for other in (sample.get("others") or [])[:4]:
            print(f"    {other['champion']} items {other['items']}")
        print(
            f"  label {sample['label_name']} ({sample['label_id']})  "
            f"basket {sample.get('label_ids')}  decision {sample['decision']}"
        )
    score_baseline(train, test)


if __name__ == "__main__":
    main()
