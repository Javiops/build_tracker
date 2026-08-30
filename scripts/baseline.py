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

SEED = 16
TRAIN_FRAC = 0.8
OUT_DIR = DATA_DIR / "ml"


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


def _example(event: dict, prefix: list[dict], dragon) -> dict | None:
    bought = _item_ids(event.get("bought") or [])
    label = _label(bought, dragon)
    if not label:
        return None
    label_id, label_name, label_completed = label
    return {
        "match_id": event.get("match_id"),
        "server": "euw" if str(event.get("match_id") or "").startswith("EUW") else "kr",
        "ts": event.get("ts") or 0,
        "champion": event.get("champion_name") or "",
        "champion_id": event.get("champion_id"),
        "role": event.get("team_position") or "",
        "team_id": event.get("team_id"),
        "gold": event.get("gold"),
        "level": event.get("level"),
        "cs": event.get("cs"),
        "kills": event.get("kills") or 0,
        "deaths": event.get("deaths") or 0,
        "assists": event.get("assists") or 0,
        "inventory": _item_ids(event.get("inventory_before") or []),
        "prefix": prefix,
        "label_id": label_id,
        "label_name": label_name,
        "label_completed": label_completed,
    }


def load_examples() -> list[dict]:
    dragon = default_dragon()
    examples: list[dict] = []
    current_id = None
    prefix: list[dict] = []
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
            if match_id != current_id:
                current_id = match_id
                prefix = []
            event = json.loads(row["payload_json"])
            event["match_id"] = match_id
            example = _example(event, list(prefix), dragon)
            if example:
                examples.append(example)
            bought = _item_ids(event.get("bought") or [])
            if bought:
                prefix.append(
                    {
                        "champion": event.get("champion_name") or "",
                        "champion_id": event.get("champion_id"),
                        "team_id": event.get("team_id"),
                        "bought": bought,
                    }
                )
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
    print(
        f"games {len(games)}  usable shops {len(examples)}  labels {len(labels)}  "
        f"completed {completed}/{len(examples)}"
    )
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
            f"  {sample['champion']} @ {sample['ts'] // 1000}s  gold={sample['gold']}  "
            f"inv={sample['inventory']}"
        )
        print(f"  prefix ({len(sample['prefix'])} earlier shops)")
        for step in sample["prefix"][-4:]:
            print(f"    {step['champion']} bought {step['bought']}")
        print(f"  label {sample['label_name']} ({sample['label_id']})")
    score_baseline(train, test)


if __name__ == "__main__":
    main()
