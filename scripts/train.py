"""Train a small tree model on the exported shop rows and compare it to the baseline."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR

from sklearn.ensemble import RandomForestClassifier

ML_DIR = DATA_DIR / "ml"
INV_KEEP = 80
PREFIX_KEEP = 80
SEED = 16
ROLES = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "UTILITY": 5}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _topk_from_counts(counts: Counter, extra: list[int], k: int = 3) -> list[int]:
    local = [item_id for item_id, _n in counts.most_common(k)]
    extra_ids = [item_id for item_id in extra if item_id not in local]
    return (local + extra_ids)[:k]


def score_guesses(rows: list[dict], guesses: list[list[int]]) -> tuple[float, float]:
    hits1 = hits3 = 0
    for row, guess in zip(rows, guesses):
        if guess and row["label_id"] == guess[0]:
            hits1 += 1
        if row["label_id"] in guess:
            hits3 += 1
    n = len(rows) or 1
    return hits1 / n, hits3 / n


def baseline_guesses(train: list[dict], test: list[dict]) -> list[list[int]]:
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for row in train:
        by_champ[row["champion"]][row["label_id"]] += 1
        global_counts[row["label_id"]] += 1
    fallback = [item_id for item_id, _n in global_counts.most_common(3)]
    return [_topk_from_counts(by_champ[row["champion"]], fallback) for row in test]


def _item_vocab(train: list[dict], key: str, keep: int) -> list[int]:
    counts: Counter = Counter()
    if key == "inventory":
        for row in train:
            counts.update(row.get("inventory") or [])
    else:
        for row in train:
            for step in row.get("prefix") or []:
                counts.update(step.get("bought") or [])
    return [item_id for item_id, _n in counts.most_common(keep)]


def build_matrix(rows: list[dict], inv_vocab: list[int], prefix_vocab: list[int]) -> np.ndarray:
    inv_index = {item_id: i for i, item_id in enumerate(inv_vocab)}
    prefix_index = {item_id: i for i, item_id in enumerate(prefix_vocab)}
    width = 10 + len(inv_vocab) + len(prefix_vocab)
    matrix = np.zeros((len(rows), width), dtype=np.float32)
    for r, row in enumerate(rows):
        matrix[r, 0] = float(row.get("champion_id") or 0)
        matrix[r, 1] = float(ROLES.get(row.get("role") or "", 0))
        matrix[r, 2] = float(row.get("team_id") or 0)
        matrix[r, 3] = float(row.get("gold") or 0)
        matrix[r, 4] = float(row.get("level") or 0)
        matrix[r, 5] = float(row.get("cs") or 0)
        matrix[r, 6] = float(row.get("kills") or 0)
        matrix[r, 7] = float(row.get("deaths") or 0)
        matrix[r, 8] = float((row.get("ts") or 0) / 60000)
        prefix = row.get("prefix") or []
        inventory = row.get("inventory") or []
        matrix[r, 9] = float(len(prefix))
        for item_id in inventory:
            idx = inv_index.get(int(item_id))
            if idx is not None:
                matrix[r, 10 + idx] = 1.0
        team = row.get("team_id")
        for step in prefix:
            ally = 1.0 if step.get("team_id") == team else 0.5
            for item_id in step.get("bought") or []:
                idx = prefix_index.get(int(item_id))
                if idx is not None:
                    col = 10 + len(inv_vocab) + idx
                    matrix[r, col] = min(2.0, matrix[r, col] + ally)
    return matrix


def model_guesses(model, features: np.ndarray, classes: np.ndarray) -> list[list[int]]:
    proba = model.predict_proba(features)
    if proba.shape[1] < 3:
        top = np.argsort(proba, axis=1)[:, ::-1]
    else:
        part = np.argpartition(proba, -3, axis=1)[:, -3:]
        row_idx = np.arange(proba.shape[0])[:, None]
        order = np.argsort(proba[row_idx, part], axis=1)[:, ::-1]
        top = np.take_along_axis(part, order, axis=1)
    out = []
    for idxs in top[:, :3]:
        out.append([int(classes[i]) for i in idxs])
    return out


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
    print("Loading rows…")
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    print(f"train {len(train)} shops  test {len(test)} shops")

    base = baseline_guesses(train, test)
    b1, b3 = score_guesses(test, base)
    print()
    print("baseline  (champion frequency only)")
    print(f"  top-1  {b1:.2f}")
    print(f"  top-3  {b3:.2f}")

    inv_vocab = _item_vocab(train, "inventory", INV_KEEP)
    prefix_vocab = _item_vocab(train, "prefix", PREFIX_KEEP)
    print()
    print(f"Building features  inv={len(inv_vocab)}  prefix={len(prefix_vocab)}")
    x_train = build_matrix(train, inv_vocab, prefix_vocab)
    x_test = build_matrix(test, inv_vocab, prefix_vocab)
    y_train = np.array([row["label_id"] for row in train], dtype=np.int32)

    print("Training random forest…")
    started = time.monotonic()
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    print(f"trained in {time.monotonic() - started:.1f}s")

    guesses = model_guesses(model, x_test, model.classes_)
    m1, m3 = score_guesses(test, guesses)
    print()
    print("model  (champ + gold/time + inventory + earlier shops)")
    print(f"  top-1  {m1:.2f}   vs baseline {b1:.2f}")
    print(f"  top-3  {m3:.2f}   vs baseline {b3:.2f}")
    print()
    per_champ(test, guesses)


if __name__ == "__main__":
    main()
