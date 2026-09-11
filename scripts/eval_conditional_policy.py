"""Evaluate a train-only conditional action policy against the displayed-policy metric.

Unlike the old champion-frequency and canonical-item baselines, this table
predicts a complete multiset basket (or a no-buy action) conditioned on the
same compact state hierarchy: champion, role, inventory, causal/stale gold bin,
and minute. It is still intentionally limited -- no board, runes or matchup --
but it is a meaningful rival for a model whose UI displays baskets.

    .venv\\Scripts\\python.exe scripts\\eval_conditional_policy.py --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import SAVE_ITEM, basket_is_legal

import eval_conditional_baseline as cb  # noqa: E402
import train_prefix as tp  # noqa: E402

ML_DIR = Path(os.environ["PREFIX_ML_DIR"]).resolve() if os.environ.get("PREFIX_ML_DIR") else DATA_DIR / "ml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--confirm-final", action="store_true")
    return parser.parse_args()


def action_of(row: dict) -> tuple[int, ...]:
    """No-buy is distinct from ward-only; duplicates remain meaningful."""
    if row.get("save_kind") == "no_buy_death":
        return (SAVE_ITEM,)
    return tuple(sorted(int(item_id) for item_id in (row.get("label_ids") or []) if item_id))


def legal(action: tuple[int, ...], row: dict, dragon) -> bool:
    if action == (SAVE_ITEM,):
        return True
    budget = float(row.get("gold_est") if row.get("gold_est") is not None else row.get("gold") or 0)
    return basket_is_legal(action, row, budget, dragon)


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_final:
        raise SystemExit("--split test needs --confirm-final; select recipes on validation.")
    train_path = ML_DIR / "visits_train.jsonl"
    eval_path = ML_DIR / f"visits_{args.split}.jsonl"
    for path in (train_path, eval_path):
        if not path.exists():
            raise SystemExit(f"{path} missing — run scripts/baseline.py first.")
    dragon = default_dragon()

    # Keep the sparse level-1 policy honest: a singleton state bucket cannot
    # be treated as a robust lookup. Coarser levels are always available.
    support: Counter = Counter()
    for row in tp.stream_rows(train_path):
        support[cb.keys_for(row)[0]] += 1
    keep = {key for key, n in support.items() if n >= cb.MIN_SUPPORT_L1}

    tables: dict[tuple, Counter] = defaultdict(Counter)
    for row in tp.stream_rows(train_path):
        action = action_of(row)
        if not action:
            continue
        for key in cb.keys_for(row):
            if key[0] != "l1" or key in keep:
                tables[key][action] += 1

    total = exact_first = exact_any = 0
    purchase_n = no_buy_n = no_buy_recall = 0
    precision = recall = 0.0
    level_used: Counter = Counter()
    no_legal = 0
    for row in tp.stream_rows(eval_path):
        truth = action_of(row)
        if not truth:
            continue
        options: list[tuple[int, ...]] = []
        used = "none"
        for key in cb.keys_for(row):
            candidates = tables.get(key)
            if not candidates:
                continue
            options = [action for action, _n in candidates.most_common() if legal(action, row, dragon)][:3]
            if options:
                used = key[0]
                break
        if not options:
            no_legal += 1
            options = [tuple()]
        level_used[used] += 1
        first = options[0]
        total += 1
        exact_first += int(first == truth)
        exact_any += int(truth in options)
        if truth == (SAVE_ITEM,):
            no_buy_n += 1
            no_buy_recall += int(first == truth)
            continue
        purchase_n += 1
        actual = Counter(truth)
        predicted = Counter(first)
        hit = sum((actual & predicted).values())
        precision += hit / sum(predicted.values()) if predicted else 0.0
        recall += hit / sum(actual.values())

    result = {
        "split": args.split,
        "policy": "train-only conditional multiset backoff",
        "n": total,
        "first_action_exact": exact_first / max(total, 1),
        "any_of_three_exact": exact_any / max(total, 1),
        "first_basket_precision_given_purchase": precision / max(purchase_n, 1),
        "first_basket_recall_given_purchase": recall / max(purchase_n, 1),
        "no_buy_death_first_action_recall": no_buy_recall / max(no_buy_n, 1),
        "backoff_level_used": dict(level_used),
        "no_legal_action": no_legal,
        "note": "A behavioral imitation baseline, not a Bayes ceiling or quality oracle.",
    }
    print(json.dumps(result, indent=2))
    out = ML_DIR / f"conditional_policy_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
