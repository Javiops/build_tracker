"""Evaluate prospectively logged explicit shop sessions.

This scores what the beta user actually saw, with the exact client wallet at
query time and the subsequent observed inventory delta. It is not an outcome or
Challenger-optimality metric; it is the minimum honest evidence for whether the
displayed policy corresponds to a real, actionable decision.

    .venv\\Scripts\\python.exe scripts\\eval_shop_telemetry.py
    .venv\\Scripts\\python.exe scripts\\eval_shop_telemetry.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.telemetry import TELEMETRY_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=TELEMETRY_DIR)
    parser.add_argument("--strict", action="store_true", help="exit nonzero on an impossible displayed basket")
    return parser.parse_args()


def exact(a: Counter, b: Counter) -> float:
    return float(a == b)


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    for path in sorted(args.dir.glob("shop_sessions_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid telemetry JSON at {path}:{line_no}: {exc}") from exc
    if not records:
        raise SystemExit(f"no telemetry records in {args.dir}; start explicit shop sessions first.")

    by_session: dict[str, dict[str, dict]] = {}
    for record in records:
        session_id = record.get("session_id")
        if not session_id:
            continue
        by_session.setdefault(session_id, {})[record.get("event") or "unknown"] = record

    resolved: list[tuple[dict, dict]] = []
    abandoned = unattributed = 0
    for pair in by_session.values():
        query = pair.get("shop_query")
        outcome = pair.get("purchase_observed") or pair.get("no_buy_timeout")
        if query and outcome:
            resolved.append((query, outcome))
        elif pair.get("inventory_changed_unattributed"):
            unattributed += 1
        elif pair.get("abandoned"):
            abandoned += 1

    if not resolved:
        raise SystemExit("telemetry has no resolved purchase/no-buy sessions yet.")

    impossible = 0
    top_exact = any_exact = no_buy_recall = no_buy_n = 0.0
    basket_precision = basket_recall = purchase_n = 0.0
    model_counts: Counter = Counter()
    for query, outcome in resolved:
        model_counts[(query.get("model") or {}).get("artifact_sha256") or "unknown"] += 1
        context = query.get("context") or {}
        options = query.get("options") or []
        wallet = int(context.get("gold_exact") or 0)
        for option in options:
            cost = sum(int(item.get("cost") or 0) for item in (option.get("items") or []))
            if cost > wallet:
                impossible += 1
        first = options[0] if options else {"kind": "buy", "items": []}
        first_basket = Counter(int(item.get("item_id") or 0) for item in (first.get("items") or []))
        is_no_buy = outcome.get("event") == "no_buy_timeout"
        if is_no_buy:
            no_buy_n += 1
            no_buy_recall += float(first.get("kind") == "save")
            top_exact += float(first.get("kind") == "save")
            continue
        actual = Counter(int(item_id) for item_id in ((outcome.get("outcome") or {}).get("bought") or []))
        if not actual:
            continue
        purchase_n += 1
        top_exact += exact(first_basket, actual)
        hit = sum((first_basket & actual).values())
        basket_precision += hit / sum(first_basket.values()) if first_basket else 0.0
        basket_recall += hit / sum(actual.values())
        any_exact += float(
            any(Counter(int(item.get("item_id") or 0) for item in (option.get("items") or [])) == actual for option in options)
        )

    n = len(resolved)
    print(f"resolved explicit shop sessions  {n}")
    print(f"  purchases                     {int(purchase_n)}")
    print(f"  no-buy timeouts                {int(no_buy_n)}")
    print(f"  abandoned/superseded           {abandoned}")
    print(f"  unscored inventory changes     {unattributed}  (not assumed to be purchases)")
    print(f"  displayed baskets over wallet  {impossible}  (hard stop: must be 0)")
    print(f"  first-action exact             {top_exact / n:.3f}")
    if purchase_n:
        print(f"  first-basket precision         {basket_precision / purchase_n:.3f}")
        print(f"  first-basket recall            {basket_recall / purchase_n:.3f}")
        print(f"  any displayed basket exact     {any_exact / purchase_n:.3f}")
    if no_buy_n:
        print(f"  no-buy first-card recall       {no_buy_recall / no_buy_n:.3f}")
    print(f"  model fingerprints             {dict(model_counts)}")
    if impossible:
        print("HARD STOP: at least one explicitly displayed basket exceeded the exact query wallet.")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
