"""Poll-to-poll stability of the live recommendation, decomposed by cause.

Replays stored live-client snapshots (data/live_snapshots, ~2s cadence, dumped
by `python -m app.live --dump`) through the deployed predictor and the real
hysteresis in app.main, then asks not just how often the advice changed but
WHY it could have.

Equal inventory and level is NOT an unchanged game state: gold, clock, the
other nine players' builds, KDA and objectives all keep moving, and a swap
caused by gold crossing an item's price is correct behaviour, not flicker.
Earlier reports called the whole 2.9% "flicker with identical state"; that
claim was wrong and this script no longer supports it. Four buckets are
reported instead:

  * inventory+level unchanged — the loose, previously-quoted condition;
  * of those, changes an affordability crossing can explain (an item in either
    recommendation became affordable or stopped being so);
  * of those, changes with a moved board/KDA/objective signature;
  * residual: everything discrete about the state is identical and the advice
    still moved. Only this last bucket is flicker.

Practice Tool snapshots do not license claims about ranked stability.

    .venv\\Scripts\\python.exe scripts\\eval_stability.py [--stride 5] [--limit 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.live import snapshot_to_row

SNAPSHOT_DIR = DATA_DIR / "live_snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="max snapshots (0 = all)")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth snapshot")
    parser.add_argument("--examples", type=int, default=10, help="flips to print")
    return parser.parse_args()


def lead_of(options: list[dict]) -> tuple | None:
    if not options:
        return None
    top = options[0]
    if top.get("kind") == "save":
        return ("save",)
    items = top.get("items") or []
    return ("buy", items[0]["item_id"]) if items else None


def basket_of(options: list[dict]) -> tuple:
    if not options or options[0].get("kind") == "save":
        return ("save",)
    return tuple(sorted(item["item_id"] for item in options[0].get("items") or []))


def option_items(options: list[dict]) -> set[int]:
    return {
        int(item["item_id"])
        for opt in options or []
        for item in opt.get("items") or []
    }


def board_signature(row: dict) -> tuple:
    """Everything discrete about the state except the shopper's own gold/clock."""
    others = tuple(
        (
            player.get("champion"),
            tuple(sorted(int(i) for i in (player.get("items") or []) if i)),
            player.get("level"),
        )
        for player in row.get("others") or []
    )
    return (
        tuple(sorted(int(i) for i in row.get("inventory") or [])),
        row.get("level"),
        others,
        row.get("kills"),
        row.get("deaths"),
        tuple(sorted((row.get("ally_obj") or {}).items())),
        tuple(sorted((row.get("enemy_obj") or {}).items())),
    )


def affordability_moved(prev_row: dict, row: dict, items: set[int], dragon) -> bool:
    """Did any item in either recommendation cross its price between polls?

    Only the items actually on screen are tested: those are the ones whose
    affordability could have changed the advice.
    """
    from collections import Counter as _Counter

    from app.shop_econ import combine_cost

    for item_id in items:
        was = combine_cost(item_id, _Counter(int(i) for i in prev_row.get("inventory") or []), dragon)
        now = combine_cost(item_id, _Counter(int(i) for i in row.get("inventory") or []), dragon)
        if (was <= float(prev_row.get("gold") or 0)) != (now <= float(row.get("gold") or 0)):
            return True
    return False


def main() -> None:
    args = parse_args()
    from app.predictor import Predictor

    predictor = Predictor()
    files = sorted(SNAPSHOT_DIR.glob("snap_*.json"))[:: max(args.stride, 1)]
    if args.limit:
        files = files[: args.limit]
    if len(files) < 2:
        raise SystemExit(f"Need ≥2 snapshots in {SNAPSHOT_DIR} (run: python -m app.live --dump)")
    print(f"{len(files)} snapshots (stride {args.stride})", flush=True)

    # Pipe every poll through the deployed hysteresis too (the real main.py
    # code, not a copy) so raw-vs-shown churn comes from one replay.
    from app import main as appmain

    appmain._live_hold.update(sig=None, lead=None, options=None, pending=None, count=0)

    prev = None  # (loose_sig, lead, basket, options, row)
    prev_shown_lead = None
    pairs = loose_pairs = 0
    lead_flips = lead_flips_loose = basket_flips_loose = kind_flips_loose = 0
    shown_flips = shown_flips_loose = 0
    afford_explained = board_explained = gold_explained = residual = 0
    residual_pairs = 0
    printed = 0
    flip_examples: Counter = Counter()
    for k, path in enumerate(files):
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = snapshot_to_row(snap, predictor.dragon)
        if not row:
            continue
        options = predictor.predict_options(row, budget_slack=0)
        loose_sig = (tuple(sorted(int(i) for i in row["inventory"])), row["level"])
        lead = lead_of(options)
        basket = basket_of(options)
        shown_lead = lead_of(appmain._stable_options(row, options))
        if prev is not None:
            p_sig, p_lead, p_basket, p_options, p_row = prev
            pairs += 1
            if shown_lead != prev_shown_lead:
                shown_flips += 1
                if loose_sig == p_sig:
                    shown_flips_loose += 1
            if lead != p_lead:
                lead_flips += 1
            if loose_sig == p_sig:
                loose_pairs += 1
                # gold and clock still moved, so "unchanged" only ever meant
                # unchanged inventory and level
                same_board = board_signature(row) == board_signature(p_row)
                same_gold = float(row.get("gold") or 0) == float(p_row.get("gold") or 0)
                if same_board and same_gold:
                    residual_pairs += 1
                if lead != p_lead:
                    lead_flips_loose += 1
                    flip_examples[(p_lead, lead)] += 1
                    items = option_items(options) | option_items(p_options)
                    if affordability_moved(p_row, row, items, predictor.dragon):
                        afford_explained += 1
                    elif not same_board:
                        board_explained += 1
                    elif not same_gold:
                        # gold moved without crossing any on-screen price: the
                        # ranking shifted on the budget feature alone
                        gold_explained += 1
                    else:
                        residual += 1
                        if printed < args.examples:
                            printed += 1
                            print(
                                f"  residual flip @ {row['ts'] // 1000}s  {p_lead} -> {lead}  "
                                f"(gold {row['gold']}, discrete state identical)"
                            )
                if basket != p_basket:
                    basket_flips_loose += 1
                if (lead or ("?",))[0] != (p_lead or ("?",))[0]:
                    kind_flips_loose += 1
        prev = (loose_sig, lead, basket, options, row)
        prev_shown_lead = shown_lead
        if (k + 1) % 500 == 0:
            print(f"  …{k + 1}/{len(files)}", flush=True)

    print(flush=True)
    print(f"poll pairs {pairs}   inventory+level unchanged: {loose_pairs}", flush=True)
    print(f"   of those, fully identical discrete state: {residual_pairs}", flush=True)
    if pairs:
        print(flush=True)
        print(f"lead churn, all polls            {lead_flips / pairs:.3f}", flush=True)
        print(f"  as displayed (hysteresis)      {shown_flips / pairs:.3f}", flush=True)
    if loose_pairs:
        print(flush=True)
        print("with inventory+level unchanged (NOT 'unchanged game state')", flush=True)
        print(f"  lead churn                     {lead_flips_loose / loose_pairs:.3f}", flush=True)
        print(f"  as displayed (hysteresis)      {shown_flips_loose / loose_pairs:.3f}", flush=True)
        print(f"  basket churn                   {basket_flips_loose / loose_pairs:.3f}", flush=True)
        print(f"  buy<->save                     {kind_flips_loose / loose_pairs:.3f}", flush=True)
        print(flush=True)
        print(f"why those {lead_flips_loose} lead changes happened", flush=True)
        print(f"  an on-screen item crossed its price   {afford_explained}", flush=True)
        print(f"  the board / KDA / objectives moved    {board_explained}", flush=True)
        print(f"  gold moved, no on-screen crossing     {gold_explained}", flush=True)
        print(
            f"  nothing discrete changed (FLICKER)    {residual}"
            + (f"   = {residual / residual_pairs:.4f} of identical-state pairs" if residual_pairs else ""),
            flush=True,
        )
    if flip_examples:
        print(flush=True)
        print("most common lead changes (inventory+level unchanged):", flush=True)
        for (a, b), c in flip_examples.most_common(8):
            print(f"  {c:>5}  {a} -> {b}", flush=True)


if __name__ == "__main__":
    main()
