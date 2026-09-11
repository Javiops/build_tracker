"""A conditional, no-transformer baseline — and how plural the labels are.

The champion-frequency baseline (0.13 top-1) is a sanity check, not a rival: it
ignores inventory, gold, minute, role and legality, so beating it says only that
conditioning on state helps at all. This builds the baseline the model should
actually have to beat — a backoff table over the state a human would look at —
and measures how concentrated the labels are inside each state bucket.

`conditional_action_concentration` is deliberately NOT called a Bayes ceiling.
It measures agreement between Challenger players in similar states, in the
buckets this hierarchy happens to define; a finer state description would move
it. It bounds what a single-label metric can mean, not what the task allows.

    .venv\\Scripts\\python.exe scripts\\eval_conditional_baseline.py [--split val]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import components_of

sys.path.insert(0, str(ROOT / "scripts"))
import train_prefix as tp  # noqa: E402

ML_DIR = Path(os.environ["PREFIX_ML_DIR"]).resolve() if os.environ.get("PREFIX_ML_DIR") else DATA_DIR / "ml"
GOLD_BAND = 250
MINUTE_BAND = 2
MIN_SUPPORT_L1 = 5      # a level-1 bucket is only kept once it has this many visits
MIN_SUPPORT_CONC = 20   # concentration is only meaningful on well-populated buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--confirm-final", action="store_true")
    return parser.parse_args()


def keys_for(row: dict) -> list[tuple]:
    """Backoff hierarchy, most specific first."""
    champ = row.get("champion") or "?"
    role = row.get("role") or "?"
    inv = tuple(sorted(int(i) for i in (row.get("inventory") or []) if i))
    gold = int((row.get("gold_est") if row.get("gold_est") is not None else row.get("gold")) or 0)
    gold_band = gold // GOLD_BAND
    minute_band = int((row.get("ts") or 0) / 60000 / MINUTE_BAND)
    return [
        ("l1", champ, role, inv, gold_band, minute_band),
        ("l2", champ, role, inv),
        ("l3", champ, role),
        ("l4", champ),
        ("l5",),
    ]


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_final:
        raise SystemExit("--split test needs --confirm-final (see scripts/eval_policy.py)")
    train_path = ML_DIR / "visits_train.jsonl"
    eval_path = ML_DIR / f"visits_{args.split}.jsonl"
    for path in (train_path, eval_path):
        if not path.exists():
            raise SystemExit(f"{path} missing — run scripts/baseline.py first.")
    dragon = default_dragon()

    # Pass 1: which level-1 buckets have enough support to be worth storing.
    print("pass 1/3 — level-1 support…", flush=True)
    support: Counter = Counter()
    for row in tp.stream_rows(train_path):
        support[keys_for(row)[0]] += 1
    keep = {key for key, count in support.items() if count >= MIN_SUPPORT_L1}
    print(f"  {len(support)} level-1 buckets, {len(keep)} kept (>= {MIN_SUPPORT_L1} visits)", flush=True)
    del support

    # Pass 2: action distributions per bucket, train only.
    print("pass 2/3 — action tables…", flush=True)
    tables: dict[tuple, Counter] = defaultdict(Counter)
    n_train = 0
    for row in tp.stream_rows(train_path):
        n_train += 1
        label = int(row["label_id"])
        for key in keys_for(row):
            if key[0] == "l1" and key not in keep:
                continue
            tables[key][label] += 1
    print(f"  {n_train} train rows, {len(tables)} buckets", flush=True)

    # How plural are the labels inside a well-populated state bucket?
    # (value, weight) pairs, never parallel lists: buckets with a single action
    # have no defined normalised entropy and would silently shift every later
    # weight by one position.
    tops: list[tuple[float, int]] = []
    entropies: list[tuple[float, int]] = []
    distincts: list[int] = []
    for key, counts in tables.items():
        if key[0] != "l1":
            continue
        total = sum(counts.values())
        if total < MIN_SUPPORT_CONC:
            continue
        probs = [c / total for c in counts.values()]
        tops.append((max(probs), total))
        distincts.append(len(counts))
        if len(probs) > 1:
            ent = -sum(p * math.log(p) for p in probs) / math.log(len(probs))
            entropies.append((ent, total))

    def wmean(pairs: list[tuple[float, int]]) -> float:
        if not pairs:
            return float("nan")
        return sum(v * w for v, w in pairs) / max(sum(w for _v, w in pairs), 1)

    print(flush=True)
    print(f"conditional_action_concentration (level-1 buckets with >= {MIN_SUPPORT_CONC} visits)", flush=True)
    print(f"  buckets measured        {len(tops)}", flush=True)
    print(f"  modal-action share      {wmean(tops):.3f}   (visit-weighted)", flush=True)
    print(f"  distinct actions/bucket {sum(distincts) / max(len(distincts), 1):.1f}", flush=True)
    print(f"  normalised entropy      {wmean(entropies):.3f}   (0 = one action, 1 = uniform)", flush=True)
    print("  NOT a ceiling: a different state description moves these numbers.", flush=True)

    # Pass 3: score the backoff policy on the eval split.
    print(flush=True)
    print(f"pass 3/3 — scoring backoff baseline on {args.split}…", flush=True)
    finals: dict[int, set[int]] = {}

    def family_of(target_id: int) -> set[int]:
        if target_id not in finals:
            finals[target_id] = components_of(target_id, dragon) | {target_id}
        return finals[target_id]

    hits1 = hits3 = fam_hits = n = n_fam = 0
    level_used: Counter = Counter()
    for row in tp.with_targets(tp.stream_rows(eval_path), dragon):
        guess: list[int] = []
        used = "none"
        for key in keys_for(row):
            counts = tables.get(key)
            if not counts:
                continue
            used = key[0]
            guess = [item for item, _c in counts.most_common(3)]
            break
        n += 1
        level_used[used] += 1
        label = int(row["label_id"])
        if guess and guess[0] == label:
            hits1 += 1
        if label in guess:
            hits3 += 1
        target_id = int(row.get("target_id") or 0)
        if target_id and guess:
            n_fam += 1
            fam_hits += int(guess[0] in family_of(target_id))

    print(flush=True)
    print(f"conditional backoff baseline on {args.split} (n={n})", flush=True)
    print(f"  canonical-item top-1  {hits1 / max(n, 1):.3f}", flush=True)
    print(f"  canonical-item top-3  {hits3 / max(n, 1):.3f}", flush=True)
    if n_fam:
        print(f"  action-family match   {fam_hits / n_fam:.3f}  n={n_fam}", flush=True)
    print(f"  backoff level used    {dict(level_used)}", flush=True)

    out = ML_DIR / f"conditional_baseline_{args.split}.json"
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "n": n,
                "canonical_top1": hits1 / max(n, 1),
                "canonical_top3": hits3 / max(n, 1),
                "family_match": (fam_hits / n_fam) if n_fam else None,
                "level_used": dict(level_used),
                "conditional_action_concentration": {
                    "buckets": len(tops),
                    "modal_share": wmean(tops),
                    "distinct_per_bucket": sum(distincts) / max(len(distincts), 1),
                    "normalised_entropy": wmean(entropies),
                    "min_support": MIN_SUPPORT_CONC,
                    "note": "agreement between players in the same bucket; not a Bayes ceiling",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
