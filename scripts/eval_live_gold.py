"""SUPERSEDED — use scripts/eval_live_gold_sampled.py instead.

Kept only to reproduce the first (misleading) gold measurement. Its "exact
arrival gold" variant reads the export's `gold_arrival_true`, which is NOT
arrival gold: it is pre-visit frame gold + the visit's own net spend, inflated
~+630g on average and derived from the label. Scoring against it overstates the
live handicap (it reported −19.4pts; the honest figure is −12.0pts). The
sampled harness re-reconstructs each game and uses reconstruct.gold_est, the
label-free forward walk. See HANDOFF.md ("The gold saga").

Original docstring follows.

Quantify the live 'free win' from exact arrival gold.

Scores one trained artifact twice on the test export: once featurized exactly
like training (pre-visit frame gold, up to 60s stale), once with the
reconstructed arrival gold in the budget feature + affordability state — the
regime live inference runs in, where the client reports gold exactly.

The arrival gold (export field gold_arrival_true) is label-derived offline
(leftover + net spend), so the engraved anti-leak rule bans it as a training
input; using it here is legitimate because nothing is trained — it only
simulates what the model would have seen live. The legality mask keeps the
usual gold + GOLD_DRIFT slack in both variants, mirroring app/predictor.py.

Run after an export that carries gold_arrival_true (baseline.py ≥ 2026-09-07):

    .venv\\Scripts\\python.exe scripts\\eval_live_gold.py [--limit 100000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.ddragon import default_dragon
from app.shop_econ import inventory_state

sys.path.insert(0, str(ROOT / "scripts"))
import train_prefix as tp  # noqa: E402

ML_DIR = DATA_DIR / "ml"
EXTRA = ("gold_arrival_true",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="prefix_model.pt", help="model file in data/ml")
    parser.add_argument("--limit", type=int, default=0, help="max test rows (0 = all)")
    return parser.parse_args()


def live_variant(row: dict, dragon) -> dict:
    gold = row.get("gold_arrival_true")
    if gold is None:
        raise SystemExit(
            "test export lacks gold_arrival_true — re-run scripts/baseline.py (2026-09-07+) first"
        )
    out = dict(row)
    out["gold"] = int(gold)
    out["gold_exact"] = True  # skip the income interpolation, like app/live.py rows
    inventory = [int(i) for i in (row.get("inventory") or []) if i]
    out.update(inventory_state(inventory, int(gold), dragon))
    return out


def main() -> None:
    args = parse_args()
    test_path = ML_DIR / "visits_test.jsonl"
    if not test_path.exists():
        raise SystemExit("Run scripts/baseline.py first so the test export exists.")

    blob = torch.load(ML_DIR / args.artifact, map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    # featurize exactly like the artifact was trained (same recipe as Predictor)
    tp.USE_EXTRAS = bool(config.get("extras", True))
    tp.USE_HISTORY = bool(config.get("history", False))
    tp.USE_GOLDEST = bool(config.get("gold_est", False))
    tp.USE_RUNES = bool(config.get("runes", False))
    tp.QUERY_DIM = int(config.get("query_dim", tp.BASE_QUERY_DIM))
    tp.SAVE_SPLICE = bool(config.get("save_splice", False))

    champ_index = blob["champ_index"]
    item_index = blob["item_index"]
    label_ids = blob["label_ids"]
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}
    dragon = default_dragon()

    slim: list[dict] = []
    for row in tp.stream_rows(test_path, extra_keys=EXTRA):
        slim.append({key: row.get(key) for key in tp.SLIM_KEYS})
        if args.limit and len(slim) >= args.limit:
            break
    n = len(slim)
    print(f"test rows {n}  artifact {args.artifact} (trained {blob.get('trained_at')})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tp.PrefixModel(
        len(champ_index),
        len(item_index),
        len(label_ids),
        len(item_index) + 1,
        d_model=int(config.get("d_model", 64)),
        layers=int(config.get("layers", 2)),
        ff_dim=int(config.get("ff_dim", 128)),
        heads=int(config.get("heads", 4)),
        query_dim=tp.QUERY_DIM,
    ).to(device)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    assert not unexpected, f"artifact has unknown weights: {unexpected}"
    model.eval()

    def rows(live: bool):
        k = 0
        for row in tp.stream_rows(test_path, extra_keys=EXTRA):
            yield live_variant(row, dragon) if live else row
            k += 1
            if args.limit and k >= args.limit:
                break

    results = {}
    for name, live in (("stale frame gold (training regime)", False), ("exact arrival gold (live regime)", True)):
        ds = tp.ShopDataset.from_stream(rows(live), n, champ_index, item_index, label_index, dragon)
        guesses = tp.predict_top3(model, ds, device, label_ids)
        del ds
        t1, t3 = tp.score_guesses(slim, guesses)
        print(flush=True)
        print(name, flush=True)
        print(f"  top-1  {t1:.3f}   top-3  {t3:.3f}", flush=True)
        tp.per_decision(slim, guesses)
        results[live] = (t1, t3)

    d1 = results[True][0] - results[False][0]
    d3 = results[True][1] - results[False][1]
    print(flush=True)
    print(f"live-gold uplift  top-1 {d1:+.3f}  top-3 {d3:+.3f}", flush=True)


if __name__ == "__main__":
    main()
