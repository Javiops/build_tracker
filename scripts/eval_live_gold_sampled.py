"""Compare budget regimes on freshly reconstructed games.

Three budgets are fed to the same artifact:

  stale : the 60s-stale frame gold (the pre-2026-09-09 training budget)
  est   : reconstruct.gold_est — a CAUSAL OFFLINE ESTIMATE (prequential-v2).
          Close to what live reads, but not the same thing.
  f+s   : 'gold_arrival_true' (frame + the visit's own net spend) — label-
          derived and inflated ~+630g; kept only to tie back to the first,
          invalid measurement.

**This harness does not measure live exact-gold performance and no result from
it may be described that way.** Match-V5 has no ground-truth gold at an
arbitrary timestamp, so the "live regime" can only be approximated here. A real
live number needs telemetry from a running client (see HANDOFF.md).

It also scores predict_top3 against the canonical label, not the deployed
policy — use scripts/eval_policy.py for anything product-facing.

    .venv\\Scripts\\python.exe scripts\\eval_live_gold_sampled.py [--games 250]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.db import db
from app.ddragon import default_dragon
from app.ladder import routing_for_match_id
from app.reconstruct import reconstruct_game
from app.riot import RiotClient, RiotError
from app.shop_econ import inventory_state

sys.path.insert(0, str(ROOT / "scripts"))
import baseline  # noqa: E402
import train_prefix as tp  # noqa: E402

ML_DIR = DATA_DIR / "ml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--artifact", default="prefix_model.pt")
    return parser.parse_args()


def test_match_ids() -> list[str]:
    """Distinct match ids of the test split, streamed off the export."""
    ids: dict[str, None] = {}
    pat = re.compile(r'"match_id":\s*"([^"]+)"')
    with (ML_DIR / "visits_test.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            m = pat.search(line[:200])
            if m:
                ids.setdefault(m.group(1))
    return list(ids)


def with_variant(row: dict, gold: int, dragon) -> dict:
    """Force ONE budget into every channel the featurization can read.

    Both `gold` and `gold_est` must be set: a GOLDX artifact reads gold_est and
    ignores gold, a stale artifact does the opposite, so setting only one leaves
    the arm at the mercy of which artifact is being scored.
    """
    out = dict(row)
    out["gold"] = int(gold)
    out["gold_est"] = int(gold)
    out["gold_exact"] = True
    inventory = [int(i) for i in (row.get("inventory") or []) if i]
    out.update(inventory_state(inventory, int(gold), dragon))
    return out


def main() -> None:
    args = parse_args()
    import random

    ids = test_match_ids()
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    sample = ids[: args.games]
    print(f"sampling {len(sample)} of {len(ids)} TEST-split games", flush=True)

    dragon = default_dragon()
    perks_by_match: dict[str, dict] = {}
    with db() as conn:
        for match_id in sample:
            row = conn.execute(
                "SELECT participants_json FROM games WHERE match_id = ?", (match_id,)
            ).fetchone()
            if row:
                perks_by_match[match_id] = {
                    p["puuid"]: {
                        "keystone_id": p.get("keystone_id") or 0,
                        "sub_style": p.get("sub_style") or 0,
                        "summ1": p.get("summ1") or 0,
                        "summ2": p.get("summ2") or 0,
                    }
                    for p in json.loads(row["participants_json"])
                }

    client = RiotClient()
    rows: list[dict] = []  # baseline example + gold_est/gold_stale/gold_fs
    fetched = 0
    try:
        for k, match_id in enumerate(sample, start=1):
            regional, _platform = routing_for_match_id(match_id)
            try:
                match = client.match(regional, match_id)
                timeline = client.timeline(regional, match_id) if match else None
            except RiotError as exc:
                if exc.status in (401, 403):
                    raise
                continue
            if not match or not timeline:
                continue
            fetched += 1
            _game, events = reconstruct_game(match, timeline, dragon)
            for ev in events:
                if ev.get("type") != "shop":
                    continue
                ev["match_id"] = match_id
                ev["perks"] = (perks_by_match.get(match_id) or {}).get(ev.get("puuid"))
                example = baseline._example(ev, dragon)
                if not example:
                    continue
                example["_gold_est"] = ev.get("gold_est")
                example["_gold_fs"] = int(ev.get("gold") or 0)
                example["gold_est_version"] = ev.get("gold_est_version")
                rows.append(example)
            if k % 25 == 0:
                print(f"  …{k}/{len(sample)} games, {len(rows)} visits", flush=True)
    finally:
        client.close()
    print(f"{fetched} games, {len(rows)} visits", flush=True)

    blob = torch.load(ML_DIR / args.artifact, map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    # parity with production: the same function app/predictor.py calls. Do NOT
    # hand-copy flags here — this harness once dropped USE_GOLDX and would have
    # scored a GOLDX artifact in the regime it was not trained in.
    tp.apply_config(config)
    assert tp.USE_GOLDX == bool(config.get("gold_x", False)), "GOLDX flag not applied"
    assert tp.USE_RUNES == bool(config.get("runes", False)), "runes flag not applied"
    assert tp.QUERY_DIM == int(config.get("query_dim", tp.BASE_QUERY_DIM)), "query dim mismatch"
    print(
        f"featurization: gold_x={tp.USE_GOLDX} gold_est={tp.USE_GOLDEST} "
        f"runes={tp.USE_RUNES} query_dim={tp.QUERY_DIM}",
        flush=True,
    )
    champ_index = blob["champ_index"]
    item_index = blob["item_index"]
    label_ids = blob["label_ids"]
    label_index = {item_id: i for i, item_id in enumerate(label_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tp.PrefixModel(
        len(champ_index), len(item_index), len(label_ids), len(item_index) + 1,
        d_model=int(config.get("d_model", 64)), layers=int(config.get("layers", 2)),
        ff_dim=int(config.get("ff_dim", 128)), heads=int(config.get("heads", 4)),
        query_dim=tp.QUERY_DIM,
    ).to(device)
    _missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    assert not unexpected
    model.eval()

    def normalized(raw_rows):
        # same slimming stream_rows applies (others trimmed, label_ids ints)
        for raw in raw_rows:
            row = {key: raw.get(key) for key in (
                "match_id", "champion", "champion_id", "role", "team_id", "gold",
                "total_gold", "ally_obj", "enemy_obj", "level", "ts", "kills", "deaths",
                "inventory", "label_id", "label_ids", "can_complete", "n_completable",
                "cheapest_complete", "gold_after_complete", "n_inventory", "decision",
                "save_kind", "keystone_id", "sub_style", "summ1", "summ2", "gold_exact",
                # the gold fields and their version tag must survive slimming,
                # or a GOLDX artifact hits the version gate on every row
                "gold_est", "gold_est_version",
            )}
            row["others"] = [
                {
                    "champion": p.get("champion") or "",
                    "champion_id": p.get("champion_id"),
                    "team_id": p.get("team_id"),
                    "role": p.get("role") or "",
                    "level": p.get("level"),
                    "gold": p.get("gold"),
                    "items": (p.get("items") or [])[: tp.MAX_ITEMS],
                }
                for p in (raw.get("others") or [])[: tp.MAX_OTHERS]
            ]
            ids2 = raw.get("label_ids") or ([raw["label_id"]] if raw.get("label_id") else [])
            row["label_ids"] = [int(i) for i in ids2 if i]
            yield row

    slim = [{key: r.get(key) for key in tp.SLIM_KEYS} for r in rows]
    variants = [
        ("stale frame gold (pre-audit training budget)", lambda r: with_variant(r, int(r["gold"]), dragon)),
        (
            "causal offline gold estimate (NOT exact live gold)",
            lambda r: with_variant(r, int(r["_gold_est"] or r["gold"]), dragon),
        ),
        ("frame+spend (label-derived, invalid)", lambda r: with_variant(r, int(r["_gold_fs"]), dragon)),
    ]
    results = {}
    for name, fn in variants:
        ds = tp.ShopDataset.from_stream(
            normalized(fn(r) for r in rows), len(rows), champ_index, item_index, label_index, dragon
        )
        guesses = tp.predict_top3(model, ds, device, label_ids)
        del ds
        t1, t3 = tp.score_guesses(slim, guesses)
        print(flush=True)
        print(f"{name}: top-1 {t1:.3f}  top-3 {t3:.3f}", flush=True)
        tp.per_decision(slim, guesses)
        results[name] = (t1, t3)

    base_t1 = results[variants[0][0]][0]
    print(flush=True)
    for name, _fn in variants[1:]:
        print(f"delta vs stale-frame budget: {name}  top-1 {results[name][0] - base_t1:+.3f}", flush=True)
    print(flush=True)
    print(
        "Reminder: these are canonical-label top-k under three offline budgets. "
        "None of them is live performance, and none scores the deployed policy "
        "(scripts/eval_policy.py does).",
        flush=True,
    )


if __name__ == "__main__":
    main()
