"""Export shop rows from the narrator timeline and score a champion-frequency baseline.

Streaming discipline (same rule as train_prefix): the export outgrew the
machine's 15.8GB RAM as Python dicts, so no code path may hold every example at
once. Examples stream to a temp file while per-match stats accumulate, then a
second pass routes each line into the train/test split. Only counters, the
match-id set, and slim (champion, label) test rows stay in memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.db import db
from app.ddragon import default_dragon, dragon_for_patch
from app.reconstruct import RECONSTRUCTION_VERSION
from app.shop_econ import (
    GOLD_DRIFT,
    PINK,
    SAVE_ITEM,
    decision_kind,
    inventory_state,
    leftover_gold,
)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the rest — the newest games, and it stays frozen
OUT_DIR = DATA_DIR / "ml"
MANIFEST_PATH = OUT_DIR / "split_manifest.json"
ASSIGNMENT_PATH = OUT_DIR / "split_assignment.json"
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


def _unmodeled_regular_slots(items: list, dragon) -> int:
    """Count retained regular slots omitted by the model's item adapter.

    Replay payloads preserve one entry per owned copy. Consumables share their
    pinned item stack size, while hidden non-stack items still occupy a slot.
    A consume/consumeOnFull static flag is not evidence of actual consumption:
    only earlier replay events may remove an item from this input snapshot.
    """
    hidden = Counter()
    for item in items or []:
        if not isinstance(item, dict) or not item.get("skip"):
            continue
        item_id = int(item.get("item_id") or 0)
        if not item_id or 1200 <= item_id <= 1204:
            continue
        if "Trinket" in (dragon.item(item_id) or {}).get("tags", []):
            continue
        hidden[item_id] += 1
    slots = 0
    for item_id, copies in hidden.items():
        stack = max(1, int((dragon.item(item_id) or {}).get("stacks") or 1))
        slots += (copies + stack - 1) // stack
    return slots


def _label(bought: list[int], dragon) -> tuple[int, str, bool] | None:
    if not bought:
        return None
    completed = []
    rest = []
    for item_id in bought:
        info = dragon.classify(item_id)
        if info.get("target_skip", info.get("skip")):
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
                "bot_quest_complete": player.get("bot_quest_complete"),
                "slot_state_version": player.get("slot_state_version"),
                "gold": player.get("gold"),
                "level": player.get("level"),
                "items": items,
            }
        )
    return out


def _example(event: dict, dragon) -> dict | None:
    if event.get("training_eligible") is False:
        return None
    is_save = bool(event.get("is_save"))
    bought = _item_ids(event.get("label_bought", event.get("bought") or []))
    if is_save:
        label_id, label_name, label_completed = SAVE_ITEM, "SAVE", False
        save_kind = "no_buy_death"
    else:
        label = _label(bought, dragon)
        if not label:
            return None
        label_id, label_name, label_completed = label
        save_kind = "ward_only" if bought and all(i in PINK for i in bought) else "build_spend"
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
        "reconstruction_version": event.get("reconstruction_version"),
        "label_source_version": event.get("label_source_version"),
        "inventory_version": event.get("inventory_version"),
        "slot_state_version": event.get("slot_state_version"),
        "bot_quest_complete": event.get("bot_quest_complete"),
        "unmodeled_regular_slots": _unmodeled_regular_slots(event.get("inventory_before") or [], dragon),
        "patch": event.get("patch"),
        "ddragon_version": event.get("ddragon_version"),
        "server": "euw" if str(event.get("match_id") or "").startswith("EUW") else "kr",
        "ts": event.get("ts") or 0,
        "champion": event.get("champion_name") or "",
        "champion_id": event.get("champion_id"),
        "role": event.get("team_position") or "",
        "team_id": event.get("team_id"),
        "gold": gold,
        "gold_left": leftover_gold(event),
        # Frame + net spend — NOT arrival gold (inflated by spend − gap income,
        # ~+630 mean; proven 2026-09-08) and label-derived: EVAL/tie-back only,
        # never a model input (engraved anti-leak rule).
        "gold_arrival_true": int(event.get("gold") or 0),
        # Causal pre-decision gold estimate (reconstruct.gold_est). An
        # APPROXIMATION, not exact arrival gold. The version tag is what makes
        # it usable: rows reconstructed before 2026-09-09 carry the superseded
        # leaky-v1 walk and no tag, and the trainer refuses them under GOLDX.
        "gold_est": event.get("gold_est"),
        "gold_est_version": event.get("gold_est_version"),
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
        # decision_kind lumps "bought nothing" together with "bought only control
        # wards", which are different actions with different costs and different
        # advice. save_kind keeps them apart: only no_buy_death carries the SAVE
        # sentinel and trains the save head; ward_only keeps the wards as real
        # item labels. (Match-V5 cannot observe a manual recall with no purchase,
        # so genuine no-buys away from a death are simply invisible here.)
        "save_kind": save_kind,
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
        else [item_id for item_id in bought if not dragon.classify(item_id).get("target_skip")],
        "label_name": label_name,
        "label_completed": label_completed,
    }


def iter_examples(dragon, only_match_ids: set[str] | None = None, only_patch: str | None = None):
    """Stream eligible examples, optionally from an explicitly frozen game set.

    The normal exporter leaves ``only_match_ids`` unset.  A bounded clean probe
    can pass a causal-refresh checkpoint slice without ever exposing mixed
    reconstruction versions to its trainer.
    """
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
        if only_match_ids is not None:
            full.intersection_update(only_match_ids)
        if only_patch is not None:
            full.intersection_update(r["match_id"] for r in conn.execute(
                "SELECT match_id FROM games WHERE patch = ?", (only_patch,)))
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
            if event.get("reconstruction_version") != RECONSTRUCTION_VERSION:
                raise ValueError(
                    f"{match_id}: expected {RECONSTRUCTION_VERSION} decision context. "
                    "Repair from source before export; gold checkpoint coverage is insufficient."
                )
            if (event.get("patch") != ".".join(dragon.version.split(".")[:2])
                    or event.get("ddragon_version") != dragon.version
                    or event.get("static_data_sha256") != dragon.signature):
                raise ValueError(f"{match_id}: patch/static-data mismatch; refusing mixed or unbound export")
            event["match_id"] = match_id
            event["perks"] = (perks_by_match.get(match_id) or {}).get(event.get("puuid"))
            example = _example(event, dragon)
            if example:
                yield example


def game_creation_map(match_ids: set[str]) -> dict[str, int]:
    with db() as conn:
        return {
            row["match_id"]: int(row["game_creation"] or 0)
            for row in conn.execute("SELECT match_id, game_creation FROM games")
            if row["match_id"] in match_ids
        }


def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def temporal_split(created: dict[str, int]) -> dict[str, str]:
    """Assign whole games to train/val/test by game_creation, oldest first.

    Games sharing a creation timestamp always land in the same split, so a
    boundary can never cut a group of simultaneous games in half.
    """
    by_ts: dict[int, list[str]] = defaultdict(list)
    for match_id, ts in created.items():
        by_ts[ts].append(match_id)
    total = len(created)
    train_cut = total * TRAIN_FRAC
    val_cut = total * (TRAIN_FRAC + VAL_FRAC)
    assignment: dict[str, str] = {}
    seen = 0
    for ts in sorted(by_ts):
        group = by_ts[ts]
        split = "train" if seen < train_cut else ("val" if seen < val_cut else "test")
        for match_id in group:
            assignment[match_id] = split
        seen += len(group)
    return assignment


def load_or_build_split(created: dict[str, int], rebuild: bool) -> tuple[dict[str, str], dict]:
    """The split is FROZEN once written, and freezing it has a price.

    A frozen split only means something if it stays temporally sound. Games that
    arrive after the manifest was drawn are newer than the test cut, so putting
    them anywhere inside the split would let training see the future relative to
    the held-out set — keeping the test IDs unchanged does not rescue that. They
    are therefore parked in `post_cutoff` and written to no artifact at all.

    The consequence is deliberate and worth stating plainly: while a manifest is
    frozen, nightly ingest grows the corpus but NOT the trainable data. Turning
    fresh games into training data means redrawing the split with
    --rebuild-split, which starts a new evaluation version and makes every
    earlier benchmark incomparable. That is a decision to take on purpose, not a
    side effect of running the exporter.
    """
    prior: dict[str, str] = {}
    previous_manifest: dict = {}
    if MANIFEST_PATH.exists():
        try:
            previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}
    if ASSIGNMENT_PATH.exists() and not rebuild:
        prior = json.loads(ASSIGNMENT_PATH.read_text(encoding="utf-8"))

    eval_version = int(previous_manifest.get("eval_version") or 1)
    supersedes = None
    if not prior:
        assignment = temporal_split(created)
        deferred = 0
        if rebuild and previous_manifest:
            eval_version += 1
            supersedes = {
                "eval_version": previous_manifest.get("eval_version"),
                "test_id_sha256": (previous_manifest.get("id_sha256") or {}).get("test"),
                "created_at": previous_manifest.get("created_at"),
            }
            archived = OUT_DIR / f"split_manifest_v{previous_manifest.get('eval_version', 1)}.json"
            archived.write_text(json.dumps(previous_manifest, indent=2), encoding="utf-8")
            print()
            print("=" * 72)
            print(f"SPLIT REDRAWN — evaluation version {eval_version}")
            print("Results from earlier versions are NOT comparable with results from this")
            print(f"one: the test set is a different set of games. Previous manifest: {archived.name}")
            print("=" * 72)
            print()
        note = f"fresh temporal split (eval_version {eval_version})"
    else:
        assignment = {m: s for m, s in prior.items() if m in created}
        deferred = 0
        for match_id in created:
            if match_id not in assignment:
                assignment[match_id] = "post_cutoff"
                deferred += 1
        note = (
            f"frozen split reused; {deferred} newer games parked in post_cutoff "
            "(excluded from train/val/test)"
        )

    by_split: dict[str, list[str]] = defaultdict(list)
    for match_id, split in assignment.items():
        by_split[split].append(match_id)
    stamps = {split: [created[m] for m in ids if m in created] for split, ids in by_split.items()}
    in_split = [created[m] for m, s in assignment.items() if s != "post_cutoff" and m in created]
    manifest = {
        "eval_version": eval_version,
        "supersedes": supersedes,
        "strategy": "temporal by games.game_creation, whole games, ties kept together",
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1 - TRAIN_FRAC - VAL_FRAC, 4)},
        "frozen": bool(prior),
        "note": note,
        "cutoff_utc": (
            datetime.utcfromtimestamp(max(in_split) / 1000).isoformat() + "Z" if in_split else None
        ),
        "post_cutoff_count": len(by_split.get("post_cutoff", [])),
        "post_cutoff_policy": (
            "games newer than the frozen cutoff are written to NO artifact; putting them in "
            "train would make training see the future relative to val/test. Use "
            "--rebuild-split to bring them in, at the cost of a new evaluation version."
        ),
        "counts": {split: len(ids) for split, ids in sorted(by_split.items())},
        "id_sha256": {split: _sha(ids) for split, ids in sorted(by_split.items())},
        "creation_range_utc": {
            split: [
                datetime.utcfromtimestamp(min(ts) / 1000).isoformat() + "Z",
                datetime.utcfromtimestamp(max(ts) / 1000).isoformat() + "Z",
            ]
            for split, ts in sorted(stamps.items())
            if ts
        },
        "disjointness": "by match_id; NOT player-disjoint (the same player recurs across splits)",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return assignment, manifest


def route_split(tmp: Path, assignment: dict[str, str]) -> tuple[dict, Counter, list[dict], dict]:
    """Second pass: route the temp export into train/val/test files.

    Rows from games in `post_cutoff` (newer than the frozen split) are counted
    and dropped — they must not reach training or evaluation. Anything with no
    assignment at all is treated the same way rather than defaulting into train.

    Baseline counters come from train only; the slim scoring rows come from
    VALIDATION, because the champion-frequency baseline is a model-selection
    reference and must not be read off the final test set.
    """
    by_champ: dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    val_slim: list[dict] = []
    rows_per_split: Counter = Counter()
    paths = {name: OUT_DIR / f"visits_{name}.jsonl" for name in ("train", "val", "test")}
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        with tmp.open(encoding="utf-8") as src:
            for line in src:
                row = json.loads(line)
                split = assignment.get(row["match_id"], "post_cutoff")
                rows_per_split[split] += 1
                if split not in handles:
                    continue
                handles[split].write(line)
                if split == "train":
                    by_champ[row["champion"]][row["label_id"]] += 1
                    global_counts[row["label_id"]] += 1
                elif split == "val":
                    val_slim.append({"champion": row["champion"], "label_id": row["label_id"]})
    finally:
        for handle in handles.values():
            handle.close()
    return by_champ, global_counts, val_slim, dict(rows_per_split)


def score_baseline(by_champ: dict[str, Counter], global_counts: Counter, test: list[dict]) -> None:
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
    from app.pipeline_guard import require_pipeline_clear
    require_pipeline_clear("export")
    raise SystemExit('In-place exports are retired. Use scripts/publish_patch_dataset.py with a qualified coverage generation.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", required=True, help="One reviewed patch; exports never mix patches implicitly")
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Redraw the temporal split from scratch, starting a NEW evaluation "
        "version. Every earlier benchmark stops being comparable, so this is a "
        "deliberate act: the nightly pipeline never passes it. Use it when the "
        "post_cutoff backlog is large enough to be worth retraining on.",
    )
    args = parser.parse_args()

    print("Loading shop events…")
    dragon = dragon_for_patch(args.patch)
    if MANIFEST_PATH.exists() and not args.rebuild_split:
        prior_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if prior_manifest.get("patch") != args.patch:
            raise SystemExit("Changing export patch requires a deliberate --rebuild-split and new evaluation version")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "visits_all.tmp.jsonl"
    n = 0
    games: set[str] = set()
    labels: set[int] = set()
    completed = 0
    basket_items = 0
    decisions: Counter = Counter()
    save_kinds: Counter = Counter()
    gold_versions: Counter = Counter()
    reconstruction_versions: Counter = Counter()
    sample: dict | None = None
    with tmp.open("w", encoding="utf-8") as handle:
        for row in iter_examples(dragon, only_patch=args.patch):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            games.add(row["match_id"])
            labels.add(row["label_id"])
            completed += bool(row["label_completed"])
            decisions[row["decision"]] += 1
            save_kinds[row["save_kind"]] += 1
            gold_versions[row.get("gold_est_version") or "untagged(leaky-v1)"] += 1
            reconstruction_versions[row.get("reconstruction_version") or "unversioned"] += 1
            basket_items += len(row.get("label_ids") or [row["label_id"]])
            if n <= 13:
                sample = row
    print(f"games {len(games)}  usable shops {n}  labels {len(labels)}  completed {completed}/{n}")
    print(
        "decisions  "
        + "  ".join(f"{name} {decisions.get(name, 0)}" for name in ("complete", "component", "start", "save"))
    )
    print(f"basket  avg {basket_items / max(n, 1):.2f} items per shop")
    print(
        "save_kind  "
        + "  ".join(f"{name} {save_kinds.get(name, 0)}" for name in ("no_buy_death", "ward_only", "build_spend"))
    )
    created = game_creation_map(games)
    missing_ts = [m for m in games if m not in created]
    if missing_ts:
        print(f"WARNING: {len(missing_ts)} games have no creation timestamp; they go to train")
        for match_id in missing_ts:
            created[match_id] = 0
    assignment, manifest = load_or_build_split(created, rebuild=args.rebuild_split)
    by_champ, global_counts, val, rows_per_split = route_split(tmp, assignment)
    tmp.unlink()

    manifest["rows_per_split"] = rows_per_split
    manifest["gold_est_versions"] = dict(gold_versions)
    manifest["reconstruction_versions"] = dict(reconstruction_versions)
    manifest["patch"] = args.patch
    manifest["ddragon_version"] = dragon.version
    manifest["static_data_sha256"] = dragon.signature
    manifest["save_kinds"] = dict(save_kinds)
    manifest["export_fingerprint"] = _sha([f"{m}:{assignment[m]}" for m in assignment])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ASSIGNMENT_PATH.write_text(json.dumps(assignment), encoding="utf-8")

    print(f"split  {manifest['note']}")
    for name in ("train", "val", "test", "post_cutoff"):
        if name == "post_cutoff" and not manifest["counts"].get(name):
            continue
        suffix = "  (EXCLUDED from every artifact)" if name == "post_cutoff" else ""
        print(
            f"       {name:<11} {manifest['counts'].get(name, 0)} games / "
            f"{rows_per_split.get(name, 0)} shops{suffix}"
        )
    if manifest["post_cutoff_count"]:
        print(
            f"       cutoff {manifest['cutoff_utc']} — newer games are held out of training "
            "until you redraw the split (--rebuild-split, new eval version)"
        )
    print(f"gold_est versions in export: {dict(gold_versions)}")
    print(f"wrote {MANIFEST_PATH}")
    for name in ("train", "val", "test"):
        print(f"wrote {OUT_DIR / f'visits_{name}.jsonl'}")
    if sample:
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
    # The final test partition is never read during routine export or model
    # selection.  This diagnostic baseline is deliberately validation-only.
    score_baseline(by_champ, global_counts, val)


if __name__ == "__main__":
    main()
