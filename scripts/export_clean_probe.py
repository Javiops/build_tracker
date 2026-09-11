"""Export an isolated, newest-first causal probe without touching data/ml.

The primary recovery must finish before it can produce a release candidate.
This exporter exists solely to obtain early, disposable evidence from the first
fully replayed recent games.  It reads the first N IDs written by the current
newest-to-oldest causal checkpoint, requires every shop row to be
``prequential-v2``, and writes a temporal train/validation split under a
separate directory.  It creates no test split and its manifest permanently
marks its artifacts as ineligible for promotion.

    .venv\\Scripts\\python.exe scripts\\export_clean_probe.py --games 3000 \\
        --out-dir data\\ml\\probes\\causal_v2_newest_3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from app.backfill import REFRESH_DONE
from app.db import db
from app.ddragon import default_dragon
import baseline


PROBE_SCHEMA = "causal-newest-probe-v1"
TRAIN_FRAC = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_prefix(count: int) -> list[str]:
    if not REFRESH_DONE.exists():
        raise SystemExit(f"missing {REFRESH_DONE}; causal refresh has not started.")
    seen: set[str] = set()
    ordered: list[str] = []
    for match_id in REFRESH_DONE.read_text(encoding="utf-8").split():
        if match_id and match_id not in seen:
            ordered.append(match_id)
            seen.add(match_id)
        if len(ordered) >= count:
            return ordered
    raise SystemExit(f"need {count} causally refreshed games; checkpoint currently has {len(ordered)}.")


def assert_clean_shop_rows(match_ids: list[str]) -> None:
    """Fail closed if the frozen probe would contain even one old row."""
    with db() as conn:
        conn.execute("CREATE TEMP TABLE probe_games (match_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO probe_games(match_id) VALUES (?)", ((match_id,) for match_id in match_ids))
        bad = conn.execute(
            """
            SELECT p.match_id
            FROM probe_games AS p
            LEFT JOIN game_events AS e ON e.match_id = p.match_id AND e.type = 'shop'
            GROUP BY p.match_id
            HAVING COUNT(e.event_index) = 0
               OR SUM(CASE WHEN json_extract(e.payload_json, '$.gold_est_version') = 'prequential-v2'
                           THEN 1 ELSE 0 END) != COUNT(e.event_index)
            ORDER BY p.match_id
            LIMIT 8
            """
        ).fetchall()
    if bad:
        examples = ", ".join(row["match_id"] for row in bad)
        raise SystemExit(f"probe contains unclean/missing shop rows; first IDs: {examples}")


def temporal_train_val(created: dict[str, int]) -> dict[str, str]:
    groups: dict[int, list[str]] = defaultdict(list)
    for match_id, timestamp in created.items():
        groups[timestamp].append(match_id)
    target = len(created) * TRAIN_FRAC
    assignment: dict[str, str] = {}
    seen = 0
    for timestamp in sorted(groups):
        split = "train" if seen < target else "val"
        for match_id in groups[timestamp]:
            assignment[match_id] = split
        seen += len(groups[timestamp])
    return assignment


def short_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]


def main() -> None:
    from app.pipeline_guard import require_pipeline_clear
    require_pipeline_clear("probe export")
    args = parse_args()
    if args.games < 100:
        raise SystemExit("a probe needs at least 100 whole games.")
    selected = checkpoint_prefix(args.games)
    assert_clean_shop_rows(selected)
    selected_set = set(selected)
    created = baseline.game_creation_map(selected_set)
    if set(created) != selected_set:
        missing = sorted(selected_set - set(created))[:5]
        raise SystemExit(f"probe games missing game_creation timestamps: {missing}")
    assignment = temporal_train_val(created)

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"probe output already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    temporary = out_dir / "visits_all.tmp.jsonl"
    gold_versions: Counter = Counter()
    rows_per_split: Counter = Counter()
    eligible_games: set[str] = set()
    dragon = baseline.dragon_for_patch(args.patch)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in baseline.iter_examples(dragon, only_match_ids=selected_set):
            version = row.get("gold_est_version")
            if version != "prequential-v2":
                raise SystemExit(f"unclean probe row {row.get('match_id')} with {version!r}")
            eligible_games.add(row["match_id"])
            gold_versions[version] += 1
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    handles = {
        split: (out_dir / f"visits_{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val")
    }
    try:
        with temporary.open(encoding="utf-8") as source:
            for line in source:
                match_id = json.loads(line)["match_id"]
                split = assignment[match_id]
                handles[split].write(line)
                rows_per_split[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
        temporary.unlink(missing_ok=True)

    if len(eligible_games) < args.games * 0.95:
        raise SystemExit(
            f"only {len(eligible_games)}/{args.games} selected games had complete eligible shop rows; probe is too thin."
        )
    manifest = {
        "schema": PROBE_SCHEMA,
        "probe": True,
        "promotion": "forbidden",
        "purpose": "early causal-gold diagnostic only; never a release benchmark",
        "selection": "first N IDs from newest-to-oldest causal refresh checkpoint",
        "requested_games": args.games,
        "eligible_games": len(eligible_games),
        "selected_match_id_sha256": short_sha(selected),
        "eligible_match_id_sha256": short_sha(sorted(eligible_games)),
        "eval_version": f"probe-causal-newest-{args.games}",
        "strategy": "whole-game temporal 80/20 train/validation inside the probe; no test partition",
        "counts": {
            "games": {split: sum(1 for value in assignment.values() if value == split) for split in ("train", "val")},
            "rows": dict(rows_per_split),
        },
        "gold_est_versions": dict(gold_versions),
        "reconstruction_versions": {baseline.RECONSTRUCTION_VERSION: sum(rows_per_split.values())},
        "patch": args.patch,
        "ddragon_version": dragon.version,
        "static_data_sha256": dragon.signature,
        "export_fingerprint": short_sha([f"{match_id}:{assignment[match_id]}" for match_id in selected]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "probe_match_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote causal probe to {out_dir}")


if __name__ == "__main__":
    main()
