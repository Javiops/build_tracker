"""Read-only source coverage and bounded replay benchmark. Never calls Riot."""
import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import DB_PATH, RAW_MATCH_V5_DIR, CACHE_DIR, PATCH_DATA_VERSIONS
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game
from app.riot import RawMatchArchive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-games", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "data/ml/repair_source_audit.json")
    args = parser.parse_args()
    if not 1 <= args.sample_games <= 50:
        raise SystemExit("sample-games must be between 1 and 50")
    conn = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA query_only=ON")
    games = conn.execute("SELECT match_id, patch FROM games ORDER BY match_id").fetchall()
    files = {kind: {p.name[:-8] for p in (RAW_MATCH_V5_DIR / kind).glob("*.json.gz")}
             for kind in ("match", "timeline")}
    counts, covered, missing = {}, [], []
    for mid, patch in games:
        entry = counts.setdefault(patch, {"games": 0, "both_files_present": 0, "missing_pair": 0})
        entry["games"] += 1
        if mid in files["match"] and mid in files["timeline"]:
            entry["both_files_present"] += 1
            covered.append((mid, patch))
        else:
            entry["missing_pair"] += 1
            missing.append(mid)
    random.Random(16).shuffle(covered)
    archive = RawMatchArchive(RAW_MATCH_V5_DIR)
    samples = []
    for mid, patch in covered[:args.sample_games]:
        version = PATCH_DATA_VERSIONS.get(patch)
        if not version or not all((CACHE_DIR / f"{kind}-{version}.json").exists() for kind in ("item", "champion")):
            samples.append({"match_id": mid, "status": "reviewed static data not locally available"})
            continue  # absolutely no network fallback in this audit
        match, timeline = archive.read("match", mid), archive.read("timeline", mid)
        if not match or not timeline or match.get("metadata", {}).get("matchId") != mid or timeline.get("metadata", {}).get("matchId") != mid:
            samples.append({"match_id": mid, "status": "invalid raw pair"})
            continue
        dragon = dragon_for_patch(patch)
        # Reject a DB/source patch discrepancy before core's resolver can load data.
        if ".".join(match["info"]["gameVersion"].split(".")[:2]) != patch:
            samples.append({"match_id": mid, "status": "DB/source patch mismatch"})
            continue
        start = time.perf_counter()
        _, fresh = reconstruct_game(match, timeline, dragon)
        elapsed = time.perf_counter() - start
        old = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload_json FROM game_events WHERE match_id=? AND type='shop'", (mid,))]
        old_by_key = {(e["participant_id"], e["ts"]): e for e in old}
        changed = Counter()
        matched = 0
        for e in fresh:
            previous = old_by_key.get((e["participant_id"], e["ts"]))
            if previous is None:
                continue
            matched += 1
            for key in ("gold_left", "gold_est", "level", "cs", "kills", "deaths", "assists", "score", "bought"):
                if json.dumps(previous.get(key), sort_keys=True) != json.dumps(e.get(key), sort_keys=True):
                    changed[key] += 1
        samples.append({"match_id": mid, "patch": patch, "status": "replayed",
                        "static_data_sha256": dragon.signature,
                        "reconstruction_seconds": elapsed, "old_visits": len(old),
                        "new_visits": len(fresh), "matched_visits": matched, "changed_fields": dict(changed)})
    conn.close()
    report = {"at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "live DB ID inventory; cache presence only, integrity checked on bounded sample; not final repair validation",
              "by_patch": counts, "missing_pair_ids": missing, "samples": samples,
              "network_requests": 0, "database_writes": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out), "by_patch": counts, "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
