"""Fail unless every current full-lobby game completed the causal refresh.

`refresh_done.txt` is an operational checkpoint, not evidence by itself: it
must contain every currently stored game ID before an export can call the
corpus fully reconstructed.  This intentionally checks sets rather than only
counts, so an old/stale checkpoint cannot pass by coincidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backfill import REFRESH_DONE, all_game_ids


def main() -> None:
    if not REFRESH_DONE.exists():
        raise SystemExit(f"missing {REFRESH_DONE}; a full causal refresh has not run.")
    games = set(all_game_ids())
    done = set(REFRESH_DONE.read_text(encoding="utf-8").split())
    missing = sorted(games - done)
    stale = sorted(done - games)
    print(f"full-lobby games={len(games)}  refresh checkpoint={len(done)}")
    if stale:
        print(f"checkpoint-only IDs={len(stale)} (harmless historical entries)")
    if missing:
        print(f"MISSING causal refresh IDs={len(missing)}; first: {', '.join(missing[:5])}")
        raise SystemExit(2)
    print("ok causal refresh coverage: every current full-lobby game is checkpointed")


if __name__ == "__main__":
    main()
