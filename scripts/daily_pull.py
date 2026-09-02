"""Daily pipeline: pull KR+EUW Challenger/Grandmaster + pros, re-export, retrain.

Meant to run unattended from Windows Task Scheduler. Logs to data/daily_pull.log.
Uses a rolling 26h window (not local midnight) so a nightly run never misses
yesterday's games; already-stored matches are deduped by the DB. Match lists are
uncapped: every ranked game a player has in the window is pulled.

After ingest it re-exports the ML JSONL (scripts/baseline.py) and retrains the
board model (scripts/train_prefix.py), which saves data/ml/prefix_model.pt.
Pass --no-train to only pull.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.ladder import ingest_ladder
from app.pros import ingest_pros
from app.riot import RiotError

LOG_PATH = DATA_DIR / "daily_pull.log"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
WINDOW_HOURS = 26
TIERS = ("challenger", "grandmaster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily ladder + pros pull, export, retrain.")
    parser.add_argument("--hours", type=int, default=WINDOW_HOURS, help="Lookback window in hours")
    parser.add_argument(
        "--max-games", type=int, default=0, help="Match-list depth per player (0 = all in window)"
    )
    parser.add_argument("--no-train", action="store_true", help="Only pull; skip export + training")
    return parser.parse_args()


def log(event: dict | str) -> None:
    msg = event if isinstance(event, str) else (event.get("message") or str(event))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_stage(name: str, script: str, env: dict[str, str] | None = None) -> bool:
    log(f"--- {name} start ---")
    proc = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        env={**os.environ, **(env or {})},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"[{name}] {line}")
    code = proc.wait()
    log(f"--- {name} exited {code} ---")
    return code == 0


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    start_time = int((datetime.now(timezone.utc) - timedelta(hours=args.hours)).timestamp())
    depth = args.max_games if args.max_games > 0 else "all"
    log(f"=== daily pull start (window {args.hours}h, {depth} games/player) ===")
    failures = 0

    for region in ("kr", "euw"):
        try:
            result = ingest_ladder(
                region=region,
                size=0,
                tiers=TIERS,
                max_games_per_player=args.max_games,
                start_time=start_time,
                progress=log,
            )
            log(f"{region} ladder done: +{result['ingested']} perspectives")
        except RiotError as exc:
            failures += 1
            log(f"{region} ladder FAILED: {exc}")
            if exc.status in (401, 403):
                log("API key rejected — aborting remaining stages. Renew RIOT_API_KEY in .env.")
                return 1
        except Exception as exc:
            failures += 1
            log(f"{region} ladder FAILED: {exc}")

    try:
        result = ingest_pros(progress=log)
        log(f"pros done: +{result.get('ingested', 0)} perspectives")
    except RiotError as exc:
        failures += 1
        log(f"pros FAILED: {exc}")
        if exc.status in (401, 403):
            return 1
    except Exception as exc:
        failures += 1
        log(f"pros FAILED: {exc}")

    if not args.no_train:
        if run_stage("export", "baseline.py"):
            train_env = {"PREFIX_EPOCHS": "12", "PREFIX_COSINE": "1"}
            if not run_stage("train", "train_prefix.py", env=train_env):
                failures += 1
        else:
            failures += 1
            log("export failed — skipping training")

    minutes = (time.monotonic() - started) / 60
    log(f"=== daily pipeline finished in {minutes:.1f} min, {failures} stage failures ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
