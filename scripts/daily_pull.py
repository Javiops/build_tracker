"""Daily pipeline: pull KR+EUW Challenger/Grandmaster + pros, re-export, retrain.

Meant to run unattended from Windows Task Scheduler. Logs to data/daily_pull.log.
Uses a rolling 26h window (not local midnight) so a nightly run never misses
yesterday's games; already-stored matches are deduped by the DB. Match lists are
uncapped: every ranked game a player has in the window is pulled.

After ingest it re-exports the ML JSONL (scripts/baseline.py) and trains a
CANDIDATE model (scripts/train_prefix.py), saved as
data/ml/prefix_candidate_<YYYYMMDD>.pt. It deliberately does not touch the
served artifact: promoting a candidate over data/ml/prefix_model.pt is a manual
step, taken after scripts/eval_policy.py has scored it. Pass --no-train to only
pull.
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
    parser.add_argument(
        "--hours",
        type=int,
        default=0,
        help="Lookback window in hours (0 = auto: cover everything since the last completed run)",
    )
    parser.add_argument(
        "--max-games", type=int, default=0, help="Match-list depth per player (0 = all in window)"
    )
    parser.add_argument("--no-train", action="store_true", help="Only pull; skip export + training")
    parser.add_argument("--patch", default="16.17,16.18", help="Explicit transition patches; one match-list pass, DTO-filtered")
    parser.add_argument("--export-patch", help="Single reviewed training patch; omission keeps this run collection-only")
    return parser.parse_args()


def log(event: dict | str) -> None:
    msg = event if isinstance(event, str) else (event.get("message") or str(event))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def auto_window_hours() -> int:
    """Window that covers everything since the last completed run.

    The 04:30 task can fire late (PC off overnight + StartWhenAvailable) or a
    run can die to an expired key; anchoring on the start of the last run that
    reached the "pipeline finished" line keeps late runs lossless — a 401-abort
    never prints that line, so its games stay inside the next window.
    """
    try:
        with LOG_PATH.open("rb") as handle:
            handle.seek(max(0, LOG_PATH.stat().st_size - 512_000))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return WINDOW_HOURS
    anchor = None
    start = None
    for line in tail.splitlines():
        if "=== daily pull start" in line:
            start = line[1:20]
        elif "=== daily pipeline finished" in line and start:
            anchor = start
    if not anchor:
        return WINDOW_HOURS
    try:
        last = datetime.strptime(anchor, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return WINDOW_HOURS
    gap_h = (datetime.now() - last).total_seconds() / 3600
    return max(WINDOW_HOURS, min(int(gap_h) + 3, 24 * 7))


def run_stage(name: str, script: str, env: dict[str, str] | None = None, args: list[str] | None = None) -> bool:
    log(f"--- {name} start ---")
    proc = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / script), *(args or [])],
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
    from app.config import selected_patches
    patch = ",".join(selected_patches(args.patch))
    if args.export_patch and len(selected_patches(args.export_patch)) != 1:
        raise SystemExit("--export-patch requires exactly one reviewed patch")
    started = time.monotonic()
    hours = args.hours or auto_window_hours()
    start_time = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    depth = args.max_games if args.max_games > 0 else "all"
    log(f"=== daily pull start (patches {patch}, window {hours}h, {depth} games/player) ===")
    failures = 0

    for region in ("kr", "euw"):
        try:
            result = ingest_ladder(
                patch=patch,
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
        result = ingest_pros(patch=patch, progress=log)
        log(f"pros done: +{result.get('ingested', 0)} perspectives")
    except RiotError as exc:
        failures += 1
        log(f"pros FAILED: {exc}")
        if exc.status in (401, 403):
            return 1
    except Exception as exc:
        failures += 1
        log(f"pros FAILED: {exc}")

    if not args.no_train and not args.export_patch:
        log("Collection complete; export/training deferred until --export-patch is explicitly selected.")
    if not args.no_train and args.export_patch:
        if run_stage("export", "baseline.py", args=["--patch", args.export_patch]):
            # Two-stage recipe (model G, 2026-09-05): joint aux-head training
            # taxes the trunk 4-5pts at any loss weight, so train the trunk
            # clean, then graft the save/plan heads onto the frozen weights.
            base_env = {
                "PREFIX_COSINE": "1",
                "PREFIX_DMODEL": "128",
                "PREFIX_LAYERS": "4",
                "PREFIX_FF": "256",
                "PREFIX_HEADS": "8",
                "PREFIX_GOLDEST": "1",
                "PREFIX_RUNES": "1",
                # Budget/mask/affordability come from reconstruct.gold_est, the
                # causal pre-decision estimate (prequential-v2), so the training
                # budget sits closer to the exact gold live reads than the
                # 60s-stale frame value does. Whether that helps the deployed
                # policy is NOT yet established: the first measurement used a
                # leaky estimator and an unpaired comparison, and the honest
                # ablation (stale vs prequential, same decoder, frozen test)
                # has not been run. Keep it on so candidates are comparable,
                # and do not quote a gain until eval_policy.py shows one.
                "PREFIX_GOLDX": "1",
                # Per-label pos_weight caps by rarity (8 for frequent labels,
                # 24 for rare): beat the flat cap of 24 on save (+5) and on
                # every basket metric, costing ~2pts of component top-1.
                "PREFIX_POSW_SCHED": "8:24",
            }
            # The nightly produces a CANDIDATE, never the served model. An
            # unattended job that overwrites prefix_model.pt promotes whatever
            # last night's data happened to produce, with no evaluation gate and
            # no way to attribute a regression — and it silently reverted a
            # hand-deployed model twice. Promotion is a separate, explicit step
            # after scripts/eval_policy.py has scored the candidate.
            stamp = datetime.now().strftime("%Y%m%d")
            trunk_env = {
                **base_env,
                "PREFIX_EPOCHS": "16",
                "PREFIX_SAVEW": "0",
                "PREFIX_TARGETW": "0",
                "PREFIX_OUT": f"prefix_trunk_{stamp}.pt",
            }
            graft_env = {
                **base_env,
                "PREFIX_EPOCHS": "6",
                "PREFIX_SAVEW": "0.5",
                "PREFIX_TARGETW": "0.25",
                "PREFIX_INIT_FROM": f"prefix_trunk_{stamp}.pt",
                "PREFIX_FREEZE": "1",
                "PREFIX_OUT": f"prefix_candidate_{stamp}.pt",
            }
            if run_stage("train-trunk", "train_prefix.py", env=trunk_env):
                if run_stage("graft-heads", "train_prefix.py", env=graft_env):
                    log(
                        f"candidate ready: data/ml/prefix_candidate_{stamp}.pt — NOT deployed. "
                        "Score it (scripts/eval_policy.py --artifact ...) before promoting it "
                        "over prefix_model.pt."
                    )
                else:
                    failures += 1
            else:
                failures += 1
                log("trunk training failed — skipping head graft")
        else:
            failures += 1
            log("export failed — skipping training")

    minutes = (time.monotonic() - started) / 60
    log(f"=== daily pipeline finished in {minutes:.1f} min, {failures} stage failures ===")
    return 1 if failures else 0


if __name__ == "__main__":
    from app.collector_lock import CollectorLease
    with CollectorLease():
        raise SystemExit(main())
